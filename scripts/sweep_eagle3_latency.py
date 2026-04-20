"""Sweep EAGLE3 latency across (topk, steps, budget).

Measures decomposed latency (target_forward, eagle3_draft, verify_overhead,
post_verify) for each (topk, steps, budget) combination. Uses oracle_patch.py
timing instrumentation. Supports resume (skips configs already in output).

Output: JSON with per-config measurements + summary table.

Usage:
    python3 scripts/sweep_eagle3_latency.py \
        --model Qwen/Qwen3-8B \
        --draft-model Tengyunw/qwen3_8b_eagle3 \
        --topks 2,4,8 \
        --steps 3,4,5 \
        --budgets 1,4,16,64,256 \
        --output results/qwen3_8b/eagle3_sweep3d.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI


PROMPT = "Write a detailed explanation of how quicksort algorithm works step by step."
TIMING_LOG = Path("/tmp/sglang_oracle_timing.jsonl")
TREE_LOG = Path("/tmp/sglang_oracle_vanilla.jsonl")


def wait_for_server(url: str, timeout: int = 600) -> bool:
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def measure_tpot(
    url: str, model: str,
    n_warmup: int = 2, n_measure: int = 5, max_tokens: int = 200,
) -> float:
    client = OpenAI(base_url=f"{url}/v1", api_key="dummy")
    for _ in range(n_warmup):
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=max_tokens, temperature=0.0,
        )
    tpots = []
    for i in range(n_measure):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=max_tokens, temperature=0.0,
        )
        elapsed = time.perf_counter() - t0
        n_gen = resp.usage.completion_tokens
        if n_gen > 0:
            tpots.append((elapsed / n_gen) * 1000)
    return statistics.median(tpots) if tpots else 0.0


def kill_server(proc: subprocess.Popen):
    import psutil
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    proc.wait(timeout=30)


def read_timing_log(skip_warmup: int = 400) -> dict:
    """Read timing log and return median values for all timing fields.

    Dynamically includes any numeric field ending in `_ms`, so detail-patch
    fields (vd_outer_*, vd_inner_*) are automatically aggregated when present.
    Also flattens per-step `accept_lengths` / `committed_tokens` lists and
    aggregates mean/median across steps.
    """
    base_fields = [
        "eagle3_draft_ms", "target_forward_ms", "verify_total_ms",
        "verify_greedy_ms", "verify_overhead_ms", "step_total_ms",
        "post_verify_ms",
    ]
    data: dict[str, list[float]] = {f: [] for f in base_fields}
    discovered: set[str] = set()
    accept_flat: list[float] = []
    committed_flat: list[float] = []
    try:
        with open(TIMING_LOG) as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                # Flatten accept_lengths / committed_tokens (one entry per req
                # per step) for real-mode latency analysis
                for al in (e.get("accept_lengths") or []):
                    if isinstance(al, (int, float)):
                        accept_flat.append(float(al))
                for ct in (e.get("committed_tokens") or []):
                    if isinstance(ct, (int, float)):
                        committed_flat.append(float(ct))
                for k, v in e.items():
                    if not isinstance(v, (int, float)):
                        continue
                    if k in base_fields or k.startswith("vd_"):
                        if k not in data:
                            data[k] = []
                            discovered.add(k)
                        data[k].append(float(v))
    except FileNotFoundError:
        pass

    # Skip warmup on every series individually
    for k in list(data.keys()):
        vals = data[k]
        data[k] = vals[skip_warmup:] if len(vals) > skip_warmup else vals
    accept_flat_post = accept_flat[skip_warmup:] if len(accept_flat) > skip_warmup else accept_flat
    committed_flat_post = committed_flat[skip_warmup:] if len(committed_flat) > skip_warmup else committed_flat

    result = {}
    for k, vals in data.items():
        result[k] = statistics.median(vals) if vals else 0.0
    result["n_samples"] = min(
        len(data["eagle3_draft_ms"]), len(data["target_forward_ms"]))
    result["_detail_keys"] = sorted(discovered)

    # Real-accept statistics (latency-only mode)
    if accept_flat_post:
        result["accept_length_mean"] = statistics.mean(accept_flat_post)
        result["accept_length_median"] = statistics.median(accept_flat_post)
        result["accept_length_max"] = max(accept_flat_post)
    if committed_flat_post:
        result["committed_tokens_mean"] = statistics.mean(committed_flat_post)
        result["committed_tokens_median"] = statistics.median(committed_flat_post)
    return result


def max_tree_capacity(topk: int, steps: int) -> int:
    """Maximum tree nodes for given topk and steps."""
    total = 0
    level_size = topk
    for _ in range(steps):
        total += level_size
        level_size *= topk
    return total


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--topks", default="1,2,4,8,16",
                        help="Comma-separated topk (branching factor) values to sweep")
    parser.add_argument("--steps", default="1,2,3,4,5,6,7,8",
                        help="Comma-separated steps (tree depth) values to sweep")
    parser.add_argument("--budgets", default="1,2,4,8,16,32,64,128,256,512,1024",
                        help="Comma-separated num_draft_tokens to sweep")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-measure", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-args", nargs="*", default=[])
    args = parser.parse_args()

    topks = [int(x) for x in args.topks.split(",")]
    steps_list = [int(x) for x in args.steps.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    url = f"http://localhost:{args.port}"

    # Enumerate all (topk, steps, budget) combinations where budget ≤ capacity
    all_configs = []
    for topk in topks:
        for steps in steps_list:
            cap = max_tree_capacity(topk, steps)
            for B in budgets:
                if B <= cap:
                    all_configs.append((topk, steps, B, cap))
    print(f"Total configs to run: {len(all_configs)} "
          f"(topks={topks}, steps={steps_list}, budgets={budgets})",
          file=sys.stderr)

    # Resume: load previous results and skip already-done configs
    output_path = os.path.abspath(args.output)
    cached_results = {}
    cached_vanilla = None
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                prev = json.load(f)
            if prev.get("model") == args.model:
                cached_vanilla = prev.get("vanilla_tpot_ms")
                for r in prev.get("results", []):
                    key = (r["topk"], r["steps"], r["budget"])
                    cached_results[key] = r
                print(f"Resuming: vanilla={'cached' if cached_vanilla else 'missing'}, "
                      f"{len(cached_results)} configs already done.",
                      file=sys.stderr)
        except (json.JSONDecodeError, KeyError):
            pass

    # Install oracle patch (SGLANG_ORACLE_VANILLA=1 needed so install_hook
    # actually injects timing instrumentation into worker source files)
    hook_env = os.environ.copy()
    hook_env["SGLANG_ORACLE_VANILLA"] = "1"
    subprocess.run([sys.executable, "-m",
                    "hybrid_spec_decoding.sglang_integration.install_hook"],
                   env=hook_env, check=True)

    # --- Measure vanilla baseline (or use cached) ---
    if cached_vanilla is not None:
        vanilla_tpot = cached_vanilla
        print(f"Using cached vanilla TPOT: {vanilla_tpot:.2f} ms/tok\n", file=sys.stderr)
    else:
        print("=" * 70, file=sys.stderr)
        print("Measuring vanilla (no speculation)...", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        env = os.environ.copy()
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        vanilla_cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.model,
            "--tp-size", str(args.tp_size),
            "--mem-fraction-static", "0.85",
            "--disable-cuda-graph",
            "--host", "0.0.0.0", "--port", str(args.port),
        ] + args.extra_args

        vanilla_log = "/tmp/sglang_vanilla_server.log"
        vanilla_fh = open(vanilla_log, "w")
        proc = subprocess.Popen(vanilla_cmd, env=env,
                                stdout=vanilla_fh, stderr=vanilla_fh)
        if not wait_for_server(url):
            kill_server(proc)
            vanilla_fh.close()
            print("ERROR: Vanilla server failed. Log tail:", file=sys.stderr)
            try:
                with open(vanilla_log) as f:
                    lines = f.readlines()
                for line in lines[-50:]:
                    print(f"  {line}", end="", file=sys.stderr)
            except Exception:
                pass
            sys.exit(1)

        vanilla_fh.close()
        vanilla_tpot = measure_tpot(url, args.model, args.n_warmup,
                                    args.n_measure, args.max_tokens)
        print(f"  Vanilla TPOT: {vanilla_tpot:.2f} ms/tok\n", file=sys.stderr)
        kill_server(proc)
        time.sleep(5)

    # --- Sweep (topk, steps, budget) ---
    # Preserve previously-completed configs (in original order)
    results = [cached_results[(t, s, b)]
               for (t, s, b, _) in all_configs
               if (t, s, b) in cached_results]

    remaining = [cfg for cfg in all_configs
                 if (cfg[0], cfg[1], cfg[2]) not in cached_results]
    print(f"Remaining configs: {len(remaining)}/{len(all_configs)}\n",
          file=sys.stderr)

    def _save_output():
        out = {
            "model": args.model,
            "draft_model": args.draft_model,
            "topks": topks,
            "steps_list": steps_list,
            "budgets": budgets,
            "vanilla_tpot_ms": round(vanilla_tpot, 2),
            "results": results,
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

    for done, (topk, steps, B, cap) in enumerate(remaining, 1):
        print("=" * 70, file=sys.stderr)
        print(f"[{done}/{len(remaining)}] topk={topk}, steps={steps}, "
              f"budget={B} (cap={cap})", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        TIMING_LOG.unlink(missing_ok=True)
        TREE_LOG.unlink(missing_ok=True)

        env = os.environ.copy()
        # Activate the oracle-patch hook (needed to install our timing wrapper)
        env["SGLANG_ORACLE_VANILLA"] = "1"
        # Run REAL speculative decoding: no force-accept, no tree/p_t extraction.
        # Timing instrumentation is kept so step_total_ms / target_forward_ms /
        # verify_total_ms / eagle3_draft_ms + real accept_lengths are logged.
        env["SGLANG_LATENCY_ONLY"] = "1"
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.model,
            "--tp-size", str(args.tp_size),
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", args.draft_model,
            "--speculative-num-steps", str(steps),
            "--speculative-eagle-topk", str(topk),
            "--speculative-num-draft-tokens", str(B),
            "--mem-fraction-static", "0.85",
            "--disable-cuda-graph",
            "--host", "0.0.0.0", "--port", str(args.port),
        ] + args.extra_args

        log_fh = open("/tmp/sglang_sweep_server.log", "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh)

        if not wait_for_server(url):
            kill_server(proc)
            log_fh.close()
            print(f"  ERROR: Server failed. Log tail:", file=sys.stderr)
            try:
                with open("/tmp/sglang_sweep_server.log") as f:
                    lines = f.readlines()
                for line in lines[-30:]:
                    print(f"    {line}", end="", file=sys.stderr)
            except Exception:
                pass
            print(file=sys.stderr)
            results.append({
                "topk": topk, "steps": steps, "budget": B,
                "max_capacity": cap, "error": "server_failed",
            })
            _save_output()
            time.sleep(3)
            continue

        step_tpot = measure_tpot(url, args.model, args.n_warmup,
                                 args.n_measure, args.max_tokens)
        kill_server(proc)
        time.sleep(5)

        timing = read_timing_log(
            skip_warmup=args.n_warmup * args.max_tokens)

        # Save per-config tree log + raw timing log to a mounted path
        tree_out_dir = Path(output_path).parent / "trees"
        tree_out_dir.mkdir(parents=True, exist_ok=True)
        tree_out_path = tree_out_dir / f"topk{topk}_steps{steps}_budget{B}.jsonl"
        try:
            if TREE_LOG.exists():
                import shutil
                shutil.copyfile(TREE_LOG, tree_out_path)
        except Exception as e:
            print(f"  WARNING: Failed to save tree log: {e}", file=sys.stderr)

        timing_out_dir = Path(output_path).parent / "timing_logs"
        timing_out_dir.mkdir(parents=True, exist_ok=True)
        timing_out_path = timing_out_dir / f"topk{topk}_steps{steps}_budget{B}.jsonl"
        try:
            if TIMING_LOG.exists():
                import shutil
                shutil.copyfile(TIMING_LOG, timing_out_path)
        except Exception as e:
            print(f"  WARNING: Failed to save timing log: {e}", file=sys.stderr)

        entry = {
            "topk": topk,
            "steps": steps,
            "budget": B,
            "max_capacity": cap,
            "step_ms": round(step_tpot, 2),
            "target_forward_ms": round(timing["target_forward_ms"], 2),
            "eagle3_draft_ms": round(timing["eagle3_draft_ms"], 2),
            "verify_total_ms": round(timing["verify_total_ms"], 2),
            "verify_greedy_ms": round(timing["verify_greedy_ms"], 3),
            "verify_overhead_ms": round(timing["verify_overhead_ms"], 2),
            "step_total_ms": round(timing["step_total_ms"], 2),
            "post_verify_ms": round(timing["post_verify_ms"], 2),
            "overhead_ms": round(step_tpot - timing["target_forward_ms"] - timing["eagle3_draft_ms"], 2),
            "n_samples": timing["n_samples"],
        }
        # Carry over any detail-patch medians (vd_outer_*, vd_inner_*)
        for k in timing.get("_detail_keys", []):
            entry[k] = round(timing[k], 3)
        # Real-mode accept statistics (latency-only runs)
        for k in ("accept_length_mean", "accept_length_median", "accept_length_max",
                  "committed_tokens_mean", "committed_tokens_median"):
            if k in timing:
                entry[k] = round(timing[k], 3)
        results.append(entry)
        _save_output()  # Save after each successful config (resume safety)

        voh = timing["verify_overhead_ms"]
        pv = timing["post_verify_ms"]
        gms = timing["verify_greedy_ms"]
        print(f"  step={step_tpot:.2f}ms "
              f"(draft={timing['eagle3_draft_ms']:.2f} + "
              f"t_fwd={timing['target_forward_ms']:.2f} + "
              f"v_greedy={gms:.3f} + "
              f"v_overhead={voh:.2f} + "
              f"post_verify={pv:.2f}) "
              f"[{timing['n_samples']} samples]\n", file=sys.stderr)

    # Final save (also saved after each config)
    _save_output()

    # --- Summary table ---
    print("\n" + "=" * 150, file=sys.stderr)
    print("EAGLE3 LATENCY DECOMPOSITION (3D sweep)", file=sys.stderr)
    print("=" * 150, file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Vanilla TPOT: {vanilla_tpot:.2f} ms/tok", file=sys.stderr)
    print(file=sys.stderr)

    print(f"{'topk':>4} | {'steps':>5} | {'budget':>6} | {'cap':>7} | "
          f"{'step':>8} | {'draft':>8} | {'t_fwd':>8} | "
          f"{'v_greedy':>8} | {'v_ovhd':>8} | {'post_v':>8} | "
          f"{'samples':>7}",
          file=sys.stderr)
    print("-" * 150, file=sys.stderr)

    for r in results:
        if "error" in r:
            print(f"{r['topk']:>4} | {r['steps']:>5} | {r['budget']:>6} | "
                  f"{r.get('max_capacity', ''):>7} | "
                  f"{'ERROR':>8}", file=sys.stderr)
        else:
            print(f"{r['topk']:>4} | {r['steps']:>5} | {r['budget']:>6} | "
                  f"{r['max_capacity']:>7} | "
                  f"{r['step_ms']:>7.2f}ms | {r['eagle3_draft_ms']:>7.2f}ms | "
                  f"{r['target_forward_ms']:>7.2f}ms | "
                  f"{r['verify_greedy_ms']:>7.3f}ms | "
                  f"{r['verify_overhead_ms']:>7.2f}ms | "
                  f"{r['post_verify_ms']:>7.2f}ms | "
                  f"{r['n_samples']:>7}", file=sys.stderr)

    print("=" * 150, file=sys.stderr)
    print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
