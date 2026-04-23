"""Measure EAGLE3 target_cost_ms / draft_cost_ms in real speculative-decoding mode.

For each (budget, steps) pair: boot one SGLang server with EAGLE3 + oracle
LATENCY-only timing instrumentation, serve the 2 prompts (1 warmup + 1
measured) from each workload, aggregate per-step timings from the oracle
timing log, and emit one JSON row per (workload, budget, steps).

``target_cost_ms = median(target_forward_ms + verify_overhead_ms)``
``draft_cost_ms  = median(eagle3_draft_ms)``

The server is kept alive across all workloads for the same (budget, steps)
pair; it is restarted only when (budget, steps) changes. This is the
single biggest cost in the sweep, so server on/off is minimized.

Usage:
    python3 simulation/scripts/measure_eagle3_cost.py \\
        --model Qwen/Qwen3-8B \\
        --draft-model Tengyunw/qwen3_8b_eagle3 \\
        --workloads specbench,bfcl_v4 \\
        --budgets 4,16,32,64,128,256,512 \\
        --steps 2,4,6,8 \\
        --topks 4,8,16 \\
        --output results/latency/eagle3_cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from _workload_prompts import load_workload_prompts


TIMING_LOG = Path(
    os.environ.get("SGLANG_ORACLE_TIMING_LOG", "/tmp/sglang_oracle_timing.jsonl"))


def wait_for_server(url: str, timeout: int = 900) -> bool:
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_server(proc: subprocess.Popen):
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def max_tree_capacity(topk: int, steps: int) -> int:
    """Upper bound on the ``num_draft_tokens`` budget SGLang will accept for
    a given (topk, steps) EAGLE3 run.

    SGLang's ``organize_draft_results`` builds a score_pool of size
    ``topk + (steps-1) * topk²`` and then asks for ``topk(score_pool,
    num_draft_tokens - 1)``. If num_draft_tokens > score_pool + 1, torch.topk
    raises ``RuntimeError: selected index k out of range`` and the scheduler
    dies. So the actual budget ceiling is ``topk + (steps-1)*topk² + 1``.

    (The naive "total descendants" count ``sum_{i=1..steps} topk^i`` is much
    larger but isn't the limiting factor — SGLang never materializes the
    deeper levels into the score pool.)
    """
    return topk + (steps - 1) * (topk ** 2) + 1


def read_timing_window(
    log_path: Path,
    start_offset: int,
) -> list[dict]:
    """Read decode-phase entries from ``log_path`` starting at ``start_offset``."""
    entries: list[dict] = []
    try:
        with open(log_path, "rb") as f:
            f.seek(start_offset)
            tail = f.read().decode("utf-8", errors="ignore")
    except FileNotFoundError:
        return entries
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("phase", "decode") == "decode":
            entries.append(e)
    return entries


def log_offset() -> int:
    try:
        return TIMING_LOG.stat().st_size
    except FileNotFoundError:
        return 0


def summarize_entries(entries: list[dict]) -> dict:
    """Reduce a list of decode-phase entries to median latencies + accept stats."""
    tf = [e["target_forward_ms"] for e in entries if "target_forward_ms" in e]
    vt = [e["verify_total_ms"] for e in entries if "verify_total_ms" in e]
    draft = [e["eagle3_draft_ms"] for e in entries if "eagle3_draft_ms" in e]
    step = [e["step_total_ms"] for e in entries if "step_total_ms" in e]

    # target_cost per entry = target_forward + verify_overhead = verify_total.
    # Prefer verify_total_ms directly (arithmetically equivalent).
    target_costs = vt

    accept_flat: list[float] = []
    committed_flat: list[float] = []
    for e in entries:
        for a in (e.get("accept_lengths") or []):
            if isinstance(a, (int, float)):
                accept_flat.append(float(a))
        for c in (e.get("committed_tokens") or []):
            if isinstance(c, (int, float)):
                committed_flat.append(float(c))

    out: dict = {"n_samples": len(entries)}
    if target_costs:
        out["target_cost_ms"] = statistics.median(target_costs)
    if draft:
        out["draft_cost_ms"] = statistics.median(draft)
    if step:
        out["step_ms"] = statistics.median(step)
    if tf:
        out["target_forward_ms"] = statistics.median(tf)
    if accept_flat:
        out["accept_length_mean"] = statistics.mean(accept_flat)
        out["accept_length_median"] = statistics.median(accept_flat)
        out["accept_length_max"] = max(accept_flat)
    if committed_flat:
        out["committed_tokens_mean"] = statistics.mean(committed_flat)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Target model (single; sweep by rerunning)")
    parser.add_argument("--draft-model", required=True,
                        help="EAGLE3 draft-model path")
    parser.add_argument("--workloads", default="specbench,bfcl_v4,swebench")
    parser.add_argument("--budgets", default="4,16,32,64,128,256,512")
    parser.add_argument("--steps", default="2,4,6,8")
    parser.add_argument("--topk", type=int, default=None,
                        help="Single topk (legacy). Ignored if --topks is set.")
    parser.add_argument("--topks", default=None,
                        help="Comma-separated topk sweep (e.g. '4,8,16'). "
                             "Overrides --topk. Each row in the output "
                             "carries its own topk.")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Generation cap for both warmup and measurement calls")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-args", nargs="*", default=[])
    args = parser.parse_args()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    budgets = [int(b) for b in args.budgets.split(",")]
    steps_list = [int(s) for s in args.steps.split(",")]
    if args.topks:
        topks = [int(k) for k in args.topks.split(",") if k.strip()]
    elif args.topk is not None:
        topks = [int(args.topk)]
    else:
        topks = [16]
    url = f"http://localhost:{args.port}"

    # Load 2 prompts per workload
    workload_prompts: dict[str, list[dict]] = {}
    for w in workloads:
        ps = load_workload_prompts(w, n_samples=2)
        if len(ps) < 2:
            print(f"SKIP {w}: need 2 prompts, got {len(ps)}", file=sys.stderr)
            continue
        workload_prompts[w] = ps
    if not workload_prompts:
        print("ERROR: no workloads with enough prompts.", file=sys.stderr)
        sys.exit(1)

    # Resume from existing output if compatible
    output_path = Path(args.output).resolve()
    cached: dict[tuple, dict] = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                prev = json.load(f)
            if (prev.get("model") == args.model
                    and prev.get("draft_model") == args.draft_model):
                for r in prev.get("results", []):
                    if "error" in r:
                        continue
                    # Prefer per-row topk (new schema). Fall back to file-level
                    # topk (legacy) so old outputs can still be resumed under
                    # the same topks grid.
                    row_topk = r.get("topk", prev.get("topk"))
                    if row_topk is None:
                        continue
                    key = (int(row_topk), r["workload"], r["budget"], r["steps"])
                    cached[key] = r
                print(f"Resuming: {len(cached)} entries cached.", file=sys.stderr)
        except (json.JSONDecodeError, KeyError):
            pass

    # Enumerate all (topk, budget, steps) triples with capacity check
    tbs_triples: list[tuple[int, int, int]] = []
    for K in topks:
        for B in budgets:
            for S in steps_list:
                cap = max_tree_capacity(K, S)
                if B > cap:
                    print(f"SKIP capacity: budget={B} > max_capacity(topk={K},steps={S})={cap}",
                          file=sys.stderr)
                    continue
                tbs_triples.append((K, B, S))

    # Install hook (idempotent; ensures SGLang worker sources are patched)
    hook_env = os.environ.copy()
    hook_env["SGLANG_ORACLE_VANILLA"] = "1"
    subprocess.run(
        [sys.executable, "-m", "simulation.oracle.install_hook"],
        env=hook_env, check=True)

    results: list[dict] = []
    # Carry over cached entries in stable order (by input workload × triple)
    for w in workload_prompts:
        for K, B, S in tbs_triples:
            if (K, w, B, S) in cached:
                results.append(cached[(K, w, B, S)])

    def _save_output():
        out = {
            "model": args.model,
            "draft_model": args.draft_model,
            "topks": topks,
            "workloads": list(workload_prompts.keys()),
            "budgets": budgets,
            "steps": steps_list,
            "results": results,
        }
        # Keep "topk" key only when a single value is swept (legacy readers).
        if len(topks) == 1:
            out["topk"] = topks[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

    _save_output()  # initial save (carries cached entries)

    timing_dir = output_path.parent / "timing_logs"
    timing_dir.mkdir(parents=True, exist_ok=True)

    for pair_idx, (K, B, S) in enumerate(tbs_triples, 1):
        # Skip entire (K, B, S) triple if every workload is already cached
        all_cached = all((K, w, B, S) in cached for w in workload_prompts)
        if all_cached:
            print(f"[{pair_idx}/{len(tbs_triples)}] topk={K} budget={B} steps={S}: "
                  f"all workloads cached — server skip",
                  file=sys.stderr)
            continue

        print("=" * 72, file=sys.stderr)
        print(f"[{pair_idx}/{len(tbs_triples)}] topk={K} budget={B} steps={S} "
              f"cap={max_tree_capacity(K, S)}",
              file=sys.stderr)
        print("=" * 72, file=sys.stderr)

        env = os.environ.copy()
        env["SGLANG_ORACLE_VANILLA"] = "1"
        env["SGLANG_LATENCY_ONLY"] = "1"
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.model,
            "--tp-size", str(args.tp_size),
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", args.draft_model,
            "--speculative-num-steps", str(S),
            "--speculative-eagle-topk", str(K),
            "--speculative-num-draft-tokens", str(B),
            "--mem-fraction-static", str(args.mem_fraction_static),
            "--disable-cuda-graph",
            "--watchdog-timeout", "600",
            "--host", "0.0.0.0", "--port", str(args.port),
        ] + args.extra_args

        # Truncate the timing log for this (K, B, S) window
        try:
            TIMING_LOG.unlink(missing_ok=True)
        except TypeError:  # py<3.8 (unlikely here)
            if TIMING_LOG.exists():
                TIMING_LOG.unlink()

        # Include port so parallel runs (e.g. 8B on :30000 + 14B on :30001)
        # don't clobber each other's per-triple server logs under /tmp.
        log_path = Path(
            f"/tmp/sglang_eagle3_cost_p{args.port}_k{K}_b{B}_s{S}.log")
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh)
        try:
            if not wait_for_server(url):
                kill_server(proc)
                log_fh.close()
                print(f"ERROR: server failed. See {log_path}", file=sys.stderr)
                for w in workload_prompts:
                    if (K, w, B, S) not in cached:
                        results.append({
                            "workload": w, "topk": K, "budget": B, "steps": S,
                            "error": "server_failed",
                        })
                _save_output()
                continue
            log_fh.close()

            client = OpenAI(base_url=f"{url}/v1", api_key="dummy")

            for w, prompts in workload_prompts.items():
                if (K, w, B, S) in cached:
                    continue
                warmup_msgs = prompts[0]["messages"]
                measure_msgs = prompts[1]["messages"]

                # Warmup (result discarded, but fills JIT + KV cache)
                try:
                    client.chat.completions.create(
                        model=args.model, messages=warmup_msgs,
                        max_tokens=args.max_tokens, temperature=0.0)
                except Exception as e:
                    print(f"  WARN {w} warmup failed: {e}", file=sys.stderr)

                # Mark timing log offset right before the measured call
                start = log_offset()

                try:
                    client.chat.completions.create(
                        model=args.model, messages=measure_msgs,
                        max_tokens=args.max_tokens, temperature=0.0)
                except Exception as e:
                    print(f"  ERROR {w} measure failed: {e}", file=sys.stderr)
                    results.append({
                        "workload": w, "topk": K, "budget": B, "steps": S,
                        "error": f"chat_failed: {e}",
                    })
                    _save_output()
                    continue

                window = read_timing_window(TIMING_LOG, start)
                summ = summarize_entries(window)
                entry = {
                    "workload": w,
                    "topk": K,
                    "budget": B,
                    "steps": S,
                    "max_capacity": max_tree_capacity(K, S),
                }
                entry.update(summ)
                if "target_cost_ms" in summ and "draft_cost_ms" in summ and "step_ms" in summ:
                    entry["overhead_ms"] = round(
                        summ["step_ms"] - summ["target_cost_ms"] - summ["draft_cost_ms"],
                        3)
                results.append(entry)
                _save_output()
                print(f"  {w:10s}  n={summ.get('n_samples', 0):>3d}  "
                      f"target={summ.get('target_cost_ms', 0):>7.2f}ms  "
                      f"draft={summ.get('draft_cost_ms', 0):>7.2f}ms  "
                      f"accept_mean={summ.get('accept_length_mean', 0):>4.2f}",
                      file=sys.stderr)
        finally:
            kill_server(proc)
            # Copy per-config raw timing log for post-hoc analysis
            if TIMING_LOG.exists():
                try:
                    shutil.copyfile(TIMING_LOG, timing_dir / f"k{K}_b{B}_s{S}.jsonl")
                except Exception as e:
                    print(f"  WARN: timing log copy failed: {e}", file=sys.stderr)
            time.sleep(3)

    _save_output()
    print(f"\nOutput: {args.output}", file=sys.stderr)
    print(f"Raw timing logs: {timing_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
