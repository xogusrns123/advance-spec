"""Sweep EAGLE3 latency across topk, steps, and num_draft_tokens.

Measures decomposed latency (target_forward, eagle3_draft) for all
combinations of the specified parameter ranges. Uses oracle_patch.py
timing instrumentation.

Output: JSON with per-config measurements + summary table.

Usage:
    python3 simulation/scripts/sweep_eagle3_latency.py \
        --model Qwen/Qwen3-8B \
        --draft-model Tengyunw/qwen3_8b_eagle3 \
        --tp-size 1 \
        --topks 1,2,4,8 \
        --steps 1,2,3,4,5 \
        --budgets 4,8,16,32,64,128,256 \
        --output simulation/results/qwen3_8b/eagle3_sweep.json

    # Quick test:
    python3 simulation/scripts/sweep_eagle3_latency.py \
        --model Qwen/Qwen3-8B \
        --draft-model Tengyunw/qwen3_8b_eagle3 \
        --topks 4,8 --steps 3,5 --budgets 16,64 \
        --output simulation/results/qwen3_8b/eagle3_sweep_quick.json
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


def wait_for_server(url: str, timeout: int = 300) -> bool:
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


def read_timing_log(skip_warmup: int = 400) -> tuple:
    """Read timing log and return median (draft_ms, fwd_ms)."""
    draft_times, fwd_times = [], []
    try:
        with open(TIMING_LOG) as f:
            for line in f:
                e = json.loads(line.strip())
                if e.get("eagle3_draft_ms") is not None:
                    draft_times.append(e["eagle3_draft_ms"])
                if e.get("target_forward_ms") is not None:
                    fwd_times.append(e["target_forward_ms"])
    except FileNotFoundError:
        pass

    draft_times = draft_times[skip_warmup:] if len(draft_times) > skip_warmup else draft_times
    fwd_times = fwd_times[skip_warmup:] if len(fwd_times) > skip_warmup else fwd_times

    med_draft = statistics.median(draft_times) if draft_times else 0.0
    med_fwd = statistics.median(fwd_times) if fwd_times else 0.0
    n_samples = min(len(draft_times), len(fwd_times))
    return med_draft, med_fwd, n_samples


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
    parser.add_argument("--topks", default="1,2,4,8",
                        help="Comma-separated topk values to sweep")
    parser.add_argument("--steps", default="1,2,3,4,5",
                        help="Comma-separated steps values to sweep")
    parser.add_argument("--budgets", default="4,8,16,32,64,128,256",
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

    # Install oracle patch
    subprocess.run([sys.executable, "-m",
                    "simulation.oracle.install_hook"],
                   check=True)

    # --- Measure vanilla baseline ---
    print("=" * 70, file=sys.stderr)
    print("Measuring vanilla (no speculation)...", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    vanilla_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--tp-size", str(args.tp_size),
        "--mem-fraction-static", "0.8",
        "--disable-cuda-graph",
        "--host", "0.0.0.0", "--port", str(args.port),
    ] + args.extra_args

    proc = subprocess.Popen(vanilla_cmd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)
    if not wait_for_server(url):
        kill_server(proc)
        print("ERROR: Vanilla server failed", file=sys.stderr)
        sys.exit(1)

    vanilla_tpot = measure_tpot(url, args.model, args.n_warmup,
                                args.n_measure, args.max_tokens)
    print(f"  Vanilla TPOT: {vanilla_tpot:.2f} ms/tok\n", file=sys.stderr)
    kill_server(proc)
    time.sleep(5)

    # --- Sweep all combinations ---
    results = []
    total_configs = 0
    for topk in topks:
        for steps in steps_list:
            cap = max_tree_capacity(topk, steps)
            valid_budgets = [b for b in budgets if b <= cap]
            total_configs += len(valid_budgets)

    done = 0
    for topk in topks:
        for steps in steps_list:
            cap = max_tree_capacity(topk, steps)
            valid_budgets = [b for b in budgets if b <= cap]

            if not valid_budgets:
                print(f"SKIP topk={topk}, steps={steps}: "
                      f"max_capacity={cap} < min_budget={min(budgets)}",
                      file=sys.stderr)
                continue

            for B in valid_budgets:
                done += 1
                print("=" * 70, file=sys.stderr)
                print(f"[{done}/{total_configs}] topk={topk}, steps={steps}, "
                      f"budget={B} (cap={cap})", file=sys.stderr)
                print("=" * 70, file=sys.stderr)

                TIMING_LOG.unlink(missing_ok=True)

                env = os.environ.copy()
                env["SGLANG_ORACLE_VANILLA"] = "1"
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
                    "--mem-fraction-static", "0.8",
                    "--disable-cuda-graph",
                    "--host", "0.0.0.0", "--port", str(args.port),
                ] + args.extra_args

                log_fh = open("/tmp/sglang_sweep_server.log", "w")
                proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh)

                if not wait_for_server(url):
                    kill_server(proc)
                    print(f"  ERROR: Server failed\n", file=sys.stderr)
                    results.append({
                        "topk": topk, "steps": steps, "budget": B,
                        "error": "server_failed",
                    })
                    time.sleep(3)
                    continue

                step_tpot = measure_tpot(url, args.model, args.n_warmup,
                                         args.n_measure, args.max_tokens)
                kill_server(proc)
                time.sleep(5)

                med_draft, med_fwd, n_samples = read_timing_log(
                    skip_warmup=args.n_warmup * args.max_tokens)

                entry = {
                    "topk": topk,
                    "steps": steps,
                    "budget": B,
                    "max_capacity": cap,
                    "step_ms": round(step_tpot, 2),
                    "target_forward_ms": round(med_fwd, 2),
                    "eagle3_draft_ms": round(med_draft, 2),
                    "overhead_ms": round(step_tpot - med_fwd - med_draft, 2),
                    "n_samples": n_samples,
                }
                results.append(entry)

                print(f"  step={step_tpot:.2f}ms "
                      f"(t_fwd={med_fwd:.2f} + e3_draft={med_draft:.2f} "
                      f"+ overhead={step_tpot - med_fwd - med_draft:.2f}) "
                      f"[{n_samples} samples]\n", file=sys.stderr)

    # --- Output ---
    output = {
        "model": args.model,
        "draft_model": args.draft_model,
        "vanilla_tpot_ms": round(vanilla_tpot, 2),
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # --- Summary table ---
    print("\n" + "=" * 90, file=sys.stderr)
    print("EAGLE3 LATENCY SWEEP RESULTS", file=sys.stderr)
    print("=" * 90, file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Vanilla TPOT: {vanilla_tpot:.2f} ms/tok", file=sys.stderr)
    print(file=sys.stderr)

    print(f"{'topk':>4} | {'steps':>5} | {'budget':>6} | {'cap':>5} | "
          f"{'step':>8} | {'t_fwd':>8} | {'e3_dft':>8} | {'ovhd':>7} | {'samples':>7}",
          file=sys.stderr)
    print("-" * 90, file=sys.stderr)

    for r in results:
        if "error" in r:
            print(f"{r['topk']:>4} | {r['steps']:>5} | {r['budget']:>6} | "
                  f"{'':>5} | {'ERROR':>8}", file=sys.stderr)
        else:
            print(f"{r['topk']:>4} | {r['steps']:>5} | {r['budget']:>6} | "
                  f"{r['max_capacity']:>5} | "
                  f"{r['step_ms']:>7.2f}ms | {r['target_forward_ms']:>7.2f}ms | "
                  f"{r['eagle3_draft_ms']:>7.2f}ms | {r['overhead_ms']:>6.2f}ms | "
                  f"{r['n_samples']:>7}", file=sys.stderr)

    print("=" * 90, file=sys.stderr)
    print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
