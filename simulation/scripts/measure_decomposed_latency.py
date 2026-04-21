"""Measure decomposed latencies: target_forward(B), eagle3_draft(B), draft_lm(B).

Uses oracle_patch.py timing instrumentation to separate EAGLE3 draft and
target model verify costs per budget. Also measures draft LM per-token cost.

Produces latency_config.json with:
    vanilla_step_ms: target TPOT (no speculation)
    target_forward_ms: {B: ms} pure target model verify cost
    eagle3_draft_ms: {B: ms} EAGLE3 draft generation cost
    eagle3_step_ms: {B: ms} full step = draft + verify (for reference)
    draft_lm_tpot_ms: draft model per-token cost

Usage:
    python3 simulation/scripts/measure_decomposed_latency.py \
        --model Qwen/Qwen3-8B \
        --draft-model Tengyunw/qwen3_8b_eagle3 \
        --tp-size 1 \
        --budgets 1,2,4,8,16,32,64,128,256 \
        --draft-lm Qwen/Qwen3-0.6B \
        --output simulation/results/qwen3_8b/latency_config.json
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
    n_warmup: int = 3, n_measure: int = 10, max_tokens: int = 200,
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
            ms_per_tok = (elapsed / n_gen) * 1000
            tpots.append(ms_per_tok)
            print(f"    [{i+1}/{n_measure}] {n_gen} tokens, "
                  f"{ms_per_tok:.2f} ms/tok", file=sys.stderr)
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


def read_timing_log() -> list[dict]:
    entries = []
    try:
        with open(TIMING_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        pass
    return entries


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--algorithm", default="EAGLE3")
    parser.add_argument("--budgets", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--eagle-topk", type=int, default=8,
                        help="EAGLE3 top-k (must match pipeline Stage 1)")
    parser.add_argument("--eagle-steps", type=int, default=5,
                        help="EAGLE3 speculation steps (must match pipeline Stage 1)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-measure", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draft-lm", default=None)
    parser.add_argument("--draft-lm-tp-size", type=int, default=1)
    parser.add_argument("--extra-args", nargs="*", default=[])
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    url = f"http://localhost:{args.port}"

    # Install oracle patch
    subprocess.run([sys.executable, "-m",
                    "simulation.oracle.install_hook"],
                   check=True)

    # --- Step 1: Vanilla (no speculation) ---
    print("=" * 60, file=sys.stderr)
    print("Measuring vanilla (no speculation)...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

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
    print(f"\n  Vanilla TPOT: {vanilla_tpot:.2f} ms/tok\n", file=sys.stderr)
    kill_server(proc)
    time.sleep(5)

    # --- Step 2: Per-budget EAGLE3 (with timing decomposition) ---
    target_forward_ms = {}
    eagle3_draft_ms = {}
    eagle3_step_ms = {}

    for B in budgets:
        print("=" * 60, file=sys.stderr)
        print(f"Measuring budget={B} ({args.algorithm})...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # Clear timing log
        TIMING_LOG.unlink(missing_ok=True)

        env = os.environ.copy()
        env["SGLANG_ORACLE_VANILLA"] = "1"
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.model,
            "--tp-size", str(args.tp_size),
            "--speculative-algorithm", args.algorithm,
            "--speculative-num-steps", str(args.eagle_steps),
            "--speculative-eagle-topk", str(args.eagle_topk),
            "--speculative-num-draft-tokens", str(B),
            "--mem-fraction-static", "0.8",
            "--disable-cuda-graph",
            "--host", "0.0.0.0", "--port", str(args.port),
        ]
        if args.draft_model and args.algorithm != "NEXTN":
            cmd.extend(["--speculative-draft-model-path", args.draft_model])
        cmd.extend(args.extra_args)

        log_fh = open("/tmp/sglang_latency_bench.log", "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh)

        if not wait_for_server(url):
            kill_server(proc)
            print(f"ERROR: Server failed at budget={B}", file=sys.stderr)
            continue

        # Measure TPOT (= eagle3_step / 1 token)
        step_tpot = measure_tpot(url, args.model, args.n_warmup,
                                 args.n_measure, args.max_tokens)

        kill_server(proc)
        time.sleep(5)

        # Read timing log for decomposed measurements
        timings = read_timing_log()
        draft_times = [t["eagle3_draft_ms"] for t in timings
                       if t.get("eagle3_draft_ms") is not None]
        fwd_times = [t["target_forward_ms"] for t in timings
                     if t.get("target_forward_ms") is not None]

        # Skip warmup entries (first n_warmup × ~tokens_per_request)
        skip = args.n_warmup * args.max_tokens
        draft_times = draft_times[skip:] if len(draft_times) > skip else draft_times
        fwd_times = fwd_times[skip:] if len(fwd_times) > skip else fwd_times

        if draft_times and fwd_times:
            med_draft = statistics.median(draft_times)
            med_fwd = statistics.median(fwd_times)
        else:
            # Fallback: approximate from step TPOT
            med_draft = max(step_tpot - vanilla_tpot, 0.0)
            med_fwd = vanilla_tpot

        eagle3_step_ms[str(B)] = step_tpot
        eagle3_draft_ms[str(B)] = med_draft
        target_forward_ms[str(B)] = med_fwd

        print(f"\n  Budget {B}: step={step_tpot:.2f}ms "
              f"(target_fwd={med_fwd:.2f} + eagle3_draft={med_draft:.2f}) "
              f"[{len(draft_times)} samples]\n", file=sys.stderr)

    # --- Step 3: Draft LM ---
    draft_lm_tpot = None
    if args.draft_lm:
        print("=" * 60, file=sys.stderr)
        print(f"Measuring draft LM: {args.draft_lm}...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        env = os.environ.copy()
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.draft_lm,
            "--tp-size", str(args.draft_lm_tp_size),
            "--mem-fraction-static", "0.8",
            "--disable-cuda-graph",
            "--host", "0.0.0.0", "--port", str(args.port),
        ]
        proc = subprocess.Popen(cmd, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=os.setsid)
        if not wait_for_server(url):
            kill_server(proc)
            print("ERROR: Draft LM server failed", file=sys.stderr)
        else:
            draft_lm_tpot = measure_tpot(url, args.draft_lm, args.n_warmup,
                                         args.n_measure, args.max_tokens)
            print(f"\n  Draft LM TPOT: {draft_lm_tpot:.2f} ms/tok\n",
                  file=sys.stderr)
            kill_server(proc)
            time.sleep(5)

    # --- Output ---
    output = {
        "vanilla_step_ms": vanilla_tpot,
        "target_forward_ms": target_forward_ms,
        "eagle3_draft_ms": eagle3_draft_ms,
        "eagle3_step_ms": eagle3_step_ms,
        "measurement": "sglang_decomposed",
        "note": "target_forward_ms: median target model verify time per step. "
                "eagle3_draft_ms: median EAGLE3 draft generation time per step. "
                "Both measured via oracle_patch.py timing instrumentation.",
    }
    if draft_lm_tpot is not None:
        output["draft_lm_tpot_ms"] = draft_lm_tpot
        output["draft_lm"] = args.draft_lm

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("DECOMPOSED LATENCY RESULTS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Vanilla TPOT: {vanilla_tpot:.2f} ms/tok", file=sys.stderr)
    if draft_lm_tpot is not None:
        print(f"Draft LM TPOT: {draft_lm_tpot:.2f} ms/tok ({args.draft_lm})",
              file=sys.stderr)
    print(file=sys.stderr)
    print(f"{'Budget':>6} | {'Step':>8} | {'T_fwd':>8} | {'E3_draft':>8} | "
          f"{'Sum':>8} | {'Diff':>6}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    for B in budgets:
        step = eagle3_step_ms.get(str(B), 0)
        t_fwd = target_forward_ms.get(str(B), 0)
        e3 = eagle3_draft_ms.get(str(B), 0)
        diff = step - (t_fwd + e3)
        print(f"{B:>6} | {step:>7.2f}ms | {t_fwd:>7.2f}ms | {e3:>7.2f}ms | "
              f"{t_fwd+e3:>7.2f}ms | {diff:>+5.1f}ms", file=sys.stderr)

    if draft_lm_tpot is not None:
        print(file=sys.stderr)
        print("Draft model cost per budget:", file=sys.stderr)
        for B in budgets:
            dm_cost = B * draft_lm_tpot
            print(f"  B={B}: {B} × {draft_lm_tpot:.2f} = {dm_cost:.1f}ms",
                  file=sys.stderr)

    print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
