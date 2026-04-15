"""Measure SGLang verify latency at different draft budgets.

Launches the SGLang server at each budget B, sends warmup + measurement
requests, records TPOT, and produces latency_config.json with real
SGLang-measured latencies.

Uses SGLANG_DRAFT_BUDGET env var (patched in oracle_patch.py) to
override speculative_num_draft_tokens at runtime.

Usage:
    # Inside container, all 4 GPUs:
    python3 scripts/measure_sglang_verify_latency.py \
        --model zai-org/GLM-4.7-Flash \
        --draft-model thoughtworks/GLM-4.7-Flash-Eagle3 \
        --tp-size 4 \
        --budgets 1,2,4,8,16 \
        --output results/glm4_flash/full_pipeline_test/latency_config_sglang.json

    # Qwen3-8B, single GPU:
    python3 scripts/measure_sglang_verify_latency.py \
        --model Qwen/Qwen3-8B \
        --draft-model Tengyunw/qwen3_8b_eagle3 \
        --tp-size 1 \
        --budgets 1,2,4,8,16 \
        --output results/qwen3_8b/latency_config_sglang.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time

from openai import OpenAI


PROMPT = "Write a detailed explanation of how quicksort algorithm works step by step."


def wait_for_server(url: str, timeout: int = 300) -> bool:
    """Poll server health until ready or timeout."""
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
    url: str,
    model: str,
    n_warmup: int = 3,
    n_measure: int = 10,
    max_tokens: int = 200,
) -> float:
    """Measure median TPOT (ms/token) via OpenAI-compatible API."""
    client = OpenAI(base_url=f"{url}/v1", api_key="dummy")

    # Warmup
    for _ in range(n_warmup):
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=max_tokens,
            temperature=0.0,
        )

    # Measure
    tpots = []
    for i in range(n_measure):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0
        n_gen = resp.usage.completion_tokens
        if n_gen > 0:
            ms_per_tok = (elapsed / n_gen) * 1000
            tpots.append(ms_per_tok)
            print(f"    [{i+1}/{n_measure}] {n_gen} tokens, "
                  f"{ms_per_tok:.2f} ms/tok", file=sys.stderr)

    return statistics.median(tpots) if tpots else 0.0


def launch_server(
    model: str,
    draft_model: str | None,
    tp_size: int,
    algorithm: str,
    draft_budget: int | None,
    port: int,
    extra_args: list[str],
) -> subprocess.Popen:
    """Launch SGLang server with given config."""
    env = os.environ.copy()
    env["SGLANG_ORACLE_VANILLA"] = "1"
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    if draft_budget is not None:
        env["SGLANG_DRAFT_BUDGET"] = str(draft_budget)
    else:
        env.pop("SGLANG_DRAFT_BUDGET", None)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--tp-size", str(tp_size),
        "--speculative-algorithm", algorithm,
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "4",
        "--speculative-num-draft-tokens", "16",
        "--mem-fraction-static", "0.8",
        "--disable-cuda-graph",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    if draft_model and algorithm != "NEXTN":
        cmd.extend(["--speculative-draft-model-path", draft_model])
    cmd.extend(extra_args)

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


def kill_server(proc: subprocess.Popen):
    """Kill server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Target model path")
    parser.add_argument("--draft-model", default=None,
                        help="EAGLE3 draft model path")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--algorithm", default="EAGLE3",
                        help="Speculative algorithm (EAGLE3 or NEXTN)")
    parser.add_argument("--budgets", default="1,2,4,8,16",
                        help="Comma-separated draft budget values")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-measure", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--output", required=True,
                        help="Output latency_config.json path")
    parser.add_argument("--extra-args", nargs="*", default=[],
                        help="Extra args passed to sglang.launch_server")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    url = f"http://localhost:{args.port}"

    # --- Step 1: Measure vanilla (no speculation) ---
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
        "--host", "0.0.0.0",
        "--port", str(args.port),
    ] + args.extra_args

    proc = subprocess.Popen(
        vanilla_cmd, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    if not wait_for_server(url):
        kill_server(proc)
        print("ERROR: Vanilla server failed to start", file=sys.stderr)
        sys.exit(1)

    vanilla_tpot = measure_tpot(url, args.model, args.n_warmup,
                                args.n_measure, args.max_tokens)
    print(f"\n  Vanilla TPOT: {vanilla_tpot:.2f} ms/tok\n", file=sys.stderr)
    kill_server(proc)
    time.sleep(5)

    # --- Step 2: Measure each budget ---
    verify_latencies = {}

    for B in budgets:
        print("=" * 60, file=sys.stderr)
        print(f"Measuring budget={B} ({args.algorithm})...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        proc = launch_server(
            model=args.model,
            draft_model=args.draft_model,
            tp_size=args.tp_size,
            algorithm=args.algorithm,
            draft_budget=B,
            port=args.port,
            extra_args=args.extra_args,
        )

        if not wait_for_server(url):
            kill_server(proc)
            print(f"ERROR: Server failed at budget={B}", file=sys.stderr)
            continue

        tpot = measure_tpot(url, args.model, args.n_warmup,
                            args.n_measure, args.max_tokens)
        print(f"\n  Budget {B} TPOT: {tpot:.2f} ms/tok\n", file=sys.stderr)

        # In oracle vanilla mode (SGLANG_ORACLE_VANILLA=1), accept_length=0,
        # so each step produces exactly 1 token (bonus only).
        # Therefore TPOT = step_cost / 1 = step_cost.
        # This gives us the exact draft(B) + verify(B) cost per step.

        verify_latencies[str(B)] = tpot

        kill_server(proc)
        time.sleep(5)

    # --- Output ---
    output = {
        "vanilla_step_ms": vanilla_tpot,
        "verify_latencies_ms": verify_latencies,
        "measurement": "sglang",
        "note": "Measured via SGLang server in oracle vanilla mode "
                "(accept_length=0, 1 token/step). "
                "TPOT = step_cost = draft(B) + verify(B) per step.",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Vanilla TPOT: {vanilla_tpot:.2f} ms/tok", file=sys.stderr)
    print(file=sys.stderr)
    print(f"{'Budget':>6} | {'Step cost (ms)':>14} | {'Overhead':>10} | {'Draft cost':>10}",
          file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    for B in budgets:
        step = verify_latencies.get(str(B), 0)
        if step > 0:
            overhead = step / vanilla_tpot
            draft_est = step - vanilla_tpot
            print(f"{B:>6} | {step:>14.2f} | {overhead:>9.2f}x | {draft_est:>9.1f}ms",
                  file=sys.stderr)
    print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
