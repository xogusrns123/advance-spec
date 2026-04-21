"""Measure small-draft-LM per-token generation latency.

Boots a single vanilla SGLang server with the given draft model (default
Qwen/Qwen3-0.6B) and times ``max_tokens=N`` chat completion calls for each
(workload, num_draft_tokens) combination. Server is on/off exactly once.

Output JSON:
    {"model": ..., "results": [
        {"workload", "num_draft_tokens",
         "total_ms", "per_token_ms", "n_actual_tokens"}, ...]}

Usage:
    python3 simulation/scripts/measure_draft_model_cost.py \\
        --model Qwen/Qwen3-0.6B \\
        --workloads specbench,bfcl_v4 \\
        --num-draft-tokens 1,3,5 \\
        --output results/latency/draft_model_cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from _workload_prompts import load_workload_prompts


def wait_for_server(url: str, timeout: int = 300) -> bool:
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Draft LM to measure (default Qwen/Qwen3-0.6B)")
    parser.add_argument("--workloads", default="specbench,bfcl_v4,swebench")
    parser.add_argument("--num-draft-tokens", default="1,3,5",
                        help="Comma-separated max_tokens values to measure")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-args", nargs="*", default=[])
    args = parser.parse_args()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    num_draft_tokens = [int(n) for n in args.num_draft_tokens.split(",")]

    # Load 2 prompts per workload (warmup + measure)
    workload_prompts = {}
    for w in workloads:
        ps = load_workload_prompts(w, n_samples=2)
        if len(ps) < 2:
            print(f"SKIP {w}: need 2 prompts, got {len(ps)}", file=sys.stderr)
            continue
        workload_prompts[w] = ps
    if not workload_prompts:
        print("ERROR: no workloads with enough prompts.", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{args.port}"
    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    print("=" * 70, file=sys.stderr)
    print(f"Launching draft LM server: {args.model} "
          f"(TP={args.tp_size}, mem_frac={args.mem_fraction_static})",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--tp-size", str(args.tp_size),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--disable-cuda-graph",
        "--host", "0.0.0.0", "--port", str(args.port),
    ] + args.extra_args

    log_path = Path("/tmp/sglang_draft_model_cost.log")
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh)
    try:
        if not wait_for_server(url):
            kill_server(proc)
            log_fh.close()
            print(f"ERROR: server failed to start. See {log_path}",
                  file=sys.stderr)
            with open(log_path) as f:
                for line in f.readlines()[-40:]:
                    print(f"  {line}", end="", file=sys.stderr)
            sys.exit(1)
        log_fh.close()

        client = OpenAI(base_url=f"{url}/v1", api_key="dummy")
        results = []
        for w, prompts in workload_prompts.items():
            warmup_prompt = prompts[0]["messages"]
            measure_prompt = prompts[1]["messages"]
            print(f"\n{w}:", file=sys.stderr)

            # Warmup (max_tokens=5, result discarded)
            try:
                client.chat.completions.create(
                    model=args.model, messages=warmup_prompt,
                    max_tokens=5, temperature=0.0)
            except Exception as e:
                print(f"  WARN warmup failed: {e}", file=sys.stderr)

            for N in num_draft_tokens:
                try:
                    t0 = time.perf_counter()
                    resp = client.chat.completions.create(
                        model=args.model, messages=measure_prompt,
                        max_tokens=N, temperature=0.0)
                    t1 = time.perf_counter()
                    total_ms = (t1 - t0) * 1000
                    n_actual = resp.usage.completion_tokens if resp.usage else N
                    per_token_ms = total_ms / max(n_actual, 1)
                    entry = {
                        "workload": w,
                        "num_draft_tokens": N,
                        "n_actual_tokens": int(n_actual),
                        "total_ms": round(total_ms, 3),
                        "per_token_ms": round(per_token_ms, 3),
                    }
                    print(f"  N={N:>2d}  n_actual={n_actual:>2d}  "
                          f"total={total_ms:>8.2f}ms  "
                          f"per_tok={per_token_ms:>7.3f}ms",
                          file=sys.stderr)
                except Exception as e:
                    print(f"  ERROR N={N}: {e}", file=sys.stderr)
                    entry = {
                        "workload": w, "num_draft_tokens": N,
                        "error": str(e),
                    }
                results.append(entry)

        output = {
            "model": args.model,
            "workloads": list(workload_prompts.keys()),
            "num_draft_tokens": num_draft_tokens,
            "results": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nOutput: {args.output}", file=sys.stderr)

    finally:
        kill_server(proc)


if __name__ == "__main__":
    main()
