"""
Measure per-step latencies for speculative decoding methods.

Sends warmup + measurement requests to an SGLang server and computes
median step latencies for vanilla, EAGLE3, and suffix decoding.

Usage:
    # Start server first, then run calibration:
    python3 -m simulation.pipeline.calibrate_latency \
        --url http://localhost:30000/v1 \
        --model zai-org/GLM-4.7-Flash \
        --warmup 5 \
        --measure 20 \
        --output simulation/results/latency_config.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from openai import OpenAI


def measure_latencies(
    url: str,
    model: str,
    n_warmup: int = 5,
    n_measure: int = 20,
    max_tokens: int = 256,
) -> dict[str, float]:
    """Measure per-token latency by sending requests and timing responses.

    Returns dict with per-step latency estimates in seconds.
    """
    client = OpenAI(base_url=url, api_key="dummy")

    prompts = [
        "Write a Python function that sorts a list using quicksort.",
        "Explain the concept of speculative decoding in 5 sentences.",
        "What are the main differences between TCP and UDP?",
        "Write a bash script that finds all files larger than 100MB.",
        "Describe how a hash table works internally.",
    ]

    # Warmup
    print(f"Warming up ({n_warmup} requests)...")
    for i in range(n_warmup):
        prompt = prompts[i % len(prompts)]
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )

    # Measure
    print(f"Measuring ({n_measure} requests)...")
    tpot_list = []

    for i in range(n_measure):
        prompt = prompts[i % len(prompts)]

        t_start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        t_end = time.perf_counter()

        wall_time = t_end - t_start
        n_tokens = response.usage.completion_tokens

        if n_tokens > 0:
            tpot = wall_time / n_tokens
            tpot_list.append(tpot)
            print(f"  [{i+1}/{n_measure}] {n_tokens} tokens, "
                  f"{wall_time:.3f}s, TPOT={tpot*1000:.2f}ms")

    if not tpot_list:
        print("No valid measurements!")
        return {}

    median_tpot = statistics.median(tpot_list)
    mean_tpot = statistics.mean(tpot_list)
    p95_tpot = sorted(tpot_list)[int(len(tpot_list) * 0.95)]

    print(f"\nResults:")
    print(f"  Median TPOT: {median_tpot*1000:.3f} ms")
    print(f"  Mean TPOT:   {mean_tpot*1000:.3f} ms")
    print(f"  P95 TPOT:    {p95_tpot*1000:.3f} ms")

    return {
        "median_tpot_s": median_tpot,
        "mean_tpot_s": mean_tpot,
        "p95_tpot_s": p95_tpot,
        "n_measurements": len(tpot_list),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate step latencies for oracle simulation")
    parser.add_argument("--url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="simulation/results/latency_config.json")
    parser.add_argument("--label", default="default",
                        help="Label for this measurement (e.g., 'vanilla', 'eagle3')")
    args = parser.parse_args()

    results = measure_latencies(
        url=args.url,
        model=args.model,
        n_warmup=args.warmup,
        n_measure=args.measure,
        max_tokens=args.max_tokens,
    )

    if results:
        results["label"] = args.label
        results["model"] = args.model

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing config or create new
        config = {}
        if output_path.exists():
            with open(output_path) as f:
                config = json.load(f)

        config[args.label] = results

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nSaved to {output_path} (label: {args.label})")
        print(f"\nMedian TPOT for {args.label}: {results['median_tpot_s']:.6f}s")


if __name__ == "__main__":
    main()
