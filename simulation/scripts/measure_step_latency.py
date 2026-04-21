"""Measure per-step latency for vanilla and EAGLE3 on SGLang server.

Sends requests and extracts per-token timing from the response.

Usage:
    python simulation/scripts/measure_step_latency.py --url http://localhost:30000/v1 --model Qwen/Qwen3-8B
"""

import argparse
import json
import statistics
import time

from openai import OpenAI


def measure(url, model, n_warmup=3, n_measure=10, max_tokens=200):
    client = OpenAI(base_url=url, api_key="dummy")

    prompt = "/no_think\nWrite a detailed explanation of how quicksort works step by step."

    # Warmup
    for i in range(n_warmup):
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )

    # Measure
    latencies = []
    for i in range(n_measure):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0

        n_gen = resp.usage.completion_tokens
        n_prompt = resp.usage.prompt_tokens
        tok_per_s = n_gen / elapsed if elapsed > 0 else 0
        ms_per_tok = (elapsed / n_gen * 1000) if n_gen > 0 else 0
        latencies.append(ms_per_tok)

        print(f"  [{i+1}/{n_measure}] {n_gen} tokens in {elapsed:.2f}s = "
              f"{tok_per_s:.1f} tok/s, {ms_per_tok:.2f} ms/tok")

    avg = statistics.mean(latencies)
    med = statistics.median(latencies)
    std = statistics.stdev(latencies) if len(latencies) > 1 else 0
    print(f"\n  Result: avg={avg:.2f} ms/tok, median={med:.2f} ms/tok, std={std:.2f}")
    return med


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    med = measure(args.url, args.model, args.warmup, args.measure, args.max_tokens)
    print(f"\nFinal: {med:.4f} ms/tok = {med/1000:.6f} s/tok")


if __name__ == "__main__":
    main()
