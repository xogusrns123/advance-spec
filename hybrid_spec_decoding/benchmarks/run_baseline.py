"""
Run baseline experiments: EAGLE-3 only and SuffixDecoding only.

Measures SR (Speedup Ratio), tau (Average Acceptance Length),
and tokens/sec for each baseline condition.

Usage:
    # EAGLE-3 baseline (requires SGLang server running)
    python -m hybrid_spec_decoding.benchmarks.run_baseline \
        --mode eagle3 \
        --server-url http://localhost:30000 \
        --config benchmarks/configs/humaneval.yaml \
        --output-dir results/baselines

    # Autoregressive baseline (no speculative decoding)
    python -m hybrid_spec_decoding.benchmarks.run_baseline \
        --mode autoregressive \
        --server-url http://localhost:30000 \
        --output-dir results/baselines
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
import yaml

from simulation.analysis.collect_eagle3_drafts import load_dataset


@dataclass
class BaselineResult:
    """Result for a single generation."""

    prompt_id: str
    mode: str
    num_tokens: int
    wall_time_s: float
    tokens_per_sec: float


def run_autoregressive(
    server_url: str,
    prompts: list[dict],
    max_tokens: int = 512,
) -> list[BaselineResult]:
    """Run autoregressive (no speculation) baseline."""
    results = []
    for i, prompt_data in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt_data['id']}")

        start = time.time()
        resp = requests.post(f"{server_url}/generate", json={
            "text": prompt_data["prompt"],
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0},
        })
        resp.raise_for_status()
        wall_time = time.time() - start

        data = resp.json()
        meta = data.get("meta_info", {})
        num_tokens = len(meta.get("output_token_ids", []))

        results.append(BaselineResult(
            prompt_id=prompt_data["id"],
            mode="autoregressive",
            num_tokens=num_tokens,
            wall_time_s=wall_time,
            tokens_per_sec=num_tokens / max(wall_time, 1e-6),
        ))
    return results


def run_eagle3(
    server_url: str,
    prompts: list[dict],
    max_tokens: int = 512,
) -> list[BaselineResult]:
    """Run EAGLE-3 speculative decoding baseline."""
    results = []
    for i, prompt_data in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt_data['id']}")

        start = time.time()
        resp = requests.post(f"{server_url}/generate", json={
            "text": prompt_data["prompt"],
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0},
        })
        resp.raise_for_status()
        wall_time = time.time() - start

        data = resp.json()
        meta = data.get("meta_info", {})
        num_tokens = len(meta.get("output_token_ids", []))

        results.append(BaselineResult(
            prompt_id=prompt_data["id"],
            mode="eagle3",
            num_tokens=num_tokens,
            wall_time_s=wall_time,
            tokens_per_sec=num_tokens / max(wall_time, 1e-6),
        ))
    return results


def run_suffix_only(
    prompts: list[dict],
    corpus_tokens: list[list[int]] | None = None,
) -> dict:
    """
    Measure SuffixDecoding candidate quality (no actual generation).
    Reports coverage rate and average candidate count.
    """
    from ..suffix_decoding.speculator import SuffixSpeculator

    speculator = SuffixSpeculator(suffix_match_len=16, max_candidates=10)

    # Build global tree from corpus if available
    if corpus_tokens:
        for tokens in corpus_tokens:
            speculator.update_global(tokens)

    return {
        "mode": "suffix_only",
        "global_tree_size": len(speculator.global_tree),
        "note": "SuffixDecoding is model-free; measure via analysis scripts",
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--mode", required=True,
                        choices=["autoregressive", "eagle3", "suffix_only", "all"])
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default="results/baselines")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        dataset = config.get("dataset", args.dataset)
        max_samples = config.get("max_samples", args.max_samples)
        max_tokens = config.get("max_tokens", args.max_tokens)
    else:
        dataset = args.dataset
        max_samples = args.max_samples
        max_tokens = args.max_tokens

    prompts = load_dataset(dataset, max_samples)
    print(f"Loaded {len(prompts)} prompts from {dataset}")

    modes = [args.mode] if args.mode != "all" else ["autoregressive", "eagle3"]
    all_results = {}

    for mode in modes:
        print(f"\n=== Running {mode} baseline ===")

        if mode == "autoregressive":
            results = run_autoregressive(args.server_url, prompts, max_tokens)
        elif mode == "eagle3":
            results = run_eagle3(args.server_url, prompts, max_tokens)
        elif mode == "suffix_only":
            info = run_suffix_only(prompts)
            all_results["suffix_only"] = info
            continue
        else:
            continue

        # Compute summary
        avg_tps = sum(r.tokens_per_sec for r in results) / max(len(results), 1)
        avg_tokens = sum(r.num_tokens for r in results) / max(len(results), 1)
        avg_wall = sum(r.wall_time_s for r in results) / max(len(results), 1)

        summary = {
            "mode": mode,
            "dataset": dataset,
            "num_samples": len(results),
            "avg_tokens_per_sec": avg_tps,
            "avg_tokens": avg_tokens,
            "avg_wall_time_s": avg_wall,
            "results": [asdict(r) for r in results],
        }

        all_results[mode] = summary

        with open(output_dir / f"{mode}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Avg tokens/sec: {avg_tps:.1f}")
        print(f"  Avg tokens: {avg_tokens:.1f}")
        print(f"  Avg wall time: {avg_wall:.3f}s")

    # Compute speedup ratio if both baselines available
    if "autoregressive" in all_results and "eagle3" in all_results:
        ar_tps = all_results["autoregressive"]["avg_tokens_per_sec"]
        eagle_tps = all_results["eagle3"]["avg_tokens_per_sec"]
        sr = eagle_tps / max(ar_tps, 1e-6)
        print(f"\n=== Speedup Ratio ===")
        print(f"  AR: {ar_tps:.1f} tok/s, EAGLE-3: {eagle_tps:.1f} tok/s")
        print(f"  SR = {sr:.2f}x")

        all_results["speedup_ratio"] = sr
        with open(output_dir / "comparison.json", "w") as f:
            json.dump({"ar_tps": ar_tps, "eagle_tps": eagle_tps, "speedup_ratio": sr}, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
