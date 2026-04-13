"""
Run tree fusion experiments.

Compares all 5 conditions:
(a) EAGLE-3 only
(b) SuffixDecoding only
(c) Parallel merge (RASD-style)
(d) Sequential extension
(e) Combined (parallel + sequential)

Usage:
    python -m hybrid_spec_decoding.benchmarks.run_fusion \
        --server-url http://localhost:30000 \
        --config benchmarks/configs/humaneval.yaml \
        --output-dir results/fusion
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
import yaml

from ..analysis.collect_eagle3_drafts import load_dataset
from ..sglang_integration.hybrid_speculator import ExperimentConfig, HybridSpeculator


@dataclass
class FusionResult:
    """Result for a single generation under a specific condition."""

    prompt_id: str
    condition: str
    num_tokens: int
    wall_time_s: float
    tokens_per_sec: float
    eagle_tokens: int = 0
    suffix_added_tokens: int = 0


@dataclass
class ExperimentSummary:
    """Summary for one experimental condition."""

    condition: str
    dataset: str
    num_samples: int
    avg_tokens_per_sec: float
    avg_tokens: float
    avg_wall_time_s: float
    speedup_ratio: float = 0.0  # vs autoregressive
    speedup_vs_eagle: float = 0.0  # vs EAGLE-3 only


CONDITIONS = {
    "a_eagle3_only": "none",
    "c_parallel_merge": "parallel",
    "d_sequential_ext": "sequential",
    "e_combined": "combined",
}


def run_condition(
    condition_name: str,
    fusion_mode: str,
    server_url: str,
    prompts: list[dict],
    config: ExperimentConfig,
    warmup_corpus: list[list[int]] | None = None,
) -> list[FusionResult]:
    """Run a single experimental condition."""
    config.fusion_mode = fusion_mode
    speculator = HybridSpeculator(config)

    # Pre-populate suffix tree with warmup corpus
    if warmup_corpus and fusion_mode != "none":
        for tokens in warmup_corpus:
            speculator.patched_worker and speculator.patched_worker.speculator.update_global(tokens)

    results = []
    for i, prompt_data in enumerate(prompts):
        print(f"    [{i+1}/{len(prompts)}] {prompt_data['id']}")

        start = time.time()
        resp = requests.post(f"{server_url}/generate", json={
            "text": prompt_data["prompt"],
            "sampling_params": {
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
            },
        })
        resp.raise_for_status()
        wall_time = time.time() - start

        data = resp.json()
        meta = data.get("meta_info", {})
        num_tokens = len(meta.get("output_token_ids", []))

        results.append(FusionResult(
            prompt_id=prompt_data["id"],
            condition=condition_name,
            num_tokens=num_tokens,
            wall_time_s=wall_time,
            tokens_per_sec=num_tokens / max(wall_time, 1e-6),
        ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run tree fusion experiments")
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default="results/fusion")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Specific conditions to run (default: all)")
    parser.add_argument("--warmup-corpus", default=None,
                        help="JSON file with token sequences for suffix tree warmup")
    parser.add_argument("--suffix-match-len", type=int, default=16)
    parser.add_argument("--pruning-topk", type=int, default=10)
    parser.add_argument("--num-draft-tokens", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
    else:
        yaml_config = {}

    dataset = yaml_config.get("dataset", args.dataset)
    max_samples = yaml_config.get("max_samples", args.max_samples)
    max_tokens = yaml_config.get("max_tokens", args.max_tokens)

    prompts = load_dataset(dataset, max_samples)
    print(f"Loaded {len(prompts)} prompts from {dataset}")

    # Load warmup corpus
    warmup_corpus = None
    if args.warmup_corpus and Path(args.warmup_corpus).exists():
        with open(args.warmup_corpus) as f:
            warmup_corpus = json.load(f)
        print(f"Loaded warmup corpus: {len(warmup_corpus)} sequences")

    # Base experiment config
    exp_config = ExperimentConfig(
        max_new_tokens=max_tokens,
        suffix_match_len=args.suffix_match_len,
        pruning_topk=args.pruning_topk,
        num_draft_tokens=args.num_draft_tokens,
    )

    # Select conditions
    conditions = args.conditions or list(CONDITIONS.keys())
    all_summaries = {}

    for cond_name in conditions:
        if cond_name not in CONDITIONS:
            print(f"Unknown condition: {cond_name}, skipping")
            continue

        fusion_mode = CONDITIONS[cond_name]
        print(f"\n=== Condition: {cond_name} (fusion_mode={fusion_mode}) ===")

        results = run_condition(
            cond_name, fusion_mode, args.server_url,
            prompts, exp_config, warmup_corpus
        )

        # Summarize
        avg_tps = sum(r.tokens_per_sec for r in results) / max(len(results), 1)
        avg_tokens = sum(r.num_tokens for r in results) / max(len(results), 1)
        avg_wall = sum(r.wall_time_s for r in results) / max(len(results), 1)

        summary = ExperimentSummary(
            condition=cond_name,
            dataset=dataset,
            num_samples=len(results),
            avg_tokens_per_sec=avg_tps,
            avg_tokens=avg_tokens,
            avg_wall_time_s=avg_wall,
        )
        all_summaries[cond_name] = summary

        # Save per-condition results
        with open(output_dir / f"{cond_name}.json", "w") as f:
            json.dump({
                "summary": asdict(summary),
                "results": [asdict(r) for r in results],
            }, f, indent=2)

        print(f"  Avg tokens/sec: {avg_tps:.1f}")
        print(f"  Avg tokens: {avg_tokens:.1f}")

    # Compute relative speedups
    eagle_baseline = all_summaries.get("a_eagle3_only")
    if eagle_baseline:
        eagle_tps = eagle_baseline.avg_tokens_per_sec
        for name, summary in all_summaries.items():
            summary.speedup_vs_eagle = summary.avg_tokens_per_sec / max(eagle_tps, 1e-6)

    # Save comparison table
    comparison = {
        name: {
            "avg_tokens_per_sec": s.avg_tokens_per_sec,
            "speedup_vs_eagle": s.speedup_vs_eagle,
        }
        for name, s in all_summaries.items()
    }
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print comparison table
    print("\n=== Comparison ===")
    print(f"{'Condition':<25} {'tok/s':>8} {'vs EAGLE-3':>12}")
    print("-" * 47)
    for name, s in all_summaries.items():
        print(f"{name:<25} {s.avg_tokens_per_sec:>8.1f} {s.speedup_vs_eagle:>11.2f}x")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
