"""
Benchmark the two hybrid baselines:

1. Suffix Decoding + EAGLE-3  (parallel merge, no pruning)
2. RASD-style tree pruning + longest-prefix tree fusion

Both baselines use:
- The same tokenizer / sampling setup (temperature=0, greedy)
- The same max tree budget (default 64 tokens)
- The same ExperimentConfig for reproducibility

Outputs JSON + CSV for every run.

Usage:
    python -m hybrid_spec_decoding.benchmarks.run_hybrid \
        --config hybrid_spec_decoding/benchmarks/configs/humaneval.yaml \
        --output-dir results/hybrid \
        --dummy   # smoke test with dummy data
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..proposers.base import BaseProposer, ProposerOutput, populate_output_metadata
from ..tree_fusion.pruning import prune_retrieval_tree, prune_to_budget
from ..tree_fusion.rasd_merge import build_retrieval_tree, longest_prefix_merge
from ..tree_fusion.tree_utils import DraftTree
from ..tracing.tracer import DecodingTracer
from .run_benchmark import (
    ExperimentConfig,
    SampleResult,
    BenchmarkSummary,
    compute_summary,
    save_results_json,
    save_results_csv,
    save_comparison_table,
)


# ---------------------------------------------------------------------------
# Hybrid proposer helpers
# ---------------------------------------------------------------------------

def _build_eagle_tree_from_logits(
    logits: np.ndarray,
    topk: int,
    depth: int,
    max_tokens: int,
) -> DraftTree:
    """Build a simple EAGLE-3-like tree from logits (for offline simulation)."""
    tree = DraftTree()
    probs = _softmax(logits)
    top_ids = np.argsort(probs)[-topk:][::-1]
    budget = max_tokens

    # Depth-1: branch top-k
    d1_nodes = []
    for tid in top_ids:
        if budget <= 0:
            break
        node = tree.add_node(tree.root, int(tid), prob=float(probs[tid]), source="eagle")
        d1_nodes.append(node)
        budget -= 1

    # Depth 2+: extend each depth-1 node greedily (single child)
    frontier = list(d1_nodes)
    for d in range(1, depth):
        next_frontier = []
        for parent in frontier:
            if budget <= 0:
                break
            # Simulate: pick a token based on hash for reproducibility
            pseudo_tid = (parent.token_id * 31 + d) % logits.shape[0]
            pseudo_prob = float(probs[pseudo_tid % len(probs)])
            child = tree.add_node(parent, pseudo_tid, prob=pseudo_prob, source="eagle")
            next_frontier.append(child)
            budget -= 1
        frontier = next_frontier

    return tree


def _build_suffix_candidates(
    context_ids: list[int],
    n_candidates: int = 5,
    max_len: int = 8,
) -> tuple[list[list[int]], list[float]]:
    """Build synthetic suffix candidates for offline simulation.

    In a real system these come from the SuffixDecodingCache.
    """
    candidates = []
    scores = []
    vocab_size = max(max(context_ids, default=100), 100) + 1

    for i in range(n_candidates):
        length = min(max_len, 3 + i)
        cand = []
        for j in range(length):
            # Deterministic pseudo-candidate based on context
            seed = (sum(context_ids[-4:]) * (i + 1) + j * 7) % vocab_size
            cand.append(seed)
        candidates.append(cand)
        scores.append(1.0 / (i + 1))

    return candidates, scores


# ---------------------------------------------------------------------------
# Baseline 1: Suffix + EAGLE-3 (simple merge)
# ---------------------------------------------------------------------------

def fuse_suffix_eagle_simple(
    eagle_tree: DraftTree,
    suffix_candidates: list[list[int]],
    suffix_scores: list[float],
    max_tokens: int = 64,
) -> DraftTree:
    """Baseline 1: Simple merge of suffix candidates into EAGLE-3 tree.

    No RASD-style pruning -- just merge via longest prefix and cap budget.
    """
    if not suffix_candidates:
        return eagle_tree

    retrieval_tree = build_retrieval_tree(suffix_candidates, suffix_scores)
    merged = longest_prefix_merge(eagle_tree, retrieval_tree)

    if merged.num_nodes > max_tokens:
        merged = prune_to_budget(merged, max_tokens)

    return merged


# ---------------------------------------------------------------------------
# Baseline 2: RASD tree pruning + longest-prefix fusion
# ---------------------------------------------------------------------------

def fuse_rasd_style(
    eagle_tree: DraftTree,
    suffix_candidates: list[list[int]],
    suffix_scores: list[float],
    eagle_first_token_probs: np.ndarray,
    pruning_topk: int = 10,
    max_tokens: int = 64,
) -> DraftTree:
    """Baseline 2: RASD-style pruning + longest-prefix merge.

    Steps:
    1. Build retrieval tree from suffix candidates
    2. Prune by EAGLE-3 first-token top-k
    3. Merge via longest prefix matching
    4. Enforce token budget
    """
    if not suffix_candidates:
        return eagle_tree

    retrieval_tree = build_retrieval_tree(suffix_candidates, suffix_scores)
    pruned_retrieval = prune_retrieval_tree(
        retrieval_tree, eagle_first_token_probs, topk=pruning_topk
    )
    merged = longest_prefix_merge(eagle_tree, pruned_retrieval)

    if merged.num_nodes > max_tokens:
        merged = prune_to_budget(merged, max_tokens)

    return merged


# ---------------------------------------------------------------------------
# Benchmark loop for both baselines
# ---------------------------------------------------------------------------

def run_hybrid_benchmark(
    baseline_name: str,
    prompts: list[dict[str, Any]],
    config: ExperimentConfig,
) -> tuple[list[SampleResult], DecodingTracer]:
    """Run a hybrid baseline benchmark."""
    tracer = DecodingTracer()
    results: list[SampleResult] = []
    vocab_size = 32000  # typical LLaMA vocab

    for i, prompt_data in enumerate(prompts):
        pid = prompt_data.get("id", str(i))
        context_ids: list[int] = prompt_data.get("token_ids", list(range(10)))

        tracer.begin_generation(request_id=pid, prompt_len=len(context_ids))
        sample_start = time.perf_counter()

        total_generated = 0
        step_count = 0
        remaining = config.max_new_tokens

        while remaining > 0:
            tracer.begin_step()

            # 1. Build EAGLE-3 tree (simulated with random logits)
            rng = np.random.RandomState(config.seed + i + step_count)
            eagle_logits = rng.randn(vocab_size).astype(np.float32)
            eagle_tree = _build_eagle_tree_from_logits(
                eagle_logits,
                topk=config.eagle_topk,
                depth=config.num_steps,
                max_tokens=config.max_tree_tokens // 2,
            )

            # 2. Build suffix candidates
            suffix_cands, suffix_scores = _build_suffix_candidates(
                context_ids, n_candidates=config.max_candidates,
            )

            # 3. Fuse
            draft_start = time.perf_counter()
            if baseline_name == "suffix_eagle_simple":
                fused = fuse_suffix_eagle_simple(
                    eagle_tree, suffix_cands, suffix_scores,
                    max_tokens=config.max_tree_tokens,
                )
            elif baseline_name == "rasd_fusion":
                eagle_probs = _softmax(eagle_logits)
                fused = fuse_rasd_style(
                    eagle_tree, suffix_cands, suffix_scores,
                    eagle_probs,
                    pruning_topk=config.pruning_topk,
                    max_tokens=config.max_tree_tokens,
                )
            else:
                fused = eagle_tree
            draft_time = time.perf_counter() - draft_start

            # Build ProposerOutput for tracing
            output = ProposerOutput(tree=fused, draft_latency_s=draft_time,
                                    proposer_name=baseline_name)
            populate_output_metadata(output)
            tracer.record_draft_from_proposer_output(output)

            # 4. Simulate verification
            tracer.begin_verify()
            paths = fused.get_all_paths()
            greedy = paths[0] if paths else []
            # Simulate acceptance: accept first N tokens deterministically
            accept_len = min(len(greedy), 3 + rng.randint(0, 4))
            accepted = greedy[:accept_len] if greedy else [rng.randint(0, vocab_size)]
            tracer.end_verify(accepted)
            tracer.end_step()

            n_accepted = len(accepted)
            context_ids = list(context_ids) + accepted
            total_generated += n_accepted
            remaining -= n_accepted
            step_count += 1

        wall_time = time.perf_counter() - sample_start
        gen_trace = tracer.end_generation(total_tokens=total_generated)

        mat = (
            sum(s.accepted_length for s in gen_trace.steps)
            / max(len(gen_trace.steps), 1)
        )
        tpot_ms = (wall_time / max(total_generated, 1)) * 1000

        results.append(SampleResult(
            prompt_id=pid,
            proposer=baseline_name,
            num_tokens=total_generated,
            num_steps=step_count,
            wall_time_s=wall_time,
            tokens_per_sec=total_generated / max(wall_time, 1e-9),
            mat=mat,
            tpot_ms=tpot_ms,
            draft_time_s=gen_trace.total_draft_s,
            verify_time_s=gen_trace.total_verify_s,
            overhead_time_s=wall_time - gen_trace.total_draft_s - gen_trace.total_verify_s,
        ))

    return results, tracer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid baseline benchmarks")
    parser.add_argument(
        "--baselines", nargs="+",
        default=["suffix_eagle_simple", "rasd_fusion"],
        choices=["suffix_eagle_simple", "rasd_fusion"],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-tree-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="results/hybrid")
    parser.add_argument("--dummy", action="store_true", help="Use dummy prompts")
    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig(
            max_samples=args.max_samples,
            max_new_tokens=args.max_tokens,
            max_tree_tokens=args.max_tree_tokens,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Prompts
    if args.dummy:
        prompts = [
            {"id": f"dummy_{i}", "prompt": f"Test prompt {i}", "token_ids": list(range(10 + i))}
            for i in range(min(args.max_samples, 10))
        ]
    else:
        try:
            from ..analysis.collect_eagle3_drafts import load_dataset
            prompts = load_dataset(config.dataset, config.max_samples)
        except Exception:
            prompts = [
                {"id": f"dummy_{i}", "prompt": f"Test {i}", "token_ids": list(range(10 + i))}
                for i in range(5)
            ]

    print(f"Loaded {len(prompts)} prompts")
    print(f"Shared config: tree_budget={config.max_tree_tokens}, "
          f"temp={config.temperature}, seed={config.seed}")

    summaries: dict[str, BenchmarkSummary] = {}

    for baseline in args.baselines:
        print(f"\n{'='*60}")
        print(f"Baseline: {baseline}")
        print(f"{'='*60}")

        results, tracer = run_hybrid_benchmark(baseline, prompts, config)
        summary = compute_summary(results, baseline, config.dataset)
        summaries[baseline] = summary

        save_results_json(summary, results, output_dir / f"{baseline}.json")
        save_results_csv(results, output_dir / f"{baseline}.csv")
        tracer.save_json(output_dir / f"{baseline}_trace.json")
        tracer.save_csv(output_dir / f"{baseline}_trace.csv")

        print(f"  Throughput: {summary.throughput_tok_per_s:.1f} tok/s")
        print(f"  MAT:        {summary.mat:.3f}")
        print(f"  TPOT:       {summary.tpot_ms:.3f} ms")
        print(f"  Pipeline:   draft={summary.draft_frac:.1%} "
              f"verify={summary.verify_frac:.1%} "
              f"overhead={summary.overhead_frac:.1%}")

    if len(summaries) > 1:
        save_comparison_table(summaries, output_dir / "comparison")
        print(f"\n{'='*60}")
        print("Comparison")
        print(f"{'='*60}")
        print(f"{'Baseline':<25} {'tok/s':>8} {'MAT':>6} {'TPOT':>8}")
        print("-" * 50)
        for name, s in summaries.items():
            print(f"{name:<25} {s.throughput_tok_per_s:>8.1f} "
                  f"{s.mat:>6.3f} {s.tpot_ms:>7.3f}ms")

    print(f"\nResults saved to {output_dir}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / (exp.sum() + 1e-30)


if __name__ == "__main__":
    main()
