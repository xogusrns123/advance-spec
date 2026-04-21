"""
Compute agreement scores between EAGLE-3 and SuffixDecoding.

Agreement = how often multiple sources predict the same next token.
High agreement -> confident prediction -> reduce branching, increase depth.
Low agreement -> uncertain -> widen branching for coverage.

This analysis informs the Agreement-Guided Tree Construction strategy.

Usage:
    python -m simulation.analysis.compute_agreement \
        --eagle3-results results/eagle3_drafts \
        --suffix-results results/suffix_candidates \
        --output-dir results/agreement
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AgreementRecord:
    """Agreement measurement at a single position."""

    position: int
    eagle_top1: int              # EAGLE-3's top-1 prediction
    suffix_top1: int             # SuffixDecoding's top-1 prediction
    agrees: bool                 # whether top-1 predictions match
    eagle_top5: list[int]        # EAGLE-3's top-5
    suffix_top5: list[int]       # SuffixDecoding's top-5
    overlap_top5: int            # |intersection of top-5|
    ground_truth: int            # actual next token
    eagle_correct: bool          # EAGLE-3 top-1 == ground truth
    suffix_correct: bool         # SuffixDecoding top-1 == ground truth
    agreement_when_correct: bool # agreement AND at least one is correct


def compute_agreement_for_request(
    eagle_record: dict,
    suffix_record: dict,
) -> list[AgreementRecord]:
    """Compute per-position agreement for a single request."""
    generated_tokens = eagle_record["generated_tokens"]
    steps = eagle_record.get("steps", [])
    suffix_positions = {p["position"]: p for p in suffix_record.get("positions", [])}

    records = []

    for step in steps:
        step_idx = step["step_idx"]
        draft_tokens = step["draft_tokens"]
        draft_probs = step.get("draft_probs", [])

        for depth_idx, draft_token in enumerate(draft_tokens):
            pos = step_idx + depth_idx
            if pos >= len(generated_tokens) - 1:
                break

            ground_truth = generated_tokens[pos + 1]

            # EAGLE-3 prediction
            eagle_top1 = draft_token
            # Get EAGLE-3 top-5 from tree paths if available
            eagle_top5 = _extract_eagle_topk(step, depth_idx, k=5)

            # SuffixDecoding prediction
            suffix_pos = suffix_positions.get(pos)
            if suffix_pos is None:
                continue

            suffix_cands = suffix_pos.get("candidates", [])
            suffix_top1 = suffix_cands[0][0] if suffix_cands and len(suffix_cands[0]) > 0 else -1
            suffix_top5 = [c[0] for c in suffix_cands[:5] if len(c) > 0]

            agrees = eagle_top1 == suffix_top1 and eagle_top1 >= 0 and suffix_top1 >= 0
            eagle_correct = eagle_top1 == ground_truth
            suffix_correct = suffix_top1 == ground_truth

            overlap = len(set(eagle_top5) & set(suffix_top5))

            records.append(AgreementRecord(
                position=pos,
                eagle_top1=eagle_top1,
                suffix_top1=suffix_top1,
                agrees=agrees,
                eagle_top5=eagle_top5,
                suffix_top5=suffix_top5,
                overlap_top5=overlap,
                ground_truth=ground_truth,
                eagle_correct=eagle_correct,
                suffix_correct=suffix_correct,
                agreement_when_correct=agrees and (eagle_correct or suffix_correct),
            ))

    return records


def _extract_eagle_topk(step: dict, depth_idx: int, k: int = 5) -> list[int]:
    """Extract EAGLE-3's top-k at a given depth from tree paths."""
    tree_paths = step.get("draft_tree_paths", [])
    if not tree_paths or depth_idx >= len(tree_paths[0]):
        return [step["draft_tokens"][depth_idx]] if depth_idx < len(step["draft_tokens"]) else []

    # Collect unique tokens at this depth across all paths
    tokens_at_depth = set()
    for path in tree_paths:
        if depth_idx < len(path):
            tokens_at_depth.add(path[depth_idx])
    return list(tokens_at_depth)[:k]


def compute_correlation(
    all_records: list[AgreementRecord],
) -> dict:
    """
    Compute correlation between agreement and actual correctness.
    Key question: does agreement predict acceptance?
    """
    agree_correct = sum(1 for r in all_records if r.agrees and (r.eagle_correct or r.suffix_correct))
    agree_wrong = sum(1 for r in all_records if r.agrees and not r.eagle_correct and not r.suffix_correct)
    disagree_correct = sum(1 for r in all_records if not r.agrees and (r.eagle_correct or r.suffix_correct))
    disagree_wrong = sum(1 for r in all_records if not r.agrees and not r.eagle_correct and not r.suffix_correct)

    total_agree = agree_correct + agree_wrong
    total_disagree = disagree_correct + disagree_wrong

    return {
        "agree_and_correct": agree_correct,
        "agree_and_wrong": agree_wrong,
        "disagree_and_correct": disagree_correct,
        "disagree_and_wrong": disagree_wrong,
        "p_correct_given_agree": agree_correct / max(total_agree, 1),
        "p_correct_given_disagree": disagree_correct / max(total_disagree, 1),
        "agreement_rate": total_agree / max(len(all_records), 1),
        "agreement_precision": agree_correct / max(total_agree, 1),
    }


def compute_overlap_distribution(
    all_records: list[AgreementRecord],
) -> dict:
    """Distribution of top-5 overlap counts."""
    overlap_counts = defaultdict(int)
    for r in all_records:
        overlap_counts[r.overlap_top5] += 1

    total = len(all_records)
    return {
        "overlap_distribution": {
            k: {"count": v, "rate": v / max(total, 1)}
            for k, v in sorted(overlap_counts.items())
        },
        "mean_overlap": np.mean([r.overlap_top5 for r in all_records]) if all_records else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute EAGLE-3 vs SuffixDecoding agreement"
    )
    parser.add_argument("--eagle3-results", required=True)
    parser.add_argument("--suffix-results", required=True)
    parser.add_argument("--output-dir", default="results/agreement")
    args = parser.parse_args()

    eagle3_dir = Path(args.eagle3_results)
    suffix_dir = Path(args.suffix_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix_files = {f.stem: f for f in suffix_dir.glob("*.json")}

    all_records = []

    for ef in sorted(eagle3_dir.glob("*.json")):
        if ef.name == "summary.json":
            continue

        with open(ef) as f:
            eagle_record = json.load(f)

        prompt_id = eagle_record["prompt_id"]
        if prompt_id not in suffix_files:
            continue

        with open(suffix_files[prompt_id]) as f:
            suffix_record = json.load(f)

        records = compute_agreement_for_request(eagle_record, suffix_record)
        all_records.extend(records)

    if not all_records:
        print("No records to analyze. Check input directories.")
        return

    # Compute statistics
    total = len(all_records)
    agreement_rate = sum(1 for r in all_records if r.agrees) / total
    eagle_accuracy = sum(1 for r in all_records if r.eagle_correct) / total
    suffix_accuracy = sum(1 for r in all_records if r.suffix_correct) / total

    correlation = compute_correlation(all_records)
    overlap = compute_overlap_distribution(all_records)

    result = {
        "total_positions": total,
        "agreement_rate": agreement_rate,
        "eagle_accuracy": eagle_accuracy,
        "suffix_accuracy": suffix_accuracy,
        "correlation": correlation,
        "overlap": overlap,
    }

    with open(output_dir / "agreement.json", "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("=== Agreement Analysis ===")
    print(f"  Total positions: {total}")
    print(f"  Agreement rate: {agreement_rate:.3f}")
    print(f"  EAGLE-3 accuracy: {eagle_accuracy:.3f}")
    print(f"  SuffixDecoding accuracy: {suffix_accuracy:.3f}")
    print(f"\n  P(correct | agree): {correlation['p_correct_given_agree']:.3f}")
    print(f"  P(correct | disagree): {correlation['p_correct_given_disagree']:.3f}")
    print(f"  Agreement precision: {correlation['agreement_precision']:.3f}")
    print(f"  Mean top-5 overlap: {overlap['mean_overlap']:.2f}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
