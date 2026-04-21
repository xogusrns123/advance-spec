"""
Collect SuffixDecoding candidates for the same inputs used in EAGLE-3 collection.

Runs SuffixDecoding standalone on each prompt's context + generated tokens,
recording what candidates it would produce at each position.

Usage:
    python -m simulation.analysis.collect_suffix_candidates \
        --eagle3-results results/eagle3_drafts \
        --output-dir simulation/results/suffix_candidates
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hybrid_spec_decoding.suffix_decoding.speculator import SuffixSpeculator


@dataclass
class SuffixCandidateRecord:
    """Suffix candidates at a single position."""

    position: int
    context_suffix: list[int]      # suffix used for matching
    candidates: list[list[int]]    # candidate continuations
    scores: list[float]            # frequency scores
    match_length: int              # how deep the suffix matched
    has_ground_truth: bool         # whether ground truth is in candidates
    ground_truth_rank: int         # rank of ground truth (-1 if not found)


@dataclass
class PerRequestResult:
    """Full suffix analysis for one request."""

    prompt_id: str
    generated_tokens: list[int]
    positions: list[SuffixCandidateRecord]
    coverage_rate: float = 0.0     # fraction of positions where GT is found


def analyze_single_request(
    prompt_id: str,
    generated_tokens: list[int],
    speculator: SuffixSpeculator,
    suffix_match_len: int = 16,
) -> PerRequestResult:
    """
    For each position in the generated sequence, query SuffixDecoding
    and check if the ground truth next token is in the candidates.
    """
    positions = []
    hits = 0

    for pos in range(len(generated_tokens) - 1):
        context = generated_tokens[: pos + 1]
        ground_truth = generated_tokens[pos + 1]

        result = speculator.speculate(context, suffix_len=suffix_match_len)

        # Check if ground truth is in any candidate
        has_gt = False
        gt_rank = -1
        for rank, cand in enumerate(result.candidates):
            if len(cand) > 0 and cand[0] == ground_truth:
                has_gt = True
                gt_rank = rank
                break

        if has_gt:
            hits += 1

        record = SuffixCandidateRecord(
            position=pos,
            context_suffix=context[-suffix_match_len:],
            candidates=result.candidates,
            scores=result.scores,
            match_length=result.match_length,
            has_ground_truth=has_gt,
            ground_truth_rank=gt_rank,
        )
        positions.append(record)

    coverage = hits / max(len(positions), 1)

    return PerRequestResult(
        prompt_id=prompt_id,
        generated_tokens=generated_tokens,
        positions=positions,
        coverage_rate=coverage,
    )


def build_speculator_from_corpus(
    all_tokens: list[list[int]],
    suffix_match_len: int = 16,
) -> SuffixSpeculator:
    """
    Build a SuffixSpeculator with a global tree from all prior outputs.
    Simulates the cross-request pattern matching of SuffixDecoding.
    """
    speculator = SuffixSpeculator(
        suffix_match_len=suffix_match_len,
        max_candidates=10,
        max_spec_length=10,
        adaptive_length=True,
    )

    for tokens in all_tokens:
        speculator.update_global(tokens)

    return speculator


def main():
    parser = argparse.ArgumentParser(
        description="Collect SuffixDecoding candidates"
    )
    parser.add_argument("--eagle3-results", required=True,
                        help="Directory with EAGLE-3 collection results")
    parser.add_argument("--output-dir", default="simulation/results/suffix_candidates")
    parser.add_argument("--suffix-match-len", type=int, default=16)
    parser.add_argument("--incremental", action="store_true",
                        help="Build global tree incrementally (simulates online)")
    args = parser.parse_args()

    eagle3_dir = Path(args.eagle3_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all EAGLE-3 results
    records = []
    for f in sorted(eagle3_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        with open(f) as fh:
            records.append(json.load(fh))

    print(f"Loaded {len(records)} EAGLE-3 records")

    # Collect all token sequences for global tree
    all_tokens = [r["generated_tokens"] for r in records if r["generated_tokens"]]

    if args.incremental:
        # Online mode: build tree incrementally
        speculator = SuffixSpeculator(
            suffix_match_len=args.suffix_match_len,
            max_candidates=10,
            max_spec_length=10,
        )
        results = []

        for i, record in enumerate(records):
            tokens = record["generated_tokens"]
            if not tokens:
                continue

            print(f"[{i+1}/{len(records)}] Analyzing: {record['prompt_id']}")

            # Analyze with current tree state
            speculator.reset_local()
            speculator.update_local(tokens)
            result = analyze_single_request(
                record["prompt_id"], tokens, speculator, args.suffix_match_len
            )
            results.append(result)

            # Update global tree for next request
            speculator.update_global(tokens)

            # Save
            out_path = output_dir / f"{record['prompt_id']}.json"
            with open(out_path, "w") as fh:
                json.dump(asdict(result), fh, indent=2)
    else:
        # Batch mode: build full global tree first
        speculator = build_speculator_from_corpus(all_tokens, args.suffix_match_len)
        results = []

        for i, record in enumerate(records):
            tokens = record["generated_tokens"]
            if not tokens:
                continue

            print(f"[{i+1}/{len(records)}] Analyzing: {record['prompt_id']}")

            speculator.reset_local()
            speculator.update_local(tokens)
            result = analyze_single_request(
                record["prompt_id"], tokens, speculator, args.suffix_match_len
            )
            results.append(result)

            out_path = output_dir / f"{record['prompt_id']}.json"
            with open(out_path, "w") as fh:
                json.dump(asdict(result), fh, indent=2)

    # Summary
    avg_coverage = sum(r.coverage_rate for r in results) / max(len(results), 1)
    summary = {
        "num_samples": len(results),
        "avg_coverage_rate": avg_coverage,
        "suffix_match_len": args.suffix_match_len,
        "incremental": args.incremental,
    }
    with open(output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nDone. Results saved to {output_dir}")
    print(f"  Avg suffix coverage: {avg_coverage:.3f}")


if __name__ == "__main__":
    main()
