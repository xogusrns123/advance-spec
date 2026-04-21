"""
Compute complementarity between EAGLE-3 and SuffixDecoding.

Measures the 4-case framework:
  Case 1: E=O, S=O  (both have ground truth -- RASD merge covers this)
  Case 2a: E=O, S_parallel=X, S_sequential=O  (sequential extension's unique value)
  Case 2b: E=O, S=X  (EAGLE-3 alone is sufficient)
  Case 3: E=X, S=O  (SuffixDecoding compensates EAGLE-3 failure)
  Case 4: E=X, S=X  (both fail)

Also measures per-depth acceptance rate P_accept(d) and suffix match rate P_match(d).

Usage:
    python -m simulation.analysis.compute_complementarity \
        --eagle3-results results/eagle3_drafts \
        --suffix-results results/suffix_candidates \
        --output-dir simulation/results/complementarity
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from hybrid_spec_decoding.suffix_decoding.speculator import SuffixSpeculator


@dataclass
class CaseStats:
    """Aggregated case statistics."""

    case1: int = 0    # E=O, S=O
    case2a: int = 0   # E=O, S_par=X, S_seq=O
    case2b: int = 0   # E=O, S=X
    case3: int = 0    # E=X, S=O
    case4: int = 0    # E=X, S=X
    total: int = 0

    @property
    def case1_rate(self) -> float:
        return self.case1 / max(self.total, 1)

    @property
    def case2a_rate(self) -> float:
        return self.case2a / max(self.total, 1)

    @property
    def case2b_rate(self) -> float:
        return self.case2b / max(self.total, 1)

    @property
    def case3_rate(self) -> float:
        return self.case3 / max(self.total, 1)

    @property
    def case4_rate(self) -> float:
        return self.case4 / max(self.total, 1)

    @property
    def fusion_value(self) -> float:
        """Fraction of positions where fusion helps over EAGLE-3 alone."""
        return (self.case1 + self.case3) / max(self.total, 1)

    @property
    def sequential_unique_value(self) -> float:
        """Fraction where sequential extension adds value over parallel."""
        return self.case2a / max(self.total, 1)

    def to_dict(self) -> dict:
        return {
            "case1_both_correct": self.case1,
            "case2a_sequential_value": self.case2a,
            "case2b_eagle_sufficient": self.case2b,
            "case3_suffix_compensates": self.case3,
            "case4_both_fail": self.case4,
            "total": self.total,
            "rates": {
                "case1": self.case1_rate,
                "case2a": self.case2a_rate,
                "case2b": self.case2b_rate,
                "case3": self.case3_rate,
                "case4": self.case4_rate,
            },
            "fusion_value": self.fusion_value,
            "sequential_unique_value": self.sequential_unique_value,
        }


@dataclass
class DepthStats:
    """Per-depth acceptance and match statistics."""

    depth: int
    total: int = 0
    eagle_accepted: int = 0
    suffix_matched: int = 0
    both_correct: int = 0

    @property
    def p_accept(self) -> float:
        return self.eagle_accepted / max(self.total, 1)

    @property
    def p_match(self) -> float:
        return self.suffix_matched / max(self.total, 1)


def compute_cases(
    eagle3_dir: Path,
    suffix_dir: Path,
    speculator: SuffixSpeculator | None = None,
    suffix_match_len: int = 16,
) -> tuple[CaseStats, dict[int, DepthStats]]:
    """
    Compute case distribution across all samples.

    Requires per-step draft logs from EAGLE-3 (with accepted/rejected info)
    and suffix candidate results.

    For Case 2a detection, we need to re-run SuffixDecoding with extended
    context (context + EAGLE-3 draft tokens), hence the optional speculator.
    """
    stats = CaseStats()
    depth_stats: dict[int, DepthStats] = defaultdict(lambda: DepthStats(depth=0))

    eagle3_files = sorted(eagle3_dir.glob("*.json"))
    suffix_files = {f.stem: f for f in suffix_dir.glob("*.json")}

    for ef in eagle3_files:
        if ef.name == "summary.json":
            continue

        with open(ef) as f:
            eagle_record = json.load(f)

        prompt_id = eagle_record["prompt_id"]
        if prompt_id not in suffix_files:
            continue

        with open(suffix_files[prompt_id]) as f:
            suffix_record = json.load(f)

        generated_tokens = eagle_record["generated_tokens"]
        if not generated_tokens:
            continue

        steps = eagle_record.get("steps", [])
        suffix_positions = {p["position"]: p for p in suffix_record.get("positions", [])}

        # Analyze each step
        for step in steps:
            step_idx = step["step_idx"]
            draft_tokens = step["draft_tokens"]
            accepted_tokens = step["accepted_tokens"]
            target_token = step["target_token"]

            # For each draft position (depth)
            for depth_idx, draft_token in enumerate(draft_tokens):
                depth = depth_idx + 1
                ds = depth_stats[depth]
                ds.depth = depth
                ds.total += 1

                eagle_correct = (
                    depth_idx < len(accepted_tokens)
                    and draft_token == target_token
                ) if depth_idx == 0 else (depth_idx < len(accepted_tokens))

                if eagle_correct:
                    ds.eagle_accepted += 1

                # Check suffix: look up the corresponding position
                pos = step_idx + depth_idx
                suffix_pos = suffix_positions.get(pos)
                suffix_has_gt = suffix_pos["has_ground_truth"] if suffix_pos else False

                if suffix_has_gt:
                    ds.suffix_matched += 1

                # Classify into cases
                stats.total += 1
                if eagle_correct and suffix_has_gt:
                    stats.case1 += 1
                    ds.both_correct += 1
                elif eagle_correct and not suffix_has_gt:
                    # Check Case 2a: can sequential extension find it?
                    if speculator is not None and pos < len(generated_tokens) - 1:
                        context = generated_tokens[:step_idx]
                        draft_path = draft_tokens[:depth_idx + 1]
                        seq_result = speculator.speculate_from_extended_context(
                            context, draft_path, suffix_match_len
                        )
                        seq_has_gt = any(
                            len(c) > 0 and c[0] == target_token
                            for c in seq_result.candidates
                        )
                        if seq_has_gt:
                            stats.case2a += 1
                        else:
                            stats.case2b += 1
                    else:
                        stats.case2b += 1
                elif not eagle_correct and suffix_has_gt:
                    stats.case3 += 1
                else:
                    stats.case4 += 1

    return stats, dict(depth_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Compute EAGLE-3 vs SuffixDecoding complementarity"
    )
    parser.add_argument("--eagle3-results", required=True)
    parser.add_argument("--suffix-results", required=True)
    parser.add_argument("--output-dir", default="simulation/results/complementarity")
    parser.add_argument("--suffix-match-len", type=int, default=16)
    parser.add_argument("--check-sequential", action="store_true",
                        help="Also check Case 2a (requires re-running suffix matching)")
    args = parser.parse_args()

    eagle3_dir = Path(args.eagle3_results)
    suffix_dir = Path(args.suffix_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally build speculator for Case 2a analysis
    speculator = None
    if args.check_sequential:
        print("Building speculator for Case 2a analysis...")
        all_tokens = []
        for f in sorted(eagle3_dir.glob("*.json")):
            if f.name == "summary.json":
                continue
            with open(f) as fh:
                record = json.load(fh)
            if record["generated_tokens"]:
                all_tokens.append(record["generated_tokens"])

        speculator = SuffixSpeculator(suffix_match_len=args.suffix_match_len)
        for tokens in all_tokens:
            speculator.update_global(tokens)

    print("Computing case distribution...")
    stats, depth_stats = compute_cases(
        eagle3_dir, suffix_dir, speculator, args.suffix_match_len
    )

    # Save results
    result = {
        "cases": stats.to_dict(),
        "depth_stats": {
            d: {
                "depth": ds.depth,
                "total": ds.total,
                "p_accept": ds.p_accept,
                "p_match": ds.p_match,
                "eagle_accepted": ds.eagle_accepted,
                "suffix_matched": ds.suffix_matched,
            }
            for d, ds in sorted(depth_stats.items())
        },
    }

    with open(output_dir / "complementarity.json", "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n=== Case Distribution ===")
    print(f"  Case 1 (both correct):       {stats.case1:5d} ({stats.case1_rate:.3f})")
    print(f"  Case 2a (sequential value):   {stats.case2a:5d} ({stats.case2a_rate:.3f})")
    print(f"  Case 2b (EAGLE sufficient):   {stats.case2b:5d} ({stats.case2b_rate:.3f})")
    print(f"  Case 3 (suffix compensates):  {stats.case3:5d} ({stats.case3_rate:.3f})")
    print(f"  Case 4 (both fail):           {stats.case4:5d} ({stats.case4_rate:.3f})")
    print(f"  Total:                        {stats.total:5d}")
    print(f"\n  Fusion value (C1+C3):         {stats.fusion_value:.3f}")
    print(f"  Sequential unique (C2a):      {stats.sequential_unique_value:.3f}")

    print("\n=== Depth Statistics ===")
    for d in sorted(depth_stats.keys()):
        ds = depth_stats[d]
        print(f"  Depth {d}: P_accept={ds.p_accept:.3f}, P_match={ds.p_match:.3f} (n={ds.total})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
