"""Prepare SWE-Bench (Lite) dataset for oracle simulation / latency measurement.

Downloads the SWE-Bench_Lite evaluation split from HuggingFace and saves it
as JSONL in the schema expected by ``simulation/agents/swebench_agent.py``
(``instance_id``, ``repo``, ``base_commit``, ``turns``, ``category``) with an
extra ``problem_statement`` field so ``_workload_prompts.py`` can also
consume it for latency measurement.

Prerequisites:
    pip install datasets

Usage:
    python3 simulation/scripts/prepare_swebench_data.py
    python3 simulation/scripts/prepare_swebench_data.py \\
        --subset princeton-nlp/SWE-bench_Verified --split test \\
        --output-dir data/swebench --num-samples 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_swebench(
    subset: str,
    split: str,
    output_dir: Path,
    num_samples: int | None,
) -> None:
    print(f"Loading HuggingFace dataset {subset} [{split}]...")
    from datasets import load_dataset
    ds = load_dataset(subset, split=split)

    rows = []
    for i, row in enumerate(ds):
        if num_samples is not None and len(rows) >= num_samples:
            break
        instance_id = row.get("instance_id") or f"swebench_{i}"
        problem = (row.get("problem_statement")
                   or row.get("issue")
                   or row.get("text")
                   or "")
        if not problem:
            continue
        rows.append({
            "instance_id": instance_id,
            "repo": row.get("repo", ""),
            "base_commit": row.get("base_commit", ""),
            "problem_statement": problem,
            "turns": [problem],
            "category": row.get("version") or "swebench",
            # Optional fields kept verbatim when present
            **{k: row[k] for k in ("hints_text", "test_patch", "patch",
                                    "FAIL_TO_PASS", "PASS_TO_PASS",
                                    "environment_setup_commit", "created_at")
               if k in row},
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dataset.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} records to {out_path}")
    if rows:
        sample = rows[0]
        print(f"Schema: {sorted(sample.keys())}")
        print(f"First instance_id: {sample['instance_id']}")
        print(f"First repo@commit: {sample['repo']}@{sample['base_commit'][:7]}")
        print(f"First problem (head): {sample['problem_statement'][:120]}...")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="princeton-nlp/SWE-bench_Verified",
                        help="HuggingFace dataset name "
                             "(default: princeton-nlp/SWE-bench_Verified)")
    parser.add_argument("--split", default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--output-dir", default="data/swebench",
                        help="Output directory (default: data/swebench)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit to first N samples (default: all)")
    args = parser.parse_args()

    prepare_swebench(
        subset=args.subset,
        split=args.split,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
