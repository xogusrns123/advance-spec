#!/usr/bin/env python3
"""Prepare BFCL multi-turn dataset for oracle simulation.

Extracts BFCL v4 multi-turn categories from the bfcl-eval package
and saves as JSONL format.

Prerequisites:
    pip install bfcl-eval

Usage:
    python3 scripts/prepare_bfcl_data.py
    python3 scripts/prepare_bfcl_data.py --output-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {path}")


def prepare_bfcl_multi_turn(output_dir: Path) -> None:
    """Load BFCL v4 multi-turn categories from bfcl-eval package."""
    print("Preparing BFCL Multi-Turn dataset...")

    try:
        import bfcl_eval
        data_dir = Path(bfcl_eval.__file__).parent / "data"
    except ImportError:
        print("  ERROR: bfcl-eval not installed. Run: pip install bfcl-eval")
        return

    categories = [
        ("multi_turn_base", "BFCL_v4_multi_turn_base.json"),
        ("multi_turn_miss_func", "BFCL_v4_multi_turn_miss_func.json"),
        ("multi_turn_miss_param", "BFCL_v4_multi_turn_miss_param.json"),
        ("multi_turn_long_context", "BFCL_v4_multi_turn_long_context.json"),
    ]

    answer_dir = data_dir / "possible_answer"

    records = []
    ground_truths = []
    idx = 0

    for cat_name, filename in categories:
        fpath = data_dir / filename
        if not fpath.exists():
            print(f"  SKIP {filename}: not found")
            continue

        # Load ground truth
        gt_path = answer_dir / filename
        gt_map = {}
        if gt_path.exists():
            with open(gt_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        gt_entry = json.loads(line)
                        gt_map[gt_entry["id"]] = gt_entry["ground_truth"]

        count = 0
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                bfcl_id = row["id"]
                record = {
                    "question_id": idx,
                    "category": f"bfcl/{cat_name}",
                    "bfcl_id": bfcl_id,
                    "question": row["question"],
                    "function": row.get("function", []),
                    "initial_config": row.get("initial_config", {}),
                    "involved_classes": row.get("involved_classes", []),
                    "id": bfcl_id,
                }

                if "missed_function" in row:
                    record["missed_function"] = row["missed_function"]

                records.append(record)

                if bfcl_id in gt_map:
                    ground_truths.append({
                        "question_id": idx,
                        "bfcl_id": bfcl_id,
                        "ground_truth": gt_map[bfcl_id],
                    })

                idx += 1
                count += 1

        print(f"  {filename}: {count} records")

    save_jsonl(records, output_dir / "bfcl_multi_turn" / "dataset.jsonl")
    if ground_truths:
        save_jsonl(
            ground_truths,
            output_dir / "bfcl_multi_turn" / "ground_truth.jsonl",
        )

    print(f"\nTotal: {len(records)} records, {len(ground_truths)} ground truths")


def main():
    parser = argparse.ArgumentParser(description="Prepare BFCL dataset")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    prepare_bfcl_multi_turn(Path(args.output_dir))


if __name__ == "__main__":
    main()
