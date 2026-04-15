#!/usr/bin/env python3
"""Prepare BFCL datasets for oracle simulation.

Two separate datasets:
  - BFCLv3 (multi_turn): Multi-turn function calling with simulators
    Categories: base, miss_func, miss_param, long_context
    Tools: GorillaFileSystem, TwitterAPI, VehicleControlAPI, etc.

  - BFCLv4 (agent): Single-turn agent tasks with real API calls
    Categories: web_search, memory
    Tools: WebSearchAPI (DuckDuckGo), MemoryAPI

Prerequisites:
    pip install bfcl-eval

Usage:
    python3 scripts/prepare_bfcl_data.py                    # both
    python3 scripts/prepare_bfcl_data.py --benchmark v3     # multi_turn only
    python3 scripts/prepare_bfcl_data.py --benchmark v4     # agent only
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


def _load_bfcl_categories(
    data_dir: Path,
    categories: list[tuple[str, str]],
    prefix: str,
) -> tuple[list[dict], list[dict]]:
    """Load BFCL categories from bfcl-eval package."""
    answer_dir = data_dir / "possible_answer"
    records = []
    ground_truths = []
    idx = 0

    for cat_name, filename in categories:
        fpath = data_dir / filename
        if not fpath.exists():
            print(f"  SKIP {filename}: not found")
            continue

        gt_path = answer_dir / filename
        gt_map = {}
        if gt_path.exists():
            with open(gt_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        gt_entry = json.loads(line)
                        gt_map[gt_entry["id"]] = gt_entry.get("ground_truth")

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
                    "category": f"{prefix}/{cat_name}",
                    "bfcl_id": bfcl_id,
                    "question": row["question"],
                    "function": row.get("function", []),
                    "initial_config": row.get("initial_config", {}),
                    "involved_classes": row.get("involved_classes", []),
                    "id": bfcl_id,
                }

                if "missed_function" in row:
                    record["missed_function"] = row["missed_function"]
                if "scenario" in row:
                    record["scenario"] = row["scenario"]

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

    return records, ground_truths


def prepare_bfcl_v3(output_dir: Path) -> None:
    """BFCLv3: Multi-turn function calling with simulators."""
    print("Preparing BFCLv3 (multi_turn) dataset...")

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

    records, ground_truths = _load_bfcl_categories(
        data_dir, categories, prefix="bfcl_v3")

    out_dir = output_dir / "bfcl_multi_turn"
    save_jsonl(records, out_dir / "dataset.jsonl")
    if ground_truths:
        save_jsonl(ground_truths, out_dir / "ground_truth.jsonl")

    print(f"  Total: {len(records)} records, {len(ground_truths)} ground truths\n")


def prepare_bfcl_v4(output_dir: Path) -> None:
    """BFCLv4: Agent tasks (web_search, memory)."""
    print("Preparing BFCLv4 (agent) dataset...")

    try:
        import bfcl_eval
        data_dir = Path(bfcl_eval.__file__).parent / "data"
    except ImportError:
        print("  ERROR: bfcl-eval not installed. Run: pip install bfcl-eval")
        return

    categories = [
        ("web_search", "BFCL_v4_web_search.json"),
        ("memory", "BFCL_v4_memory.json"),
    ]

    records, ground_truths = _load_bfcl_categories(
        data_dir, categories, prefix="bfcl_v4")

    out_dir = output_dir / "bfcl_agent"
    save_jsonl(records, out_dir / "dataset.jsonl")
    if ground_truths:
        save_jsonl(ground_truths, out_dir / "ground_truth.jsonl")

    print(f"  Total: {len(records)} records, {len(ground_truths)} ground truths\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare BFCL datasets")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--benchmark", default="all",
                        choices=["all", "v3", "v4"],
                        help="Which benchmark to prepare (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.benchmark in ("all", "v3"):
        prepare_bfcl_v3(output_dir)
    if args.benchmark in ("all", "v4"):
        prepare_bfcl_v4(output_dir)


if __name__ == "__main__":
    main()
