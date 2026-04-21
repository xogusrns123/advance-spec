"""Prepare SpecBench / MT-Bench dataset for oracle simulation.

Downloads MT-Bench prompts from HuggingFace and converts to the
SpecBench JSONL format expected by specbench_agent.py.

Usage:
    python3 simulation/scripts/prepare_specbench_data.py
    python3 simulation/scripts/prepare_specbench_data.py --output-dir data/specbench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_specbench(output_dir: Path) -> None:
    """Download MT-Bench prompts and save as SpecBench JSONL."""
    from datasets import load_dataset

    print("Downloading MT-Bench prompts from HuggingFace...")
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dataset.jsonl"

    records = []
    for row in ds:
        records.append({
            "question_id": row["prompt_id"],
            "category": row["category"],
            "turns": row["prompt"],
        })

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Category distribution
    from collections import Counter
    cats = Counter(r["category"] for r in records)
    print(f"Saved {len(records)} questions to {out_path}")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/specbench",
                        help="Output directory (default: data/specbench)")
    args = parser.parse_args()
    prepare_specbench(Path(args.output_dir))


if __name__ == "__main__":
    main()
