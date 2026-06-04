"""Build per-benchmark exclude files for first-10-task sim runs.

For each benchmark, reads the captured agent_results to enumerate all
request_ids and the _completed.tsv from step_dataset to identify our
already-processed 10 tasks. Writes an exclude file listing all OTHER
request_ids (= every task EXCEPT our chosen 10).

Output: simulation/results/step_dataset/qwen3_14b/_exclude/{bench}.txt
"""
from __future__ import annotations

import argparse
import ijson
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", required=True, type=Path,
                    help="e.g. /workspace/simulation/results/qwen3_14b")
    ap.add_argument("--step-root", required=True, type=Path,
                    help="e.g. /workspace/simulation/results/step_dataset/qwen3_14b")
    args = ap.parse_args()

    out_dir = args.step_root / "_exclude"
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench in ("bfcl_v4", "specbench", "swebench_verified"):
        completed_tsv = args.step_root / bench / "_completed.tsv"
        keep: set = set()
        with open(completed_tsv) as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    keep.add(line.split("\t")[0])

        capture_path = (args.capture_root
                        / f"{bench}_steps8_topk16_capture"
                        / "agent_results_eagle3.json")
        all_rids: list = []
        with open(capture_path, "rb") as f:
            for q in ijson.items(f, "questions.item"):
                rid = q.get("bfcl_id") or q.get("instance_id") or str(
                    q.get("question_id", ""))
                if rid:
                    all_rids.append(rid)

        exclude = [r for r in all_rids if r not in keep]
        out_path = out_dir / f"{bench}.txt"
        with open(out_path, "w") as f:
            for r in exclude:
                f.write(r + "\n")

        print(f"{bench}: total={len(all_rids)} keep={len(keep)} "
              f"exclude={len(exclude)} → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
