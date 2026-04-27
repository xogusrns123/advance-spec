#!/usr/bin/env python3
"""Prepare all evaluation datasets in unified jsonl format.

Outputs (in /workspace/data/):
- swebench_verified/dataset.jsonl   (500 SWE-Bench Verified, full)
- swebench_lite/dataset.jsonl       (300 SWE-Bench Lite, full)
- longbench_lcc/dataset.jsonl       (500 LongBench LCC, full)
- longbench_repobench/dataset.jsonl (500 RepoBench-P, full)
"""
from datasets import load_dataset
from collections import Counter
import json
import os
import zipfile


def write_swebench(dataset_name: str, out_dir: str) -> int:
    ds = load_dataset(dataset_name, split="test")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dataset.jsonl")
    repos = Counter()
    with open(path, "w") as f:
        for e in ds:
            d = {k: e[k] for k in e if k != "created_at"}
            if "turns" not in d:
                d["turns"] = [d.get("problem_statement", "")]
            if "category" not in d:
                d["category"] = dataset_name.split("_")[-1]
            f.write(json.dumps(d) + "\n")
            repos[e["repo"]] += 1
    print(f"{dataset_name}: wrote {len(ds)} to {path}")
    print(f"  repos:")
    for r, n in repos.most_common():
        print(f"    {r}: {n}")
    return len(ds)


def write_longbench(zip_path: str, member: str, label: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dataset.jsonl")
    n = 0
    with zipfile.ZipFile(zip_path) as zf, open(path, "w") as out:
        with zf.open(member) as f:
            for line in f:
                e = json.loads(line)
                ctx = e.get("context", "") or ""
                inp = e.get("input", "") or ""
                lang = e.get("language", "") or ""
                # Format prompt: "complete the next line"
                if inp:
                    prompt = (
                        f"Please complete the following {lang.capitalize()} "
                        f"code. Output ONLY the next line of code, no "
                        f"explanation, no comments, no code fences.\n\n"
                        f"Question:\n{inp}\n\nContext:\n```{lang}\n{ctx}\n```"
                    )
                else:
                    prompt = (
                        f"Please complete the following {lang.capitalize()} "
                        f"code. Output ONLY the next line of code, no "
                        f"explanation, no comments, no code fences.\n\n"
                        f"```{lang}\n{ctx}\n```"
                    )
                entry = {
                    "question_id": str(e["_id"]),
                    "category": f"{label}/{lang}",
                    "turns": [prompt],
                }
                out.write(json.dumps(entry) + "\n")
                n += 1
    print(f"{label}: wrote {n} to {path}")
    return n


if __name__ == "__main__":
    print("=== SWE-Bench Verified (full 500) ===")
    write_swebench("princeton-nlp/swe-bench_verified",
                   "/workspace/data/swebench_verified")

    print("\n=== SWE-Bench Lite (full 300) ===")
    write_swebench("princeton-nlp/swe-bench_lite",
                   "/workspace/data/swebench_lite")

    print("\n=== LongBench LCC (full 500) ===")
    write_longbench("/workspace/data/longbench_data.zip",
                    "data/lcc.jsonl", "longbench/lcc",
                    "/workspace/data/longbench_lcc")

    print("\n=== LongBench RepoBench-P (full 500) ===")
    write_longbench("/workspace/data/longbench_data.zip",
                    "data/repobench-p.jsonl", "longbench/repobench",
                    "/workspace/data/longbench_repobench")

    print("\nDone.")
