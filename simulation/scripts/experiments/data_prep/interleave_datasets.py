#!/usr/bin/env python3
"""Interleave per-category/per-repo so the first N items in each dataset
cover all sub-categories evenly. Early-stop guarantees diversity.

Order rule: round-robin by category (e.g., cat0[0], cat1[0], cat2[0], ...,
catK[0], cat0[1], cat1[1], ...) — first K items cover every category.
"""
import json
import os
from collections import OrderedDict


def interleave_jsonl(src: str, dst: str, key_fn) -> None:
    """Read src jsonl, group by key_fn, interleave round-robin."""
    buckets: OrderedDict[str, list[str]] = OrderedDict()
    with open(src) as f:
        for line in f:
            e = json.loads(line)
            k = key_fn(e)
            buckets.setdefault(k, []).append(line)

    interleaved: list[str] = []
    cats = list(buckets.keys())
    max_len = max(len(b) for b in buckets.values())
    for i in range(max_len):
        for c in cats:
            if i < len(buckets[c]):
                interleaved.append(buckets[c][i])

    with open(dst, "w") as f:
        f.writelines(interleaved)
    print(f"{dst}: {len(interleaved)} items, {len(cats)} categories")
    for c, b in buckets.items():
        print(f"  {c}: {len(b)}")


# specbench
interleave_jsonl(
    "/workspace/data/specbench/dataset.jsonl",
    "/workspace/data/specbench/dataset_interleaved.jsonl",
    lambda e: e.get("category", "?"),
)

# bfcl_v4 stratified
interleave_jsonl(
    "/workspace/data/bfcl_agent/dataset_stratified.jsonl",
    "/workspace/data/bfcl_agent/dataset_stratified_interleaved.jsonl",
    lambda e: e.get("category", "?"),
)

# bfcl_v3 stratified
interleave_jsonl(
    "/workspace/data/bfcl_multi_turn/dataset_stratified.jsonl",
    "/workspace/data/bfcl_multi_turn/dataset_stratified_interleaved.jsonl",
    lambda e: e.get("category", "?"),
)

# swebench_verified — by repo
interleave_jsonl(
    "/workspace/data/swebench_verified/dataset.jsonl",
    "/workspace/data/swebench_verified/dataset_interleaved.jsonl",
    lambda e: e.get("repo", "?"),
)

# LCC — by language
interleave_jsonl(
    "/workspace/data/longbench_lcc/dataset.jsonl",
    "/workspace/data/longbench_lcc/dataset_interleaved.jsonl",
    lambda e: e.get("category", "?"),
)

# RepoBench — by language
interleave_jsonl(
    "/workspace/data/longbench_repobench/dataset.jsonl",
    "/workspace/data/longbench_repobench/dataset_interleaved.jsonl",
    lambda e: e.get("category", "?"),
)
