"""Stream-split a Stage 1 agent_results_eagle3.json into N shard files.

Why: collect_draft_model.py loads the full JSON via json.load(), which uses
~3x file size in RAM. For specbench (17GB) / swebench (25GB) captures, four
parallel shards exhaust 200GB+ RAM. Pre-splitting via ijson keeps splitter
RAM at ~1 question (<1GB) and lets each shard load only its slice.

Sharding: round-robin by question index (idx % num_shards). Each shard's
output is a fully valid JSON file with the same metadata + its subset of
questions. The downstream collect_draft_model can run on each slice without
the --shard flag (or with --shard 0/1 for compatibility).

Usage:
    python3 -m simulation.pipeline.split_agent_results \
        --src   simulation/results/qwen3_14b/.../agent_results_eagle3.json \
        --out-pattern simulation/results/qwen3_14b/.../agent_results_eagle3_shard{i}.json \
        --num-shards 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ijson


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out-pattern", required=True,
                   help="Output path with literal '{i}' for shard index, e.g. "
                        "'.../agent_results_eagle3_shard{i}.json'")
    p.add_argument("--num-shards", type=int, default=4)
    args = p.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"missing: {src}")
    if "{i}" not in args.out_pattern:
        sys.exit("--out-pattern must contain '{i}'")

    # Resume: if all shard files already exist and are non-empty, skip.
    existing = [Path(args.out_pattern.replace("{i}", str(i)))
                for i in range(args.num_shards)]
    if all(p.exists() and p.stat().st_size > 0 for p in existing):
        print(f"All {args.num_shards} shard files already present, skipping.",
              file=sys.stderr)
        return

    # Pass 1: read metadata
    # use_float=True returns Python floats (json.dump-friendly) instead of
    # Decimal, which ijson uses by default.
    with open(src, "rb") as f:
        try:
            metadata = next(ijson.items(f, "metadata", use_float=True))
        except StopIteration:
            metadata = {}
    print(f"metadata: model={metadata.get('model')!r} "
          f"benchmark={metadata.get('benchmark')!r} "
          f"num_requests={metadata.get('num_requests')}", file=sys.stderr)

    # Open output files, write metadata header + start of questions array
    fps = []
    counts = [0] * args.num_shards
    for i in range(args.num_shards):
        out_path = Path(args.out_pattern.replace("{i}", str(i)))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fp = open(out_path, "w")
        fp.write('{"metadata": ')
        json.dump(metadata, fp)
        fp.write(', "questions": [')
        fps.append(fp)

    t0 = time.time()
    bytes_in = 0
    src_size = src.stat().st_size
    last_report = t0

    with open(src, "rb") as f:
        for idx, q in enumerate(ijson.items(f, "questions.item",
                                            use_float=True)):
            target = idx % args.num_shards
            if counts[target] > 0:
                fps[target].write(",")
            json.dump(q, fps[target])
            counts[target] += 1
            del q
            now = time.time()
            if now - last_report >= 30.0:
                bytes_in = f.tell()
                pct = bytes_in / src_size * 100 if src_size else 0
                rate = bytes_in / (now - t0) / (1024 * 1024) if now > t0 else 0
                print(f"  [{idx + 1} questions] {pct:.1f}% "
                      f"({bytes_in / 1e9:.1f} / {src_size / 1e9:.1f} GB) "
                      f"{rate:.0f} MB/s", file=sys.stderr)
                last_report = now

    for fp in fps:
        fp.write("]}")
        fp.close()

    print(f"Done. shard counts={counts}, elapsed={time.time() - t0:.0f}s",
          file=sys.stderr)
    for i in range(args.num_shards):
        out_path = Path(args.out_pattern.replace("{i}", str(i)))
        size_gb = out_path.stat().st_size / 1e9
        print(f"  shard{i}: {counts[i]} questions, {size_gb:.2f} GB",
              file=sys.stderr)


if __name__ == "__main__":
    main()
