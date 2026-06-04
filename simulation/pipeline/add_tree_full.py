"""Backfill 01b_tree_full.jsonl.gz from existing 03_anchor_raw + 06b_node_raw.

For each step, reconstruct the full extended tree (backbone + suffix) as
per-node arrays, emit one self-contained JSONL row. No suffix-cache
re-simulation needed.

Output schema (per row):
  run_id, sample_id, call_idx, step_id,
  backbone_size, extended_size,
  tree_token_ids: List[int]      # length = extended_size
  tree_parents:   List[int]
  tree_depth:     List[int]
  tree_source:    List[str]      # "eagle" or "suffix"
  tree_path_prob: List[float]    # path_prob_eagle for backbone, suffix_cum_prob×anchor_pp for suffix
  tree_local_prob: List[float]   # anchor_local_prob_eagle (backbone) or suffix_edge_prob (suffix)
  tree_freq:      List[int]      # null for backbone, suffix_freq for suffix
  tree_is_accepted: List[bool]   # on greedy accepted path

CLI:
  python -m simulation.pipeline.add_tree_full \\
      --bench-dir /workspace/simulation/results/step_dataset/qwen3_14b/bfcl_v4 \\
      --shards 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def _read_jsonl_gz(path: Path) -> Iterator[dict]:
    if path.stat().st_size == 0:
        return
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _shard_for(rid: str, n_shards: int) -> int:
    return (hash(rid) & 0x7FFFFFFF) % n_shards


def _stream_step_groups(shard: Path) -> Iterator[Tuple[Tuple, List[dict]]]:
    """Yield (step_key, list_of_rows) groups by streaming one shard.

    Assumes rows for one step are CONSECUTIVE in the file (true for our
    builder which writes per-step in a tight inner loop).
    """
    if not shard.exists() or shard.stat().st_size == 0:
        return
    current_key = None
    buffer: List[dict] = []
    for row in _read_jsonl_gz(shard):
        key = (row["sample_id"], row["call_idx"], row["step_id"])
        if current_key is None:
            current_key = key
        if key != current_key:
            yield current_key, buffer
            buffer = []
            current_key = key
        buffer.append(row)
    if buffer:
        yield current_key, buffer


def _process_shard_pair(
    anchor_shard: Path, node_shard: Path, writer_for_rid,
    t0: float, emit_count: List[int],
) -> None:
    """Streaming merge of one shard pair. Memory: only one step's rows
    buffered at a time per stream (a few thousand rows ≈ 1MB)."""
    anchor_stream = _stream_step_groups(anchor_shard)
    node_stream = _stream_step_groups(node_shard)

    a_key, a_rows = next(anchor_stream, (None, None))
    n_key, n_rows = next(node_stream, (None, None))

    while a_key is not None or n_key is not None:
        if a_key == n_key:
            key = a_key
            anchors = a_rows
            nodes = n_rows
            a_key, a_rows = next(anchor_stream, (None, None))
            n_key, n_rows = next(node_stream, (None, None))
        elif n_key is None or (a_key is not None and a_key < n_key):
            key = a_key
            anchors = a_rows
            nodes = []
            a_key, a_rows = next(anchor_stream, (None, None))
        else:
            key = n_key
            anchors = []
            nodes = n_rows
            n_key, n_rows = next(node_stream, (None, None))

        rid, cidx, sidx = key

        # Filter out virtual root (anchor_node_id = -1)
        backbone = sorted(
            (a for a in anchors if a["anchor_node_id"] >= 0),
            key=lambda a: a["anchor_node_id"],
        )
        suffix = sorted(nodes, key=lambda n: n["node_id"])

        if not backbone and not suffix:
            continue

        # Build arrays. Backbone nodes are indexed 0..n_bb-1 in extended tree.
        # Suffix nodes already carry node_id with extended-tree indexing
        # (n_bb..n-1).
        backbone_size = len(backbone)
        extended_size = backbone_size + len(suffix)

        token_ids = [0] * extended_size
        parents = [-1] * extended_size
        depth = [0] * extended_size
        source = [""] * extended_size
        path_prob = [None] * extended_size
        local_prob = [None] * extended_size
        freq = [None] * extended_size
        is_accepted = [False] * extended_size

        # Backbone fill: anchor_node_id IS the index (already 0-based, sorted)
        run_id = (backbone or suffix)[0]["run_id"]
        for a in backbone:
            i = int(a["anchor_node_id"])
            if i >= extended_size:
                continue
            token_ids[i] = int(a["anchor_token_id"])
            parents[i] = int(a["anchor_parent_node_id"])
            depth[i] = int(a["anchor_depth"])
            source[i] = "eagle"
            path_prob[i] = a.get("anchor_path_prob_eagle")
            local_prob[i] = a.get("anchor_local_prob_eagle")
            is_accepted[i] = bool(a.get("anchor_hit", False))

        # Suffix fill: use node_id directly
        for n in suffix:
            i = int(n["node_id"])
            if i >= extended_size:
                continue
            token_ids[i] = int(n["token_id"])
            parents[i] = int(n["parent_node_id"])
            depth[i] = int(n["depth"])
            source[i] = "suffix"
            # path_prob for suffix: derive from anchor's path_prob × suffix_cum_prob
            anchor_id = int(n["anchor_node_id"])
            if anchor_id < 0:
                # virtual root anchor; anchor_pp = 1.0
                anchor_pp = 1.0
            elif anchor_id < backbone_size and path_prob[anchor_id] is not None:
                anchor_pp = path_prob[anchor_id]
            else:
                anchor_pp = None
            sfx_cum = n.get("suffix_cum_prob")
            if anchor_pp is not None and sfx_cum is not None:
                path_prob[i] = float(anchor_pp) * float(sfx_cum)
            local_prob[i] = n.get("suffix_edge_prob")
            freq[i] = n.get("suffix_freq")
            is_accepted[i] = bool(n.get("is_accepted", False))

        row = {
            "run_id": run_id,
            "sample_id": rid,
            "call_idx": int(cidx),
            "step_id": int(sidx),
            "backbone_size": backbone_size,
            "extended_size": extended_size,
            "tree_token_ids": token_ids,
            "tree_parents": parents,
            "tree_depth": depth,
            "tree_source": source,
            "tree_path_prob": path_prob,
            "tree_local_prob": local_prob,
            "tree_freq": freq,
            "tree_is_accepted": is_accepted,
        }
        writer_for_rid(rid).write(
            (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))
        emit_count[0] += 1
        if emit_count[0] % 1000 == 0:
            print(f"  emitted {emit_count[0]} steps total "
                  f"elapsed={time.time() - t0:.1f}s",
                  file=sys.stderr)


def backfill(bench_dir: Path, n_shards: int) -> None:
    t0 = time.time()
    print(f"[backfill] bench_dir={bench_dir} shards={n_shards}",
          file=sys.stderr)

    # Open 01b writers
    writers = []
    for s in range(n_shards):
        p = bench_dir / f"01b_tree_full.shard{s:02d}.jsonl.gz"
        writers.append(gzip.open(p, "wb"))

    def writer_for_rid(rid: str):
        return writers[_shard_for(rid, n_shards)]

    emit_count = [0]

    # Process shard pairs
    anchor_shards = sorted(bench_dir.glob("03_anchor_raw.shard*.jsonl.gz"))
    node_shards = sorted(bench_dir.glob("06b_node_raw.shard*.jsonl.gz"))
    # Pair by shard index
    shard_indices = sorted(set(
        [int(p.name.split("shard")[1].split(".")[0]) for p in anchor_shards] +
        [int(p.name.split("shard")[1].split(".")[0]) for p in node_shards]
    ))
    for idx in shard_indices:
        anchor_p = bench_dir / f"03_anchor_raw.shard{idx:02d}.jsonl.gz"
        node_p = bench_dir / f"06b_node_raw.shard{idx:02d}.jsonl.gz"
        print(f"[backfill] processing shard pair idx={idx} "
              f"({anchor_p.name}, {node_p.name})", file=sys.stderr)
        _process_shard_pair(anchor_p, node_p, writer_for_rid, t0, emit_count)

    for w in writers:
        w.close()
    print(f"[backfill] DONE emitted {emit_count[0]} rows in "
          f"{time.time() - t0:.1f}s", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True, type=Path)
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()
    backfill(args.bench_dir, args.shards)


if __name__ == "__main__":
    main()
