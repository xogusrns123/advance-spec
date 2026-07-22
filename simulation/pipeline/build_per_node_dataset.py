"""Build a per-node JSONL dataset from oracle captures.

For each step in each capture, build the MAXIMUM tree we can construct:
  * EAGLE backbone = the full captured pool resliced to (steps=8, topk=16)
    via ``pool_reslicer.reslice_eagle3_pool`` (identity slice when (s,k) ==
    capture's S,K).
  * Suffix grafts at every backbone node AND at the virtual root, with no
    F/T filter (min_token_prob=0.0, max_spec_tokens=0).
  * Dedup rule: on (parent_token_id, token_id) collision EAGLE wins.

Suffix cache is re-simulated per call: fresh ``SuffixDecodingCache`` at
the start of each (request_id, call_idx) group, advanced by exactly ONE
ground-truth token between steps (no speculative pollution).

One JSONL row per node. ``target_p_t`` is intentionally dropped — the
captured ``eagle3_tree_p_t`` only covers the verified node (oracle force-1
mode) and re-running target verification is out of scope.

CLI:
    python -m simulation.pipeline.build_per_node_dataset \
        --benchmark bfcl_v4 \
        --capture-root /path/to/<bench>_steps8_topk16_capture \
        --output-dir /path/to/per_node_dataset/qwen3_14b \
        --limit-steps 100        # probe mode
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from simulation.pipeline._agent_io import extract_requests
from simulation.pipeline.assemble_records import collect_step_records
from simulation.pipeline.per_node_features import (
    compute_cond_probs_from_path,
    compute_children,
    compute_depths,
    compute_entropy_at_parent,
    compute_n_descendants,
    compute_sibling_rank,
    compute_top1_prob_at_parent,
    greedy_accepted_path,
    latency_lookup_target_ms,
)


SUFFIX_SPEC_KWARGS: Dict[str, Any] = dict(
    max_spec_factor=4.0,
    min_token_prob=0.0,
    max_spec_tokens=None,  # None → cache.max_tree_depth (= 64)
)


def _build_running_context_for_call(
    records: List[dict],
) -> List[List[int]]:
    """Reconstruct the full running context (prompt + accepted GT so far)
    for every step in a call.

    The first record per call has the full prompt in ``context_token_ids``
    (assembler's ``first_in_call=True`` path). Subsequent records keep
    only the trailing 128 tokens. We reconstruct by appending one GT
    token (the actually accepted next-token) between steps.

    Returns a list aligned with ``records`` where entry ``i`` is the full
    running context at the start of step ``i``.
    """
    if not records:
        return []
    full_prompt = list(records[0]["context_token_ids"])
    contexts: List[List[int]] = [full_prompt]
    running = list(full_prompt)
    for i, rec in enumerate(records[:-1]):
        gt = rec.get("ground_truth_future") or []
        if gt:
            running.append(int(gt[0]))
        contexts.append(list(running))
    return contexts


def _graft_suffix_at_anchor(
    cache,
    cache_req_id,
    anchor_idx: int,
    anchor_path_tokens: List[int],
    ext_context: List[int],
    ext_token_ids: List[int],
    ext_parents: List[int],
    ext_source: List[str],
    ext_extension_anchor: List[int],
    ext_match_length: List[Optional[int]],
    ext_suffix_cum_prob: List[Optional[float]],
    ext_suffix_freq: List[Optional[int]],
    ext_path_prob_root: List[Optional[float]],
    anchor_path_prob: Optional[float],
    existing_children_by_parent: Dict[int, Dict[int, int]],
) -> None:
    """Speculate suffix at one anchor and graft into the extended tree.

    ``anchor_idx`` is the index of the anchor in the extended tree (-1
    for virtual root). ``anchor_path_tokens`` is the token sequence from
    root to anchor (inclusive of anchor) — used to temporarily extend
    the suffix cache so speculation can see the anchor's continuation.

    Mutates the ``ext_*`` lists and ``existing_children_by_parent``
    in place.
    """
    # Temporary cache extension covers the path from root to anchor (i.e.,
    # what we'd "have decoded" if we'd accepted the entire backbone up
    # to this anchor). For virtual-root grafts, the extension is empty.
    if anchor_path_tokens:
        cache.extend_active_response(cache_req_id, anchor_path_tokens)
    try:
        draft = cache.speculate(cache_req_id, ext_context, **SUFFIX_SPEC_KWARGS)
    finally:
        if anchor_path_tokens:
            cache.pop_active_response(cache_req_id)

    if not draft.token_ids:
        return

    # Map suffix-local index -> extended-tree index. Skip nodes that
    # collide with an existing EAGLE/suffix child of the same parent.
    suffix_idx_to_ext: Dict[int, int] = {}
    for j, tok in enumerate(draft.token_ids):
        sp = draft.parents[j]
        # Parent in extended tree:
        if sp < 0:
            ext_parent = anchor_idx
        else:
            mapped = suffix_idx_to_ext.get(sp)
            if mapped is None:
                # Parent suffix node was deduped away -> drop this child
                # too (no orphans).
                continue
            ext_parent = mapped

        existing = existing_children_by_parent.setdefault(ext_parent, {})
        if tok in existing:
            # Dedup: an existing child already covers (ext_parent, tok).
            # Reuse it for any descendants but do NOT add a new node.
            suffix_idx_to_ext[j] = existing[tok]
            continue

        new_idx = len(ext_token_ids)
        ext_token_ids.append(int(tok))
        ext_parents.append(int(ext_parent))
        ext_source.append("suffix")
        ext_extension_anchor.append(int(anchor_idx))
        ext_match_length.append(int(draft.match_len))
        # draft.probs[j] is the per-node cumulative path prob within the
        # graft (relative to the anchor).
        sfx_cum = float(draft.probs[j]) if j < len(draft.probs) else None
        ext_suffix_cum_prob.append(sfx_cum)
        # raw count: number of suffixes through this node (from Arctic patch).
        cnt = int(draft.counts[j]) if j < len(draft.counts) else None
        ext_suffix_freq.append(cnt)
        # Root-relative path prob = anchor_path_prob * sfx_cum. Anchor=-1
        # is virtual root so anchor_path_prob=1.0.
        base_pp = 1.0 if anchor_idx < 0 else (anchor_path_prob or 0.0)
        if sfx_cum is None:
            ext_path_prob_root.append(None)
        else:
            ext_path_prob_root.append(float(base_pp) * float(sfx_cum))

        existing[tok] = new_idx
        suffix_idx_to_ext[j] = new_idx


def _path_from_root(node_idx: int, parents: List[int],
                    tokens: List[int]) -> List[int]:
    """Tokens from root down to node_idx, inclusive of node_idx."""
    path: List[int] = []
    cur = node_idx
    while cur >= 0:
        path.append(tokens[cur])
        cur = parents[cur]
    path.reverse()
    return path


def build_extended_tree(
    cache,
    cache_req_id,
    running_context: List[int],
    backbone_token_ids: List[int],
    backbone_parents: List[int],
    backbone_path_draft_p_t: List[float],
) -> Dict[str, Any]:
    """Construct the max extended tree at this step.

    Returns a dict with all per-node feature arrays needed for emission.
    """
    n_bb = len(backbone_token_ids)

    ext_token_ids: List[int] = list(backbone_token_ids)
    ext_parents: List[int] = list(backbone_parents)
    ext_source: List[str] = ["eagle"] * n_bb
    ext_extension_anchor: List[int] = [-2] * n_bb  # sentinel: backbone has no anchor
    ext_match_length: List[Optional[int]] = [None] * n_bb
    ext_suffix_cum_prob: List[Optional[float]] = [None] * n_bb
    ext_suffix_freq: List[Optional[int]] = [None] * n_bb
    # Root-relative path prob: backbone uses path_draft_p_t directly.
    ext_path_prob_root: List[Optional[float]] = [
        float(p) if p is not None else None for p in backbone_path_draft_p_t
    ]

    existing_children_by_parent: Dict[int, Dict[int, int]] = defaultdict(dict)
    for i, p in enumerate(backbone_parents):
        existing_children_by_parent[int(p)][int(backbone_token_ids[i])] = i

    # Virtual-root graft first (anchor_idx=-1), then per-backbone-node grafts.
    anchor_list: List[Tuple[int, List[int], Optional[float]]] = []
    anchor_list.append((-1, [], 1.0))  # virtual root: no path-from-anchor
    for i in range(n_bb):
        anchor_path = _path_from_root(i, backbone_parents, backbone_token_ids)
        anchor_pp = ext_path_prob_root[i]
        anchor_list.append((i, anchor_path, anchor_pp))

    # The suffix cache sees ``ext_context`` to find matches.
    # For backbone anchors, ext_context = running_context + path-from-root-to-anchor
    # (so the cache's most-recent tokens align with where this anchor "would be"
    # in the decoded sequence). For virtual-root grafts, ext_context = running_context.
    for anchor_idx, anchor_path, anchor_pp in anchor_list:
        if anchor_idx < 0:
            ext_context = list(running_context)
        else:
            ext_context = list(running_context) + list(anchor_path)

        # temporary_extension expects only the NEW tokens being added; we
        # pass anchor_path (root-to-anchor) so the cache sees those tokens
        # in its tree during this speculate call.
        _graft_suffix_at_anchor(
            cache=cache,
            cache_req_id=cache_req_id,
            anchor_idx=anchor_idx,
            anchor_path_tokens=anchor_path,
            ext_context=ext_context,
            ext_token_ids=ext_token_ids,
            ext_parents=ext_parents,
            ext_source=ext_source,
            ext_extension_anchor=ext_extension_anchor,
            ext_match_length=ext_match_length,
            ext_suffix_cum_prob=ext_suffix_cum_prob,
            ext_suffix_freq=ext_suffix_freq,
            ext_path_prob_root=ext_path_prob_root,
            anchor_path_prob=anchor_pp,
            existing_children_by_parent=existing_children_by_parent,
        )

    return {
        "token_ids": ext_token_ids,
        "parents": ext_parents,
        "source": ext_source,
        "extension_anchor": ext_extension_anchor,
        "match_length": ext_match_length,
        "suffix_cum_prob": ext_suffix_cum_prob,
        "suffix_freq": ext_suffix_freq,
        "path_prob_root": ext_path_prob_root,
        "n_backbone": n_bb,
    }


def emit_rows_for_step(
    record: dict,
    benchmark: str,
    ext: Dict[str, Any],
    latency_target_forward: Optional[Dict[int, float]],
    truncated: bool,
) -> Iterator[dict]:
    """Yield per-node JSONL rows for one step's extended tree."""
    rid = record["request_id"]
    cidx = int(record["call_idx"])
    sidx = int(record["step_idx"])

    n = len(ext["token_ids"])
    parents = ext["parents"]
    token_ids = ext["token_ids"]
    source = ext["source"]
    path_prob_root = ext["path_prob_root"]

    depths = compute_depths(parents)
    desc = compute_n_descendants(parents)
    children = compute_children(parents)

    # EAGLE cond prob (only meaningful for EAGLE nodes; for suffix nodes
    # path_prob_root mixes anchor & suffix terms — we still derive a
    # "cond prob" via path ratio but only label it as p_eagle for backbone).
    cond_probs_all = compute_cond_probs_from_path(parents, path_prob_root)
    # Restrict cond_probs to backbone-only entries for EAGLE sibling stats.
    eagle_cond: List[Optional[float]] = [
        cp if (source[i] == "eagle" and cp is not None) else None
        for i, cp in enumerate(cond_probs_all)
    ]
    rank_eagle = compute_sibling_rank(parents, eagle_cond)
    top1_eagle = compute_top1_prob_at_parent(parents, eagle_cond)
    entropy_eagle = compute_entropy_at_parent(parents, eagle_cond)

    # Suffix edge prob = suffix_cum_prob[i] / suffix_cum_prob[parent] (only
    # when both are suffix nodes under the same anchor). For suffix nodes
    # whose parent is backbone, edge_prob equals the cum_prob itself (no
    # ratio inside the graft).
    sfx_cum = ext["suffix_cum_prob"]
    suffix_edge_prob: List[Optional[float]] = [None] * n
    for i in range(n):
        if source[i] != "suffix":
            continue
        p = parents[i]
        if p < 0 or source[p] != "suffix":
            # Edge from anchor (backbone or root) into this suffix node
            suffix_edge_prob[i] = sfx_cum[i]
            continue
        pcum = sfx_cum[p]
        if pcum is None or pcum <= 0 or sfx_cum[i] is None:
            suffix_edge_prob[i] = None
        else:
            suffix_edge_prob[i] = float(sfx_cum[i]) / float(pcum)

    # Greedy walk against GT future
    gt = record.get("ground_truth_future") or []
    accepted_path = greedy_accepted_path(token_ids, parents, gt)
    accepted_set = set(accepted_path)
    step_accepted_length = len(accepted_path)

    n_siblings: List[int] = []
    for i in range(n):
        p = parents[i]
        # n_siblings = count of OTHER children of this parent.
        n_siblings.append(max(0, len(children.get(p, [])) - 1))

    target_cost = (
        latency_lookup_target_ms(n, latency_target_forward)
        if latency_target_forward is not None else None
    )

    for i in range(n):
        is_backbone = source[i] == "eagle"
        ext_anchor = ext["extension_anchor"][i]
        if is_backbone:
            extension_anchor_out: Optional[int] = None
        else:
            extension_anchor_out = int(ext_anchor)

        row = {
            # A. Metadata
            "benchmark": benchmark,
            "request_id": rid,
            "call_idx": cidx,
            "step_idx": sidx,
            "node_id": i,
            "parent_id": int(parents[i]),
            "depth": int(depths[i]),
            "token_id": int(token_ids[i]),
            "source": source[i],

            # B. EAGLE features
            "p_eagle": eagle_cond[i] if is_backbone else None,
            "path_prob_eagle": (
                float(path_prob_root[i])
                if (is_backbone and path_prob_root[i] is not None) else None
            ),
            "rank_eagle": rank_eagle[i] if is_backbone else None,
            "entropy_at_parent_eagle": (
                entropy_eagle[i] if is_backbone else None
            ),
            "p_eagle_top1": top1_eagle[i] if is_backbone else None,
            "logit_eagle": None,

            # C. Suffix features
            "match_length": (
                ext["match_length"][i] if not is_backbone else None
            ),
            "suffix_freq": (
                ext["suffix_freq"][i] if not is_backbone else None
            ),
            "suffix_edge_prob": (
                suffix_edge_prob[i] if not is_backbone else None
            ),
            "suffix_cum_prob": (
                sfx_cum[i] if not is_backbone else None
            ),

            # D. Structure
            "n_siblings": int(n_siblings[i]),
            "n_descendants": int(desc[i]),
            "is_backbone": bool(is_backbone),
            "extension_anchor": extension_anchor_out,

            # E. Ground truth
            "is_accepted": bool(i in accepted_set),
            "is_in_accepted_path": bool(i in accepted_set),
            "step_accepted_length": int(step_accepted_length),
            # target_p_t intentionally omitted — only verified node has it
            # (oracle force-1 mode) and re-running target is out of scope.

            # F. Step-level (denormalized on each row)
            "total_nodes": int(n),
            "target_cost_ms": target_cost,
            "truncated": bool(truncated),
        }
        yield row


def _make_shard_writers(
    output_dir: Path, prefix: str, n_shards: int, gzip_compress: bool,
    append: bool = False,
) -> Tuple[List[Any], List[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "ab" if append else "wb"
    writers: List[Any] = []
    paths: List[Path] = []
    for s in range(n_shards):
        p = output_dir / f"{prefix}.shard{s:02d}.jsonl{'.gz' if gzip_compress else ''}"
        if gzip_compress:
            w = gzip.open(p, mode)
        else:
            w = open(p, mode)
        writers.append(w)
        paths.append(p)
    return writers, paths


def _shard_for_request(rid: str, n_shards: int) -> int:
    return (hash(rid) & 0x7FFFFFFF) % n_shards


_RIDCIDX_PAT = None


def _ridcidx_pat():
    """Compile + cache the regex used by shard scans."""
    global _RIDCIDX_PAT
    if _RIDCIDX_PAT is None:
        import re
        _RIDCIDX_PAT = re.compile(
            rb'"request_id":"([^"]*)","call_idx":(-?\d+)')
    return _RIDCIDX_PAT


def _scan_shard_keys(
    output_dir: Path, benchmark: str
) -> Tuple[set, Dict[Path, Tuple[str, int]]]:
    """Scan existing shards. Return:
      * all_keys: set of (rid, cidx) pairs seen anywhere.
      * last_per_shard: per-shard, the LAST (rid, cidx) seen in that file
        (best-effort: gzip tail may be truncated by kill — caller treats
        these as suspect).

    Fast regex-only scan; no JSON parse.
    """
    pat = _ridcidx_pat()
    all_keys: set = set()
    last_per_shard: Dict[Path, Tuple[str, int]] = {}
    for p in sorted(output_dir.glob(f"{benchmark}.shard*.jsonl.gz")):
        if p.stat().st_size == 0:
            continue
        last: Optional[Tuple[str, int]] = None
        try:
            with gzip.open(p, "rb") as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        key = (m.group(1).decode(), int(m.group(2)))
                        all_keys.add(key)
                        last = key
        except (OSError, EOFError):
            # gzip tail truncated — stop reading this file
            pass
        if last is not None:
            last_per_shard[p] = last
    return all_keys, last_per_shard


def _read_manifest(manifest_path: Path) -> set:
    """Read the completed-call manifest. One line per call: rid\\tcidx."""
    out: set = set()
    if not manifest_path.exists():
        return out
    with open(manifest_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            try:
                out.add((parts[0], int(parts[1])))
            except ValueError:
                continue
    return out


def _rewrite_shards_excluding(
    output_dir: Path, benchmark: str, exclude_keys: set,
) -> None:
    """Rewrite each shard, dropping rows whose (rid, cidx) is in
    exclude_keys. Atomic per-shard: write to .tmp, fsync, rename.

    Tolerates a corrupt gzip tail (kept rows up to the corruption point).
    """
    if not exclude_keys:
        return
    pat = _ridcidx_pat()
    for p in sorted(output_dir.glob(f"{benchmark}.shard*.jsonl.gz")):
        if p.stat().st_size == 0:
            continue
        tmp = p.with_name(p.name + ".tmp")
        kept = 0
        dropped = 0
        try:
            with gzip.open(p, "rb") as fin, gzip.open(tmp, "wb") as fout:
                for line in fin:
                    m = pat.search(line)
                    if m:
                        key = (m.group(1).decode(), int(m.group(2)))
                        if key in exclude_keys:
                            dropped += 1
                            continue
                    fout.write(line)
                    kept += 1
                fout.flush()
            # fsync the renamed-to file
            with open(tmp, "rb") as fchk:
                os.fsync(fchk.fileno())
        except (OSError, EOFError):
            pass
        os.replace(tmp, p)
        if kept or dropped:
            print(f"[resume] shard {p.name}: kept {kept} rows, "
                  f"dropped {dropped} rows", file=sys.stderr)


def _read_inflight(inflight_path: Path) -> Optional[Tuple[str, int]]:
    """Read the in-flight (rid, cidx) marker, if any. Returns None if
    missing or empty."""
    if not inflight_path.exists():
        return None
    try:
        with open(inflight_path) as f:
            line = f.readline().rstrip("\n")
    except OSError:
        return None
    if not line:
        return None
    parts = line.split("\t")
    if len(parts) != 2:
        return None
    try:
        return (parts[0], int(parts[1]))
    except ValueError:
        return None


def _write_inflight(inflight_path: Path, key: Tuple[str, int]) -> None:
    """Mark a (rid, cidx) as in-flight. Atomic-ish: write to .tmp, fsync,
    rename."""
    tmp = inflight_path.with_name(inflight_path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(f"{key[0]}\t{key[1]}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, inflight_path)


def _prepare_resume_state(
    output_dir: Path, benchmark: str,
) -> Tuple[set, Path, Path]:
    """Build the set of (rid, cidx) considered complete on disk, after
    cleaning partial rows. Returns (completed_keys, manifest_path, inflight_path).

    Three startup paths:
      1. manifest + inflight marker both exist (clean previous kill via
         this code): trust manifest, purge ONE key (the in-flight), fast.
      2. manifest exists, no inflight marker: clean shutdown last time,
         no purge needed.
      3. no manifest (legacy/first resume): full regex scan; heuristic =
         last-per-shard keys are suspect → purge them, seed manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{benchmark}._completed.tsv"
    inflight_path = output_dir / f"{benchmark}._inflight.tsv"

    if manifest_path.exists():
        completed = _read_manifest(manifest_path)
        inflight = _read_inflight(inflight_path)
        if inflight is not None:
            # The in-flight call was definitively partial — purge it.
            print(f"[resume] purging in-flight call {inflight} from shards",
                  file=sys.stderr)
            _rewrite_shards_excluding(output_dir, benchmark, {inflight})
            # Also drop from completed in case caller already added it (defensive)
            completed.discard(inflight)
            # Remove the inflight marker
            try:
                os.unlink(inflight_path)
            except FileNotFoundError:
                pass
        else:
            print(f"[resume] manifest authoritative ({len(completed)} "
                  f"completed); no in-flight marker", file=sys.stderr)
        return completed, manifest_path, inflight_path

    # No manifest: legacy migration via full scan + heuristic.
    all_keys, last_per_shard = _scan_shard_keys(output_dir, benchmark)
    suspect = set(last_per_shard.values())
    completed = all_keys - suspect
    if all_keys:
        print(f"[resume] no manifest found; full-scan migration: "
              f"{len(all_keys)} total keys, "
              f"{len(suspect)} suspect (partial), "
              f"{len(completed)} complete", file=sys.stderr)
    if suspect:
        _rewrite_shards_excluding(output_dir, benchmark, suspect)
    # Seed manifest with the complete set so subsequent resumes are exact.
    with open(manifest_path, "w") as f:
        for rid, cidx in sorted(completed):
            f.write(f"{rid}\t{cidx}\n")
        f.flush()
        os.fsync(f.fileno())
    return completed, manifest_path, inflight_path


def _stream_calls(
    agent_trajectory_path: str,
    tokenizer=None,
) -> Iterator[Tuple[Tuple[str, int], List[dict]]]:
    """Stream per-call record groups, one (rid, call_idx) at a time.

    Avoids holding all records in memory: parses one question via ijson,
    extracts its requests, runs collect_step_records on that single
    question (with eagle3_reslice = full pool identity), groups by call,
    and yields each (rid, call_idx) → step list. Frees the question
    dict between yields.
    """
    import ijson
    with open(agent_trajectory_path, "rb") as f:
        for q in ijson.items(f, "questions.item"):
            single_data = {"questions": [q], "per_request": []}
            reqs = extract_requests(
                single_data, set(), None, tokenizer, None, None, None)
            if not reqs:
                continue
            partial = collect_step_records(
                reqs,
                suffix_by_key=None,
                dm_by_key=None,
                mtp_requests=None,
                dm_capture_requests=None,
                eagle3_reslice=(8, 16, 8, 16),
            )
            # Group by (rid, call_idx)
            groups: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
            order: List[Tuple[str, int]] = []
            for r in partial:
                key = (r["request_id"], int(r["call_idx"]))
                if key not in groups:
                    order.append(key)
                groups[key].append(r)
            for key in order:
                groups[key].sort(key=lambda x: int(x["step_idx"]))
                yield key, groups[key]


def run(
    benchmark: str,
    capture_root: Path,
    output_dir: Path,
    soft_cap_nodes: int,
    n_shards: int,
    gzip_compress: bool,
    limit_steps: int,
    limit_requests: int,
    probe_stats_path: Optional[Path],
    model: Optional[str] = None,
) -> None:
    from arctic_inference.suffix_decoding.cache import SuffixDecodingCache

    tokenizer = None
    if model:
        from transformers import AutoTokenizer
        print(f"[builder] loading tokenizer: {model}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model)

    agent_trajectory_path = str(capture_root / "agent_results_eagle3.json")
    latency_data_path = capture_root / "latency_data.json"
    target_forward: Optional[Dict[int, float]] = None
    if latency_data_path.exists():
        with open(latency_data_path) as f:
            lat = json.load(f)
        # Keys are strings of pow-of-2 ints
        target_forward = {int(k): float(v)
                          for k, v in (lat.get("target_forward_ms") or {}).items()}

    print(f"[builder] benchmark={benchmark}", file=sys.stderr)
    print(f"[builder] agent_trajectory={agent_trajectory_path}", file=sys.stderr)
    print(f"[builder] output_dir={output_dir}", file=sys.stderr)
    print(f"[builder] soft_cap={soft_cap_nodes} shards={n_shards} "
          f"limit_steps={limit_steps} limit_requests={limit_requests}",
          file=sys.stderr)

    # Resume prep: clean partial rows, load completed-set, open manifest.
    completed_keys, manifest_path, inflight_path = _prepare_resume_state(
        output_dir, benchmark)
    manifest_fh = open(manifest_path, "a", buffering=1)
    if completed_keys:
        print(f"[builder] resume: skipping {len(completed_keys)} "
              f"already-completed calls", file=sys.stderr)

    # Shard writers (APPEND mode — we keep already-clean rows)
    writers, shard_paths = _make_shard_writers(
        output_dir, benchmark, n_shards, gzip_compress, append=True)

    step_sizes: List[int] = []
    steps_emitted = 0
    rows_emitted = 0
    truncated_count = 0
    requests_seen: set = set()
    t0 = time.time()

    stop_flag = False
    try:
        for (rid, cidx), recs in _stream_calls(agent_trajectory_path, tokenizer):
            if stop_flag:
                break
            if (rid, cidx) in completed_keys:
                continue
            if (limit_requests is not None and limit_requests > 0 and
                    rid not in {r for r, _ in requests_seen} and
                    len({r for r, _ in requests_seen}) >= limit_requests):
                break
            if not recs:
                continue
            requests_seen.add((rid, cidx))

            # Mark this call as in-flight BEFORE we touch shards. If
            # killed mid-call, next resume reads this marker and purges
            # the partial rows from shards.
            _write_inflight(inflight_path, (rid, cidx))

            cache = SuffixDecodingCache(
                max_tree_depth=64, enable_undo=True)
            cache_req_id = f"{rid}__call{cidx}"
            running_contexts = _build_running_context_for_call(recs)

            # Prime the cache with the full prompt of step 0.
            prompt = running_contexts[0] if running_contexts else []
            try:
                cache.start_request(
                    cache_req_id, np.array(prompt, dtype=np.int32))
            except ValueError:
                # Already active (shouldn't happen since cache is fresh)
                pass

            try:
                for step_pos, rec in enumerate(recs):
                    if (limit_steps is not None and limit_steps > 0 and
                            steps_emitted >= limit_steps):
                        stop_flag = True
                        break

                    e3 = (rec.get("per_proposer") or {}).get("eagle3")
                    if not e3 or not e3.get("token_ids"):
                        # Skip steps with no EAGLE tree
                        continue
                    bb_tids = list(e3["token_ids"])
                    bb_pids = list(e3["parents"])
                    bb_pp = list(e3.get("path_draft_p_t") or [])
                    if len(bb_pp) != len(bb_tids):
                        # Defensive: pad/clip
                        bb_pp = bb_pp[:len(bb_tids)] + [None] * max(
                            0, len(bb_tids) - len(bb_pp))

                    running_context = running_contexts[step_pos]

                    ext = build_extended_tree(
                        cache=cache,
                        cache_req_id=cache_req_id,
                        running_context=running_context,
                        backbone_token_ids=bb_tids,
                        backbone_parents=bb_pids,
                        backbone_path_draft_p_t=bb_pp,
                    )

                    truncated = False
                    if soft_cap_nodes and len(ext["token_ids"]) > soft_cap_nodes:
                        ext, truncated = _apply_soft_cap(ext, soft_cap_nodes)

                    step_sizes.append(len(ext["token_ids"]))
                    if truncated:
                        truncated_count += 1

                    shard = _shard_for_request(rid, n_shards)
                    for row in emit_rows_for_step(
                            rec, benchmark, ext, target_forward, truncated):
                        writers[shard].write(
                            (json.dumps(row, separators=(",", ":"))
                             + "\n").encode("utf-8"))
                        rows_emitted += 1

                    steps_emitted += 1
                    if steps_emitted % 50 == 0:
                        sz = step_sizes
                        p50 = sorted(sz)[len(sz) // 2] if sz else 0
                        elapsed = time.time() - t0
                        rate = steps_emitted / max(elapsed, 1e-6)
                        print(f"[builder] steps={steps_emitted} rows={rows_emitted} "
                              f"p50_tree_size={p50} truncated={truncated_count} "
                              f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                              file=sys.stderr)

                    # Advance cache by ONE real GT token
                    gt = rec.get("ground_truth_future") or []
                    if gt:
                        cache.add_active_response(cache_req_id, [int(gt[0])])
            finally:
                try:
                    cache.stop_request(cache_req_id)
                except Exception:
                    pass

            # Commit this call: flush gzip → fsync → manifest → drop inflight.
            if not stop_flag:
                # gzip flush forces a sync block; underlying fd fsync persists it.
                for w in writers:
                    try:
                        w.flush()
                    except Exception:
                        pass
                for w in writers:
                    raw = getattr(w, "fileobj", None) or getattr(w, "_fp", None)
                    fd = getattr(raw, "fileno", None)
                    if callable(fd):
                        try:
                            os.fsync(fd())
                        except OSError:
                            pass
                manifest_fh.write(f"{rid}\t{cidx}\n")
                manifest_fh.flush()
                os.fsync(manifest_fh.fileno())
                completed_keys.add((rid, cidx))
                # Inflight marker no longer applies — drop it.
                try:
                    os.unlink(inflight_path)
                except FileNotFoundError:
                    pass

            if stop_flag:
                break
    finally:
        for w in writers:
            w.close()
        try:
            manifest_fh.close()
        except Exception:
            pass

    # Write probe stats if requested
    if probe_stats_path is not None and step_sizes:
        sz = sorted(step_sizes)
        def pct(p):
            idx = max(0, min(len(sz) - 1, int(len(sz) * p / 100)))
            return sz[idx]
        stats = {
            "benchmark": benchmark,
            "n_steps": len(sz),
            "n_rows": rows_emitted,
            "tree_size": {
                "min": int(sz[0]),
                "p50": int(pct(50)),
                "p90": int(pct(90)),
                "p95": int(pct(95)),
                "p99": int(pct(99)),
                "max": int(sz[-1]),
                "mean": float(sum(sz) / len(sz)),
            },
            "n_truncated_steps": int(truncated_count),
            "soft_cap": int(soft_cap_nodes) if soft_cap_nodes else 0,
            "elapsed_sec": time.time() - t0,
            "shards": [str(p) for p in shard_paths],
        }
        probe_stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(probe_stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[builder] wrote probe stats -> {probe_stats_path}",
              file=sys.stderr)

    print(f"[builder] DONE steps={steps_emitted} rows={rows_emitted} "
          f"truncated={truncated_count} elapsed={time.time() - t0:.1f}s",
          file=sys.stderr)


def _apply_soft_cap(
    ext: Dict[str, Any], cap: int
) -> Tuple[Dict[str, Any], bool]:
    """Drop lowest-path_prob_root suffix leaves until tree size <= cap.

    Never drops backbone nodes (source='eagle'). Never orphans: only drops
    leaves, repeatedly.
    """
    parents = ext["parents"]
    n = len(parents)
    if n <= cap:
        return ext, False

    source = ext["source"]
    pp = ext["path_prob_root"]

    # Build children map
    children = defaultdict(list)
    for i, p in enumerate(parents):
        children[p].append(i)

    alive = [True] * n
    n_alive = n

    # Iteratively drop the lowest path_prob_root suffix leaf.
    while n_alive > cap:
        candidate: Optional[int] = None
        best_pp: Optional[float] = None
        for i in range(n):
            if not alive[i]:
                continue
            if source[i] != "suffix":
                continue
            # leaf check: no alive children
            kids = children.get(i, [])
            if any(alive[k] for k in kids):
                continue
            score = pp[i] if pp[i] is not None else -1.0
            if candidate is None or score < best_pp:
                candidate = i
                best_pp = score
        if candidate is None:
            # No suffix leaf left — can't trim further without orphaning
            # backbone (which we forbid).
            break
        alive[candidate] = False
        n_alive -= 1

    if n_alive == n:
        return ext, False

    # Compact arrays
    old_to_new: Dict[int, int] = {}
    new_to_old: List[int] = []
    for i in range(n):
        if alive[i]:
            old_to_new[i] = len(new_to_old)
            new_to_old.append(i)

    def _remap_list(lst):
        return [lst[i] for i in new_to_old]

    new_parents = []
    for i in new_to_old:
        p = parents[i]
        new_parents.append(-1 if p < 0 else old_to_new[p])

    new_ext = {
        "token_ids": _remap_list(ext["token_ids"]),
        "parents": new_parents,
        "source": _remap_list(ext["source"]),
        "extension_anchor": [
            ext["extension_anchor"][i] if ext["extension_anchor"][i] < 0
            else (old_to_new[ext["extension_anchor"][i]]
                  if ext["extension_anchor"][i] in old_to_new else -2)
            for i in new_to_old
        ],
        "match_length": _remap_list(ext["match_length"]),
        "suffix_cum_prob": _remap_list(ext["suffix_cum_prob"]),
        "suffix_freq": _remap_list(ext["suffix_freq"]),
        "path_prob_root": _remap_list(ext["path_prob_root"]),
        "n_backbone": ext["n_backbone"],
    }
    return new_ext, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    choices=["specbench", "bfcl_v4", "swebench_verified"])
    ap.add_argument("--capture-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--soft-cap-nodes", type=int, default=0,
                    help="0 = no cap")
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--no-gzip", action="store_true")
    ap.add_argument("--limit-steps", type=int, default=0,
                    help="0 = no limit")
    ap.add_argument("--limit-requests", type=int, default=0,
                    help="0 = no limit")
    ap.add_argument("--probe-stats", type=Path, default=None,
                    help="Write tree-size distribution JSON for probe runs")
    ap.add_argument("--model", type=str, default=None,
                    help="HF model id for tokenizer-based prompt "
                         "reconstruction (improves suffix-cache priming)")
    args = ap.parse_args()

    run(
        benchmark=args.benchmark,
        capture_root=args.capture_root,
        output_dir=args.output_dir,
        soft_cap_nodes=args.soft_cap_nodes,
        n_shards=args.shards,
        gzip_compress=not args.no_gzip,
        limit_steps=args.limit_steps,
        limit_requests=args.limit_requests,
        probe_stats_path=args.probe_stats,
        model=args.model,
    )


if __name__ == "__main__":
    main()
