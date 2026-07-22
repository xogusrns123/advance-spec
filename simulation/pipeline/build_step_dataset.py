"""Build step+anchor+suffix-node dataset from oracle captures.

Per benchmark, emits 4 sharded JSONL.gz files and 3 TSV summary files:

  01_step_raw.shardNN.jsonl.gz       one row per step
  02_depth_summary.tsv               aggregate of 01 (and accepted-by-depth)
  03_anchor_raw.shardNN.jsonl.gz     one row per (step, EAGLE anchor + virtual root)
  04_anchor_depth_summary.tsv        aggregate of 03 by anchor_depth
  05_feature_bins_summary.tsv        feature-vs-accept bin analyses
  06_cost_raw.shardNN.jsonl.gz       per-step tree size + multi-scenario latency
  06b_node_raw.shardNN.jsonl.gz      one row per SUFFIX node only

A centralized 00_run_index.tsv (one row per builder invocation) sits at
``--output-dir`` root (above per-benchmark subdirs) and shares run_ids
across benchmarks.

Algorithm-agnostic: NO method-tuning fields in raw rows (no rho, lambda,
alpha, F filter, T filter except as capture-time metadata in run_index).

Resume mechanism: per-call manifest (_completed.tsv) + in-flight marker
(_inflight.tsv); on restart the in-flight call's partial rows are
purged from all 4 sharded file families atomically.

Builder caps work at ``--max-requests`` unique request_ids per benchmark
(default 10) since the full sweep takes hours.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from simulation.pipeline.build_per_node_dataset import (
    _build_running_context_for_call,
    _make_shard_writers,
    _ridcidx_pat,
    _read_inflight,
    _read_manifest,
    _rewrite_shards_excluding,
    _stream_calls,
    _write_inflight,
    build_extended_tree,
)
from simulation.pipeline.per_node_features import (
    compute_children,
    compute_depths,
    compute_n_descendants,
    greedy_accepted_path,
    latency_lookup_target_ms,
)


# ---------------------------------------------------------------------------
# Run index
# ---------------------------------------------------------------------------

_RUN_INDEX_COLUMNS = [
    "run_id",
    "date",
    "captured_at",
    "model",
    "target_model",
    "benchmark",
    "split",
    "sample_range",
    "method",
    "proposer_family",
    "base_proposer",
    "extension_proposer",
    "backbone_steps",
    "backbone_topk",
    "suffix_max_spec_factor_at_capture",
    "suffix_min_token_prob_at_capture",
    "suffix_max_spec_tokens_at_capture",
    "seed",
    "notes",
]


def _read_run_index(path: Path) -> Dict[str, dict]:
    """Read 00_run_index.tsv; return {run_id: row_dict}. Tolerates missing
    file."""
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            row = dict(zip(header, parts))
            rid = row.get("run_id")
            if rid:
                out[rid] = row
    return out


def _append_run_index(
    path: Path, row: Dict[str, Any], force: bool
) -> None:
    """Append one row to the run index. Creates the file with a header if
    missing. Errors if the run_id is already present (unless force)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_run_index(path)
    if row["run_id"] in existing and not force:
        raise SystemExit(
            f"[run_index] run_id={row['run_id']} already exists in "
            f"{path}. Pass --force-overwrite-run-id to override.")
    new_file = not path.exists()
    with open(path, "a") as f:
        if new_file:
            f.write("\t".join(_RUN_INDEX_COLUMNS) + "\n")
        f.write(
            "\t".join(str(row.get(c, "")) for c in _RUN_INDEX_COLUMNS)
            + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Resume across 4 file families
# ---------------------------------------------------------------------------

_FILE_FAMILIES = [
    "01_step_raw",
    "03_anchor_raw",
    "06_cost_raw",
    "06b_node_raw",
]


def _scan_shard_keys_for_family(
    bench_dir: Path, family: str
) -> Tuple[set, Dict[Path, Tuple[str, int]]]:
    """Scan one family's shards for (rid, cidx) keys."""
    pat = _ridcidx_pat()
    all_keys: set = set()
    last_per_shard: Dict[Path, Tuple[str, int]] = {}
    for p in sorted(bench_dir.glob(f"{family}.shard*.jsonl.gz")):
        if p.stat().st_size == 0:
            continue
        last = None
        try:
            with gzip.open(p, "rb") as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        key = (m.group(1).decode(), int(m.group(2)))
                        all_keys.add(key)
                        last = key
        except (OSError, EOFError):
            pass
        if last is not None:
            last_per_shard[p] = last
    return all_keys, last_per_shard


def _prepare_resume_state_step(
    bench_dir: Path,
) -> Tuple[set, Path, Path]:
    """Manifest+inflight-aware resume. Returns (completed, manifest_path,
    inflight_path).

    Same logic as build_per_node_dataset but applies purge to all 4 file
    families.
    """
    bench_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bench_dir / "_completed.tsv"
    inflight_path = bench_dir / "_inflight.tsv"

    if manifest_path.exists():
        completed = _read_manifest(manifest_path)
        inflight = _read_inflight(inflight_path)
        if inflight is not None:
            print(f"[resume] purging in-flight call {inflight} from all "
                  f"4 file families", file=sys.stderr)
            for fam in _FILE_FAMILIES:
                _rewrite_shards_excluding(bench_dir, fam, {inflight})
            completed.discard(inflight)
            try:
                os.unlink(inflight_path)
            except FileNotFoundError:
                pass
        else:
            print(f"[resume] manifest authoritative ({len(completed)} "
                  f"completed); no in-flight marker", file=sys.stderr)
        return completed, manifest_path, inflight_path

    # No manifest: derive heuristically. Suspect = union of last-per-shard
    # across all 4 file families.
    all_keys_union: set = set()
    suspect_union: set = set()
    for fam in _FILE_FAMILIES:
        all_k, last_per = _scan_shard_keys_for_family(bench_dir, fam)
        all_keys_union |= all_k
        suspect_union |= set(last_per.values())
    completed = all_keys_union - suspect_union
    if all_keys_union:
        print(f"[resume] no manifest found; full-scan migration: "
              f"{len(all_keys_union)} total keys, "
              f"{len(suspect_union)} suspect (partial), "
              f"{len(completed)} complete", file=sys.stderr)
    for fam in _FILE_FAMILIES:
        if suspect_union:
            _rewrite_shards_excluding(bench_dir, fam, suspect_union)
    with open(manifest_path, "w") as f:
        for rid, cidx in sorted(completed):
            f.write(f"{rid}\t{cidx}\n")
        f.flush()
        os.fsync(f.fileno())
    return completed, manifest_path, inflight_path


# ---------------------------------------------------------------------------
# Row emission
# ---------------------------------------------------------------------------

def _shard_for(rid: str, n_shards: int) -> int:
    return (hash(rid) & 0x7FFFFFFF) % n_shards


def _per_node_depths_backbone(
    parents: List[int], source: List[str]
) -> Tuple[List[Optional[int]], Dict[int, int]]:
    """Depths within the BACKBONE subtree only (suffix nodes get None).
    Also returns a map from anchor backbone idx -> n_descendants in backbone."""
    n = len(parents)
    depths: List[Optional[int]] = [None] * n
    n_desc_bb: List[int] = [0] * n
    # Process backbone in topological order (parent before child).
    # Backbone nodes have parent < node_id always due to BFS order.
    for i in range(n):
        if source[i] != "eagle":
            continue
        p = parents[i]
        if p < 0:
            depths[i] = 1
        elif p < n and source[p] == "eagle" and depths[p] is not None:
            depths[i] = depths[p] + 1
        else:
            depths[i] = 1  # fallback (shouldn't happen for backbone)
    # n_descendants in backbone subtree (only counting backbone children)
    # Walk children in reverse topo (deepest first)
    order = sorted(
        (i for i in range(n) if source[i] == "eagle"),
        key=lambda i: -(depths[i] if depths[i] is not None else 0),
    )
    for i in order:
        p = parents[i]
        if p >= 0 and p < n and source[p] == "eagle":
            n_desc_bb[p] += n_desc_bb[i] + 1
    return depths, {i: n_desc_bb[i] for i in range(n) if source[i] == "eagle"}


def _suffix_accepted_below_anchor(
    accepted_path: List[int],
    parents: List[int],
    source: List[str],
    anchor_idx: int,
) -> int:
    """Count suffix-sourced nodes in the accepted_path that are descendants
    of anchor_idx (anchor_idx=-1 means virtual root)."""
    if not accepted_path:
        return 0
    # Build "is_descendant_of_anchor" by walking parents
    accepted_set = set(accepted_path)
    count = 0
    for node in accepted_path:
        if source[node] != "suffix":
            continue
        # Walk up from node to either anchor_idx or root
        cur = node
        is_descendant = False
        while cur >= 0:
            par = parents[cur]
            if par == anchor_idx:
                is_descendant = True
                break
            cur = par
        if anchor_idx == -1 and not is_descendant:
            # virtual root case: anything not reached above is still its descendant
            # Re-check: virtual root is "parent=-1" so a node whose ancestry chain hits
            # parent=-1 at some point is a descendant of virtual root.
            # Actually ALL nodes are descendants of virtual root by definition.
            is_descendant = True
        if is_descendant:
            count += 1
    return count


def _per_anchor_suffix_stats(
    ext: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """For each anchor (backbone idx and -1 for virtual root), collect
    aggregate stats over suffix children grafted at that anchor.

    Returns: {anchor_idx: {match_len, candidate_len, total_score, avg_freq,
                            max_freq, available, child_node_ids}}.
    """
    n = len(ext["token_ids"])
    source = ext["source"]
    parents = ext["parents"]
    ext_anchor = ext["extension_anchor"]
    sfx_cum = ext["suffix_cum_prob"]
    sfx_freq = ext["suffix_freq"]
    match_len = ext["match_length"]

    by_anchor: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "candidate_len": 0,
        "total_score": 0.0,
        "freqs": [],
        "match_len": None,
        "child_ids": [],
    })

    for i in range(n):
        if source[i] != "suffix":
            continue
        a = int(ext_anchor[i])
        agg = by_anchor[a]
        agg["candidate_len"] += 1
        if sfx_cum[i] is not None:
            agg["total_score"] += float(sfx_cum[i])
        if sfx_freq[i] is not None:
            agg["freqs"].append(int(sfx_freq[i]))
        if agg["match_len"] is None and match_len[i] is not None:
            agg["match_len"] = int(match_len[i])
        agg["child_ids"].append(i)

    # Finalize
    final: Dict[int, Dict[str, Any]] = {}
    for a, agg in by_anchor.items():
        freqs = agg["freqs"]
        final[a] = {
            "available": agg["candidate_len"] > 0,
            "match_len": agg["match_len"],
            "candidate_len": agg["candidate_len"],
            "total_score": float(agg["total_score"]),
            "avg_freq": (sum(freqs) / len(freqs)) if freqs else None,
            "max_freq": max(freqs) if freqs else None,
            "child_ids": agg["child_ids"],
        }
    return final


def emit_step_anchor_node_cost_rows(
    record: dict,
    run_id: str,
    benchmark: str,
    ext: Dict[str, Any],
    latency_target_forward: Optional[Dict[int, float]],
    latency_eagle3_draft_ms: Optional[float],
    step_build_latency_ms: float,
) -> Tuple[dict, List[dict], List[dict], dict]:
    """For one step, build the 4 row groups: step_raw, anchor_raws,
    suffix_node_raws, cost_raw."""
    rid = record["request_id"]
    cidx = int(record["call_idx"])
    sidx = int(record["step_idx"])

    parents = ext["parents"]
    token_ids = ext["token_ids"]
    source = ext["source"]
    n_bb = ext["n_backbone"]
    path_prob_root = ext["path_prob_root"]

    n = len(token_ids)
    depths_ext = compute_depths(parents)
    depths_bb, n_desc_bb_map = _per_node_depths_backbone(parents, source)
    n_desc_ext = compute_n_descendants(parents)
    children = compute_children(parents)

    # Greedy walk against GT future
    gt = record.get("ground_truth_future") or []
    accepted_path = greedy_accepted_path(token_ids, parents, gt)
    accepted_set = set(accepted_path)

    accepted_max_depth_bb = max(
        (depths_bb[i] for i in accepted_path if source[i] == "eagle"),
        default=0,
    )
    accepted_max_depth_ext = max(
        (depths_ext[i] for i in accepted_path),
        default=0,
    )

    backbone_max_depth = max(
        (d for d, s in zip(depths_bb, source) if s == "eagle" and d is not None),
        default=0,
    )

    # Per-anchor suffix aggregate
    anchor_sfx = _per_anchor_suffix_stats(ext)

    # ---- 01_step_raw -------------------------------------------------------
    step_row = {
        "run_id": run_id,
        "sample_id": rid,
        "call_idx": cidx,
        "step_id": sidx,
        "backbone_size": int(n_bb),
        "backbone_max_depth": int(backbone_max_depth),
        "extended_size": int(n),
        "num_suffix_nodes": int(n - n_bb),
        "num_suffix_calls": int(n_bb + 1),  # +1 for virtual root
        "accepted_len_via_max_tree": int(len(accepted_path)),
        "accepted_max_depth_backbone": int(accepted_max_depth_bb),
        "accepted_max_depth_extended": int(accepted_max_depth_ext),
        "step_build_latency_ms": float(step_build_latency_ms),
    }

    # ---- 03_anchor_raw -----------------------------------------------------
    anchor_rows: List[dict] = []
    # Virtual root row (anchor_node_id = -1)
    vroot_stats = anchor_sfx.get(-1, {})
    vroot_acc_below = _suffix_accepted_below_anchor(
        accepted_path, parents, source, -1)
    anchor_rows.append({
        "run_id": run_id,
        "sample_id": rid,
        "call_idx": cidx,
        "step_id": sidx,
        "anchor_node_id": -1,
        "anchor_depth": 0,
        "anchor_token_id": None,
        "anchor_parent_node_id": None,
        "anchor_path_prob_eagle": 1.0,
        "anchor_local_prob_eagle": None,
        "n_descendants_backbone": int(n_bb),
        "anchor_hit": True,  # virtual root always "hit"
        "suffix_called": True,
        "suffix_available": bool(vroot_stats.get("available", False)),
        "suffix_match_len": vroot_stats.get("match_len"),
        "suffix_candidate_len": int(vroot_stats.get("candidate_len", 0)),
        "suffix_total_score": float(vroot_stats.get("total_score", 0.0)),
        "suffix_avg_freq": vroot_stats.get("avg_freq"),
        "suffix_max_freq": vroot_stats.get("max_freq"),
        "suffix_accepted_len_from_anchor": int(vroot_acc_below),
    })
    # Per-backbone-node anchor rows
    for i in range(n_bb):
        a_stats = anchor_sfx.get(i, {})
        # Compute conditional prob: path_prob[i] / path_prob[parent]; if parent
        # is virtual root, conditional == path_prob[i].
        p = parents[i]
        if p < 0:
            local_prob = path_prob_root[i]
        else:
            pp = path_prob_root[p]
            if pp is None or pp <= 0 or path_prob_root[i] is None:
                local_prob = None
            else:
                local_prob = float(path_prob_root[i]) / float(pp)
        a_acc_below = _suffix_accepted_below_anchor(
            accepted_path, parents, source, i)
        anchor_rows.append({
            "run_id": run_id,
            "sample_id": rid,
            "call_idx": cidx,
            "step_id": sidx,
            "anchor_node_id": int(i),
            "anchor_depth": int(depths_bb[i] or 0),
            "anchor_token_id": int(token_ids[i]),
            "anchor_parent_node_id": int(p),
            "anchor_path_prob_eagle": (
                float(path_prob_root[i])
                if path_prob_root[i] is not None else None),
            "anchor_local_prob_eagle": local_prob,
            "n_descendants_backbone": int(n_desc_bb_map.get(i, 0)),
            "anchor_hit": bool(i in accepted_set),
            "suffix_called": True,
            "suffix_available": bool(a_stats.get("available", False)),
            "suffix_match_len": a_stats.get("match_len"),
            "suffix_candidate_len": int(a_stats.get("candidate_len", 0)),
            "suffix_total_score": float(a_stats.get("total_score", 0.0)),
            "suffix_avg_freq": a_stats.get("avg_freq"),
            "suffix_max_freq": a_stats.get("max_freq"),
            "suffix_accepted_len_from_anchor": int(a_acc_below),
        })

    # ---- 06b_node_raw (suffix nodes only) ----------------------------------
    suffix_node_rows: List[dict] = []
    # Compute sibling counts within suffix grafts (= other children of same parent)
    for i in range(n_bb, n):
        if source[i] != "suffix":
            continue
        par = parents[i]
        # suffix_edge_prob = suffix_cum_prob[i] / suffix_cum_prob[parent]
        # if parent is suffix; else equals suffix_cum_prob[i]
        if par >= 0 and source[par] == "suffix":
            p_cum = ext["suffix_cum_prob"][par]
            if p_cum is not None and p_cum > 0 and ext["suffix_cum_prob"][i] is not None:
                edge_prob = float(ext["suffix_cum_prob"][i]) / float(p_cum)
            else:
                edge_prob = None
        else:
            edge_prob = (
                float(ext["suffix_cum_prob"][i])
                if ext["suffix_cum_prob"][i] is not None else None)

        # n_siblings_in_graft: count of other children of the same parent
        # whose source == "suffix" (backbone siblings don't count).
        siblings_same_parent = [
            k for k in children.get(par, []) if source[k] == "suffix"
        ]
        n_siblings_in_graft = max(0, len(siblings_same_parent) - 1)
        # n_descendants_in_graft: count descendants restricted to suffix
        # source. We approximate with n_descendants_ext minus any backbone
        # descendants (none — backbone is above suffix in tree topology).
        n_descendants_in_graft = int(n_desc_ext[i])

        anc = int(ext["extension_anchor"][i])
        a_depth = 0 if anc < 0 else int(depths_bb[anc] or 0)
        suffix_node_rows.append({
            "run_id": run_id,
            "sample_id": rid,
            "call_idx": cidx,
            "step_id": sidx,
            "node_id": int(i),
            "parent_node_id": int(par),
            "depth": int(depths_ext[i]),
            "token_id": int(token_ids[i]),
            "anchor_node_id": anc,
            "anchor_depth": a_depth,
            "suffix_edge_prob": edge_prob,
            "suffix_cum_prob": (
                float(ext["suffix_cum_prob"][i])
                if ext["suffix_cum_prob"][i] is not None else None),
            "suffix_freq": (
                int(ext["suffix_freq"][i])
                if ext["suffix_freq"][i] is not None else None),
            "is_accepted": bool(i in accepted_set),
            "n_siblings_in_graft": int(n_siblings_in_graft),
            "n_descendants_in_graft": int(n_descendants_in_graft),
        })

    # ---- 06_cost_raw -------------------------------------------------------
    if latency_target_forward is not None:
        target_max = latency_lookup_target_ms(n, latency_target_forward)
        target_bb_only = latency_lookup_target_ms(n_bb, latency_target_forward)
    else:
        target_max = None
        target_bb_only = None
    draft_bb_ms = float(latency_eagle3_draft_ms) if latency_eagle3_draft_ms else None
    # Suffix latency estimate: per-call CPU overhead (~20µs/node generated).
    sfx_total_us = (n - n_bb) * 20.0
    draft_suffix_ms = sfx_total_us / 1000.0

    def _safe_sum(*xs: Optional[float]) -> Optional[float]:
        if any(x is None for x in xs):
            return None
        return float(sum(x for x in xs))

    cost_row = {
        "run_id": run_id,
        "sample_id": rid,
        "call_idx": cidx,
        "step_id": sidx,
        "accepted_len_via_max_tree": int(len(accepted_path)),
        "num_eagle_nodes_generated": int(n_bb),
        "num_suffix_nodes_generated": int(n - n_bb),
        "num_suffix_calls": int(n_bb + 1),
        "target_latency_max_tree_ms": target_max,
        "target_latency_backbone_only_ms": target_bb_only,
        "draft_latency_backbone_ms": draft_bb_ms,
        "draft_latency_suffix_total_ms": draft_suffix_ms,
        "total_latency_max_tree_ms": _safe_sum(target_max, draft_bb_ms, draft_suffix_ms),
        "total_latency_backbone_only_ms": _safe_sum(target_bb_only, draft_bb_ms),
    }

    return step_row, anchor_rows, suffix_node_rows, cost_row


# ---------------------------------------------------------------------------
# Summary aggregators
# ---------------------------------------------------------------------------

class DepthAggregator:
    """Aggregate per-step acceptance into depth survival tables.

    Two proposer slices: "eagle" (depth within backbone, accepted backbone-only),
    "extended" (depth within extended tree, accepted via max tree).
    """

    def __init__(self) -> None:
        # counts[proposer][depth] = {num_eligible, num_accepted_ge_depth}
        self.counts: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"eligible": 0, "ge": 0}))
        self.total_steps = 0

    def update(self, step_row: dict) -> None:
        self.total_steps += 1
        bb_depth = int(step_row["backbone_max_depth"])
        ext_depth = int(step_row["accepted_max_depth_extended"])  # max acc
        bb_acc = int(step_row["accepted_max_depth_backbone"])
        for d in range(1, max(bb_depth, ext_depth) + 1):
            # eagle proposer: eligible iff backbone_max_depth >= d
            if bb_depth >= d:
                self.counts["eagle"][d]["eligible"] += 1
                if bb_acc >= d:
                    self.counts["eagle"][d]["ge"] += 1
            # extended proposer: eligible iff backbone_max_depth >= d (we always
            # have at least backbone depth; suffix can go deeper)
            ext_size_depth = max(bb_depth, ext_depth)  # depth available
            if ext_size_depth >= d:
                self.counts["extended"][d]["eligible"] += 1
                if ext_depth >= d:
                    self.counts["extended"][d]["ge"] += 1

    def flush(self, path: Path, run_id: str, benchmark: str) -> None:
        header = [
            "run_id", "benchmark", "proposer", "depth",
            "num_steps", "num_eligible_steps", "num_accepted_ge_depth",
            "survival", "eligible_survival", "conditional_accept",
        ]
        existing_run = self._read_existing_run_ids(path)
        new_file = not path.exists()
        with open(path, "a") as f:
            if new_file:
                f.write("\t".join(header) + "\n")
            if run_id in existing_run:
                return  # avoid duplicate rows on resume
            for prop in sorted(self.counts):
                cnts = self.counts[prop]
                # Conditional needs prev-depth count
                prev_ge = None
                for d in sorted(cnts):
                    c = cnts[d]
                    surv = c["ge"] / self.total_steps if self.total_steps else 0.0
                    el_surv = c["ge"] / c["eligible"] if c["eligible"] else 0.0
                    if prev_ge is None:
                        cond = el_surv  # depth 1 condition equals eligible survival
                    elif prev_ge == 0:
                        cond = 0.0
                    else:
                        cond = c["ge"] / prev_ge
                    f.write("\t".join([
                        run_id, benchmark, prop, str(d),
                        str(self.total_steps), str(c["eligible"]), str(c["ge"]),
                        f"{surv:.6f}", f"{el_surv:.6f}", f"{cond:.6f}",
                    ]) + "\n")
                    prev_ge = c["ge"]
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _read_existing_run_ids(path: Path) -> set:
        out: set = set()
        if not path.exists():
            return out
        with open(path) as f:
            next(f, None)  # header
            for line in f:
                parts = line.split("\t", 1)
                if parts:
                    out.add(parts[0])
        return out


class AnchorDepthAggregator:
    """Aggregate per-anchor rows by anchor_depth."""

    def __init__(self) -> None:
        # by_depth[d] = {n, hit, avail, match_lens, scores, acc_cond_hit (list), uncond_contrib}
        self.by_depth: Dict[int, dict] = defaultdict(lambda: {
            "n": 0, "hit": 0, "avail": 0,
            "match_lens": [], "scores": [],
            "acc_lens_cond_hit": [], "uncond_contrib": [],
        })

    def update(self, anchor_row: dict) -> None:
        d = int(anchor_row["anchor_depth"])
        agg = self.by_depth[d]
        agg["n"] += 1
        hit = bool(anchor_row["anchor_hit"])
        if hit:
            agg["hit"] += 1
        if bool(anchor_row["suffix_available"]):
            agg["avail"] += 1
        if anchor_row.get("suffix_match_len") is not None:
            agg["match_lens"].append(float(anchor_row["suffix_match_len"]))
        agg["scores"].append(float(anchor_row.get("suffix_total_score", 0.0)))
        acc_below = int(anchor_row.get("suffix_accepted_len_from_anchor", 0))
        if hit:
            agg["acc_lens_cond_hit"].append(acc_below)
        agg["uncond_contrib"].append(acc_below if hit else 0)

    def flush(self, path: Path, run_id: str, benchmark: str) -> None:
        header = [
            "run_id", "benchmark", "anchor_depth",
            "num_anchors", "anchor_hit_rate", "suffix_availability_rate",
            "avg_suffix_match_len", "avg_suffix_score",
            "avg_suffix_accept_len_cond_hit",
            "avg_suffix_uncond_contribution",
        ]
        existing = self._read_existing_run_ids(path)
        new_file = not path.exists()
        with open(path, "a") as f:
            if new_file:
                f.write("\t".join(header) + "\n")
            if run_id in existing:
                return
            for d in sorted(self.by_depth):
                a = self.by_depth[d]
                hit_rate = a["hit"] / a["n"] if a["n"] else 0.0
                avail_rate = a["avail"] / a["n"] if a["n"] else 0.0
                avg_ml = (
                    sum(a["match_lens"]) / len(a["match_lens"])
                    if a["match_lens"] else 0.0)
                avg_sc = (
                    sum(a["scores"]) / len(a["scores"])
                    if a["scores"] else 0.0)
                avg_acc_cond = (
                    sum(a["acc_lens_cond_hit"]) / len(a["acc_lens_cond_hit"])
                    if a["acc_lens_cond_hit"] else 0.0)
                avg_uncond = (
                    sum(a["uncond_contrib"]) / len(a["uncond_contrib"])
                    if a["uncond_contrib"] else 0.0)
                f.write("\t".join([
                    run_id, benchmark, str(d),
                    str(a["n"]), f"{hit_rate:.6f}", f"{avail_rate:.6f}",
                    f"{avg_ml:.4f}", f"{avg_sc:.6f}",
                    f"{avg_acc_cond:.4f}", f"{avg_uncond:.4f}",
                ]) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _read_existing_run_ids(path: Path) -> set:
        out: set = set()
        if not path.exists():
            return out
        with open(path) as f:
            next(f, None)
            for line in f:
                parts = line.split("\t", 1)
                if parts:
                    out.add(parts[0])
        return out


class FeatureBinAggregator:
    """Equal-frequency quantile binning for selected features. Two-pass:
    first pass collects values, second pass emits."""

    FEATURES = {
        # anchor-level features
        "suffix_total_score": "anchor",  # 03 row
        "suffix_match_len": "anchor",
        "anchor_path_prob_eagle": "anchor",
        # node-level features
        "suffix_freq": "node",          # 06b row
        "n_descendants_in_graft": "node",
    }
    N_BINS = 10

    def __init__(self) -> None:
        # values[feature_name] = list of (value, suffix_accepted_len_proxy)
        # For anchor features, proxy = suffix_accepted_len_from_anchor
        # For node features, proxy = int(is_accepted)
        self.values: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def update_anchor(self, anchor_row: dict) -> None:
        proxy = float(anchor_row.get("suffix_accepted_len_from_anchor", 0))
        for feat, kind in self.FEATURES.items():
            if kind != "anchor":
                continue
            v = anchor_row.get(feat)
            if v is None:
                continue
            self.values[feat].append((float(v), proxy))

    def update_node(self, node_row: dict) -> None:
        proxy = 1.0 if node_row.get("is_accepted") else 0.0
        for feat, kind in self.FEATURES.items():
            if kind != "node":
                continue
            v = node_row.get(feat)
            if v is None:
                continue
            self.values[feat].append((float(v), proxy))

    def flush(self, path: Path, run_id: str, benchmark: str) -> None:
        header = [
            "run_id", "benchmark", "feature_name", "bin_id",
            "bin_low", "bin_high", "num_rows", "avg_feature_value",
            "avg_suffix_accepted_len", "positive_rate_accept_gt0",
            "top_bin_lift",
        ]
        existing = self._read_existing_run_ids(path)
        new_file = not path.exists()
        with open(path, "a") as f:
            if new_file:
                f.write("\t".join(header) + "\n")
            if run_id in existing:
                return
            for feat, vals in self.values.items():
                if not vals:
                    continue
                # Compute global average proxy for lift
                global_avg = sum(p for _, p in vals) / len(vals)
                # Sort by feature value to assign bins
                sorted_vals = sorted(vals, key=lambda x: x[0])
                n = len(sorted_vals)
                bin_size = max(1, n // self.N_BINS)
                bins: List[List[Tuple[float, float]]] = []
                for b in range(self.N_BINS):
                    start = b * bin_size
                    end = (b + 1) * bin_size if b < self.N_BINS - 1 else n
                    if start >= n:
                        break
                    bins.append(sorted_vals[start:end])
                # Last bin avg as "top bin"
                top_bin_avg = (
                    sum(p for _, p in bins[-1]) / len(bins[-1])
                    if bins else 0.0)
                top_lift = (
                    top_bin_avg / global_avg if global_avg > 0 else 0.0)
                for bi, bvals in enumerate(bins):
                    feat_vals = [v for v, _ in bvals]
                    proxies = [p for _, p in bvals]
                    bin_low = min(feat_vals)
                    bin_high = max(feat_vals)
                    avg_feat = sum(feat_vals) / len(feat_vals)
                    avg_proxy = sum(proxies) / len(proxies)
                    pos_rate = sum(1 for p in proxies if p > 0) / len(proxies)
                    f.write("\t".join([
                        run_id, benchmark, feat, str(bi),
                        f"{bin_low:.6f}", f"{bin_high:.6f}",
                        str(len(bvals)), f"{avg_feat:.6f}",
                        f"{avg_proxy:.4f}", f"{pos_rate:.4f}",
                        f"{top_lift:.4f}",
                    ]) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _read_existing_run_ids(path: Path) -> set:
        out: set = set()
        if not path.exists():
            return out
        with open(path) as f:
            next(f, None)
            for line in f:
                parts = line.split("\t", 1)
                if parts:
                    out.add(parts[0])
        return out


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    benchmark: str,
    capture_root: Path,
    output_dir: Path,
    run_id: str,
    method_label: str,
    base_proposer: str,
    notes: str,
    n_shards: int,
    max_requests: int,
    limit_steps: int,
    model: Optional[str],
    target_model: Optional[str],
    split: str,
    sample_range: str,
    proposer_family: str,
    extension_proposer: str,
    seed: int,
    force_overwrite_run_id: bool,
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
    eagle3_draft_ms: Optional[float] = None
    if latency_data_path.exists():
        with open(latency_data_path) as f:
            lat = json.load(f)
        target_forward = {
            int(k): float(v)
            for k, v in (lat.get("target_forward_ms") or {}).items()
        }
        # Use a representative draft ms — eagle3_draft_ms map; pick the
        # smallest budget (4) since draft cost is roughly flat.
        e3d = lat.get("eagle3_draft_ms") or {}
        if e3d:
            eagle3_draft_ms = float(next(iter(e3d.values())))

    bench_dir = output_dir / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Run index registration
    run_index_path = output_dir / "00_run_index.tsv"
    run_row = {
        "run_id": run_id,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "model": model or "",
        "target_model": target_model or "",
        "benchmark": benchmark,
        "split": split,
        "sample_range": sample_range,
        "method": method_label,
        "proposer_family": proposer_family,
        "base_proposer": base_proposer,
        "extension_proposer": extension_proposer,
        "backbone_steps": 8,
        "backbone_topk": 16,
        "suffix_max_spec_factor_at_capture": 4.0,
        "suffix_min_token_prob_at_capture": 0.0,
        "suffix_max_spec_tokens_at_capture": 64,
        "seed": seed,
        "notes": notes,
    }
    _append_run_index(run_index_path, run_row, force_overwrite_run_id)
    print(f"[builder] registered run_id={run_id} in {run_index_path}",
          file=sys.stderr)

    # Resume prep
    completed_keys, manifest_path, inflight_path = _prepare_resume_state_step(
        bench_dir)
    manifest_fh = open(manifest_path, "a", buffering=1)
    if completed_keys:
        print(f"[builder] resume: skipping {len(completed_keys)} "
              f"already-completed calls", file=sys.stderr)
    completed_rids: set = {rid for rid, _ in completed_keys}

    # Shard writers (4 file families)
    writers: Dict[str, List[Any]] = {}
    shard_paths: Dict[str, List[Path]] = {}
    for fam in _FILE_FAMILIES:
        w, p = _make_shard_writers(
            bench_dir, fam, n_shards, gzip_compress=True, append=True)
        writers[fam] = w
        shard_paths[fam] = p

    # Aggregators
    depth_agg = DepthAggregator()
    anchor_depth_agg = AnchorDepthAggregator()
    feat_agg = FeatureBinAggregator()

    steps_emitted = 0
    rids_seen: set = set(completed_rids)
    t0 = time.time()
    stop_flag = False

    def _commit_call(rid: str, cidx: int) -> None:
        # Flush + fsync all 4 file families' writers
        for fam in _FILE_FAMILIES:
            for w in writers[fam]:
                try:
                    w.flush()
                except Exception:
                    pass
            for w in writers[fam]:
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
        try:
            os.unlink(inflight_path)
        except FileNotFoundError:
            pass

    try:
        for (rid, cidx), recs in _stream_calls(agent_trajectory_path, tokenizer):
            if stop_flag:
                break
            if (rid, cidx) in completed_keys:
                rids_seen.add(rid)
                if len(rids_seen) >= max_requests:
                    print(f"[builder] hit max_requests={max_requests} "
                          f"(from manifest); stopping", file=sys.stderr)
                    break
                continue
            # Pre-check task limit BEFORE starting a new call
            if rid not in rids_seen and len(rids_seen) >= max_requests:
                print(f"[builder] hit max_requests={max_requests}; "
                      f"stopping before new task", file=sys.stderr)
                break
            if not recs:
                continue
            rids_seen.add(rid)

            _write_inflight(inflight_path, (rid, cidx))

            cache = SuffixDecodingCache(
                max_tree_depth=64, enable_undo=True)
            cache_req_id = f"{rid}__call{cidx}"
            running_contexts = _build_running_context_for_call(recs)

            prompt = running_contexts[0] if running_contexts else []
            try:
                cache.start_request(
                    cache_req_id, np.array(prompt, dtype=np.int32))
            except ValueError:
                pass

            try:
                for step_pos, rec in enumerate(recs):
                    if (limit_steps is not None and limit_steps > 0
                            and steps_emitted >= limit_steps):
                        stop_flag = True
                        break

                    e3 = (rec.get("per_proposer") or {}).get("eagle3")
                    if not e3 or not e3.get("token_ids"):
                        continue
                    bb_tids = list(e3["token_ids"])
                    bb_pids = list(e3["parents"])
                    bb_pp = list(e3.get("path_draft_p_t") or [])
                    if len(bb_pp) != len(bb_tids):
                        bb_pp = bb_pp[:len(bb_tids)] + [None] * max(
                            0, len(bb_tids) - len(bb_pp))

                    running_context = running_contexts[step_pos]
                    step_t0 = time.time()
                    ext = build_extended_tree(
                        cache=cache,
                        cache_req_id=cache_req_id,
                        running_context=running_context,
                        backbone_token_ids=bb_tids,
                        backbone_parents=bb_pids,
                        backbone_path_draft_p_t=bb_pp,
                    )
                    step_build_ms = (time.time() - step_t0) * 1000.0

                    step_row, anchor_rows, suffix_node_rows, cost_row = (
                        emit_step_anchor_node_cost_rows(
                            record=rec,
                            run_id=run_id,
                            benchmark=benchmark,
                            ext=ext,
                            latency_target_forward=target_forward,
                            latency_eagle3_draft_ms=eagle3_draft_ms,
                            step_build_latency_ms=step_build_ms,
                        ))

                    shard = _shard_for(rid, n_shards)
                    writers["01_step_raw"][shard].write(
                        (json.dumps(step_row, separators=(",", ":"))
                         + "\n").encode("utf-8"))
                    for r in anchor_rows:
                        writers["03_anchor_raw"][shard].write(
                            (json.dumps(r, separators=(",", ":"))
                             + "\n").encode("utf-8"))
                    for r in suffix_node_rows:
                        writers["06b_node_raw"][shard].write(
                            (json.dumps(r, separators=(",", ":"))
                             + "\n").encode("utf-8"))
                    writers["06_cost_raw"][shard].write(
                        (json.dumps(cost_row, separators=(",", ":"))
                         + "\n").encode("utf-8"))

                    # Aggregators
                    depth_agg.update(step_row)
                    for ar in anchor_rows:
                        anchor_depth_agg.update(ar)
                        feat_agg.update_anchor(ar)
                    for nr in suffix_node_rows:
                        feat_agg.update_node(nr)

                    steps_emitted += 1
                    if steps_emitted % 50 == 0:
                        elapsed = time.time() - t0
                        rate = steps_emitted / max(elapsed, 1e-6)
                        print(f"[builder] steps={steps_emitted} "
                              f"rids={len(rids_seen)} "
                              f"elapsed={elapsed:.1f}s "
                              f"rate={rate:.2f}/s",
                              file=sys.stderr)

                    gt = rec.get("ground_truth_future") or []
                    if gt:
                        cache.add_active_response(
                            cache_req_id, [int(gt[0])])
            finally:
                try:
                    cache.stop_request(cache_req_id)
                except Exception:
                    pass

            if not stop_flag:
                _commit_call(rid, cidx)

            if stop_flag:
                break
            if len(rids_seen) >= max_requests:
                print(f"[builder] hit max_requests={max_requests}; "
                      f"stopping after task completion", file=sys.stderr)
                break
    finally:
        for fam in _FILE_FAMILIES:
            for w in writers[fam]:
                w.close()
        try:
            manifest_fh.close()
        except Exception:
            pass

    # Flush summary aggregators
    depth_agg.flush(bench_dir / "02_depth_summary.tsv", run_id, benchmark)
    anchor_depth_agg.flush(
        bench_dir / "04_anchor_depth_summary.tsv", run_id, benchmark)
    feat_agg.flush(
        bench_dir / "05_feature_bins_summary.tsv", run_id, benchmark)

    print(f"[builder] DONE benchmark={benchmark} run_id={run_id} "
          f"steps={steps_emitted} rids={len(rids_seen)} "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    choices=["specbench", "bfcl_v4", "swebench_verified"])
    ap.add_argument("--capture-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--run-id", type=str, default=None,
                    help="Defaults to <iso timestamp>_<benchmark>_<base_proposer>")
    ap.add_argument("--method-label", type=str, default="extension_max_tree")
    ap.add_argument("--base-proposer", type=str, default="eagle3")
    ap.add_argument("--proposer-family", type=str, default="hybrid")
    ap.add_argument("--extension-proposer", type=str, default="suffix")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--target-model", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--split", type=str, default="")
    ap.add_argument("--sample-range", type=str, default="0-9")
    ap.add_argument("--notes", type=str, default="")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--max-requests", type=int, default=10,
                    help="hard cap on unique request_ids per benchmark")
    ap.add_argument("--limit-steps", type=int, default=0,
                    help="0 = no limit (probe-mode cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-overwrite-run-id", action="store_true")
    args = ap.parse_args()

    if args.run_id is None:
        args.run_id = (
            f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}_"
            f"{args.benchmark}_{args.base_proposer}")

    run(
        benchmark=args.benchmark,
        capture_root=args.capture_root,
        output_dir=args.output_dir,
        run_id=args.run_id,
        method_label=args.method_label,
        base_proposer=args.base_proposer,
        notes=args.notes,
        n_shards=args.shards,
        max_requests=args.max_requests,
        limit_steps=args.limit_steps,
        model=args.model,
        target_model=args.target_model,
        split=args.split,
        sample_range=args.sample_range,
        proposer_family=args.proposer_family,
        extension_proposer=args.extension_proposer,
        seed=args.seed,
        force_overwrite_run_id=args.force_overwrite_run_id,
    )


if __name__ == "__main__":
    main()
