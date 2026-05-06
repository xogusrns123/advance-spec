"""Per-method, per-depth accept rates for the depth-axis comparison.

For every step we evaluate four trees against the same ground-truth
future and accumulate per-depth seq / ind / cond stats:

  eagle3     : standalone eagle3 backbone tree.
  suffix     : standalone suffix tree from sx.speculate(base_ctx).
  extension  : merged tree (eagle3 backbone + per-base-node suffix
               grafts, NO dedup). Per-depth indicator is set-OR over
               all (b, e) cells with b + e == depth.
  anchor     : DEDUPED extension tree, restricted to the subtree
               rooted at the 'anchor' node (= the deepest eagle3
               backbone node greedily matched against the ground
               truth, OR the virtual root if eagle3 missed at depth
               1). Position p is depth FROM anchor and is matched
               against gt[eg_chain + p - 1].

Output: same schema as before, with by_method = {eagle3, suffix,
extension, anchor}. depth_ge / cond_denom describe each method's own
denominator policy:

  eagle3, suffix, extension : depth_ge[d-1] += 1 once per step where
    len(gt) ≥ d.
  anchor                    : depth_ge[p-1] += 1 once per step where
    len(gt) - eg_chain ≥ p (so position p has a target gt token).

cond_denom and cond_accept follow the same per-step "greedy walk
attempted / matched" semantics as position_accept_rates.

Suffix cache update: same one-bonus-token policy as
compute_position_accepts.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.evaluation.tree_knapsack import (  # noqa: E402
    position_accept_rates,
    position_accept_rates_2d,
)
from simulation.pipeline.assemble_records import (  # noqa: E402
    assemble_records_from_artifacts,
)
from simulation.scripts.compute_extension_position_accepts_2d import (  # noqa: E402
    _build_extension_tree,
)


SUFFIX_FACTOR = 4.0
SUFFIX_MIN_PROB = 0.0


def _empty_stats(max_d: int) -> Dict[str, List[int]]:
    return {
        "seq_accept":  [0] * max_d,
        "ind_accept":  [0] * max_d,
        "depth_ge":    [0] * max_d,
        "cond_accept": [0] * max_d,
        "cond_denom":  [0] * max_d,
    }


def _accumulate(stats, seq, ind, ca, cd, dn):
    """Bump per-depth counters. ``dn`` is the unconditional denom-cap
    (= min(len(gt), max_d) for absolute-depth methods, or
       min(len(gt) - eg_chain, max_d) for the anchor method)."""
    for d in range(dn):
        stats["depth_ge"][d] += 1
        stats["seq_accept"][d] += seq[d]
        stats["ind_accept"][d] += ind[d]
        stats["cond_accept"][d] += ca[d]
        stats["cond_denom"][d] += cd[d]


def _project_2d_to_1d(seq2, ind2, ca2, cd2, dg2, max_d):
    """Per-depth set-OR projection of an (b, e) grid onto depth = b+e.
    depth-d indicator is 1 iff *any* cell at that depth is 1."""
    rows = len(seq2)
    cols = len(seq2[0]) if rows else 0
    seq = [0] * max_d
    ind = [0] * max_d
    ca  = [0] * max_d
    cd  = [0] * max_d
    for b in range(rows):
        for e in range(cols):
            d = b + e
            if d == 0 or d > max_d:
                continue
            i = d - 1
            if seq2[b][e]: seq[i] = 1
            if ind2[b][e]: ind[i] = 1
            if ca2[b][e]:  ca[i]  = 1
            if cd2[b][e]:  cd[i]  = 1
    return seq, ind, ca, cd


def _find_anchor(base_tids, base_pids, gt, max_b):
    """Greedy walk on the backbone tree against the ground truth.
    Returns (eg_chain, anchor_idx).

    eg_chain : number of backbone tokens matched in order (= depth
               reached by greedy walk on backbone).
    anchor_idx : index into base_tids of the deepest matched backbone
                 node. -1 if eg_chain == 0 (anchor is the virtual root).
    """
    cur = -1
    eg_chain = 0
    cap = min(max_b, len(gt))
    for d in range(1, cap + 1):
        gt_tok = gt[d - 1]
        matched = -1
        for i, p in enumerate(base_pids):
            if p == cur and base_tids[i] == gt_tok:
                matched = i
                break
        if matched < 0:
            break
        cur = matched
        eg_chain = d
    return eg_chain, cur


def _walk_extension_for_kt(tids: List[int], pids: List[int],
                            coords: List[Tuple[int, int]],
                            gt: list) -> Tuple[int, int]:
    """Greedy walk on the (no-dedup) extension tree. Returns (k, tau):
      k   = consecutive backbone (e == 0) matches from root.
      tau = subsequent graft (e >= 1) matches once the walk leaves the
            backbone. Total accepted = k + tau.

    Tie-break: base nodes are inserted first into the extension tree by
    ``_build_extension_tree``, so when iterating children-by-parent in
    insertion order, backbone children are tried before graft children
    (greedy walk follows backbone preferentially when both match)."""
    if not tids or not gt:
        return 0, 0
    children_by_p: Dict[int, List[int]] = {}
    for i, p in enumerate(pids):
        children_by_p.setdefault(p, []).append(i)
    cur = -1
    k = 0
    tau = 0
    in_backbone = True
    for gt_tok in gt:
        cands = children_by_p.get(cur, [])
        matched = -1
        for c in cands:
            if tids[c] == gt_tok:
                matched = c
                break
        if matched < 0:
            break
        b, e = coords[matched]
        if e == 0 and in_backbone:
            k += 1
        else:
            in_backbone = False
            tau += 1
        cur = matched
    return k, tau


def _extract_anchor_subtree(tids: List[int], pids: List[int],
                            anchor_idx: int
                            ) -> Tuple[List[int], List[int]]:
    """Return (sub_tids, sub_pids) for the subtree rooted at anchor.

    If anchor_idx == -1 (virtual-root anchor), the full tree is
    returned unchanged. Otherwise the anchor node is dropped and its
    direct children become the new virtual-root children. Tree
    is re-indexed in topological order.
    """
    if anchor_idx < 0:
        return list(tids), list(pids)
    n = len(tids)
    # BFS collect descendants (excluding anchor itself).
    children_idx: Dict[int, List[int]] = {}
    for i, p in enumerate(pids):
        children_idx.setdefault(p, []).append(i)
    desc_order: List[int] = []
    queue = list(children_idx.get(anchor_idx, []))
    seen = set(queue)
    while queue:
        front = queue
        queue = []
        for n_ in front:
            desc_order.append(n_)
            for c in children_idx.get(n_, []):
                if c not in seen:
                    seen.add(c)
                    queue.append(c)
    # Re-index
    old_to_new = {old: new for new, old in enumerate(desc_order)}
    sub_tids: List[int] = []
    sub_pids: List[int] = []
    for old in desc_order:
        sub_tids.append(int(tids[old]))
        old_p = pids[old]
        if old_p == anchor_idx:
            sub_pids.append(-1)
        else:
            sub_pids.append(old_to_new.get(old_p, -1))
    return sub_tids, sub_pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-results", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--responses", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--reslice-steps", type=int, default=8)
    ap.add_argument("--reslice-topk", type=int, default=8)
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=16)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--max-depth", type=int, default=64)
    ap.add_argument("--max-b", type=int, default=8)
    ap.add_argument("--max-e", type=int, default=16)
    ap.add_argument("--output", required=True)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--max-questions", type=int, default=None)
    args = ap.parse_args()

    eagle3_reslice = (args.capture_steps, args.capture_topk,
                      args.reslice_steps, args.reslice_topk)

    print(f"[depth] loading capture: {args.agent_results}", file=sys.stderr)
    t0 = time.time()
    records = assemble_records_from_artifacts(
        agent_results_path=args.agent_results,
        suffix_drafts_path=None, draft_model_drafts_path=None,
        mtp_agent_results_path=None, exclude_path=args.exclude,
        model=args.model, dataset_path=args.dataset,
        responses_path=args.responses, eagle3_reslice=eagle3_reslice,
    )
    print(f"[depth] {len(records)} records in {time.time() - t0:.1f}s",
          file=sys.stderr)

    max_d = args.max_depth

    by_seq: Dict[tuple, List[dict]] = defaultdict(list)
    for rec in records:
        by_seq[(rec["request_id"], rec.get("call_idx", 0))].append(rec)
    for k in by_seq:
        by_seq[k].sort(key=lambda r: r.get("step_idx", 0))

    try:
        from hybrid_spec_decoding.suffix_decoding.suffix_tree import (
            SuffixDecodingCache,
        )
    except Exception as e:
        sys.exit(f"[depth] FATAL: cannot import SuffixDecodingCache: {e}")

    sx = SuffixDecodingCache(
        max_tree_depth=max(max_d, 64), max_cached_requests=100000)

    methods = ("eagle3", "suffix", "extension", "anchor")
    stats = {m: _empty_stats(max_d) for m in methods}

    # ── Survival-decomposition traces (per-step k, tau, and standalone
    # chain lengths). (k, tau) measured on the DEDUPED extension tree.
    # Histograms accumulated incrementally so the output stays small.
    # k beyond max_b is excluded (per spec).
    K_BIN = args.max_b + 1                 # k = 0..max_b inclusive
    TAU_BIN = args.max_e + 1               # tau = 0..max_e inclusive
    L_BIN = max_d + 1                      # L = 0..max_d inclusive
    trace = {
        "n_steps": 0,
        "n_excluded_k_overflow": 0,
        "n_excluded_tau_overflow": 0,
        "anchor_hist":          [0] * K_BIN,                # len K_BIN
        "anchor_tau_hist":      [[0] * TAU_BIN for _ in range(K_BIN)],
        "L_eagle_hist":         [0] * L_BIN,
        "L_suffix_hist":        [0] * L_BIN,
        "L_extension_hist":     [0] * L_BIN,
    }

    n_seqs = 0
    n_steps = 0
    n_no_base = 0
    last_log = time.time()
    seq_keys = list(by_seq.keys())
    if args.max_questions:
        seq_keys = seq_keys[:args.max_questions]

    for ki, key in enumerate(seq_keys):
        seq = by_seq[key]
        if not seq:
            continue
        n_seqs += 1
        cache_key = f"{key[0]}_{key[1]}"
        prompt_ctx = seq[0].get("context_token_ids") or []
        sx.start_request(cache_key, np.asarray(prompt_ctx, dtype=np.int32))

        for rec in seq:
            gt = rec.get("ground_truth_future") or []
            base = (rec.get("per_proposer") or {}).get("eagle3")
            base_ctx = rec.get("context_token_ids") or []
            if not gt or not base or not base.get("token_ids") or not base_ctx:
                if gt:
                    sx.add_active_response(cache_key, [int(gt[0])])
                if not base or not base.get("token_ids"):
                    n_no_base += 1
                continue

            base_tids = list(base["token_ids"])
            base_pids = list(base["parents"])

            # ── eagle3
            e_seq, e_ind, e_ca, e_cd, e_dn = position_accept_rates(
                base_tids, base_pids, gt, max_d)
            if e_dn > 0:
                _accumulate(stats["eagle3"], e_seq, e_ind, e_ca, e_cd, e_dn)

            # ── suffix
            try:
                draft = sx.speculate(
                    cache_key, np.asarray(base_ctx, dtype=np.int32),
                    max_spec_factor=SUFFIX_FACTOR,
                    min_token_prob=SUFFIX_MIN_PROB,
                    use_tree_spec=True)
                sf_tids = list(draft.token_ids) if draft and draft.token_ids else []
                sf_pids = list(draft.parents) if draft and draft.token_ids else []
            except Exception:
                sf_tids, sf_pids = [], []
            s_seq, s_ind, s_ca, s_cd, s_dn = position_accept_rates(
                sf_tids, sf_pids, gt, max_d)
            if s_dn > 0:
                _accumulate(stats["suffix"], s_seq, s_ind, s_ca, s_cd, s_dn)

            # ── extension (no-dedup, all cells, set-OR per depth)
            tids_e, pids_e, coords_e = _build_extension_tree(
                base_tids, base_pids, base_ctx,
                sx, cache_key, args.budget, args.max_b, args.max_e,
                dedup=False)
            seq2, ind2, ca2, cd2, dg2, _ = position_accept_rates_2d(
                tids_e, pids_e, coords_e, gt, args.max_b, args.max_e)
            ex_seq, ex_ind, ex_ca, ex_cd = _project_2d_to_1d(
                seq2, ind2, ca2, cd2, dg2, max_d)
            ex_dn = min(len(gt), max_d)
            _accumulate(stats["extension"], ex_seq, ex_ind, ex_ca, ex_cd, ex_dn)

            # ── anchor: DEDUPED extension tree, subtree at anchor.
            # Anchor = deepest matched backbone node (or virtual root
            # if eg_chain == 0). Position p in the subtree corresponds
            # to gt[eg_chain + p - 1].
            eg_chain, anchor_idx = _find_anchor(
                base_tids, base_pids, gt, args.max_b)
            tids_d, pids_d, coords_d = _build_extension_tree(
                base_tids, base_pids, base_ctx,
                sx, cache_key, args.budget, args.max_b, args.max_e,
                dedup=True)
            sub_tids, sub_pids = _extract_anchor_subtree(
                tids_d, pids_d, anchor_idx)
            gt_after = gt[eg_chain:] if eg_chain < len(gt) else []
            if gt_after:
                a_seq, a_ind, a_ca, a_cd, a_dn = position_accept_rates(
                    sub_tids, sub_pids, gt_after, max_d)
                if a_dn > 0:
                    _accumulate(stats["anchor"], a_seq, a_ind, a_ca, a_cd, a_dn)

            # ── Survival-decomposition traces.
            from simulation.evaluation.tree_knapsack import greedy_tree_walk
            L_eagle = greedy_tree_walk(base_tids, base_pids, gt)
            L_suffix = greedy_tree_walk(sf_tids, sf_pids, gt)
            ext_k, ext_tau = _walk_extension_for_kt(
                tids_d, pids_d, coords_d, gt)
            L_extension = ext_k + ext_tau
            trace["n_steps"] += 1
            if ext_k >= K_BIN:
                trace["n_excluded_k_overflow"] += 1
            else:
                trace["anchor_hist"][ext_k] += 1
                if ext_tau >= TAU_BIN:
                    trace["n_excluded_tau_overflow"] += 1
                    # Still count the anchor; just don't bin tau.
                else:
                    trace["anchor_tau_hist"][ext_k][ext_tau] += 1
            if L_eagle < L_BIN:
                trace["L_eagle_hist"][L_eagle] += 1
            else:
                trace["L_eagle_hist"][L_BIN - 1] += 1
            if L_suffix < L_BIN:
                trace["L_suffix_hist"][L_suffix] += 1
            else:
                trace["L_suffix_hist"][L_BIN - 1] += 1
            if L_extension < L_BIN:
                trace["L_extension_hist"][L_extension] += 1
            else:
                trace["L_extension_hist"][L_BIN - 1] += 1

            n_steps += 1
            sx.add_active_response(cache_key, [int(gt[0])])

        sx.stop_request(cache_key)

        if time.time() - last_log > 30:
            print(f"[depth] {ki + 1}/{len(seq_keys)} seqs, "
                  f"{n_steps} steps, no_base={n_no_base}",
                  file=sys.stderr)
            last_log = time.time()

    elapsed = time.time() - t0
    print(f"[depth] DONE — {n_seqs} seqs, {n_steps} steps in "
          f"{elapsed:.1f}s, no_base={n_no_base}", file=sys.stderr)

    out = {
        "metadata": {
            "input_source": args.agent_results,
            "max_depth": max_d,
            "max_b": args.max_b, "max_e": args.max_e,
            "budget": args.budget,
            "reslice": {"S": args.capture_steps, "K": args.capture_topk,
                        "s": args.reslice_steps, "k": args.reslice_topk},
            "n_sequences": n_seqs,
            "n_steps": n_steps,
            "n_dropped_no_base_tree": n_no_base,
            "elapsed_sec": elapsed,
            "_doc": (
                "Per-method per-depth (or per-position-from-anchor) "
                "accept rates. Methods: eagle3 (standalone), suffix "
                "(standalone, sx.speculate(base_ctx)), extension "
                "(no-dedup merged tree, set-OR projection), anchor "
                "(DEDUPED merged tree's subtree rooted at the deepest "
                "eagle3-matched backbone node — or virtual root if "
                "eagle3 missed at depth 1; position p is depth from "
                "anchor, matched against gt[eg_chain + p - 1]).")
        },
        "max_depth": max_d,
        "by_method": stats,
        "trace": trace,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[depth] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
