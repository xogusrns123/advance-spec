"""Per-method confidence calibration for draft trees (EAGLE-2 Fig.6 analog).

Per-DEPTH-GROUP calibration: at each (step, parent) we have one group
of sibling draft nodes (size = topk). The group is a *trial* iff its
parent path matches gt[0:d-1] AND len(gt) >= d. The group is *accepted*
iff ANY sibling has token == gt[d-1] — matching the fact that
speculative decoding accepts the depth as soon as any candidate is
right (topk siblings cannot all "fail" if one matches).

Bin conf for a group: the max sibling conf (= "best candidate" — the
one greedy walk would try first since SGLang sorts children by score).
This avoids the per-NODE 1/topk dilution that artificially caps the
score-axis accept rate at ~1/topk for big trees.

Three confidence axes are computed per method:

  per_step  : per-NODE single-step transition prob, recovered from
              cumulative path-product as conf[i] = cum[i] / cum[parent[i]].
              Range [0, 1]. Direct EAGLE-2 Fig.6 analog. 20 bins of width
              0.05.
  path_prob : per-NODE cumulative path-product (= probs[i] for suffix,
              path_draft_p_t[i] for eagle3). Range [0, 1]. The absolute
              probability that the entire root→i path is correct — the
              metric tree pruning actually uses. 20 bins of width 0.05.
  score     : per-DRAFT scalar SuffixDraft.score = sum(probs) (suffix) /
              sum(path_draft_p_t) (eagle3). Used raw (no normalization).
              Each trial node in a draft is binned by that draft's raw
              score (all nodes in one step share the same score-bin).
              20 equal-width bins on [0, --score-max] (default 8); scores
              above score_max clamp to the last bin.

Both fields' source data:
  eagle3 : per_proposer.eagle3.path_draft_p_t (cumulative path-product
           of draft head softmax probs along the path).
  suffix : SuffixDraft.probs from arctic_inference. Per the C++ source
           (vendor/ArcticInference/csrc/suffix_decoding/suffix_tree.cc
           lines 1330-1409), probs[i] is a cumulative path-product
           of frequency-based count ratios — same semantics as eagle3's
           path_draft_p_t. SuffixDraft.score = sum(probs).

Bin count defaults to 20 (0.05 width, matching EAGLE-2 Fig.6).
Output stores raw counts (trials = denominator, accepts = numerator);
rates are derivable downstream.

Suffix cache update: same one-bonus-token policy as
compute_position_accepts.py / compute_method_depth_rates.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.pipeline.assemble_records import (  # noqa: E402
    assemble_records_from_artifacts,
)


SUFFIX_FACTOR = 4.0
SUFFIX_MIN_PROB = 0.0


def _accumulate_per_node(tids: List[int], pids: List[int],
                           confs: List[float], gt: list,
                           n_bins: int,
                           trials: List[int],
                           accepts: List[int]) -> int:
    """Per-NODE binning (original EAGLE-2 Fig.6 strict semantics).
    Each draft node is a separate trial when its parent path matches gt.
    Accept iff token[n] == gt[d-1]. Bin by node's own conf.

    Used for per_step and path_prob axes — granular per-node calibration
    where high-conf bins concentrate the 'right' candidates and show
    near-diagonal calibration.
    """
    n = len(tids)
    if n == 0 or not gt:
        return 0
    if len(pids) != n or len(confs) != n:
        return 0
    L = len(gt)
    depth = [0] * n
    path_accept = [False] * n
    n_added = 0
    for i in range(n):
        p = pids[i]
        if p == -1:
            depth[i] = 1
        else:
            depth[i] = depth[p] + 1
        d = depth[i]
        if d > L:
            path_accept[i] = False
        else:
            tok_match = (int(tids[i]) == int(gt[d - 1]))
            if p == -1:
                path_accept[i] = tok_match
            else:
                path_accept[i] = path_accept[p] and tok_match
        if p == -1:
            parent_match = True
        else:
            parent_match = path_accept[p]
        if not parent_match or d > L:
            continue
        c = confs[i]
        if c is None:
            continue
        c = float(c)
        if c != c:  # NaN guard
            continue
        if c < 0.0:
            c = 0.0
        elif c > 1.0:
            c = 1.0
        bin_idx = int(c * n_bins)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        trials[bin_idx] += 1
        n_added += 1
        if int(tids[i]) == int(gt[d - 1]):
            accepts[bin_idx] += 1
    return n_added


def _accumulate_per_depth_group(tids: List[int], pids: List[int],
                                  confs: List[float], gt: list,
                                  n_bins: int,
                                  trials: List[int],
                                  accepts: List[int]) -> int:
    """Per-DEPTH-GROUP binning. Each (step, parent) sibling group = 1 trial.
    Accept iff ANY sibling has token == gt[d-1]. Bin by max-conf sibling.

    Used for the score axis where score is per-DRAFT (broadcast to all
    nodes), so per-NODE binning would dilute by 1/topk for big trees.
    Depth-group treats the 'topk siblings at this depth' as a single
    binary attempt, matching the spec-decode semantics ('depth succeeds
    if any candidate is right'). For path-compressed chains (suffix),
    each parent has 1 child, so this degenerates to per-NODE.
    """
    n = len(tids)
    if n == 0 or not gt:
        return 0
    if len(pids) != n or len(confs) != n:
        return 0
    L = len(gt)
    depth = [0] * n
    path_accept = [False] * n
    children_by_parent: dict[int, List[int]] = {}
    for i in range(n):
        p = pids[i]
        children_by_parent.setdefault(p, []).append(i)
        if p == -1:
            depth[i] = 1
        else:
            depth[i] = depth[p] + 1
        d = depth[i]
        if d > L:
            path_accept[i] = False
        else:
            tok_match = (int(tids[i]) == int(gt[d - 1]))
            if p == -1:
                path_accept[i] = tok_match
            else:
                path_accept[i] = path_accept[p] and tok_match
    n_added = 0
    for parent_idx, siblings in children_by_parent.items():
        if not siblings:
            continue
        d = depth[siblings[0]]
        if d > L:
            continue
        if parent_idx == -1:
            parent_match = True
        else:
            parent_match = path_accept[parent_idx]
        if not parent_match:
            continue
        sib_confs: List[float] = []
        for c in siblings:
            v = confs[c]
            if v is None:
                continue
            v = float(v)
            if v != v:  # NaN guard
                continue
            sib_confs.append(v)
        if not sib_confs:
            continue
        rep = max(sib_confs)
        if rep < 0.0:
            rep = 0.0
        elif rep > 1.0:
            rep = 1.0
        bin_idx = int(rep * n_bins)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        trials[bin_idx] += 1
        n_added += 1
        gt_tok = int(gt[d - 1])
        if any(int(tids[c]) == gt_tok for c in siblings):
            accepts[bin_idx] += 1
    return n_added


# Backwards-compat alias (legacy unit tests). Routes to depth-group, the
# stricter "fair" semantics. The main loop calls per-axis explicitly.
_accumulate_calibration = _accumulate_per_depth_group


def _recover_per_step_conf(cumulative: List[float],
                             pids: List[int]) -> List[float]:
    """Recover single-step conf from cumulative path-product. Root's
    parent prob is 1.0 (virtual root not stored). Works for both
    eagle3 path_draft_p_t and suffix probs — identical semantics."""
    n = len(cumulative)
    confs = [0.0] * n
    for i in range(n):
        p = pids[i]
        cur = float(cumulative[i])
        if p == -1:
            confs[i] = cur
        else:
            par = float(cumulative[p])
            confs[i] = cur / par if par > 0 else 0.0
    return confs


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
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--score-max", type=float, default=8.0,
                    help="Upper edge of score-axis bins (raw sum-of-probs). "
                         "Scores above this clamp to the last bin. Default 8.")
    ap.add_argument("--dm-capture-results", default=None,
                    help="Path to STANDALONE draft-model agent_results JSON. "
                         "When provided, draft_model proposer is added to the "
                         "calibration analysis with tree topology + "
                         "path_draft_p_t.")
    ap.add_argument("--mtp-capture-results", default=None,
                    help="Path to MTP agent_results JSON (Qwen3.5-9B etc.). "
                         "When provided, mtp proposer is added to the "
                         "calibration analysis.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--max-questions", type=int, default=None)
    args = ap.parse_args()

    eagle3_reslice = (args.capture_steps, args.capture_topk,
                      args.reslice_steps, args.reslice_topk)

    print(f"[calib] loading capture: {args.agent_results}", file=sys.stderr)
    if args.dm_capture_results:
        print(f"[calib] dm-capture: {args.dm_capture_results}", file=sys.stderr)
    if args.mtp_capture_results:
        print(f"[calib] mtp-capture: {args.mtp_capture_results}", file=sys.stderr)
    t0 = time.time()
    records = assemble_records_from_artifacts(
        agent_results_path=args.agent_results,
        suffix_drafts_path=None, draft_model_drafts_path=None,
        mtp_agent_results_path=args.mtp_capture_results,
        dm_capture_path=args.dm_capture_results,
        exclude_path=args.exclude,
        model=args.model, dataset_path=args.dataset,
        responses_path=args.responses, eagle3_reslice=eagle3_reslice,
    )
    print(f"[calib] {len(records)} records in {time.time() - t0:.1f}s",
          file=sys.stderr)

    n_bins = args.n_bins
    score_max = float(args.score_max)
    per_step_bin_edges = [i / n_bins for i in range(n_bins + 1)]
    path_prob_bin_edges = [i / n_bins for i in range(n_bins + 1)]
    score_bin_edges = [i * score_max / n_bins for i in range(n_bins + 1)]

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
        sys.exit(f"[calib] FATAL: cannot import SuffixDecodingCache: {e}")

    sx = SuffixDecodingCache(
        max_tree_depth=64, max_cached_requests=100000)

    methods = ("eagle3", "suffix", "draft_model", "mtp")
    axes = ("per_step", "path_prob", "score")
    trials = {m: {ax: [0] * n_bins for ax in axes} for m in methods}
    accepts = {m: {ax: [0] * n_bins for ax in axes} for m in methods}

    n_seqs = 0
    n_steps = 0
    n_no_eagle3_base = 0
    n_no_path_draft_p_t = 0
    n_eagle3_trials = 0
    n_suffix_trials = 0
    n_dm_trials = 0
    n_mtp_trials = 0
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
            if not gt or not base_ctx:
                if gt:
                    sx.add_active_response(cache_key, [int(gt[0])])
                continue

            # ── eagle3
            if base and base.get("token_ids"):
                base_tids = list(base["token_ids"])
                base_pids = list(base["parents"])
                pdpt = base.get("path_draft_p_t")
                if pdpt is not None and len(pdpt) == len(base_tids):
                    e3_per_step = _recover_per_step_conf(pdpt, base_pids)
                    e3_path_prob = [float(x) for x in pdpt]
                    e3_n = len(pdpt)
                    e3_score_raw = sum(e3_path_prob) if e3_n else 0.0
                    e3_score_scaled = (e3_score_raw / score_max) if score_max > 0 else 0.0
                    e3_score_per_node = [e3_score_scaled] * e3_n
                    # per_step + path_prob: per-NODE binning (each child = 1 trial)
                    nt_ps = _accumulate_per_node(
                        base_tids, base_pids, e3_per_step, gt, n_bins,
                        trials["eagle3"]["per_step"],
                        accepts["eagle3"]["per_step"])
                    _accumulate_per_node(
                        base_tids, base_pids, e3_path_prob, gt, n_bins,
                        trials["eagle3"]["path_prob"],
                        accepts["eagle3"]["path_prob"])
                    # score: per-DEPTH-GROUP (sibling group = 1 trial, ANY-match)
                    _accumulate_per_depth_group(
                        base_tids, base_pids, e3_score_per_node, gt, n_bins,
                        trials["eagle3"]["score"],
                        accepts["eagle3"]["score"])
                    n_eagle3_trials += nt_ps  # per_step trial count
                else:
                    n_no_path_draft_p_t += 1
            else:
                n_no_eagle3_base += 1

            # ── suffix
            try:
                draft = sx.speculate(
                    cache_key, np.asarray(base_ctx, dtype=np.int32),
                    max_spec_factor=SUFFIX_FACTOR,
                    min_token_prob=SUFFIX_MIN_PROB,
                    use_tree_spec=True)
                if draft and draft.token_ids:
                    sf_tids = list(draft.token_ids)
                    sf_pids = list(draft.parents)
                    sf_cum = [float(x) for x in draft.probs]
                    if len(sf_cum) == len(sf_tids):
                        sf_per_step = _recover_per_step_conf(sf_cum, sf_pids)
                        sf_n = len(sf_cum)
                        sf_score_raw = float(draft.score)
                        sf_score_scaled = (sf_score_raw / score_max) if score_max > 0 else 0.0
                        sf_score_per_node = [sf_score_scaled] * sf_n
                        nt_ps = _accumulate_per_node(
                            sf_tids, sf_pids, sf_per_step, gt, n_bins,
                            trials["suffix"]["per_step"],
                            accepts["suffix"]["per_step"])
                        _accumulate_per_node(
                            sf_tids, sf_pids, sf_cum, gt, n_bins,
                            trials["suffix"]["path_prob"],
                            accepts["suffix"]["path_prob"])
                        _accumulate_per_depth_group(
                            sf_tids, sf_pids, sf_score_per_node, gt, n_bins,
                            trials["suffix"]["score"],
                            accepts["suffix"]["score"])
                        n_suffix_trials += nt_ps
            except Exception:
                pass

            # ── draft_model (STANDALONE capture, same schema as eagle3)
            # GT for dm: dm's OWN committed trajectory (when present), else
            # eagle3's. Required because dm capture may use placeholder target
            # whose committed tokens diverge from eagle3 trajectory unless
            # SGLANG_ORACLE_REPLAY actually aligned them.
            dm = (rec.get("per_proposer") or {}).get("draft_model")
            if dm and dm.get("token_ids"):
                dm_tids = list(dm["token_ids"])
                dm_pids = list(dm["parents"])
                dm_pdpt = dm.get("path_draft_p_t")
                gt_dm = rec.get("ground_truth_dm") or gt
                if dm_pdpt is not None and len(dm_pdpt) == len(dm_tids):
                    dm_per_step = _recover_per_step_conf(dm_pdpt, dm_pids)
                    dm_path_prob = [float(x) for x in dm_pdpt]
                    dm_n = len(dm_pdpt)
                    dm_score_raw = sum(dm_path_prob) if dm_n else 0.0
                    dm_score_scaled = (dm_score_raw / score_max) if score_max > 0 else 0.0
                    dm_score_per_node = [dm_score_scaled] * dm_n
                    n_dm_trials += _accumulate_per_node(
                        dm_tids, dm_pids, dm_per_step, gt_dm, n_bins,
                        trials["draft_model"]["per_step"],
                        accepts["draft_model"]["per_step"])
                    _accumulate_per_node(
                        dm_tids, dm_pids, dm_path_prob, gt_dm, n_bins,
                        trials["draft_model"]["path_prob"],
                        accepts["draft_model"]["path_prob"])
                    _accumulate_per_depth_group(
                        dm_tids, dm_pids, dm_score_per_node, gt_dm, n_bins,
                        trials["draft_model"]["score"],
                        accepts["draft_model"]["score"])

            # ── mtp (Qwen3.5-9B native MTP head; same schema as eagle3)
            # GT for mtp: mtp's OWN committed trajectory (= the underlying
            # 9B target's natural output). MTP captures don't use replay, so
            # their tokens diverge from the eagle3 (14B) trajectory.
            mtp = (rec.get("per_proposer") or {}).get("mtp")
            if mtp and mtp.get("token_ids"):
                mtp_tids = list(mtp["token_ids"])
                mtp_pids = list(mtp["parents"])
                mtp_pdpt = mtp.get("path_draft_p_t")
                gt_mtp = rec.get("ground_truth_mtp") or gt
                if mtp_pdpt is not None and len(mtp_pdpt) == len(mtp_tids):
                    mtp_per_step = _recover_per_step_conf(mtp_pdpt, mtp_pids)
                    mtp_path_prob = [float(x) for x in mtp_pdpt]
                    mtp_n = len(mtp_pdpt)
                    mtp_score_raw = sum(mtp_path_prob) if mtp_n else 0.0
                    mtp_score_scaled = (mtp_score_raw / score_max) if score_max > 0 else 0.0
                    mtp_score_per_node = [mtp_score_scaled] * mtp_n
                    n_mtp_trials += _accumulate_per_node(
                        mtp_tids, mtp_pids, mtp_per_step, gt_mtp, n_bins,
                        trials["mtp"]["per_step"],
                        accepts["mtp"]["per_step"])
                    _accumulate_per_node(
                        mtp_tids, mtp_pids, mtp_path_prob, gt_mtp, n_bins,
                        trials["mtp"]["path_prob"],
                        accepts["mtp"]["path_prob"])
                    _accumulate_per_depth_group(
                        mtp_tids, mtp_pids, mtp_score_per_node, gt_mtp, n_bins,
                        trials["mtp"]["score"],
                        accepts["mtp"]["score"])

            n_steps += 1
            sx.add_active_response(cache_key, [int(gt[0])])

        sx.stop_request(cache_key)

        if time.time() - last_log > 30:
            print(f"[calib] {ki + 1}/{len(seq_keys)} seqs, "
                  f"{n_steps} steps, eagle3={n_eagle3_trials}, "
                  f"suffix={n_suffix_trials}, "
                  f"dm={n_dm_trials}, mtp={n_mtp_trials}",
                  file=sys.stderr)
            last_log = time.time()

    elapsed = time.time() - t0
    print(f"[calib] DONE — {n_seqs} seqs, {n_steps} steps in "
          f"{elapsed:.1f}s, eagle3={n_eagle3_trials}, "
          f"suffix={n_suffix_trials}, "
          f"dm={n_dm_trials}, mtp={n_mtp_trials}", file=sys.stderr)

    out = {
        "metadata": {
            "input_source": args.agent_results,
            "dataset": args.dataset,
            "responses": args.responses,
            "model": args.model,
            "n_bins": n_bins,
            "score_max": score_max,
            "per_step_bin_edges": per_step_bin_edges,
            "path_prob_bin_edges": path_prob_bin_edges,
            "score_bin_edges": score_bin_edges,
            "reslice": {"S": args.capture_steps, "K": args.capture_topk,
                        "s": args.reslice_steps, "k": args.reslice_topk},
            "n_sequences": n_seqs,
            "n_steps": n_steps,
            "n_dropped_no_eagle3_base": n_no_eagle3_base,
            "n_dropped_no_path_draft_p_t": n_no_path_draft_p_t,
            "n_eagle3_trials": n_eagle3_trials,
            "n_suffix_trials": n_suffix_trials,
            "n_dm_trials": n_dm_trials,
            "n_mtp_trials": n_mtp_trials,
            "dm_capture_results": args.dm_capture_results,
            "mtp_capture_results": args.mtp_capture_results,
            "elapsed_sec": elapsed,
            "_doc": (
                "Per-method confidence calibration on two axes. "
                "per_step: per-node single-step conf (cum[i]/cum[parent]) "
                "binned on [0,1] in 20 bins of width 0.05. EAGLE-2 Fig.6 "
                "strict analog. "
                "score: per-DRAFT scalar SuffixDraft.score=sum(probs) "
                "(suffix) or sum(path_draft_p_t) (eagle3), used RAW. "
                "Binned on [0, score_max] in 20 equal-width bins; values "
                "above score_max clamp to last bin. All trial nodes in a "
                "draft share the same score-bin. "
                "Trial: parent path matches gt[0:d-1] AND len(gt)>=d. "
                "Accept: token[n] == gt[d-1]. trials = denominator, "
                "accepts = numerator."),
        },
        "n_bins": n_bins,
        "score_max": score_max,
        "per_step_bin_edges": per_step_bin_edges,
        "path_prob_bin_edges": path_prob_bin_edges,
        "score_bin_edges": score_bin_edges,
        "by_method": {
            m: {
                ax: {"trials": trials[m][ax], "accepts": accepts[m][ax]}
                for ax in axes
            }
            for m in methods
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calib] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
