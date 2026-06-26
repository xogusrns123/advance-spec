"""Extract per-draft-token FEATURES + accept label, for signal-existence analysis.

This is the feature-rich sibling of extract_calib_pairs.py. Instead of only
(probability, accept) it emits, per drafted edge, a small set of candidate
features, plus per-position context scalars (incl. output entropy over the
accumulated output-token sequence). Offline / CPU-only. We do NOT calibrate
here; downstream (analyze_feature_signal.py) just asks "does each feature carry
signal about acceptance" (AUROC / MI / within-depth).

Two draft kinds per position (same definitions as extract_calib_pairs.py):
  model  — EAGLE3 (14B) / MTP (27B) top-1 chain. Per-edge:
             depth (1-based), edge_prob = path_draft_p_t[d]/path_draft_p_t[d-1],
             path_prob = cumulative path_draft_p_t[d]. accept y[d] = 1 iff d < A_e.
  suffix — suffix decoding greedy chain (speculate at the realized context, j=0).
             Per-edge: depth, count_ratio_prob = draft.probs[node], log_count =
             log1p(draft.counts[node]), branch_factor = #siblings at the chosen
             node, subtree_size = #descendants+1. accept y[d] = 1 iff d < A_s.

Per-position scalars ("pp", broadcast to every edge of the position downstream):
  abs_pos (step_idx), gt_remaining (len(gt)),
  out_entropy_cum / out_entropy_win64 (bits; Shannon entropy of the token-freq
    distribution over ALL / the last <window> OUTPUT tokens emitted so far in
    this generation — a pure function of the realized token stream),
  match_len / score / n_nodes (the suffix speculation's scalars).

Output: gzipped JSONL, one row per position:
  {"rid","ci","pos",
   "m":{"y":[...],"depth":[...],"edge_prob":[...],"path_prob":[...]},
   "s":{"y":[...],"depth":[...],"count_ratio_prob":[...],"log_count":[...],
        "branch_factor":[...],"subtree_size":[...]},
   "pp":{"abs_pos":int,"gt_remaining":int,"out_entropy_cum":float|null,
         "out_entropy_win64":float|null,"match_len":int|null,"score":float|null,
         "n_nodes":int|null}}
"m"/"s" are omitted when that draft has no edges at the position.

Usage (inside sglang-bench container, from /workspace):
  python3 -m simulation.scripts.experiments.extract_signal_features \
      --agent-results .../bfcl_v4_steps16_topk1_capture/agent_results_eagle3.json \
      --dataset data/bfcl_agent/dataset_stratified_interleaved.jsonl \
      --model Qwen/Qwen3-14B \
      --output simulation/results/calib_verify/features_14b.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_spec_decoding.suffix_decoding.suffix_tree import (  # noqa: E402
    SuffixDecodingCache,
)
from simulation.pipeline.per_node_features import (  # noqa: E402
    compute_children,
    compute_n_descendants,
)
from simulation.scripts.experiments.run_chain_handoff_oracle import (  # noqa: E402
    _extract_chain,
    _prefix_accept,
)


def _entropy_bits(counts, total):
    """Shannon entropy (base 2) of a token-frequency distribution."""
    if total <= 0:
        return None
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _model_cols(prop, gt):
    """Per-edge feature columns for the eagle3/mtp top-1 chain (or None)."""
    chain, _ = _extract_chain(prop["token_ids"], prop["parents"],
                              prop.get("path_draft_p_t"))
    if not chain:
        return None
    ae = _prefix_accept(chain, gt)
    pp = prop.get("path_draft_p_t") or []
    y, depth, edge_prob, path_prob = [], [], [], []
    prev = 1.0
    for d in range(len(chain)):
        cum = float(pp[d]) if d < len(pp) and pp[d] is not None else 0.0
        edge = (cum / prev) if prev > 1e-12 else 0.0
        edge = min(max(edge, 0.0), 1.0)
        y.append(1 if d < ae else 0)
        depth.append(d + 1)
        edge_prob.append(round(edge, 6))
        path_prob.append(round(cum, 6))
        if cum > 0.0:
            prev = cum
    return {"y": y, "depth": depth, "edge_prob": edge_prob,
            "path_prob": path_prob}


def _dual_speculate(cache, req_id, ctx_arr, spec_kwargs):
    """Speculate on the local and global suffix trees SEPARATELY and return
    (winner_draft, win) where win is "local" / "global" / "tie".

    This mirrors arctic_inference's internal ``speculate`` (it computes a draft
    from the per-request local tree and from the cross-request global tree and
    keeps the higher-scoring one) but exposes WHICH tree won, so the downstream
    signal analysis can be bucketed by tree. The returned draft is the one that
    would actually be deployed (local on ties, matching arctic's ``>=``).
    """
    from arctic_inference.suffix_decoding._C import SuffixTree
    inner = cache._cache
    loc = inner._local_trees[req_id]
    glob = inner._global_tree
    mst = spec_kwargs["max_spec_tokens"]
    msf = spec_kwargs["max_spec_factor"]
    mso = spec_kwargs.get("max_spec_offset", 0.0)
    mtp = spec_kwargs["min_token_prob"]
    uts = spec_kwargs["use_tree_spec"]
    d_loc = SuffixTree.speculate_ndarray(loc, ctx_arr, mst, msf, mso, mtp, uts)
    d_glob = SuffixTree.speculate_ndarray(glob, ctx_arr, mst, msf, mso, mtp, uts)
    if d_loc.score > d_glob.score:
        return d_loc, "local"
    if d_glob.score > d_loc.score:
        return d_glob, "global"
    return d_loc, "tie"   # equal score -> arctic deploys local (draft1, >=)


def _suffix_cols(token_ids, parents, counts, probs, gt):
    """Per-edge feature columns for the deployable greedy suffix chain (or None).

    Greedy walk root->leaf taking the highest-count child (ties: higher prob,
    BFS order) — same selection as extract_calib_pairs._greedy_suffix_chain_pq.
    """
    n = len(token_ids)
    if n == 0:
        return None
    cnt = counts if (counts and len(counts) == n) else [0] * n
    prb = probs if (probs and len(probs) == n) else [0.0] * n
    ch = compute_children(parents)            # parent_idx -> [child idxs] (root -1)
    ndesc = compute_n_descendants(parents)
    walk = []
    node = -1
    while True:
        kids = ch.get(node)
        if not kids:
            break
        best = max(kids, key=lambda i: (cnt[i], prb[i], -i))
        walk.append(best)
        node = best
    if not walk:
        return None
    gtoks = [token_ids[b] for b in walk]
    a_s = _prefix_accept(gtoks, gt)
    y, depth, crp, logc, cnts, bf, subt = [], [], [], [], [], [], []
    for d, b in enumerate(walk):
        parent = parents[b]
        y.append(1 if d < a_s else 0)
        depth.append(d + 1)
        crp.append(round(float(prb[b]), 6))
        logc.append(round(math.log1p(cnt[b]), 6))
        cnts.append(int(cnt[b]))               # raw suffix-tree node count
        bf.append(len(ch.get(parent, [])))    # #siblings (children of the parent)
        subt.append(int(ndesc[b]) + 1)
    return {"y": y, "depth": depth, "count_ratio_prob": crp, "log_count": logc,
            "count": cnts, "branch_factor": bf, "subtree_size": subt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent-results", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--output", required=True)
    ap.add_argument("--capture-steps", type=int, default=16)
    ap.add_argument("--capture-topk", type=int, default=1)
    ap.add_argument("--chain-depth", type=int, default=16)
    ap.add_argument("--limit-requests", type=int, default=0)
    ap.add_argument("--max-positions", type=int, default=0,
                    help="global cap on rows written (0 = no cap); for smoke runs")
    ap.add_argument("--window", type=int, default=64,
                    help="window for out_entropy_win (output tokens)")
    ap.add_argument("--min-entropy-n", type=int, default=8,
                    help="min output tokens before entropy is defined (else null)")
    ap.add_argument("--max-spec-tokens", type=int, default=256)
    ap.add_argument("--max-spec-factor", type=float, default=4.0)
    ap.add_argument("--min-token-prob", type=float, default=0.0)
    ap.add_argument("--tag-winner", action="store_true",
                    help="speculate on the local & global suffix trees "
                         "separately and tag each row with which tree won "
                         "(row['win'] in {local,global,tie}); the suffix "
                         "features are the deployed winner's, identical to the "
                         "default path but with the tree label attached")
    args = ap.parse_args()

    t0 = time.time()
    from simulation.pipeline.assemble_records import (
        assemble_records_from_artifacts,
    )
    records = assemble_records_from_artifacts(
        agent_results_path=args.agent_results, suffix_drafts_path=None,
        draft_model_drafts_path=None, mtp_agent_results_path=None,
        exclude_path=None, model=args.model, dataset_path=args.dataset,
        responses_path=None,
        eagle3_reslice=(args.capture_steps, args.capture_topk,
                        args.chain_depth, 1))
    print(f"records: {len(records)} (load {time.time()-t0:.0f}s)", file=sys.stderr)

    if args.limit_requests > 0:
        seen, keep = [], []
        for rec in records:
            rid = rec["request_id"]
            if rid not in seen:
                if len(seen) >= args.limit_requests:
                    continue
                seen.append(rid)
            keep.append(rec)
        records = keep
        print(f"limit-requests={args.limit_requests}: {len(records)} records",
              file=sys.stderr)

    by_seq = defaultdict(list)
    for rec in records:
        by_seq[(rec["request_id"], rec.get("call_idx", 0))].append(rec)
    for k in by_seq:
        by_seq[k].sort(key=lambda r: r.get("step_idx", 0))

    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000,
                                enable_undo=True)
    spec_kwargs = dict(max_spec_tokens=args.max_spec_tokens,
                       max_spec_factor=args.max_spec_factor,
                       min_token_prob=args.min_token_prob, use_tree_spec=True)
    W = args.window
    min_n = args.min_entropy_n

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = gzip.open(out_path, "wt", compresslevel=4)
    meta = {"n_positions": 0, "n_model_edges": 0, "n_suffix_edges": 0,
            "n_rids": 0, "n_spec_errors": 0, "window": W,
            "tag_winner": bool(args.tag_winner),
            # per-tree-winner suffix accounting (only when --tag-winner)
            "win_positions": {"local": 0, "global": 0, "tie": 0},
            "win_suffix_edges": {"local": 0, "global": 0, "tie": 0}}

    rids_seen = set()
    n_done = 0
    stop = False
    for (rid, cid), seq in sorted(by_seq.items()):
        if stop:
            break
        if not seq:
            continue
        rids_seen.add(rid)
        cache_req_id = f"{rid}_{cid}"
        cache.start_request(cache_req_id,
                            np.asarray(seq[0].get("context_token_ids") or [],
                                       dtype=np.int32))
        out_tokens = []           # realized output tokens so far (this generation)
        cum_counter = Counter()   # running token-freq over out_tokens
        for i, rec in enumerate(seq):
            ctx = rec.get("context_token_ids") or []
            gt = rec.get("ground_truth_future") or []
            step_idx = rec.get("step_idx", 0)
            if ctx and gt:
                prop = (rec.get("per_proposer") or {}).get("eagle3")
                m_cols = (_model_cols(prop, gt)
                          if prop and prop.get("token_ids") else None)
                # suffix speculation at the realized context (j=0)
                draft = None
                tree_win = None
                try:
                    ctx_arr = np.ascontiguousarray(
                        list(ctx[-64:]), dtype=np.int32)
                    if args.tag_winner:
                        draft, tree_win = _dual_speculate(
                            cache, cache_req_id, ctx_arr, spec_kwargs)
                    else:
                        draft = cache.speculate(
                            cache_req_id, ctx_arr, **spec_kwargs)
                except Exception:
                    meta["n_spec_errors"] += 1
                    draft = None
                    tree_win = None
                s_cols = (_suffix_cols(draft.token_ids, draft.parents,
                                       draft.counts, draft.probs, gt)
                          if draft is not None and draft.token_ids else None)
                # per-position output-entropy over accumulated output tokens
                n_out = len(out_tokens)
                ecum = (_entropy_bits(cum_counter.values(), n_out)
                        if n_out >= min_n else None)
                win = out_tokens[-W:]
                ewin = (_entropy_bits(Counter(win).values(), len(win))
                        if len(win) >= min_n else None)
                pp = {
                    "abs_pos": int(step_idx), "gt_remaining": len(gt),
                    "out_entropy_cum": (round(ecum, 6) if ecum is not None
                                        else None),
                    "out_entropy_win64": (round(ewin, 6) if ewin is not None
                                          else None),
                    "match_len": (int(draft.match_len)
                                  if draft is not None else None),
                    "score": (round(float(draft.score), 4)
                              if draft is not None else None),
                    "n_nodes": (len(draft.token_ids)
                                if draft is not None else None),
                }
                if m_cols or s_cols:
                    row = {"rid": rid, "ci": cid, "pos": int(step_idx),
                           "pp": pp}
                    if m_cols:
                        row["m"] = m_cols
                        meta["n_model_edges"] += len(m_cols["y"])
                    if s_cols:
                        row["s"] = s_cols
                        meta["n_suffix_edges"] += len(s_cols["y"])
                        if tree_win is not None:
                            row["win"] = tree_win
                            meta["win_positions"][tree_win] += 1
                            meta["win_suffix_edges"][tree_win] += len(
                                s_cols["y"])
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                    meta["n_positions"] += 1
                    if (args.max_positions and
                            meta["n_positions"] >= args.max_positions):
                        stop = True

            # advance realized output (unconditional — keeps cache + entropy state)
            if i < len(seq) - 1:
                adv = seq[i + 1].get("step_idx", 0) - step_idx
            else:
                adv = 1
            if gt and adv > 0:
                toks_adv = list(gt[:adv])
                cache.add_active_response(cache_req_id, toks_adv)
                out_tokens.extend(toks_adv)
                cum_counter.update(toks_adv)
            n_done += 1
            if n_done % 50000 == 0:
                print(f"  {n_done} recs, {meta['n_positions']} rows "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
            if stop:
                break
        cache.stop_request(cache_req_id)

    fh.close()
    meta["n_rids"] = len(rids_seen)
    meta["elapsed_s"] = round(time.time() - t0, 1)
    with open(str(out_path) + ".meta.json", "w") as mf:
        json.dump(meta, mf, indent=2)
    print(f"DONE rids={meta['n_rids']} positions={meta['n_positions']} "
          f"model_edges={meta['n_model_edges']} "
          f"suffix_edges={meta['n_suffix_edges']} -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
