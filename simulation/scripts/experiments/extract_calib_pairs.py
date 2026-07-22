"""Extract per-draft-token (probability, accept) pairs for calibration, from a
chain hand-off capture (agent_results_eagle3.json). Offline / CPU-only.

Two draft kinds per position:
  model  — EAGLE3 (14B) / MTP (27B). Per-edge conditional draft probability
           p_edge[d] = path_draft_p_t[d] / path_draft_p_t[d-1] (path_draft_p_t
           is the CUMULATIVE root->node path prob; see oracle_patch
           _extract_tree_path_draft_p_t). accept y[d] = 1 iff the drafted token
           matched GT at depth d (d < A_e).
  suffix — suffix decoding. Speculate the trie at the REALIZED context (j=0,
           no GT extension = what serving would draft here). Per-edge count
           ratio prob = draft.probs[node] (= count/total, the non-Jeffreys
           suffix_p). accept y[d] = 1 iff d < A_s (greedy suffix chain matched
           GT to depth d).

Trie discipline mirrors run_chain_handoff_oracle.py (start_request ->
per-position speculate -> add_active_response(gt[:advance]) -> stop_request).

Output: gzipped JSONL, one row per position:
  {"rid": str, "ci": int, "pos": int,
   "m": [[p_edge, y], ...],     # model (eagle3/mtp) chain, one per depth
   "s": [[p_count_ratio, y], ...]}  # suffix greedy chain, one per depth
The calibration step (calib_verify.py) groups rows by rid and splits the first
--fit-n-tasks rids into the fit set, the next --test-n-tasks into the test set.

Usage (inside sglang-bench container, from /workspace):
  python3 -m simulation.scripts.experiments.extract_calib_pairs \
      --agent-trajectory simulation/results/calib_verify/snap_14b/agent_results_eagle3.json \
      --dataset data/bfcl_agent/dataset_stratified_interleaved.jsonl \
      --model Qwen/Qwen3-14B \
      --output simulation/results/calib_verify/pairs_14b.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_spec_decoding.suffix_decoding.suffix_tree import (  # noqa: E402
    SuffixDecodingCache,
)
from simulation.scripts.experiments.run_chain_handoff_oracle import (  # noqa: E402
    _build_children,
    _extract_chain,
    _prefix_accept,
)


def _greedy_suffix_chain_pq(token_ids, parents, counts, probs):
    """Deployable greedy suffix chain + per-node count-ratio prob.

    Walks root->leaf taking the highest-count child (ties: higher prob, BFS
    order), returning (tokens, probs) where probs[i] = draft.probs of the i-th
    walked node (count/total at its parent = the non-Jeffreys suffix_p)."""
    if not token_ids:
        return [], []
    n = len(token_ids)
    cnt = counts if (counts and len(counts) == n) else [0] * n
    prb = probs if (probs and len(probs) == n) else [0.0] * n
    ch = _build_children(parents)
    toks, ps = [], []
    node = -1
    while True:
        kids = ch.get(node)
        if not kids:
            break
        best = max(kids, key=lambda i: (cnt[i], prb[i], -i))
        toks.append(token_ids[best])
        ps.append(float(prb[best]))
        node = best
    return toks, ps


def _model_edges(prop, gt):
    """(p_edge, y) per depth for the eagle3/mtp top-1 chain."""
    chain, _ = _extract_chain(prop["token_ids"], prop["parents"],
                              prop.get("path_draft_p_t"))
    if not chain:
        return []
    ae = _prefix_accept(chain, gt)
    pp = prop.get("path_draft_p_t") or []
    rows = []
    prev = 1.0
    for d in range(len(chain)):
        cum = float(pp[d]) if d < len(pp) and pp[d] is not None else 0.0
        edge = (cum / prev) if prev > 1e-12 else 0.0
        edge = min(max(edge, 0.0), 1.0)
        rows.append([round(edge, 6), 1 if d < ae else 0])
        if cum > 0.0:
            prev = cum
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent-trajectory", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--output", required=True)
    ap.add_argument("--capture-steps", type=int, default=16)
    ap.add_argument("--capture-topk", type=int, default=1)
    ap.add_argument("--chain-depth", type=int, default=16)
    ap.add_argument("--limit-requests", type=int, default=0)
    ap.add_argument("--max-spec-tokens", type=int, default=256)
    ap.add_argument("--max-spec-factor", type=float, default=4.0)
    ap.add_argument("--min-token-prob", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    from simulation.pipeline.assemble_records import (
        assemble_records_from_artifacts,
    )
    records = assemble_records_from_artifacts(
        agent_trajectory_path=args.agent_trajectory, suffix_drafts_path=None,
        draft_model_drafts_path=None, mtp_agent_trajectory_path=None,
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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = gzip.open(out_path, "wt", compresslevel=4)
    meta = {"n_positions": 0, "n_model_pairs": 0, "n_suffix_pairs": 0,
            "n_rids": 0, "n_spec_errors": 0}

    rids_seen = set()
    n_done = 0
    for (rid, cid), seq in sorted(by_seq.items()):
        if not seq:
            continue
        rids_seen.add(rid)
        cache_req_id = f"{rid}_{cid}"
        cache.start_request(cache_req_id,
                            np.asarray(seq[0].get("context_token_ids") or [],
                                       dtype=np.int32))
        for i, rec in enumerate(seq):
            ctx = rec.get("context_token_ids") or []
            gt = rec.get("ground_truth_future") or []
            prop = (rec.get("per_proposer") or {}).get("eagle3")
            if ctx and gt and prop and prop.get("token_ids"):
                m_rows = _model_edges(prop, gt)
                # suffix at the realized context (j=0)
                s_rows = []
                ctx_tail = list(ctx[-64:])
                try:
                    draft = cache.speculate(
                        cache_req_id, np.asarray(ctx_tail, dtype=np.int32),
                        **spec_kwargs)
                except Exception:
                    meta["n_spec_errors"] += 1
                    draft = None
                if draft is not None and draft.token_ids:
                    gtoks, gps = _greedy_suffix_chain_pq(
                        draft.token_ids, draft.parents, draft.counts, draft.probs)
                    a_s = _prefix_accept(gtoks, gt)
                    s_rows = [[round(gps[d], 6), 1 if d < a_s else 0]
                              for d in range(len(gtoks))]
                if m_rows or s_rows:
                    fh.write(json.dumps(
                        {"rid": rid, "ci": cid, "pos": rec.get("step_idx", 0),
                         "m": m_rows, "s": s_rows},
                        separators=(",", ":")) + "\n")
                    meta["n_positions"] += 1
                    meta["n_model_pairs"] += len(m_rows)
                    meta["n_suffix_pairs"] += len(s_rows)

            if i < len(seq) - 1:
                adv = seq[i + 1].get("step_idx", 0) - rec.get("step_idx", 0)
            else:
                adv = 1
            if gt and adv > 0:
                cache.add_active_response(cache_req_id, list(gt[:adv]))
            n_done += 1
            if n_done % 50000 == 0:
                print(f"  {n_done}/{len(records)} ({time.time()-t0:.0f}s)",
                      file=sys.stderr, flush=True)
        cache.stop_request(cache_req_id)

    fh.close()
    meta["n_rids"] = len(rids_seen)
    meta["elapsed_s"] = round(time.time() - t0, 1)
    with open(str(out_path) + ".meta.json", "w") as mf:
        json.dump(meta, mf, indent=2)
    print(f"DONE rids={meta['n_rids']} positions={meta['n_positions']} "
          f"model_pairs={meta['n_model_pairs']} suffix_pairs={meta['n_suffix_pairs']} "
          f"-> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
