#!/usr/bin/env python3
"""Chain hand-off oracle dense pass (O0-O3 raw table).

For every captured token position t (oracle captures are 1-token-per-step,
so every position has a record) this script computes:

  * A_e(t)      — accept length of the EAGLE3/MTP single chain (full pool
                  resliced to (s'=chain_depth, k'=1); chain_depth defaults
                  to the capture's steps) against the ground truth.
  * A_s(t, j)   — for each hand-off depth j in 0..min(A_e, chain_len):
                  accept length of a suffix-decoding draft speculated from
                  context + gt[:j] (the j EAGLE tokens, which equal the GT
                  prefix because j <= A_e). Two variants are recorded:
                    orc — best path in the suffix TREE (greedy GT walk),
                    grd — deployable greedy chain (max-count child walk).

Trie discipline mirrors run_tree_oracle_sim.py's live-suffix pre-pass
(start_request(prompt) -> per position evaluate -> add_active_response(
gt[:step_idx delta]) -> stop_request), so the trie content at position t
is exactly what any simulated method would see there. j>0 prefixes are
temporarily inserted via SuffixDecodingCache.temporary_extension — the
same pattern _extension_step uses.

All oracles (O0 single source, O1 root selection, O2 fixed hand-off,
O3 adaptive hand-off), budget caps and trajectory walks are derived
OFFLINE from the emitted table by analyze_chain_handoff.py.

Output: gzipped JSONL, one row per position:
  {"rid": str, "ci": int, "pos": int, "el": int, "ae": int, "gl": int,
   "sfx": [[j, orc, grd, match_len, greedy_chain_len, n_nodes, score], ...]}
plus a sidecar <output>.meta.json with counters and parameters.

Usage (inside sglang-bench container, from /workspace):
  python3 -m simulation.scripts.experiments.run_chain_handoff_oracle \
      --agent-trajectory simulation/results/qwen3_14b/bfcl_v4_steps8_topk16_capture/agent_results_eagle3.json \
      --dataset data/bfcl_agent/dataset_stratified_interleaved.jsonl \
      --output simulation/results/chain_handoff_oracle/qwen3_14b/bfcl_v4_dense.jsonl.gz \
      [--limit-requests 20] [--validate-greedy 2000]
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


def _build_children(parents):
    """children[p] -> list of node indices whose parent is p (-1 = root)."""
    ch = defaultdict(list)
    for i, p in enumerate(parents):
        ch[p].append(i)
    return ch


def _tree_walk_accept(token_ids, parents, gt):
    """Accept length of the best GT-matching path in a tree.

    Same semantics as tree_knapsack.greedy_tree_walk but O(n + depth)
    via a (parent, token) -> child index, instead of O(n * depth).
    """
    if not token_ids or not gt:
        return 0
    idx = {}
    for i, p in enumerate(parents):
        key = (p, token_ids[i])
        if key not in idx:
            idx[key] = i
    accepted = 0
    node = -1
    for tok in gt:
        nxt = idx.get((node, tok))
        if nxt is None:
            break
        accepted += 1
        node = nxt
    return accepted


def _is_chain(parents):
    if not parents:
        return False
    if parents[0] != -1:
        return False
    for i in range(1, len(parents)):
        if parents[i] != i - 1:
            return False
    return True


def _extract_chain(token_ids, parents, path_probs):
    """Top-1 chain from a tree: follow the max-path_prob child each level.

    Returns (chain_tokens, was_chain). For k'=1 resliced trees this is the
    identity; the fallback covers steps whose full pool was missing (the
    loader then attaches the original topk=16 tree).
    """
    if _is_chain(parents):
        return list(token_ids), True
    ch = _build_children(parents)
    chain = []
    node = -1
    pp = path_probs if (path_probs and len(path_probs) == len(token_ids)) \
        else [0.0] * len(token_ids)
    while True:
        kids = ch.get(node)
        if not kids:
            break
        best = max(kids, key=lambda i: (pp[i] if pp[i] is not None else 0.0))
        chain.append(token_ids[best])
        node = best
    return chain, False


def _greedy_suffix_chain(token_ids, parents, counts, probs):
    """Deployable greedy chain: from the root, repeatedly take the child
    with the highest count (ties: higher prob, then BFS order)."""
    if not token_ids:
        return []
    n = len(token_ids)
    cnt = counts if (counts and len(counts) == n) else [0] * n
    prb = probs if (probs and len(probs) == n) else [0.0] * n
    ch = _build_children(parents)
    chain = []
    node = -1
    while True:
        kids = ch.get(node)
        if not kids:
            break
        best = max(kids, key=lambda i: (cnt[i], prb[i], -i))
        chain.append(token_ids[best])
        node = best
    return chain


def _prefix_accept(chain, gt):
    n = 0
    for a, b in zip(chain, gt):
        if a != b:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent-trajectory", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--output", required=True,
                    help="gzipped JSONL output path (.jsonl.gz)")
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=16)
    ap.add_argument("--chain-depth", type=int, default=None,
                    help="reslice target depth s' (default: --capture-steps)")
    ap.add_argument("--limit-requests", type=int, default=0,
                    help="smoke mode: keep only the first N distinct "
                         "request_ids (0 = all)")
    ap.add_argument("--validate-greedy", type=int, default=0,
                    help="for the first N suffix evaluations, also call "
                         "speculate(use_tree_spec=False) and compare with "
                         "the derived greedy chain (stats in meta)")
    ap.add_argument("--max-spec-tokens", type=int, default=256)
    ap.add_argument("--max-spec-factor", type=float, default=4.0)
    ap.add_argument("--min-token-prob", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    from simulation.pipeline.assemble_records import (
        assemble_records_from_artifacts,
    )
    chain_depth = args.chain_depth or args.capture_steps
    records = assemble_records_from_artifacts(
        agent_trajectory_path=args.agent_trajectory,
        suffix_drafts_path=None,
        draft_model_drafts_path=None,
        mtp_agent_trajectory_path=None,
        exclude_path=None,
        model=args.model,
        dataset_path=args.dataset,
        responses_path=None,
        eagle3_reslice=(args.capture_steps, args.capture_topk,
                        chain_depth, 1),
    )
    print(f"records: {len(records)} (load {time.time() - t0:.0f}s)",
          file=sys.stderr)

    if args.limit_requests > 0:
        keep = []
        seen = []
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

    cache = SuffixDecodingCache(
        max_tree_depth=64, max_cached_requests=100000, enable_undo=True)
    spec_kwargs = dict(
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
        use_tree_spec=True,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = gzip.open(out_path, "wt", compresslevel=4)

    meta = {
        "agent_trajectory": args.agent_trajectory,
        "eagle3_reslice": [args.capture_steps, args.capture_topk,
                           chain_depth, 1],
        "spec_kwargs": spec_kwargs,
        "limit_requests": args.limit_requests,
        "n_records": len(records),
        "n_positions": 0,          # rows emitted
        "n_skipped_no_eagle": 0,   # positions without an eagle3 tree
        "n_chain_fallback": 0,     # eagle3 tree was NOT a k'=1 chain
        "n_spec_errors": 0,
        "n_spec_calls": 0,
        "validate_greedy": {"n": 0, "exact": 0, "prefix": 0},
    }
    vg_left = args.validate_greedy

    n_done = 0
    t1 = time.time()
    for (rid, cid), seq in sorted(by_seq.items()):
        if not seq:
            continue
        cache_req_id = f"{rid}_{cid}"
        prompt = seq[0].get("context_token_ids") or []
        cache.start_request(cache_req_id, np.asarray(prompt, dtype=np.int32))
        for i, rec in enumerate(seq):
            ctx = rec.get("context_token_ids") or []
            gt = rec.get("ground_truth_future") or []
            prop = (rec.get("per_proposer") or {}).get("eagle3")
            if ctx and gt and prop and prop.get("token_ids"):
                chain, was_chain = _extract_chain(
                    prop["token_ids"], prop["parents"],
                    prop.get("path_draft_p_t"))
                if not was_chain:
                    meta["n_chain_fallback"] += 1
                ae = _prefix_accept(chain, gt)
                ctx_tail = list(ctx[-64:])
                sfx_rows = []
                for j in range(0, min(ae, len(chain)) + 1):
                    ext = list(gt[:j])
                    gt_j = gt[j:]
                    draft = None
                    meta["n_spec_calls"] += 1
                    try:
                        if j == 0:
                            draft = cache.speculate(
                                cache_req_id,
                                np.asarray(ctx_tail, dtype=np.int32),
                                **spec_kwargs)
                        else:
                            with cache.temporary_extension(cache_req_id, ext):
                                draft = cache.speculate(
                                    cache_req_id,
                                    np.asarray(ctx_tail + ext,
                                               dtype=np.int32),
                                    **spec_kwargs)
                    except Exception:
                        meta["n_spec_errors"] += 1
                        draft = None
                    if draft is None or not draft.token_ids:
                        sfx_rows.append([j, 0, 0,
                                         int(draft.match_len) if draft else 0,
                                         0, 0, 0.0])
                        continue
                    orc = _tree_walk_accept(
                        draft.token_ids, draft.parents, gt_j)
                    gchain = _greedy_suffix_chain(
                        draft.token_ids, draft.parents,
                        draft.counts, draft.probs)
                    grd = _prefix_accept(gchain, gt_j)
                    if vg_left > 0:
                        vg_left -= 1
                        try:
                            if j == 0:
                                cdraft = cache.speculate(
                                    cache_req_id,
                                    np.asarray(ctx_tail, dtype=np.int32),
                                    max_spec_tokens=args.max_spec_tokens,
                                    max_spec_factor=args.max_spec_factor,
                                    min_token_prob=args.min_token_prob,
                                    use_tree_spec=False)
                            else:
                                with cache.temporary_extension(
                                        cache_req_id, ext):
                                    cdraft = cache.speculate(
                                        cache_req_id,
                                        np.asarray(ctx_tail + ext,
                                                   dtype=np.int32),
                                        max_spec_tokens=args.max_spec_tokens,
                                        max_spec_factor=args.max_spec_factor,
                                        min_token_prob=args.min_token_prob,
                                        use_tree_spec=False)
                            meta["validate_greedy"]["n"] += 1
                            ctoks = list(cdraft.token_ids)
                            if ctoks == gchain:
                                meta["validate_greedy"]["exact"] += 1
                            m = min(len(ctoks), len(gchain))
                            if ctoks[:m] == gchain[:m]:
                                meta["validate_greedy"]["prefix"] += 1
                        except Exception:
                            pass
                    sfx_rows.append([
                        j, int(orc), int(grd), int(draft.match_len),
                        len(gchain), len(draft.token_ids),
                        round(float(draft.score), 4)])
                fh.write(json.dumps(
                    {"rid": rid, "ci": cid, "pos": rec.get("step_idx", 0),
                     "el": len(chain), "ae": int(ae), "gl": len(gt),
                     "sfx": sfx_rows},
                    separators=(",", ":")) + "\n")
                meta["n_positions"] += 1
            elif ctx and gt:
                meta["n_skipped_no_eagle"] += 1

            # Trie feed: GT tokens up to the next record's offset (mirrors
            # the sim's add_active_response(gt[:advance]) discipline).
            if i < len(seq) - 1:
                adv = (seq[i + 1].get("step_idx", 0)
                       - rec.get("step_idx", 0))
            else:
                adv = 1
            if gt and adv > 0:
                cache.add_active_response(cache_req_id, list(gt[:adv]))

            n_done += 1
            if n_done % 50000 == 0:
                el = time.time() - t1
                print(f"  {n_done}/{len(records)} positions "
                      f"({el:.0f}s, {n_done / max(el, 1e-9):.0f} pos/s)",
                      file=sys.stderr, flush=True)
        cache.stop_request(cache_req_id)

    fh.close()
    meta["elapsed_s"] = round(time.time() - t0, 1)
    meta_path = str(out_path) + ".meta.json"
    with open(meta_path, "w") as mf:
        json.dump(meta, mf, indent=2)
    print(f"DONE rows={meta['n_positions']} fallback={meta['n_chain_fallback']} "
          f"spec_errors={meta['n_spec_errors']} -> {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
