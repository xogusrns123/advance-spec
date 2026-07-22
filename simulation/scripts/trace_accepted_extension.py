"""Trace anchor prediction quality (independent of dedup).

For each per-step record in an agent_trajectory capture, runs the extension
speculate logic and records, per anchor:
  * depth (in base tree; root anchor uses depth=0 sentinel)
  * path_p_t (cumulative EAGLE3 draft path probability at the anchor's
    base node; root uses 1.0)
  * score (the SuffixDecodingCache draft.score returned at this anchor)
  * n_suffix_tokens (number of NEW nodes the anchor added after dedup —
    a graft-cost proxy; unchanged from prior versions)
  * gt_match_length (NEW): how many tokens the anchor's raw speculate
    output (a tree of token_ids/parents) matches against the
    ground-truth future when greedy-walked. Independent of other
    anchors, independent of dedup. Anchors with gt_match_length > 0
    are counted as "accepted".

This dedup-free attribution measures each anchor's prediction quality on
its own — answering "if backbone reached this anchor's base node, would
its suffix prediction hit the ground truth?" That is the right signal
for designing a keep/drop filter on anchors.

Usage:
    python3 -m simulation.scripts.trace_accepted_extension \
        --agent-trajectory <path> \
        --budget 128 --F 4.0 --T 0.0 \
        --max-records 5000 --output trace.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.pipeline.assemble_records import assemble_records_from_artifacts
from hybrid_spec_decoding.suffix_decoding.suffix_tree import SuffixDecodingCache


def _chain_match_length(token_ids, parents, gt, shift):
    """Greedy walk on a draft tree against gt[shift:].

    token_ids[i], parents[i] form a tree with virtual root parent=-1.
    Returns the number of consecutive gt tokens matched.
    """
    if not token_ids or shift >= len(gt):
        return 0
    child_idx = defaultdict(list)
    for i, p in enumerate(parents):
        child_idx[p].append(i)
    node = -1
    matched = 0
    for t_idx in range(shift, len(gt)):
        t = gt[t_idx]
        picked = None
        for c in child_idx.get(node, []):
            if token_ids[c] == t:
                picked = c
                break
        if picked is None:
            break
        matched += 1
        node = picked
    return matched


def trace_step(rec, budget, suffix_cache, cache_req_id, F, T):
    """Run extension speculate on one record. Per-anchor metrics only — no
    dedup-based walk attribution.

    Returns dict with 'anchors' list. Each anchor entry has:
      depth, path_p_t, score, n_suffix_tokens, gt_match_length, is_root,
      node_idx (-1 for root anchor, else base-node index).
    """
    gt = rec.get("ground_truth_future", [])
    base = rec.get("per_proposer", {}).get("eagle3")
    if not gt or not base or not base.get("token_ids"):
        return None

    tids_raw = base["token_ids"]
    pids_raw = base["parents"]
    n = min(budget, len(tids_raw))
    tids = list(tids_raw[:n])
    pids = [p if p < n else -1 for p in pids_raw[:n]]
    path_draft_p_t = base.get("path_draft_p_t")
    if path_draft_p_t and len(path_draft_p_t) >= n:
        path_p_t = [float(path_draft_p_t[i] or 0.0) for i in range(n)]
    else:
        path_p_t = [1.0] * n
    # Local edge probability: node_p_t[i] = path_p_t[i] / path_p_t[parent].
    # For layer-0 (parent=-1) this equals path_p_t[i] itself.
    node_p_t = [1.0] * n
    for i in range(n):
        parent = pids[i]
        parent_path = path_p_t[parent] if parent >= 0 else 1.0
        if parent_path > 1e-12:
            node_p_t[i] = path_p_t[i] / parent_path
        else:
            node_p_t[i] = 0.0

    # depth + root→node paths
    depths = [0] * n
    paths = [None] * n
    for i in range(n):
        d = 0
        cur = i
        path = []
        while cur >= 0:
            path.append(tids[cur])
            cur_parent = pids[cur]
            if cur_parent >= 0:
                d += 1
            cur = cur_parent
        depths[i] = d
        path.reverse()
        paths[i] = path

    base_context = rec.get("context_token_ids") or []

    # Tree-building bookkeeping. We retain this purely to track
    # `n_suffix_tokens` per anchor (new nodes added after dedup) — a
    # graft-cost proxy that's still informative even with the new
    # dedup-free hit metric.
    ext_tids = list(tids)
    ext_pids = list(pids)
    children = {}
    for i in range(len(ext_tids)):
        p = ext_pids[i]
        children.setdefault(p, {})[ext_tids[i]] = i

    anchors = []

    # ----- Virtual-root anchor -----
    root_anchor_idx = len(anchors)
    root_meta = {"depth": 0, "path_p_t": 1.0, "node_p_t": 1.0, "score": None,
                 "n_suffix_tokens": 0, "gt_match_length": 0,
                 "is_root": True, "node_idx": -1}
    anchors.append(root_meta)
    try:
        _root_draft = suffix_cache.speculate(
            cache_req_id,
            np.array(base_context, dtype=np.int32),
            max_spec_factor=F, min_token_prob=T, use_tree_spec=True)
    except Exception:
        _root_draft = None
    if _root_draft is not None and _root_draft.token_ids:
        root_meta["score"] = float(getattr(_root_draft, "score", 0.0))
        # Root anchor's chain matches starting at gt[0:].
        root_meta["gt_match_length"] = _chain_match_length(
            _root_draft.token_ids, _root_draft.parents, gt, shift=0)
        # Tree integration for n_suffix_tokens accounting.
        _root_local = {}
        for j, (tid, pid) in enumerate(zip(_root_draft.token_ids, _root_draft.parents)):
            if pid == -1:
                tparent = -1
            else:
                tparent = _root_local.get(pid)
                if tparent is None:
                    break
            existing = children.get(tparent, {}).get(tid)
            if existing is not None:
                _root_local[j] = existing
                continue
            new_idx = len(ext_tids)
            ext_tids.append(tid)
            ext_pids.append(tparent)
            children.setdefault(tparent, {})[tid] = new_idx
            _root_local[j] = new_idx
            root_meta["n_suffix_tokens"] += 1

    # ----- Per-base-node anchors -----
    for node_idx in range(n):
        anchor_meta = {"depth": depths[node_idx],
                       "path_p_t": path_p_t[node_idx],
                       "node_p_t": node_p_t[node_idx],
                       "score": None, "n_suffix_tokens": 0,
                       "gt_match_length": 0,
                       "is_root": False, "node_idx": node_idx}
        anchors.append(anchor_meta)

        ext_context = np.array(base_context + paths[node_idx], dtype=np.int32)
        try:
            with suffix_cache.temporary_extension(cache_req_id, paths[node_idx]):
                draft = suffix_cache.speculate(
                    cache_req_id, ext_context,
                    max_spec_factor=F, min_token_prob=T,
                    use_tree_spec=True)
        except Exception:
            continue
        if not draft.token_ids:
            continue
        anchor_meta["score"] = float(getattr(draft, "score", 0.0))

        # Per-node anchor's chain matches starting at gt[d_N + 1:]
        # (paths[node_idx] has d_N + 1 tokens; the next gt position is d_N + 1).
        shift = depths[node_idx] + 1
        anchor_meta["gt_match_length"] = _chain_match_length(
            draft.token_ids, draft.parents, gt, shift=shift)

        # Tree integration for n_suffix_tokens accounting.
        local_to_tree = {}
        for j, (tid, pid) in enumerate(zip(draft.token_ids, draft.parents)):
            if pid == -1:
                tparent = node_idx
            else:
                tparent = local_to_tree.get(pid)
                if tparent is None:
                    break
            existing = children.get(tparent, {}).get(tid)
            if existing is not None:
                local_to_tree[j] = existing
                continue
            new_idx = len(ext_tids)
            ext_tids.append(tid)
            ext_pids.append(tparent)
            children.setdefault(tparent, {})[tid] = new_idx
            local_to_tree[j] = new_idx
            anchor_meta["n_suffix_tokens"] += 1

    return {
        "anchors": anchors,
        "n_base_nodes": n,
        "n_ext_tokens": len(ext_tids),
        "max_base_depth": max(depths) if depths else 0,
    }


def stats(values):
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "var": float(a.var(ddof=1)) if a.size > 1 else 0.0,
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-trajectory", required=True)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--F", type=float, default=4.0)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--max-records", type=int, default=2000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--responses", default=None)
    ap.add_argument("--capture-steps", type=int, default=None)
    ap.add_argument("--capture-topk", type=int, default=None)
    ap.add_argument("--reslice-steps", type=int, default=None)
    ap.add_argument("--reslice-topk", type=int, default=None)
    args = ap.parse_args()

    eagle3_reslice = None
    if args.reslice_steps and args.reslice_topk:
        eagle3_reslice = (args.capture_steps or 8, args.capture_topk or 16,
                          args.reslice_steps, args.reslice_topk)
        print(f"Reslice: {eagle3_reslice}", file=sys.stderr)

    print(f"Assembling records from {args.agent_trajectory} ...", file=sys.stderr)
    records = assemble_records_from_artifacts(
        agent_trajectory_path=args.agent_trajectory,
        model=args.model,
        dataset_path=args.dataset,
        responses_path=args.responses,
        eagle3_reslice=eagle3_reslice,
    )
    print(f"  got {len(records)} records", file=sys.stderr)
    if args.max_records and len(records) > args.max_records:
        records = records[:args.max_records]
        print(f"  trimmed to {len(records)}", file=sys.stderr)

    suffix_cache = SuffixDecodingCache(
        max_tree_depth=64, max_cached_requests=100000,
        enable_undo=True,
    )

    by_seq = defaultdict(list)
    for r in records:
        by_seq[(r["request_id"], r.get("call_idx", 0))].append(r)

    METRICS = ("depth", "path_p_t", "node_p_t", "score", "n_suffix_tokens",
               "gt_match_length")

    def _mk():
        return {m: [] for m in METRICS}

    considered = _mk()
    considered_root = _mk()
    considered_pernode = _mk()
    accepted_anchor = _mk()
    accepted_root = _mk()
    accepted_pernode = _mk()

    n_records_processed = 0
    n_total_steps = 0

    for seq_key, seq_records in by_seq.items():
        seq_records.sort(key=lambda r: r.get("step_idx", 0))
        cache_req_id = f"{seq_key[0]}_{seq_key[1]}"
        first_rec = seq_records[0]
        prompt = first_rec.get("context_token_ids", []) or []
        suffix_cache.start_request(cache_req_id, np.array(prompt, dtype=np.int32))

        for rec in seq_records:
            res = trace_step(rec, args.budget, suffix_cache, cache_req_id,
                             args.F, args.T)
            n_total_steps += 1
            if res is None:
                continue
            n_records_processed += 1
            for a in res["anchors"]:
                if a.get("score") is None:
                    continue
                is_root = a["is_root"]
                tgt_c = considered_root if is_root else considered_pernode
                tgt_a = accepted_root if is_root else accepted_pernode
                for d_ in (considered, tgt_c):
                    for m in METRICS:
                        d_[m].append(a[m])
                if a["gt_match_length"] > 0:
                    for d_ in (accepted_anchor, tgt_a):
                        for m in METRICS:
                            d_[m].append(a[m])

        if n_records_processed >= args.max_records:
            break

    out = {
        "config": {"budget": args.budget, "F": args.F, "T": args.T,
                   "agent_trajectory": args.agent_trajectory,
                   "n_records_processed": n_records_processed,
                   "n_total_steps": n_total_steps},
        "considered_all": {k: stats(v) for k, v in considered.items()},
        "considered_root": {k: stats(v) for k, v in considered_root.items()},
        "considered_pernode": {k: stats(v) for k, v in considered_pernode.items()},
        "accepted_all": {k: stats(v) for k, v in accepted_anchor.items()},
        "accepted_root": {k: stats(v) for k, v in accepted_root.items()},
        "accepted_pernode": {k: stats(v) for k, v in accepted_pernode.items()},
        "raw": {
            "considered_root": considered_root,
            "considered_pernode": considered_pernode,
            "accepted_root": accepted_root,
            "accepted_pernode": accepted_pernode,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out["config"], indent=2))
    for tag, key in [("CONSIDERED (all)", "considered_all"),
                     ("CONSIDERED (root)", "considered_root"),
                     ("CONSIDERED (per-node)", "considered_pernode"),
                     ("ACCEPTED (all)", "accepted_all"),
                     ("ACCEPTED (root)", "accepted_root"),
                     ("ACCEPTED (per-node)", "accepted_pernode")]:
        print(f"\n--- {tag} ---")
        for k, v in out[key].items():
            n = v.get("n", 0)
            if n == 0:
                print(f"  {k}: n=0"); continue
            print(f"  {k}: n={n} mean={v['mean']:.3f} std={v['std']:.3f} "
                  f"p10={v['p10']:.3f} p50={v['p50']:.3f} p90={v['p90']:.3f}")
    print(f"\nanchor accept rate overall: {len(accepted_anchor['depth'])}/{len(considered['depth'])} = "
          f"{len(accepted_anchor['depth'])/max(1,len(considered['depth']))*100:.2f}%")
    print(f"anchor accept rate root:    {len(accepted_root['depth'])}/{len(considered_root['depth'])} = "
          f"{len(accepted_root['depth'])/max(1,len(considered_root['depth']))*100:.2f}%")
    print(f"anchor accept rate pernode: {len(accepted_pernode['depth'])}/{len(considered_pernode['depth'])} = "
          f"{len(accepted_pernode['depth'])/max(1,len(considered_pernode['depth']))*100:.2f}%")


if __name__ == "__main__":
    main()
