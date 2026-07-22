"""GOAL-2 depth-gated competition policies (tree regime), oracle ceiling.

At each position p (predicting gt[p:] from context gt[:p]) we have the EAGLE tree
gt-accept a_e and the suffix tree gt-accept a_s. In the tree, the gt-path reaches
depth d iff the tree accepts >= d+1 tokens (reach[d] = a >= d+1).

A depth-gated policy decides, per depth d, whether that depth is:
  - "eagle-only"  : advance iff EAGLE reaches d+1  (a_e >= d+1)
  - "competition" : advance iff EAGLE **or** suffix reaches d+1 (max(a_e,a_s) >= d+1)
                    [oracle ceiling of competition = union; gt-path is a chain so
                     union-accept == max(a_e, a_s) — see project_goal2_tree_regime]
Accept = longest contiguous run of satisfied depths.

Policies:
  all_eagle   : every depth eagle-only            -> accept == a_e
  full_comp   : every depth competition           -> accept == max(a_e,a_s) (=oracle)
  policyA     : depth 0 eagle-only, depth>=1 comp
  policyB     : even depth eagle-only, odd depth comp

Bounds: all_eagle <= policyA, policyB <= full_comp (oracle). policyA differs from
full_comp only by giving up suffix at depth 0; policyB additionally re-blocks at
every even depth (eagle must carry those). Also reports eagle's depth-0 accept
rate P(a_e>=1) to check the "EAGLE strong at depth 0" premise.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from arctic_inference.suffix_decoding import SuffixDecodingCache
from simulation.evaluation.tree_knapsack import greedy_tree_walk
from simulation.pipeline.pool_reslicer import reslice_eagle3_pool
from simulation.scripts.goal2_tree_compare import truncate_tree


def _top1_path_accept(tids, pids, fut, cap):
    """Follow the top-1 (first-child) path of a tree and count the leading prefix
    that matches the gt future. First child in index order = highest-count/prob
    child for both the reslice(k'=1) eagle chain and the SuffixDecodingCache tree."""
    node, acc, depth = -1, 0, 0
    while depth < cap:
        child = None
        for i in range(len(pids)):
            if pids[i] == node:
                child = i
                break
        if child is None:
            break
        if acc < len(fut) and tids[child] == fut[acc]:
            acc += 1; node = child; depth += 1
        else:
            break
    return acc


def _policy_accept(a_e, a_s, is_comp):
    """is_comp(d) -> True if depth d is a competition depth."""
    L = 0
    m = max(a_e, a_s)
    while True:
        d = L
        reach = m if is_comp(d) else a_e
        if reach >= d + 1:
            L += 1
        else:
            break
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--budgets", default="8,16,32,64,128,256")
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=8)
    ap.add_argument("--max-spec-factor", type=float, default=100.0)
    ap.add_argument("--chain", action="store_true",
                    help="CHAIN regime: each proposer emits one top-1 chain "
                         "(eagle = reslice k'=1; suffix = top-1 path) instead of a tree.")
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    S, K = args.capture_steps, args.capture_topk
    KP = 1 if args.chain else K  # reslice topk' (chain -> top-1 path)
    d = json.load(open(args.record))

    turns = []
    for q in d["questions"]:
        for turn in q["agent_metrics"]["steps"]:
            ents = (turn.get("spec_decode") or {}).get("oracle_vanilla_entries") or []
            gt, trees = [], []
            for e in ents:
                if not e.get("tokens"):
                    continue
                gt.append(e["tokens"][0][0])
                pool = e.get("eagle3_pool_full") or {}
                tree = ([], [], [])
                if pool.get("draft_tokens") and pool.get("parent_list") is not None \
                        and pool.get("path_probs") is not None and pool.get("pool_size"):
                    try:
                        tree = reslice_eagle3_pool(
                            list(pool["draft_tokens"]), list(pool["parent_list"]),
                            list(pool["path_probs"]), int(pool["pool_size"]), S, K, S, KP)
                    except Exception:
                        pass
                trees.append(tree)
            if len(gt) >= 2:
                turns.append((gt, trees))
    npos = sum(len(g) - 1 for g, _ in turns)
    print(f"turns={len(turns)}  positions={npos}  regime={'CHAIN' if args.chain else 'TREE'}\n")

    polA = lambda d: d >= 1
    polB = lambda d: d % 2 == 1
    print(f"{'budget':>7}{'all_eagle':>11}{'policyA':>10}{'policyB':>10}"
          f"{'full_oracle':>13}{'e_d0_acc':>10}{'A_gap%':>9}{'B_gap%':>9}")
    for B in budgets:
        cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
        rid = 0
        E, A, Bp, O, d0 = [], [], [], [], []
        for gt, trees in turns:
            cache.start_request(rid, np.asarray(gt[:1], dtype=np.int32))
            ctx = [gt[0]]
            for p in range(1, len(gt)):
                fut = gt[p:]
                tok, par, pp = trees[p] if p < len(trees) else ([], [], [])
                et, ep = truncate_tree(tok, par, pp, B)
                a_e = greedy_tree_walk(et, ep, fut) if et else 0  # chain: et is a single path
                try:
                    dr = cache.speculate(rid, np.asarray(ctx, dtype=np.int32),
                                         max_spec_tokens=B, max_spec_factor=args.max_spec_factor,
                                         min_token_prob=0.0, use_tree_spec=True)
                    if args.chain:
                        a_s = _top1_path_accept(list(dr.token_ids), list(dr.parents), fut, B) \
                            if dr.token_ids else 0
                    else:
                        a_s = greedy_tree_walk(list(dr.token_ids), list(dr.parents), fut) \
                            if dr.token_ids else 0
                except Exception:
                    a_s = 0
                E.append(a_e)
                A.append(_policy_accept(a_e, a_s, polA))
                Bp.append(_policy_accept(a_e, a_s, polB))
                O.append(max(a_e, a_s))
                d0.append(1 if a_e >= 1 else 0)
                cache.add_active_response(rid, [int(gt[p])]); ctx.append(gt[p])
            cache.stop_request(rid); rid += 1
        e, a, b, o, z = map(np.mean, (E, A, Bp, O, d0))
        # gap closed over all_eagle toward full_oracle
        gapA = 100 * (a - e) / (o - e) if o > e else 0.0
        gapB = 100 * (b - e) / (o - e) if o > e else 0.0
        print(f"{B:>7}{e:>11.3f}{a:>10.3f}{b:>10.3f}{o:>13.3f}{z:>10.3f}"
              f"{gapA:>9.1f}{gapB:>9.1f}")


if __name__ == "__main__":
    main()
