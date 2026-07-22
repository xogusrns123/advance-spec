"""GOAL-2 tree-regime comparison at MATCHED node budgets, from an oracle-vanilla
EAGLE3 capture (eagle3_pool_full = full draft tree; oracle-vanilla commits 1 gt
token/step so we get the tree at EVERY gt position on the greedy trajectory).

Per tree-node BUDGET B (swept), on the SAME gt trajectory:
  - EAGLE3 tree@B : the REAL EAGLE tree, reconstructed per-node from the captured
                    beam pool via pool_reslicer.reslice_eagle3_pool(S,K -> S,K)
                    (identity reslice = full native tree with correct parents),
                    then truncated to the top-B nodes by cumulative path_prob with
                    ancestor closure (= what organize_draft_results(ndt=B) selects).
  - suffix tree@B : live SuffixDecodingCache tree, max_spec_tokens=B (factor
                    non-binding), greedy gt-path walk.
  - UNION-ORACLE@B: per-position max(eagle3@B, suffix@B) gt-path accept (verify
                    BOTH trees; nodes = eagle3_nodes + suffix_nodes).
Reports MAT (mean accept) and actual nodes/step per method per budget.

NOTE: the captured `parent_list` is EAGLE's per-beam structure (len (S-1)*K+1),
NOT a per-node parent array — so it MUST be run through reslice_eagle3_pool to get
the true node topology. A naive per-node read of parent_list gives a wrong (chain-
like) tree and undercounts eagle3 MAT.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from arctic_inference.suffix_decoding import SuffixDecodingCache
from simulation.evaluation.tree_knapsack import greedy_tree_walk
from simulation.pipeline.pool_reslicer import reslice_eagle3_pool


def truncate_tree(token_ids, parents, path_probs, B):
    """Top-B nodes by path_prob forming a connected subtree under the VIRTUAL
    root (-1). A node is addable once its parent is kept; children of the virtual
    root (parent == -1) are always addable. Returns reindexed (tok, par) with the
    virtual-root convention (par == -1 => child of root) that greedy_tree_walk uses.
    """
    n = len(token_ids)
    if n == 0:
        return [], []
    if path_probs is None or len(path_probs) != n:
        path_probs = [1.0 / (i + 1) for i in range(n)]
    kept = set()
    # greedily add the highest-path_prob node whose parent is already kept
    # (or whose parent is the virtual root) until we reach B nodes.
    while len(kept) < min(B, n):
        best, bp = None, -1.0
        for j in range(n):
            if j in kept:
                continue
            pj = parents[j]
            if (pj == -1 or pj in kept) and path_probs[j] > bp:
                best, bp = j, path_probs[j]
        if best is None:
            break
        kept.add(best)
    order = sorted(kept)
    remap = {old: i for i, old in enumerate(order)}
    tok = [token_ids[o] for o in order]
    par = [(-1 if parents[o] == -1 else remap.get(parents[o], -1)) for o in order]
    return tok, par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--budgets", default="8,16,32,64,128,256")
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=8)
    ap.add_argument("--max-spec-factor", type=float, default=100.0)  # non-binding -> budget binds
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    S, K = args.capture_steps, args.capture_topk
    d = json.load(open(args.record))
    qs = d["questions"]

    # pre-extract per-turn gt sequences + eagle3 per-node trees (reconstructed once)
    turns = []
    n_reslice_fail = 0
    for q in qs:
        for turn in q["agent_metrics"]["steps"]:
            ents = (turn.get("spec_decode") or {}).get("oracle_vanilla_entries") or []
            gt, trees = [], []
            for e in ents:
                if not e.get("tokens"):
                    continue  # keep gt[i] and trees[i] aligned: skip token-less entries together
                gt.append(e["tokens"][0][0])
                pool = e.get("eagle3_pool_full") or {}
                dt = pool.get("draft_tokens")
                pl = pool.get("parent_list")
                pp = pool.get("path_probs")
                ps = pool.get("pool_size")
                tree = ([], [], [])
                if dt and pl is not None and pp is not None and ps:
                    try:
                        tree = reslice_eagle3_pool(
                            list(dt), list(pl), list(pp), int(ps), S, K, S, K)
                    except Exception:
                        n_reslice_fail += 1
                trees.append(tree)
            if len(gt) < 2:
                continue
            turns.append((gt, trees))
    print(f"turns={len(turns)}  positions={sum(len(g) for g,_ in turns)}"
          f"  reslice_fail={n_reslice_fail}")

    # ORACLE has two cost notions, SAME MAT (= per-position max):
    #   * verify-both  : run BOTH trees every step -> union_nodes = e3+sfx nodes.
    #   * select-oracle: perfectly pick the BETTER single proposer per position ->
    #                    cost = winner's tree nodes (1 tree). This is the ceiling
    #                    the raw/calib/OURS SELECTION methods chase (they also emit
    #                    1 tree/step). We report its node cost separately.
    print(f"\n{'budget':>7}{'e3_MAT':>9}{'e3_nodes':>10}{'sfx_MAT':>9}{'sfx_nodes':>11}"
          f"{'oracle_MAT':>11}{'sel_nodes':>11}{'both_nodes':>12}")
    for B in budgets:
        ae, asf, au, ne, ns, nsel = [], [], [], [], [], []
        cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
        rid = 0
        for gt, trees in turns:
            # Both proposers predict the SAME position p from context gt[:p] and
            # are scored on the greedy gt-path gt[p:]:
            #   - EAGLE tree[p] is drafted from gt[p-1]'s hidden state -> depth-1
            #     predicts position p (verified empirically: fut must be gt[p:]).
            #   - suffix speculates on ctx = gt[:p] (fed incrementally) -> gt[p:].
            # Start at p=1 (position 0 has no suffix context; 1 pos/turn dropped).
            cache.start_request(rid, np.asarray(gt[:1], dtype=np.int32))
            ctx = [gt[0]]
            for p in range(1, len(gt)):
                fut = gt[p:]
                tok, par, pp = trees[p] if p < len(trees) else ([], [], [])
                et, ep = truncate_tree(tok, par, pp, B)
                a_e = greedy_tree_walk(et, ep, fut) if et else 0
                try:
                    dr = cache.speculate(rid, np.asarray(ctx, dtype=np.int32),
                                         max_spec_tokens=B, max_spec_factor=args.max_spec_factor,
                                         min_token_prob=0.0, use_tree_spec=True)
                    a_s = greedy_tree_walk(list(dr.token_ids), list(dr.parents), fut) if dr.token_ids else 0
                    sn = len(dr.token_ids)
                except Exception:
                    a_s, sn = 0, 0
                ae.append(a_e); asf.append(a_s); au.append(max(a_e, a_s))
                ne.append(len(et)); ns.append(sn)
                nsel.append(len(et) if a_e >= a_s else sn)  # winner's tree cost
                cache.add_active_response(rid, [int(gt[p])]); ctx.append(gt[p])
            cache.stop_request(rid); rid += 1
        ae, asf, au = np.mean(ae), np.mean(asf), np.mean(au)
        ne, ns, nsel = np.mean(ne), np.mean(ns), np.mean(nsel)
        print(f"{B:>7}{ae:>9.3f}{ne:>10.2f}{asf:>9.3f}{ns:>11.2f}{au:>11.3f}{nsel:>11.2f}{ne+ns:>12.2f}")


if __name__ == "__main__":
    main()
