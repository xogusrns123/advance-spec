"""GOAL-2 depth-gated competition in the CHAIN regime, with a PRE-WARMED suffix
cache (warm data separated from eval data), leave-one-task-out.

For each eval task q: warm the SuffixDecodingCache's global tree with the FULL
responses of all OTHER tasks (start/add/stop -> merges into global), then evaluate
q's turns with that warm global (+ the eval turn's own self-warm active tree).
COLD = same but no warm data (fresh cache, self-warm only). all_eagle is suffix-
independent so must be identical COLD vs WARM (sanity check). Reports a_e, suffix
chain a_s, Policy A/B and full-oracle, COLD vs WARM.

Chain: eagle = reslice(S, k'=1) top-1 chain; suffix = top-1 path of the speculate
tree. Both predict position p from context gt[:p], scored on gt[p:].
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from arctic_inference.suffix_decoding import SuffixDecodingCache
from simulation.evaluation.tree_knapsack import greedy_tree_walk
from simulation.pipeline.pool_reslicer import reslice_eagle3_pool
from simulation.scripts.goal2_tree_compare import truncate_tree
from simulation.scripts.goal2_depth_policy import _top1_path_accept, _policy_accept


def _extract(d, S, K, KP):
    """Return list of tasks; each task = list of turns (gt, eagle_chain_tree)."""
    tasks = []
    for q in d["questions"]:
        turns = []
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
        if turns:
            tasks.append(turns)
    return tasks


def _eval_turn(cache, rid, gt, trees, B, factor, accum):
    cache.start_request(rid, np.asarray(gt[:1], dtype=np.int32))
    ctx = [gt[0]]
    for p in range(1, len(gt)):
        fut = gt[p:]
        tok, par, pp = trees[p] if p < len(trees) else ([], [], [])
        et, ep = truncate_tree(tok, par, pp, B)
        a_e = greedy_tree_walk(et, ep, fut) if et else 0
        try:
            dr = cache.speculate(rid, np.asarray(ctx, dtype=np.int32),
                                 max_spec_tokens=B, max_spec_factor=factor,
                                 min_token_prob=0.0, use_tree_spec=True)
            a_s = _top1_path_accept(list(dr.token_ids), list(dr.parents), fut, B) \
                if dr.token_ids else 0
        except Exception:
            a_s = 0
        accum["e"].append(a_e)
        accum["s"].append(a_s)
        accum["A"].append(_policy_accept(a_e, a_s, lambda d: d >= 1))
        accum["B"].append(_policy_accept(a_e, a_s, lambda d: d % 2 == 1))
        accum["O"].append(max(a_e, a_s))
        cache.add_active_response(rid, [int(gt[p])]); ctx.append(gt[p])
    # do NOT stop_request here: keeps eval data out of the global tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--budgets", default="8,16,32,64")
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=8)
    ap.add_argument("--max-spec-factor", type=float, default=100.0)
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    S, K = args.capture_steps, args.capture_topk
    d = json.load(open(args.record))
    tasks = _extract(d, S, K, 1)  # chain: reslice k'=1
    npos = sum(len(g) - 1 for t in tasks for g, _ in t)
    print(f"tasks={len(tasks)}  positions={npos}  regime=CHAIN  (leave-one-task-out warm)\n")

    for mode in ("COLD", "WARM"):
        print(f"===== {mode} suffix cache =====")
        print(f"{'budget':>7}{'a_e':>8}{'a_s':>8}{'all_eagle':>11}{'policyA':>10}"
              f"{'policyB':>10}{'oracle':>9}{'A_gap%':>9}")
        for B in budgets:
            acc = {k: [] for k in ("e", "s", "A", "B", "O")}
            for qi in range(len(tasks)):
                cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
                if mode == "WARM":
                    wid = 10_000_000
                    for qj in range(len(tasks)):
                        if qj == qi:
                            continue
                        for gt, _ in tasks[qj]:
                            cache.start_request(wid, np.asarray(gt[:1], dtype=np.int32))
                            if len(gt) > 1:
                                cache.add_active_response(wid, [int(x) for x in gt[1:]])
                            cache.stop_request(wid); wid += 1
                rid = qi * 1000
                for gt, trees in tasks[qi]:
                    _eval_turn(cache, rid, gt, trees, B, args.max_spec_factor, acc)
                    rid += 1
            e, s, A, Bp, O = (np.mean(acc[k]) for k in ("e", "s", "A", "B", "O"))
            gapA = 100 * (A - e) / (O - e) if O > e else 0.0
            print(f"{B:>7}{e:>8.3f}{s:>8.3f}{e:>11.3f}{A:>10.3f}{Bp:>10.3f}{O:>9.3f}{gapA:>9.1f}")
        print()


if __name__ == "__main__":
    main()
