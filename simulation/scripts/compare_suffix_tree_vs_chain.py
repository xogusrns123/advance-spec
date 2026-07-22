"""Direct measure of the suffix-decoding TREE structure's effect: replay a GT
trajectory through SuffixDecodingCache with use_tree_spec True (tree) vs False
(single chain), each advancing by its OWN accept, and compare MAT across draft
budgets. Answers 'does the tree structure actually help (vs a chain)?'.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from arctic_inference.suffix_decoding import SuffixDecodingCache
from simulation.evaluation.tree_knapsack import greedy_tree_walk


def run(rows, tree, budget, factor=8.0):
    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
    accs = []
    rid = 0
    for r in rows:
        prompt = list(r["input_ids"]); gt = list(r["output_ids"])
        if not gt:
            continue
        cache.start_request(rid, np.asarray(prompt, dtype=np.int32))
        ctx = list(prompt); pos = 0
        while pos < len(gt):
            d = cache.speculate(rid, np.asarray(ctx, dtype=np.int32),
                                max_spec_tokens=budget, max_spec_factor=factor,
                                min_token_prob=0.0, use_tree_spec=tree)
            acc = greedy_tree_walk(list(d.token_ids), list(d.parents), gt[pos:]) if d.token_ids else 0
            accs.append(acc)
            commit = min(acc + 1, len(gt) - pos); seg = gt[pos:pos + commit]
            cache.add_active_response(rid, [int(t) for t in seg]); ctx.extend(seg); pos += commit
        cache.stop_request(rid); rid += 1
    a = np.asarray(accs, float)
    return a.mean(), (a >= 1).mean(), (a >= 3).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.gt) if l.strip()]
    hdr = "%7s %10s %9s %10s   %14s   %14s" % (
        "budget", "chain MAT", "tree MAT", "tree gain", "P(acc>=1) c/t", "P(acc>=3) c/t")
    print(hdr)
    for budget in (8, 16, 32, 64, 256):
        cm, c1, c3 = run(rows, False, budget)
        tm, t1, t3 = run(rows, True, budget)
        print("%7d %10.3f %9.3f %+10.3f   %6.3f/%6.3f   %6.3f/%6.3f" % (
            budget, cm, tm, tm - cm, c1, t1, c3, t3))


if __name__ == "__main__":
    main()
