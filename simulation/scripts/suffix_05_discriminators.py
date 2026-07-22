"""Within the suffix_p=0.5 bucket, do features a PATCH could expose separate
accept (head==gt) from miss?  The current metric (suffix_p, count, total,
match_len) can't tell a genuine hard 2-way tie from a reliable count-2 match.
We query the LOCAL tree and the GLOBAL tree separately (they are the two sources
speculate() picks the max-score of) and test whether provenance / cross-tree
agreement / match_len stratify the accept rate inside the 0.5 bucket.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/workspace")
from arctic_inference.suffix_decoding import SuffixDecodingCache
from arctic_inference.suffix_decoding._C import SuffixTree

D = 64


def spec(tree, ctx, budget, factor):
    c = ctx[-D:] if len(ctx) > D else ctx
    return SuffixTree.speculate_ndarray(tree, np.asarray(c, dtype=np.int32),
                                        budget, factor, 0.0, 0.0, True)


def rate(hit, n):
    return f"{hit/n*100:5.1f}% (n={n})" if n else "     (n=0)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--factor", type=float, default=8.0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.gt) if l.strip()]

    cache = SuffixDecodingCache(max_tree_depth=D, max_cached_requests=100000)
    by_src = defaultdict(lambda: [0, 0])          # src -> [hit, n]
    by_agree = defaultdict(lambda: [0, 0])        # agree bool -> [hit,n]
    by_ml = defaultdict(lambda: [0, 0])           # ml bin -> [hit,n]
    by_otherconf = defaultdict(lambda: [0, 0])    # loser p0 >= .9? -> [hit,n]
    tot = [0, 0]
    rid = 0
    for r in rows:
        prompt = list(r["input_ids"]); gt = list(r["output_ids"])
        if not gt:
            continue
        cache.start_request(rid, np.asarray(prompt, dtype=np.int32))
        ctx = list(prompt)
        lt = cache._local_trees[rid]; gt_tree = cache._global_tree
        for pos in range(len(gt)):
            g = gt[pos]
            comb = cache.speculate(rid, np.asarray(ctx, dtype=np.int32),
                                   max_spec_tokens=args.budget, max_spec_factor=args.factor,
                                   min_token_prob=0.0, use_tree_spec=True)
            if comb.token_ids and comb.probs and abs(float(comb.probs[0]) - 0.5) < 1e-6:
                head = comb.token_ids[0]; acc = int(g == head); ml = int(comb.match_len)
                ld = spec(lt, ctx, args.budget, args.factor)
                gd = spec(gt_tree, ctx, args.budget, args.factor)
                lhas = bool(ld.token_ids); ghas = bool(gd.token_ids)
                lscore = ld.score if lhas else -1.0
                gscore = gd.score if ghas else -1.0
                src = "local" if lscore >= gscore else "global"
                tot[0] += acc; tot[1] += 1
                by_src[src][0] += acc; by_src[src][1] += 1
                # cross-tree agreement (both propose a head)
                if lhas and ghas:
                    ag = (ld.token_ids[0] == gd.token_ids[0])
                    by_agree[ag][0] += acc; by_agree[ag][1] += 1
                    # confidence of the LOSING tree
                    loser_p = (gd.probs[0] if src == "local" else ld.probs[0])
                    key = "loser_p>=0.9" if loser_p >= 0.9 else "loser_p<0.9"
                    by_otherconf[key][0] += acc; by_otherconf[key][1] += 1
                else:
                    by_agree["only_one_tree"][0] += acc; by_agree["only_one_tree"][1] += 1
                b = "1" if ml == 1 else ("2-3" if ml <= 3 else ("4-7" if ml <= 7 else "8+"))
                by_ml[b][0] += acc; by_ml[b][1] += 1
            cache.add_active_response(rid, [int(g)]); ctx.append(int(g))
        cache.stop_request(rid); rid += 1

    print(f"p=0.5 positions: {tot[1]}   overall head-accept: {rate(tot[0], tot[1])}\n")
    print("by winning tree (provenance):")
    for k, (h, n) in sorted(by_src.items()):
        print(f"    {k:<8}: {rate(h, n)}")
    print("\nby cross-tree head agreement:")
    for k in (True, False, "only_one_tree"):
        if k in by_agree:
            h, n = by_agree[k]; print(f"    {str(k):<14}: {rate(h, n)}")
    print("\nby losing-tree confidence (when both trees present):")
    for k, (h, n) in sorted(by_otherconf.items()):
        print(f"    {k:<12}: {rate(h, n)}")
    print("\nby match_len:")
    for k in ("1", "2-3", "4-7", "8+"):
        if k in by_ml:
            h, n = by_ml[k]; print(f"    ml {k:<4}: {rate(h, n)}")


if __name__ == "__main__":
    main()
