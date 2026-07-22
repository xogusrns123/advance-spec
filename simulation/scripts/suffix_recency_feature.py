"""NEW suffix-side feature (no model, no fusion): RECENCY.

Current suffix_p weights all occurrences of the matched context equally, so a
context seen twice with two different continuations reports 0.5 with no way to
choose. But autoregressive text has strong local coherence: the MORE RECENT
continuation of a context is likelier to recur. We test whether a recency-based
head (follower of the most recent prior occurrence) beats the count-based head
(most frequent follower) -- especially on the tie (no-majority) bucket.

Self-contained: own longest-suffix matcher over the per-request context
(prompt + response so far). Reports, on ALL matched positions and on the TIE
subset (n>=2, head fraction <= 0.5):
   count-head accept   vs   recency-head accept   vs   any-child ceiling
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict, Counter
sys.path.insert(0, "/workspace")

MAXL = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--limit-requests", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.gt) if l.strip()]

    buckets = {"ALL": dict(n=0, ch=0, rh=0, ceil=0, differ=0),
               "TIE (n>=2, head_frac<=0.5)": dict(n=0, ch=0, rh=0, ceil=0, differ=0),
               "n==2 (1+1)": dict(n=0, ch=0, rh=0, ceil=0, differ=0)}

    n_req = 0
    for r in rows:
        prompt = list(r["input_ids"]); gt = list(r["output_ids"])
        if not gt:
            continue
        n_req += 1
        if args.limit_requests and n_req > args.limit_requests:
            break
        gcount = [defaultdict(Counter) for _ in range(MAXL + 1)]   # L -> gram -> Counter(follower)
        grecent = [dict() for _ in range(MAXL + 1)]                # L -> gram -> most-recent follower
        seq = []

        def commit(t):
            for L in range(1, min(MAXL, len(seq)) + 1):
                g = tuple(seq[-L:])
                gcount[L][g][t] += 1
                grecent[L][g] = t
            seq.append(t)

        for t in prompt:
            commit(t)
        for pos in range(len(gt)):
            g = gt[pos]
            # longest suffix match with >=1 prior occurrence
            matched = None
            for L in range(min(MAXL, len(seq)), 0, -1):
                key = tuple(seq[-L:])
                if key in gcount[L]:
                    matched = (L, key); break
            if matched is not None:
                L, key = matched
                children = gcount[L][key]
                n = sum(children.values())
                count_head = children.most_common(1)[0][0]
                recency_head = grecent[L][key]
                head_frac = children[count_head] / n
                differ = int(count_head != recency_head)
                def rec(bk):
                    b = buckets[bk]; b["n"] += 1
                    b["ch"] += (g == count_head); b["rh"] += (g == recency_head)
                    b["ceil"] += (g in children); b["differ"] += differ
                rec("ALL")
                if n >= 2 and head_frac <= 0.5:
                    rec("TIE (n>=2, head_frac<=0.5)")
                if n == 2 and len(children) == 2:
                    rec("n==2 (1+1)")
            commit(g)

    print(f"requests={n_req}\n")
    for bk, b in buckets.items():
        n = b["n"]
        if not n:
            print(f"[{bk}] none\n"); continue
        print(f"[{bk}]  positions={n}")
        print(f"    count-head accept   : {b['ch']/n*100:5.2f}%")
        print(f"    recency-head accept : {b['rh']/n*100:5.2f}%   (delta {(b['rh']-b['ch'])/n*100:+.2f} pp)")
        print(f"    any-child ceiling   : {b['ceil']/n*100:5.2f}%")
        print(f"    count!=recency head : {b['differ']/n*100:5.1f}% of positions")
        print()


if __name__ == "__main__":
    main()
