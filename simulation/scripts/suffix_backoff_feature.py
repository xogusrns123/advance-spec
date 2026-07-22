"""NEW suffix-side feature (no model/fusion): MULTI-LENGTH BACK-OFF consensus.

At a longest-match tie (suffix_p=0.5, n=2, 1+1) the single node is uninformative.
But the SHORTER contexts have far more data. Idea: fold in the head token's
statistics at shorter context lengths (a back-off / Kneser-Ney-style view). If
the tie's head is ALSO the dominant follower at a shorter, high-count context,
the position is really confident (mislabeled 0.5); if the head is rare at short
context, it's a genuine tie. Tests whether this separates accept within the
0.5 / 1+1 bucket -- something the longest-match count-ratio cannot.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict, Counter
sys.path.insert(0, "/workspace")

MAXL = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.gt) if l.strip()]

    # within longest-match 1+1 ties, stratify head-accept by short-context evidence
    strat = {"head is argmax@L2": [0, 0], "head not argmax@L2": [0, 0],
             "head absent@L2": [0, 0]}
    pbin = defaultdict(lambda: [0, 0])   # short-context prob of head -> [hit,n]
    tie_n = 0

    for r in rows:
        prompt = list(r["input_ids"]); gt = list(r["output_ids"])
        if not gt:
            continue
        gcount = [defaultdict(Counter) for _ in range(MAXL + 1)]
        seq = []

        def commit(t):
            for L in range(1, min(MAXL, len(seq)) + 1):
                gcount[L][tuple(seq[-L:])][t] += 1
            seq.append(t)

        for t in prompt:
            commit(t)
        for pos in range(len(gt)):
            g = gt[pos]
            matched = None
            for L in range(min(MAXL, len(seq)), 0, -1):
                key = tuple(seq[-L:])
                if key in gcount[L]:
                    matched = (L, key); break
            if matched is not None:
                L, key = matched
                children = gcount[L][key]; n = sum(children.values())
                head = children.most_common(1)[0][0]
                head_frac = children[head] / n
                # focus: longest-match 1+1 tie
                if n == 2 and len(children) == 2 and head_frac <= 0.5 and L >= 2:
                    tie_n += 1
                    hit = (g == head)
                    # short context = last 2 tokens (more data)
                    k2 = tuple(seq[-2:]); ch2 = gcount[2].get(k2, Counter())
                    if head in ch2:
                        p_head2 = ch2[head] / sum(ch2.values())
                        argmax2 = ch2.most_common(1)[0][0]
                        bkt = "head is argmax@L2" if argmax2 == head else "head not argmax@L2"
                        b = int(p_head2 * 5)  # 0..5 bins
                        pbin[min(b, 4)][0] += hit; pbin[min(b, 4)][1] += 1
                    else:
                        bkt = "head absent@L2"
                    strat[bkt][0] += hit; strat[bkt][1] += 1
            commit(g)

    print(f"longest-match 1+1 tie positions (L>=2): {tie_n}\n")
    print("head-accept stratified by short-context (last-2-token) evidence:")
    for k, (h, n) in strat.items():
        print(f"    {k:<22}: {h/n*100:5.1f}%  (n={n})" if n else f"    {k:<22}: (n=0)")
    print("\nhead-accept by head's probability at the 2-gram context:")
    for b in range(5):
        h, n = pbin[b]
        lo, hi = b/5, (b+1)/5
        print(f"    p_head@L2 in [{lo:.1f},{hi:.1f}): {h/n*100:5.1f}%  (n={n})" if n else
              f"    p_head@L2 in [{lo:.1f},{hi:.1f}): (n=0)")


if __name__ == "__main__":
    main()
