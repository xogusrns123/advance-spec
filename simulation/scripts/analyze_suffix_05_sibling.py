"""Fundamental question: when suffix decoding reports suffix_p=0.5, if we turned
the single-chain guess into a TREE proposing every known continuation of the
matched context, what fraction of the time is gt among them?  (ceiling for
tree-ifying the 0.5 atom).

Robust definition (avoids the winning-tree local/global ambiguity of the cache's
counts): the matched context is C = ctx[-match_len:] (match_len from the cache).
gt is a "known child" of C iff C followed by gt (the (match_len+1)-gram) has
occurred earlier anywhere in history (prompt + all responses seen so far). That
is exactly "a tree proposing all of C's previously-seen continuations catches gt".

We report, on the cache's p0=0.5 positions (and the n=2 subset):
  P(gt == head)              -- current single-chain accept
  P(gt is a known child)     -- TREE CEILING  <-- the answer
  plus, restricted to contexts that had exactly 2 distinct known children
  (the literal "1+1 / two candidates" case), P(gt in those two).

Run inside sglang-bench.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "/workspace")

MAXL = 24


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--max-spec-tokens", type=int, default=256)
    ap.add_argument("--max-spec-factor", type=float, default=8.0)
    ap.add_argument("--limit-requests", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    from arctic_inference.suffix_decoding import SuffixDecodingCache
    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)

    # history gram -> Counter(next token), over prompt+all responses seen so far
    gnext = [defaultdict(Counter) for _ in range(MAXL + 1)]
    stream = []

    def add_token(t):
        for L in range(1, min(MAXL, len(stream)) + 1):
            gnext[L][tuple(stream[-L:])][t] += 1
        stream.append(t)

    def stats(name):
        return B.setdefault(name, dict(n=0, head_hit=0, tree_hit=0, ml_skip=0,
                                       two_n=0, two_hit=0, nchild=Counter()))
    B = {}

    rid = 0
    n_req = 0
    for line in open(args.gt):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prompt = list(r.get("input_ids") or [])
        gt = list(r.get("output_ids") or [])
        if not gt:
            continue
        n_req += 1
        if args.limit_requests and n_req > args.limit_requests:
            break
        cache.start_request(rid, np.asarray(prompt, dtype=np.int32))
        ctx = list(prompt)
        for t in prompt:                 # prompt is searchable history (local tree has it)
            add_token(t)
        for pos in range(len(gt)):
            gt_tok = gt[pos]
            try:
                d = cache.speculate(rid, np.asarray(ctx, dtype=np.int32),
                                    max_spec_tokens=args.max_spec_tokens,
                                    max_spec_factor=args.max_spec_factor,
                                    min_token_prob=0.0, use_tree_spec=True)
            except Exception:
                d = None
            if d is not None and d.token_ids and d.probs:
                p0 = float(d.probs[0]); c0 = int(d.counts[0]) if d.counts else 0
                head = d.token_ids[0]; ml = int(d.match_len)
                n_node = int(round(c0 / p0)) if p0 > 0 else 0
                buckets = []
                if abs(p0 - 0.5) < args.eps:
                    buckets.append("p=0.5 (all)")
                    if n_node == 2:
                        buckets.append("p=0.5 & n=2")
                if buckets:
                    if ml > MAXL or ml > len(stream):
                        for b in buckets:
                            stats(b)["n"] += 1; stats(b)["ml_skip"] += 1
                    else:
                        C = tuple(stream[-ml:])
                        children = gnext[ml].get(C, Counter())  # prior followers of C
                        tree_hit = gt_tok in children
                        for b in buckets:
                            s = stats(b)
                            s["n"] += 1
                            s["head_hit"] += (gt_tok == head)
                            s["tree_hit"] += tree_hit
                            s["nchild"][len(children)] += 1
                            if len(children) == 2:
                                s["two_n"] += 1
                                s["two_hit"] += tree_hit
            add_token(gt_tok)
            cache.add_active_response(rid, [int(gt_tok)])
            ctx.append(int(gt_tok))
        cache.stop_request(rid)
        rid += 1

    print(f"requests={n_req}\n")
    for name in ("p=0.5 (all)", "p=0.5 & n=2"):
        s = B.get(name)
        if not s or s["n"] == 0:
            print(f"[{name}] none\n"); continue
        n = s["n"]; usable = n - s["ml_skip"]
        print(f"[{name}]  positions={n}  (match_len>{MAXL} skipped: {s['ml_skip']})")
        print(f"    P(gt == head, single chain)        : {s['head_hit']/max(1,usable)*100:.2f}%")
        print(f"    P(gt is a known child, TREE CEILING): {s['tree_hit']/max(1,usable)*100:.2f}%   <== ANSWER")
        gain = (s['tree_hit'] - s['head_hit']) / max(1, usable) * 100
        print(f"    tree gain over chain               : +{gain:.2f} pp")
        print(f"    # known children dist (top): {dict(s['nchild'].most_common(6))}")
        if s["two_n"]:
            print(f"    among contexts with EXACTLY 2 known children (literal 1+1): "
                  f"n={s['two_n']}, P(gt in the two)={s['two_hit']/s['two_n']*100:.2f}%")
        print()


if __name__ == "__main__":
    main()
