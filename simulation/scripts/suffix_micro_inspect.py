"""Microscopic look at suffix p=0.5 positions: decode the context, the proposed
head token, and the ground-truth token to TEXT, split into HIT vs MISS, to build
intuition for what (non-fusion, suffix-side) feature separates accept from miss.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from arctic_inference.suffix_decoding import SuffixDecodingCache
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--ctxwin", type=int, default=18)
    ap.add_argument("--budget", type=int, default=64)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    rows = [json.loads(l) for l in open(args.gt) if l.strip()]
    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)

    hits, misses = [], []
    rid = 0
    for r in rows:
        prompt = list(r["input_ids"]); gt = list(r["output_ids"])
        if not gt:
            continue
        cache.start_request(rid, np.asarray(prompt, dtype=np.int32))
        ctx = list(prompt)
        for pos in range(len(gt)):
            g = gt[pos]
            d = cache.speculate(rid, np.asarray(ctx, dtype=np.int32),
                                max_spec_tokens=args.budget, max_spec_factor=8.0,
                                min_token_prob=0.0, use_tree_spec=True)
            if d.token_ids and d.probs and abs(float(d.probs[0]) - 0.5) < 1e-6:
                head = d.token_ids[0]; ml = int(d.match_len)
                ctx_txt = tok.decode(ctx[-args.ctxwin:])
                item = (ml, repr(ctx_txt), repr(tok.decode([head])), repr(tok.decode([g])))
                (hits if g == head else misses).append(item)
            cache.add_active_response(rid, [int(g)]); ctx.append(int(g))
        cache.stop_request(rid); rid += 1
        if len(hits) > args.n and len(misses) > args.n:
            break

    def show(title, items):
        print(f"\n{'='*90}\n{title}  (showing {min(len(items),args.n)})\n{'='*90}")
        print(f"{'ml':>3}  {'context (…text…)':<58} {'HEAD':>10} -> {'GT':>10}")
        for ml, c, h, g in items[:args.n]:
            print(f"{ml:>3}  {c[-58:]:<58} {h:>10} -> {g:>10}")

    show("HITS  (suffix head == gt)", hits)
    show("MISSES (suffix head != gt)", misses)


if __name__ == "__main__":
    main()
