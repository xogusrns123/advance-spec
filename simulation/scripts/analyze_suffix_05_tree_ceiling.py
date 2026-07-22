"""When suffix decoding reports suffix_p = 0.5 (mostly the matched-node count n=2,
1+1 split), what is the probability the ground-truth next token is among the
matched node's children?  That probability is the CEILING for converting the
0.5 single-chain guess into a tree (proposing all children): a tree can only
recover the miss if gt is one of the children.

Model-free dense replay of a recorded GT trajectory through SuffixDecodingCache
(commit gt one token at a time). At each position we take the draft tree's ROOT
children (parents==-1) = the first-token candidates the suffix trie offers, with
their probs/counts, and check gt membership.

Definitions per position:
  head          = draft.token_ids[0]         (the single chain guess)
  suffix_p      = draft.probs[0]             (head_child.count / matched_node.count)
  root children = tokens with parents==-1    (all first-token candidates in tree)
  n (matched node count) = round(counts[0]/probs[0])

Reports, for the p=0.5 bucket (and the strict n=2 / 1+1 subcase):
  P(gt == head)            -- current single-chain accept
  P(gt in root children)   -- tree ceiling (THE fundamental number asked)

Run inside sglang-bench:
  docker exec sglang-bench python3 /workspace/simulation/scripts/analyze_suffix_05_tree_ceiling.py \
    --gt .../qwen35_27b_3way_real_full/gt_tokens.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace")


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

    # accumulators, keyed by bucket name
    B = {}
    def bucket(name):
        return B.setdefault(name, dict(n=0, head_hit=0, in_children=0,
                                       nchild=Counter(), ncount=Counter(),
                                       other_child_hit=0))

    rid = 0
    n_req = 0
    total_pos = 0
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
                total_pos += 1
                roots = [i for i in range(len(d.parents)) if d.parents[i] == -1]
                root_toks = [d.token_ids[i] for i in roots]
                root_cnts = [int(d.counts[i]) for i in roots] if d.counts else []
                head = d.token_ids[0]
                p0 = float(d.probs[0])
                c0 = int(d.counts[0]) if d.counts else 0
                n_node = int(round(c0 / p0)) if p0 > 0 else 0
                head_hit = (gt_tok == head)
                in_ch = (gt_tok in root_toks)
                other_hit = (in_ch and not head_hit)

                def rec(name):
                    b = bucket(name)
                    b["n"] += 1
                    b["head_hit"] += head_hit
                    b["in_children"] += in_ch
                    b["other_child_hit"] += other_hit
                    b["nchild"][len(roots)] += 1
                    b["ncount"][n_node] += 1

                rec("ALL")
                if abs(p0 - 0.5) < args.eps:
                    rec("p=0.5 (all)")
                    if n_node == 2 and len(roots) == 2 and sorted(root_cnts) == [1, 1]:
                        rec("p=0.5 & n=2 (1+1 split)")
            # commit gt token
            cache.add_active_response(rid, [int(gt_tok)])
            ctx.append(int(gt_tok))
        cache.stop_request(rid)
        rid += 1

    print(f"requests={n_req}  positions with a suffix proposal={total_pos}\n")
    for name in ("ALL", "p=0.5 (all)", "p=0.5 & n=2 (1+1 split)"):
        b = B.get(name)
        if not b or b["n"] == 0:
            print(f"[{name}] (none)\n")
            continue
        n = b["n"]
        print(f"[{name}]  positions={n} ({n/total_pos*100:.1f}% of all)")
        print(f"    P(gt == head, single-chain accept) : {b['head_hit']/n*100:5.2f}%")
        print(f"    P(gt in root children, TREE CEILING): {b['in_children']/n*100:5.2f}%")
        print(f"    extra from siblings (gt=other child): {b['other_child_hit']/n*100:5.2f}%")
        nc = b["nchild"].most_common(5)
        print(f"    #root-children dist: {dict(sorted(b['nchild'].items()))}")
        print(f"    matched-node count n dist (top): {dict(b['ncount'].most_common(6))}")
        print()


if __name__ == "__main__":
    main()
