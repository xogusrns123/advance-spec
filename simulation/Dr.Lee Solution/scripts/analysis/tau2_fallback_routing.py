#!/usr/bin/env python3
"""Why does the SD-paper hybrid (fallback) drop below pure suffix in tau2
tool_call?  Mirror the fallback policy EXACTLY (suffix if probe T>=tau else
dflash) but, per round, log the route + realized accept + segment. Then break
down the tool_call segment: how often is it routed to the weak dflash draft,
and what accept length does each route realize.
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from measure_k_fusion import ArcticSuffix           # noqa: E402
from fusion_tree import build_extension_chain       # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402
from replay_segments_5way import piece_offsets, tag  # noqa: E402

REC = "results/perpos_tau2_alleval/tau2_4way"
KIND = "tau2"
TAU = 16.0
NUM_SPEC = None
MAXR = 4096


def main():
    traces = json.load(open(Path(REC).with_suffix(".traces.json")))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)

    recs = defaultdict(dict)
    for l in open(REC + ".jsonl"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    cats_by_rid = {}
    for rid in recs:
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        offs, full = piece_offsets(tr["output_ids"], tok)
        cats_by_rid[rid] = tag(full, offs, KIND)

    suffix = ArcticSuffix(); suffix.fit(warm_traces)

    # (seg, route) -> [rounds, sum_acc];  and T histogram in tool_call
    agg = defaultdict(lambda: [0, 0.0])
    Tvals_tc = []
    for rid, rby in recs.items():
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids = tr["output_ids"], tr["prompt_ids"]
        cats = cats_by_rid[rid]
        suffix.new_eval(pids)
        m = 0
        for _ in range(MAXR):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            block_full = rec["dflash_tok"]
            root = gt[m]
            ctx_list = pids + gt[:m] + [root]
            suf, T = suffix.probe(ctx_list, num_spec)
            if T >= TAU:
                route = "suffix"
                tree = build_extension_chain([], suf[:num_spec])
            else:
                route = "dflash"
                tree = build_extension_chain(block_full[:num_spec], [])
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents),
                                         gt[m + 1:])
            acc = len(path)
            seg = cats[m] if 0 <= m < len(cats) else "final"
            a = agg[(seg, route)]; a[0] += 1; a[1] += acc
            if seg == "tool_call":
                Tvals_tc.append(T)
            accepted = [tree.tokens[i] for i in path]
            nxt = [root] + accepted
            if m + 1 + acc < len(gt):
                nxt.append(gt[m + 1 + acc])
            suffix.add_response(nxt)
            m += 1 + acc + 1
            if m >= len(gt):
                break

    print(f"\ntau2 fallback (SD-hybrid, tau={TAU}) routing by segment/route")
    print(f"{'seg':<10}{'route':<9}{'rounds':>8}{'%seg':>7}{'meanAcc':>9}")
    for seg in ("think", "tool_call", "final"):
        tot = sum(agg[(seg, r)][0] for r in ("suffix", "dflash"))
        for route in ("suffix", "dflash"):
            c, s = agg[(seg, route)]
            if c:
                print(f"{seg:<10}{route:<9}{c:>8}{100*c/tot:>6.1f}%{s/c:>9.2f}")
    # tool_call T distribution
    if Tvals_tc:
        Tvals_tc.sort()
        n = len(Tvals_tc)
        below = sum(1 for t in Tvals_tc if t < TAU)
        q = lambda p: Tvals_tc[min(n - 1, int(p * n))]
        print(f"\ntool_call probe-score T: n={n}, below tau({TAU})={100*below/n:.1f}%")
        print(f"  T pctl: p10={q(.1):.1f} p25={q(.25):.1f} p50={q(.5):.1f} "
              f"p75={q(.75):.1f} p90={q(.9):.1f}")


if __name__ == "__main__":
    main()
