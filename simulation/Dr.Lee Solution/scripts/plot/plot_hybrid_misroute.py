#!/usr/bin/env python3
"""SD-paper hybrid routing-error rate, per workload x segment.

At each fallback round the hybrid commits to ONE proposer (suffix if probe
T>=tau else dflash). We compute BOTH counterfactual accept lengths at that same
round and flag a routing mistake when the road-not-taken would have accepted
strictly more:
  Type A (wrong fallback) : routed dflash (T<tau) but acc_suffix > acc_dflash
  Type B (missed fallback): routed suffix (T>=tau) but acc_dflash > acc_suffix
Ties (equal accept) are never mistakes. Reported as a fraction of rounds.

Outputs -> readable_outputs/figures/mat/hybrid_misroute/
  HYBRID_misroute_rate.png        : stacked TypeA+TypeB bars, per wl x seg
  hybrid_misroute_summary.json
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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measure_k_fusion import ArcticSuffix           # noqa: E402
from fusion_tree import build_extension_chain       # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402
from replay_segments_5way import piece_offsets, tag  # noqa: E402

REC = {"specbench": "results/perpos_specbench_alleval/specbench_4way",
       "bfcl": "results/perpos_bfcl_alleval/bfcl_4way",
       "swebench": "results/perpos_swebench_alleval/swebench_4way",
       "spider": "results/perpos_spider_alleval/spider_4way",
       "tau2": "results/perpos_tau2_alleval/tau2_4way"}
TAU = {"specbench": 32, "bfcl": 32, "swebench": 4, "spider": 16, "tau2": 16}
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
SEGS = ["think", "tool_call", "final"]
MAXR = 4096
OUT = Path("readable_outputs/figures/mat/hybrid_misroute")
CA, CB = "#E45756", "#4C78A8"   # Type A red, Type B blue


def acc_of(tree, gt_rest):
    return len(greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt_rest))


def collect(kind, stem, tau):
    traces = json.load(open(stem + ".traces.json"))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)
    recs = defaultdict(dict)
    for l in open(stem + ".jsonl"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    cats = {}
    for rid in recs:
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        offs, full = piece_offsets(tr["output_ids"], tok)
        cats[rid] = tag(full, offs, kind)
    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    # per seg: [rounds, typeA, typeB, regretA_tok, regretB_tok]
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])
    for rid, rby in recs.items():
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids = tr["output_ids"], tr["prompt_ids"]
        cs = cats[rid]
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
            rest = gt[m + 1:]
            tree_s = build_extension_chain([], suf[:num_spec])
            tree_d = build_extension_chain(block_full[:num_spec], [])
            acc_s = acc_of(tree_s, rest)
            acc_d = acc_of(tree_d, rest)
            route_suffix = T >= tau
            seg = cs[m] if 0 <= m < len(cs) else "final"
            a = agg[seg]; a[0] += 1
            if not route_suffix and acc_s > acc_d:            # Type A: wrong fallback
                a[1] += 1; a[3] += acc_s - acc_d
            elif route_suffix and acc_d > acc_s:              # Type B: missed fallback
                a[2] += 1; a[4] += acc_d - acc_s
            # advance by the ACTUAL chosen route
            if route_suffix:
                acc, tree = acc_s, tree_s
            else:
                acc, tree = acc_d, tree_d
            accepted = [tree.tokens[i] for i in greedy_tree_walk_path(
                list(tree.tokens), list(tree.parents), rest)]
            nxt = [root] + accepted
            if m + 1 + acc < len(gt):
                nxt.append(gt[m + 1 + acc])
            suffix.add_response(nxt)
            m += 1 + acc + 1
            if m >= len(gt):
                break
    return agg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sfp = OUT / "hybrid_misroute_summary.json"
    if os.environ.get("REPLOT") and sfp.exists():
        summary = json.load(open(sfp))
        print("REPLOT: loaded cached summary", flush=True)
        _render(summary)
        return
    summary = {}
    for wl, stem in REC.items():
        if not os.path.exists(stem + ".traces.json"):
            print(f"[{wl}] record missing, skip", flush=True); continue
        agg = collect(wl, stem, TAU[wl])
        summary[wl] = {"tau": TAU[wl], "segs": {}}
        print(f"[{wl}] tau={TAU[wl]}", flush=True)
        for seg in SEGS:
            n, ta, tb, ra, rb = agg.get(seg, [0, 0, 0, 0, 0])
            if not n:
                continue
            summary[wl]["segs"][seg] = {
                "rounds": n,
                "typeA_wrong_fallback_pct": 100.0 * ta / n,
                "typeB_missed_fallback_pct": 100.0 * tb / n,
                "total_misroute_pct": 100.0 * (ta + tb) / n,
                "regretA_tok": ra, "regretB_tok": rb}
            print(f"    {seg:<10} n={n:>6}  A(wrong-fb)={100*ta/n:5.1f}%  "
                  f"B(missed-fb)={100*tb/n:5.1f}%  total={100*(ta+tb)/n:5.1f}%",
                  flush=True)
    json.dump(summary, open(OUT / "hybrid_misroute_summary.json", "w"), indent=2)
    _render(summary)


def _render(summary):
    # ---- stacked bars: per workload cluster, one bar per present segment ----
    labels, A, B, group_span = [], [], [], []
    xs, x = [], 0.0
    tick_pos, tick_lab = [], []
    for wl in REC:
        segs_present = [s for s in SEGS if s in summary[wl]["segs"]]
        start = x
        for s in segs_present:
            d = summary[wl]["segs"][s]
            labels.append(s); A.append(d["typeA_wrong_fallback_pct"])
            B.append(d["typeB_missed_fallback_pct"]); xs.append(x); x += 1.0
        tick_pos.append((start + x - 1.0) / 2.0)
        tick_lab.append(WL_NAME[wl])
        group_span.append((start - 0.4, x - 0.6))
        x += 0.8      # gap

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    bA = ax.bar(xs, A, 0.8, color=CA, edgecolor="k", linewidth=0.5,
                label="Type A — wrong fallback (routed dflash, suffix was better)")
    bB = ax.bar(xs, B, 0.8, bottom=A, color=CB, edgecolor="k", linewidth=0.5,
                label="Type B — missed fallback (routed suffix, dflash was better)")
    for xi, a, b in zip(xs, A, B):
        ax.text(xi, a + b + 0.4, f"{a + b:.0f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", length=0, pad=4)
    # workload names + connector line beneath the segment tick labels
    # (x in data coords, y in axes fraction via get_xaxis_transform)
    xt = ax.get_xaxis_transform()
    for (lo, hi) in group_span:
        ax.plot([lo, hi], [-0.075, -0.075], color="k", lw=0.9,
                transform=xt, clip_on=False)
    for (p, lab) in zip(tick_pos, tick_lab):
        ax.text(p, -0.105, lab, ha="center", va="top", fontsize=11,
                fontweight="bold", transform=xt, clip_on=False)
    ax.set_ylabel("routing-mistake rate  (% of hybrid rounds)", fontsize=11)
    ax.set_title("SD-paper hybrid routing errors per workload × segment "
                 "(τ single-proposer switch)", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_ylim(0, max(a + b for a, b in zip(A, B)) * 1.18)
    ax.margins(x=0.02)
    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    fig.savefig(OUT / "HYBRID_misroute_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", OUT / "HYBRID_misroute_rate.png", flush=True)


if __name__ == "__main__":
    main()
