#!/usr/bin/env python3
"""Per-workload x per-segment distribution of the suffix probe score T
(= arctic ctx-only expected accept length; the quantity the SD-paper hybrid
thresholds at tau to decide suffix-vs-dflash). T is logged per fallback round
under each workload's pooled-best tau, so the distribution is exactly what the
hybrid sees. Shows why so many tool_call rounds fall below tau and get
mis-routed to the weak dflash draft.

Outputs -> readable_outputs/figures/mat/probe_score/
  T_dist_<wl>.png   (5) : per-workload histogram of T by segment + tau line
  T_dist_overview.png    : box grid, all workloads x segments, log-y + tau ticks
  probe_score_summary.json
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
SEGC = {"think": "#4C78A8", "tool_call": "#E45756", "final": "#59A14F"}
MAXR = 4096
OUT = Path("readable_outputs/figures/mat/probe_score")


def collect_T(kind, stem, tau):
    """Replay the fallback policy; return {seg: [T,...]} over all rounds."""
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
    Tby = defaultdict(list)
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
            if T >= tau:
                tree = build_extension_chain([], suf[:num_spec])
            else:
                tree = build_extension_chain(block_full[:num_spec], [])
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents),
                                         gt[m + 1:])
            acc = len(path)
            seg = cs[m] if 0 <= m < len(cs) else "final"
            Tby[seg].append(float(T))
            accepted = [tree.tokens[i] for i in path]
            nxt = [root] + accepted
            if m + 1 + acc < len(gt):
                nxt.append(gt[m + 1 + acc])
            suffix.add_response(nxt)
            m += 1 + acc + 1
            if m >= len(gt):
                break
    return Tby, num_spec


def pctl(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, summary = {}, {}
    for wl, stem in REC.items():
        if not os.path.exists(stem + ".traces.json"):
            print(f"[{wl}] record missing, skip", flush=True); continue
        Tby, ns = collect_T(wl, stem, TAU[wl])
        data[wl] = Tby
        summary[wl] = {"tau": TAU[wl], "num_spec": ns, "segs": {}}
        for seg in SEGS:
            a = Tby.get(seg, [])
            summary[wl]["segs"][seg] = {
                "n": len(a),
                "below_tau_pct": (100.0 * sum(1 for t in a if t < TAU[wl]) / len(a)) if a else 0.0,
                "mean": float(np.mean(a)) if a else float("nan"),
                "p10": pctl(a, 10), "p25": pctl(a, 25), "p50": pctl(a, 50),
                "p75": pctl(a, 75), "p90": pctl(a, 90)}
        print(f"[{wl}] tau={TAU[wl]}", flush=True)
        for seg in SEGS:
            s = summary[wl]["segs"][seg]
            print(f"    {seg:<10} n={s['n']:>6} below_tau={s['below_tau_pct']:5.1f}% "
                  f"med={s['p50']:.1f} p90={s['p90']:.1f}", flush=True)

    json.dump(summary, open(OUT / "probe_score_summary.json", "w"), indent=2)

    # ---- per-workload histogram ----
    for wl in REC:
        Tby = data[wl]
        allT = [t for seg in SEGS for t in Tby.get(seg, [])]
        if not allT:
            continue
        hi = np.percentile(allT, 99)
        hi = max(hi, TAU[wl] * 1.4)
        bins = np.linspace(0, hi, 46)
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        for seg in SEGS:
            a = np.clip(np.asarray(Tby.get(seg, [])), 0, hi)
            if not len(a):
                continue
            below = 100.0 * np.mean(np.asarray(Tby[seg]) < TAU[wl])
            ax.hist(a, bins=bins, density=True, histtype="stepfilled",
                    color=SEGC[seg], alpha=0.45, edgecolor=SEGC[seg], linewidth=1.4,
                    label=f"{seg}  (below τ: {below:.0f}%)")
        ax.axvline(TAU[wl], color="k", ls="--", lw=1.6, zorder=5)
        ax.text(TAU[wl], ax.get_ylim()[1] * 0.96, f" τ = {TAU[wl]}",
                fontweight="bold", va="top", fontsize=10)
        ax.set_xlabel("suffix probe score  T  (ctx-only expected accept length)", fontsize=10.5)
        ax.set_ylabel("density", fontsize=10.5)
        ax.set_title(f"{WL_NAME[wl]} — probe-score distribution by segment",
                     fontsize=12.5, fontweight="bold")
        ax.legend(fontsize=9.5, loc="upper right", frameon=True)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUT / f"T_dist_{wl}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved ->", OUT / f"T_dist_{wl}.png", flush=True)

    # ---- overview box grid (log-y, tau tick per workload) ----
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    positions, ticks, tlabels = [], [], []
    p = 0
    for wl in REC:
        base = p
        for j, seg in enumerate(SEGS):
            a = data[wl].get(seg, [])
            a = [max(t, 0.05) for t in a]      # floor for log-y
            if a:
                bp = ax.boxplot(a, positions=[p], widths=0.7, showfliers=False,
                                patch_artist=True, medianprops=dict(color="k", lw=1.4))
                for b in bp["boxes"]:
                    b.set(facecolor=SEGC[seg], alpha=0.6, edgecolor="k", linewidth=0.6)
            p += 1
        # tau marker across this workload's 3 boxes
        ax.plot([base - 0.5, p - 0.5], [TAU[wl], TAU[wl]], color="#d62728",
                lw=2.0, ls="--", zorder=6)
        ticks.append((base + p - 1) / 2.0)
        tlabels.append(f"{WL_NAME[wl]}\n(τ={TAU[wl]})")
        p += 1                                  # gap between workloads
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels, fontsize=10)
    ax.set_ylabel("suffix probe score  T   (log scale)", fontsize=11)
    ax.set_title("Probe-score T distribution per workload × segment "
                 "(red dashed = τ threshold)", fontsize=12.5, fontweight="bold")
    seg_h = [plt.Rectangle((0, 0), 1, 1, fc=SEGC[s], alpha=0.6, ec="k") for s in SEGS]
    ax.legend(seg_h, SEGS, fontsize=10, loc="upper right", frameon=True, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "T_dist_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", OUT / "T_dist_overview.png", flush=True)


if __name__ == "__main__":
    main()
