#!/usr/bin/env python3
"""per-category counterpart of SEGMENT_gain_vs_regret: collapse the SD-hybrid
mis-routing into a single scalar (routing regret = accepted tokens lost / round)
and plot it against the compose - hybrid MAT gain, one point per category
(= the record's `task` field, matching density_vs_hybridgain.json's subtask rows).

Outputs -> readable_outputs/figures/mat/boundary_gap/
  CATEGORY_gain_vs_regret.png
  category_regret_summary.json
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

REC = {"specbench": "results/perpos_specbench_alleval/specbench_4way",
       "bfcl": "results/perpos_bfcl_alleval/bfcl_4way",
       "swebench": "results/perpos_swebench_alleval/swebench_4way",
       "spider": "results/perpos_spider_alleval/spider_4way",
       "tau2": "results/perpos_tau2_alleval/tau2_4way"}
TAU = {"specbench": 32, "bfcl": 32, "swebench": 4, "spider": 16, "tau2": 16}
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
WLC = {"specbench": "#B279A2", "bfcl": "#54A24B", "swebench": "#F58518",
       "spider": "#4C78A8", "tau2": "#8c564b"}
BASE = "/workspace/simulation/results/pipeline_4way/"
OUT = Path("readable_outputs/figures/mat/boundary_gap")
MAXR = 4096


def acc_of(tree, rest):
    return len(greedy_tree_walk_path(list(tree.tokens), list(tree.parents), rest))


def collect(stem, tau):
    """-> {task: [rounds, regret_tok]}"""
    traces = json.load(open(stem + ".traces.json"))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)
    recs = defaultdict(dict)
    task_of = {}
    for l in open(stem + ".jsonl"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r
        task_of[r["rid"]] = r["task"]
    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    agg = defaultdict(lambda: [0, 0])
    for rid, rby in recs.items():
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids = tr["output_ids"], tr["prompt_ids"]
        task = task_of[rid]
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
            acc_s, acc_d = acc_of(tree_s, rest), acc_of(tree_d, rest)
            route_suffix = T >= tau
            a = agg[task]; a[0] += 1
            if route_suffix:
                acc, tree = acc_s, tree_s
                if acc_d > acc_s:
                    a[1] += acc_d - acc_s
            else:
                acc, tree = acc_d, tree_d
                if acc_s > acc_d:
                    a[1] += acc_s - acc_d
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
    sfp = OUT / "category_regret_summary.json"
    if os.environ.get("REPLOT") and sfp.exists():
        summary = json.load(open(sfp))
        print("REPLOT: cached", flush=True)
    else:
        summary = {}
        for wl, stem in REC.items():
            if not os.path.exists(stem + ".traces.json"):
                print(f"[{wl}] record missing, skip", flush=True); continue
            agg = collect(stem, TAU[wl])
            summary[wl] = {t: {"rounds": n, "regret_tok": r, "rpr": r / n}
                           for t, (n, r) in agg.items() if n}
            print(f"[{wl}] {len(summary[wl])} categories", flush=True)
        json.dump(summary, open(sfp, "w"), indent=2)

    # merge with density subtask gain
    den = json.load(open(BASE + "density_vs_hybridgain.json"))["subtask"]
    rows = []
    for r in den:
        wl, task = r["wl"], r["task"]
        s = summary.get(wl, {}).get(task)
        if not s:
            continue
        rows.append(dict(wl=wl, task=task, gain=r["gain"], rpr=s["rpr"]))
    print(f"merged {len(rows)} category cells", flush=True)

    xs = np.array([r["rpr"] for r in rows])
    ys = np.array([r["gain"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for r in rows:
        ax.scatter(r["rpr"], r["gain"], c=WLC[r["wl"]], s=70, edgecolor="k",
                   linewidth=0.5, zorder=3)
    a, b = np.polyfit(xs, ys, 1)
    gx = np.linspace(xs.min(), xs.max(), 100)
    fit = ax.plot(gx, a * gx + b, color="#333", lw=1.8, ls="--", zorder=2,
                  label="linear fit")[0]
    # pearson
    mx, my = xs.mean(), ys.mean()
    r_p = float(((xs - mx) * (ys - my)).sum() /
                (((xs - mx) ** 2).sum() ** .5 * ((ys - my) ** 2).sum() ** .5))
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("SD-hybrid routing regret  (accepted tokens lost / round)",
                  fontsize=10.5)
    ax.set_ylabel("MAT gain:  compose − SD-paper hybrid", fontsize=10.5)
    ax.set_title("per category — gain vs hybrid routing regret",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    hs = [plt.Line2D([0], [0], marker="o", ls="", mfc=WLC[w], mec="k", ms=9,
                     label=WL_NAME[w]) for w in WLC if w in {r["wl"] for r in rows}]
    hs.append(fit)
    ax.legend(handles=hs, fontsize=8.5, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "CATEGORY_gain_vs_regret.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> CATEGORY_gain_vs_regret.png  (n={len(rows)}, pearson={r_p:.3f})",
          flush=True)


if __name__ == "__main__":
    main()
