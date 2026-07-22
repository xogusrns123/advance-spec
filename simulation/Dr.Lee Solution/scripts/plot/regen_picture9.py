#!/usr/bin/env python3
"""Regenerate Picture9 with REAL data (honest, no forced fit).

Synthetic Picture9 had 3 panels: (1) |G| CDF with a "one-shot fraction", (2) |R|
CCDF, (3) per-workload "predicted speedup" bars. We rebuild all three from the
real per-position winner curves + real MAT ladder.

Definitions (canonical n-gram-repeat partition, _gr.rg_cover(s, 4); model-free,
uses only the suffix copy-depth curve s):
  |R|  run length of a region covered by a >=4-gram repeat (suffix-copyable)
  |G|  run length of the novel region in between (the head must draft it)

Panel 1/2 pool the AGENTIC workloads (spider+swebench+bfcl+tau2) -- the regime
where a head->suffix handoff can occur. Panel 3 = predicted gain from the R/G
distributions ALONE (no measured MAT). A cycle = one novel gap G followed by its
repeat run R; rounds/cycle (a round = one draft+verify):

  selection (baseline) ~ ceil(Gbar / tau_m) + 1
  composition (ours)   ~ E[ max(ceil((|G| - k*) / tau_m), 0) ] + 1

  predicted speedup = selection_rounds / composition_rounds

The model drafts a gap tau_m tokens/round; the repeat run R is harvested by the
suffix in one round (the "+1"). Composition crosses the R->G boundary in one
chain, bridging the first k* tokens of the gap for free -> only |G|-k* remain.
tau_m, k* are method constants (NOT measured); k*=3 matches Picture3's 3-token
head. Only Gbar and the |G| distribution enter; R sets the per-cycle "+1".

Run inside the container (figures dir is root-owned):
  docker exec sglang-bench bash -lc \
    "cd '/workspace/simulation/Dr.Lee Solution' && python3 scripts/plot/regen_picture9.py"
"""
from __future__ import annotations
import gzip, json, math, os, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _gr import rg_cover, gr_runs, n_boundaries

BASE = Path(__file__).resolve().parent.parent.parent
IDIR = BASE / "results" / "interp_validation"
RES = BASE / "results"
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"
BLUE, ORANGE, GRAY, NAVY = "#2A78D6", "#D85A30", "#9A988F", "#1F2A44"

WL_SRC = {  # display -> (wl key, traces stem, is_agentic)
    "AgenticSQL": ("spider", "perpos_spider_alleval/spider_4way", True),
    "SWE-Bench":  ("swebench", "perpos_swebench_alleval/swebench_4way", True),
    "τ²-bench":   ("tau2", "perpos_tau2_alleval/tau2_4way", True),
    "BFCLv4":     ("bfcl", "perpos_bfcl_full/bfcl_v4_full", True),
    "Spec-Bench": ("specbench", "perpos_specbench_full/specbench", False),
}
ONE_SHOT_MAX = 3          # |G| <= this = one-shot draftable (synthetic marker x=3)
TAU_M = 4                 # model draft tokens accepted per round (method constant)
K_STAR = 3                # composition bridge: gap tokens crossed for free (=Pic3 head)


def predict_speedup(G):
    """Rounds-per-cycle model from the |G| distribution ALONE (+ constants).
    selection ~ ceil(Gbar/tau_m)+1 ; composition ~ E[max(ceil((|G|-k*)/tau_m),0)]+1.
    Returns (speedup, sel_rounds, comp_rounds)."""
    Gbar = sum(G) / len(G)
    sel = math.ceil(Gbar / TAU_M) + 1
    comp = sum(max(math.ceil((g - K_STAR) / TAU_M), 0) for g in G) / len(G) + 1
    return sel / comp, sel, comp


def load_runs(wl, stem):
    """|G| (novel runs), |R| (>=4-gram repeat runs), boundary count (R<->G
    transitions), token count -- from the suffix copy-depth curves only."""
    cur = {}
    with gzip.open(IDIR / f"curves_{wl}.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l)
            cur[r["rid"]] = r
    tr = json.load(open(RES / f"{stem}.traces.json"))
    ev = {t["rid"]: t for t in tr["eval_traces"]}
    has_conv = any("conv" in t for t in tr["eval_traces"])
    units = defaultdict(list)
    for rid, cu in cur.items():
        t = ev.get(rid, {})
        uid = t.get("conv", rid) if has_conv else rid
        units[uid].extend(cu["s"])
    G, R, nb, ntok = [], [], 0, 0
    for s in units.values():
        cov = rg_cover(s)
        g, r = gr_runs(cov)
        G += g
        R += r
        nb += n_boundaries(cov)
        ntok += len(cov)
    return G, R, nb, ntok


def load_K(wl):
    """real MAT ladder arms from the interp_validation report (all 5 workloads)."""
    a = json.load(open(RES / "interp_validation" / f"report_{wl}.json"))["arms"]
    return {k: a[k]["K"] for k in ("dflash", "suffix", "compose")}


def cdf_step(vals):
    vals = sorted(vals)
    n = len(vals)
    xs, ys = [], []
    for i, v in enumerate(vals):
        xs.append(v)
        ys.append((i + 1) / n)
    return xs, ys


def main():
    per = {}
    poolG, poolR = [], []
    for disp, (wl, stem, ag) in WL_SRC.items():
        G, R, nb, ntok = load_runs(wl, stem)
        per[disp] = dict(wl=wl, ag=ag, G=G, R=R, K=load_K(wl),
                         mG=sum(G) / len(G), mR=sum(R) / len(R))
        if ag:
            poolG += G
            poolR += R

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # ---- Panel 1: |G| CDF (pooled agentic) --------------------------------
    ax = axes[0]
    xs, ys = cdf_step(poolG)
    ax.step(xs, ys, where="post", color=BLUE, lw=2.4)
    frac = sum(1 for g in poolG if g <= ONE_SHOT_MAX) / len(poolG)
    ax.axvline(ONE_SHOT_MAX, ls="--", color="#222", lw=1.4)
    ax.plot([0, ONE_SHOT_MAX], [frac, frac], ls=":", color="#222", lw=1.2)
    ax.annotate(f"one-shot fraction {frac:.2f}\n(|G| ≤ {ONE_SHOT_MAX} tok)",
                xy=(ONE_SHOT_MAX, frac), xytext=(ONE_SHOT_MAX + 2.5, frac - 0.20),
                fontsize=11, color="#222",
                arrowprops=dict(arrowstyle="->", color="#222", lw=1.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("|G| (tokens)", fontsize=12.5)
    ax.set_ylabel("CDF", fontsize=12.5)
    ax.set_title("Input ①  |G|: head-draft gaps", color=BLUE, fontsize=13.5)

    # ---- Panel 2: |R| CCDF (pooled agentic), log x ------------------------
    ax = axes[1]
    xs, ys = cdf_step(poolR)
    ccdf = [1 - y for y in ys]
    ax.step(xs, ccdf, where="post", color=ORANGE, lw=2.4)
    ax.set_xscale("log")
    ax.set_xlim(1, max(poolR) * 1.1)
    ax.set_ylim(0, 1.02)
    ax.axvline(sum(poolR) / len(poolR), ls=":", color="#888", lw=1.2)
    ax.set_xlabel("|R| (tokens, log)", fontsize=12.5)
    ax.set_ylabel("CCDF", fontsize=12.5)
    ax.set_title("Input ②  |R|: ≥4-gram repeat runs", color=ORANGE, fontsize=13.5)

    # ---- Panel 3: predicted speedup bars (per workload) -------------------
    ax = axes[2]
    for d in per.values():
        d["pred"], d["sel"], d["comp"] = predict_speedup(d["G"])
    order = sorted(per.items(), key=lambda kv: kv[1]["pred"])
    labs, vals, cols = [], [], []
    for disp, d in order:
        labs.append(disp)
        vals.append(d["pred"])
        cols.append(ORANGE if d["ag"] else GRAY)
    xpos = range(len(labs))
    ax.bar(xpos, vals, color=cols, width=0.62)
    for x, v in zip(xpos, vals):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    ax.axhline(1.0, color="#B8B2A6", lw=1.4)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(labs, fontsize=11, rotation=20, ha="right")
    ax.set_ylim(0.9, max(vals) * 1.06)
    ax.set_ylabel("predicted speedup (×)", fontsize=12.5)
    ax.set_title("Output  predicted gain", fontsize=13.5)

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture9.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {fp}")
    print(f"pooled agentic: |G| n={len(poolG)} mean={sum(poolG)/len(poolG):.2f} "
          f"one-shot(<= {ONE_SHOT_MAX})={frac:.3f} | "
          f"|R| n={len(poolR)} mean={sum(poolR)/len(poolR):.2f}")
    for disp, d in per.items():
        meas = d["K"]["compose"] / max(d["K"]["dflash"], d["K"]["suffix"])
        print(f"  {disp:<12} Gbar={d['mG']:5.2f} sel={d['sel']:.2f} "
              f"comp={d['comp']:.2f} pred={d['pred']:.3f}  (measured={meas:.3f})")


if __name__ == "__main__":
    main()
