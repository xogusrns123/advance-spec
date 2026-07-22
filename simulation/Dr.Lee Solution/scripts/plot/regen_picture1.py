#!/usr/bin/env python3
"""Regenerate synthetic Picture1 with REAL data — same 2-panel format.

  Left : Repeated coverage (% of output tokens in a >=4-gram repeat)
         = 100 * mean( R(p) )         R(p) = rg_cover(s, 4)  (see _gr.py)
  Right: Boundary rate (R<->G alternations / 100 tokens)
         = 100 * #{p : R(p) != R(p-1)} / L   on the SAME R/G partition

agentic = swebench / spider(AgenticSQL) / bfcl / tau2 ; non-agentic = specbench.
R/G partition is the canonical n-gram-repeat definition (_gr.rg_cover): R = tokens
covered by a repeated chunk of >=4 tokens, G = the rest. Model-free (uses only
s(p) = realized suffix copy depth along gt, no proposer).

  PYTHONPATH=/workspace python3 scripts/plot/regen_picture1.py
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _gr import rg_cover

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "results" / "interp_validation"
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"

ORANGE, GRAY, NAVY = "#D85A30", "#9A988F", "#1F2A44"

# (display label, workload file, subtask filter or None, is_agentic)
# the 5 experimented workloads (whole-workload units)
COLS = [
    ("SWE-Bench", "swebench", None, True),
    ("AgenticSQL", "spider", None, True),
    ("BFCLv4", "bfcl", None, True),
    ("τ²-bench", "tau2", None, True),
    ("Spec-Bench", "specbench", None, False),
]


def load(w):
    out = []
    with gzip.open(DATA / f"curves_{w}.jsonl.gz", "rt") as f:
        for l in f:
            out.append(json.loads(l))
    return out


def metrics(curves, tf):
    Ltot = cov = bnd = 0
    for r in curves:
        if tf and r.get("task") != tf:
            continue
        s = r["s"]
        L = len(s)
        if L < 2:
            continue
        ind = rg_cover(s)                 # True = R (>=4-gram repeat), False = G
        Ltot += L
        cov += sum(ind)
        bnd += sum(1 for i in range(1, L) if ind[i] != ind[i - 1])
    return (100.0 * cov / Ltot, 100.0 * bnd / Ltot) if Ltot else (0.0, 0.0)


def nice_top(v, step):
    import math
    return math.ceil(v / step) * step


def main():
    cache = {}
    covs, bnds, labels, colors = [], [], [], []
    for lbl, w, tf, ag in COLS:
        cache.setdefault(w, load(w))
        c, b = metrics(cache[w], tf)
        covs.append(c); bnds.append(b); labels.append(lbl)
        colors.append(ORANGE if ag else GRAY)

    x = range(len(COLS))
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.5))
    plt.rcParams.update({"font.size": 13})

    panels = [
        (axes[0], covs, "Repeated coverage (% of tokens in a ≥4-gram repeat)",
         "{:.0f}", nice_top(max(covs), 20), 20, False),
        (axes[1], bnds, "Boundary rate (R↔G alternations / 100 tokens)",
         "{:.1f}", nice_top(max(bnds), 5), 5, True),
    ]
    for ax, vals, title, fmt, top, step, show_leg in panels:
        ax.bar(x, vals, width=0.62, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v + top * 0.012, fmt.format(v), ha="center",
                    va="bottom", fontsize=13, color="#222222")
        ax.set_ylim(0, top)
        ax.set_yticks(range(0, top + 1, step))
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=12.5)
        ax.set_title(title, fontsize=15, color=NAVY, pad=12)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if show_leg:
            ax.legend(handles=[Patch(color=ORANGE, label="agentic"),
                               Patch(color=GRAY, label="non-agentic")],
                      loc="upper right", frameon=False, fontsize=13)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture1.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")
    for lbl, c, b in zip(labels, covs, bnds):
        print(f"  {lbl:<12} coverage={c:5.1f}%  boundary={b:5.2f}/100tok")


if __name__ == "__main__":
    main()
