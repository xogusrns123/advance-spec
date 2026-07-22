#!/usr/bin/env python3
"""Regenerate synthetic Picture6 as a MAT figure (not speedup).

Synthetic Picture6 plotted "speedup vs. best single proposer (×)" against the
Act-1 boundary rate. Per the standing rule (no speedup-vs-baseline graphs; MAT
is the standard axis) we plot the MAT gain instead:

    y = ΔMAT = K(compose) − max(K(dflash), K(suffix))      [tokens / step]
    x = boundary rate (alternations / 100 tokens), measured in Act 1 (Picture1)

K values are the mean-accepted-tokens ladder from the consistent Jul-9
interp_validation run (regen1_*.log; the current report_*.json for
spider/swebench/tau2 were re-run mid-session WITHOUT the compose/oracle arms,
so they are not a valid source). agentic = swebench/spider/bfcl/tau2, orange;
non-agentic = specbench, gray.

  python3 scripts/plot/regen_picture6.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"
ORANGE, GRAY, NAVY = "#D85A30", "#9A988F", "#1F2A44"

# (label, boundary_rate, dflash_K, suffix_K, compose_K, is_agentic)
# source: results/interp_validation/regen1_{wl}.log  (Jul-9 consistent run)
ROWS = [
    ("AgenticSQL", 9.64, 2.19, 2.81, 4.21, True),
    ("SWE-Bench", 8.44, 1.28, 2.46, 3.38, True),
    ("τ²-bench", 8.59, 1.92, 2.49, 3.76, True),
    ("BFCLv4", 7.15, 2.87, 2.04, 4.04, True),
    ("Spec-Bench", 7.23, 4.02, 1.13, 3.68, False),
]


def main():
    xs, ys, labs, cols = [], [], [], []
    for lab, br, df, sf, cp, ag in ROWS:
        xs.append(br)
        ys.append(cp - max(df, sf))
        labs.append(lab)
        cols.append(ORANGE if ag else GRAY)

    # least-squares fit + Pearson r
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    r = sxy / (sxx * syy) ** 0.5

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axhline(0, color="#B8B2A6", lw=1.4, zorder=0)
    xl = [min(xs) - 1.2, max(xs) + 1.2]
    ax.plot(xl, [slope * x + intercept for x in xl], ls="--", color="#6B7280",
            lw=1.8, zorder=1)
    ax.scatter(xs, ys, s=190, c=cols, zorder=3, edgecolor="white", lw=1.2)
    for x, y, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(10, 8),
                    fontsize=12.5, color="#222222")
    ax.text(0.97, 0.06, f"least-squares fit,  r = {r:.2f}  (n={n})",
            transform=ax.transAxes, ha="right", fontsize=11, color="#6B7280")
    ax.text(min(xs) - 1.0, 0.06, "compose = best single (no gain)",
            fontsize=10.5, color="#8A8378", va="bottom")

    ax.set_xlabel("boundary rate (alternations / 100 tokens)", fontsize=12.5)
    ax.set_ylabel("MAT gain vs. best single proposer\n"
                  "(compose − best, tokens / step)", fontsize=12.5)
    ax.set_xlim(*xl)
    ax.tick_params(labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture6.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}   slope={slope:.3f} r={r:.2f}")
    for lab, x, y in zip(labs, xs, ys):
        print(f"  {lab:<12} boundary={x:5.2f}  ΔMAT={y:+.2f}")


if __name__ == "__main__":
    main()
