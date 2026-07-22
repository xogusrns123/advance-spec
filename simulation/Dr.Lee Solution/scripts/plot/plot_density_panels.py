#!/usr/bin/env python3
"""BOUNDARY_density_vs_gain split into 3 independent images (workload / subtask /
segment), no annotations. density (winner-flip /1K) vs compose-over-hybrid gain.
Data: density_vs_hybridgain.json.
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("/workspace/simulation/results/pipeline_4way/density_vs_hybridgain.json"))
OUT = Path("readable_outputs/figures/mat/boundary_gap")
WLC = {"specbench": "#B279A2", "bfcl": "#54A24B", "swebench": "#F58518",
       "spider": "#4C78A8", "tau2": "#8c564b"}
SEG_MARK = {"think": "o", "tool_call": "s", "final": "^"}


def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        od = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j < n and v[od[j]] == v[od[i]]:
                j += 1
            for k in range(i, j):
                r[od[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((p - my) ** 2 for p in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}


def base(ax):
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.grid(alpha=0.2)
    ax.set_xlabel("winner-change boundary density  / 1K tok", fontsize=10.5)
    ax.set_ylabel("MAT gain:  compose − SD-paper hybrid", fontsize=10.5)


def wl_legend(ax, present, extra=None, loc="lower right"):
    hs = [plt.Line2D([0], [0], marker="o", ls="", mfc=WLC[w], mec="k", ms=9,
                     label=WL_NAME[w]) for w in WLC if w in present]
    if extra:
        hs += extra
    ax.legend(handles=hs, fontsize=8.5, loc=loc, frameon=True)


# workload
rows = D["workload"]
fig, ax = plt.subplots(figsize=(7.0, 5.4))
for r in rows:
    ax.scatter(r["dens"], r["gain"], s=120, c=WLC[r["wl"]], edgecolor="k", linewidth=0.6, zorder=3)
rho = spearman([r["dens"] for r in rows], [r["gain"] for r in rows])
ax.set_title("per workload", fontsize=12, fontweight="bold")
base(ax); wl_legend(ax, {r["wl"] for r in rows}); fig.tight_layout()
fig.savefig(OUT / "BOUNDARY_density_vs_gain_workload.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# category
import numpy as np
rows = D["subtask"]
fig, ax = plt.subplots(figsize=(7.0, 5.4))
xs = [r["dens"] for r in rows]; ys = [r["gain"] for r in rows]
for r in rows:
    ax.scatter(r["dens"], r["gain"], s=55, c=WLC[r["wl"]], edgecolor="k", linewidth=0.5, zorder=3)
a, b = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
gx = np.linspace(min(xs), max(xs), 100)
fitline = ax.plot(gx, a * gx + b, color="#333", lw=1.8, ls="--", zorder=2, label="linear fit")[0]
rho = spearman(xs, ys)
ax.set_title("per category", fontsize=12, fontweight="bold")
base(ax); wl_legend(ax, {r["wl"] for r in rows}, extra=[fitline]); fig.tight_layout()
fig.savefig(OUT / "BOUNDARY_density_vs_gain_category.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# segment
rows = D["segment"]
fig, ax = plt.subplots(figsize=(7.0, 5.4))
for r in rows:
    ax.scatter(r["dens"], r["gain"], s=90, c=WLC[r["wl"]], marker=SEG_MARK[r["seg"]],
               edgecolor="k", linewidth=0.5, zorder=3)
rho = spearman([r["dens"] for r in rows], [r["gain"] for r in rows])
segh = [plt.Line2D([0], [0], marker=m, ls="", mfc="#999", mec="k", ms=9, label=sg)
        for sg, m in SEG_MARK.items()]
wl_legend(ax, {r["wl"] for r in rows}, extra=segh, loc="upper right")
ax.set_title("per segment", fontsize=12, fontweight="bold")
base(ax); fig.tight_layout()
fig.savefig(OUT / "BOUNDARY_density_vs_gain_segment.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved 3 panels -> BOUNDARY_density_vs_gain_{workload,subtask,segment}.png")
