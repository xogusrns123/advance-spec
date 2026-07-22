#!/usr/bin/env python3
"""Companion to BOUNDARY_density_vs_gain.png: the metric that DOES explain the
compose-over-hybrid gap at every granularity -- warm4 = P(suffix can copy >= 4
tokens) (suffix-copyable share). Same 3-panel layout, same data
(gapmetric_candidates.json). Emits readable_outputs/figures/mat/WARM4_vs_gain.png.
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

D = json.load(open("/workspace/simulation/results/pipeline_4way/gapmetric_candidates.json"))
OUT = Path("readable_outputs/figures/mat/boundary_gap")
WLC = {"specbench": "#B279A2", "bfcl": "#54A24B", "swebench": "#F58518",
       "spider": "#4C78A8", "tau2": "#8c564b"}
SEG_MARK = {"think": "o", "tool_call": "s", "final": "^"}
M = "warm4"


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
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

ax = axes[0]
rows = D["workload"]
for r in rows:
    ax.scatter(100 * r[M], r["gain"], s=110, c=WLC[r["wl"]], edgecolor="k",
               linewidth=0.6, zorder=3)
    ax.annotate(r["wl"], (100 * r[M], r["gain"]), xytext=(6, 5),
                textcoords="offset points", fontsize=9.5)
rho = spearman([r[M] for r in rows], [r["gain"] for r in rows])
ax.set_title(f"per WORKLOAD   rho = {rho:+.2f}  (n=5)", fontsize=12,
             fontweight="bold", color="#1a7d1a")

ax = axes[1]
rows = D["subtask"]
for r in rows:
    ax.scatter(100 * r[M], r["gain"], s=52, c=WLC[r["wl"]], edgecolor="k",
               linewidth=0.5, zorder=3)
rho = spearman([r[M] for r in rows], [r["gain"] for r in rows])
sub = [r for r in rows if r["wl"] == "bfcl"]
rb = spearman([r[M] for r in sub], [r["gain"] for r in sub])
ax.set_title(f"per SUBTASK   pooled rho = {rho:+.2f}  (n={len(rows)})\n"
             f"within bfcl rho = {rb:+.2f} (was −0.60 with density)",
             fontsize=10.5, fontweight="bold", color="#1a7d1a")

ax = axes[2]
rows = D["segment"]
for r in rows:
    ax.scatter(100 * r[M], r["gain"], s=80, c=WLC[r["wl"]], marker=SEG_MARK[r["seg"]],
               edgecolor="k", linewidth=0.5, zorder=3)
by = {}
for r in rows:
    by.setdefault(r["wl"], {})[r["seg"]] = r
for wl, d in by.items():
    if "think" in d and "tool_call" in d:
        t, c = d["think"], d["tool_call"]
        ax.annotate("", xy=(100 * c[M], c["gain"]), xytext=(100 * t[M], t["gain"]),
                    arrowprops=dict(arrowstyle="->", color=WLC[wl], lw=1.6, alpha=0.75),
                    zorder=2)
ax.annotate("arrows now point RIGHT-UP:\ntool_call = more suffix-copyable mass\n= bigger gain (4/4 workloads)",
            (0.03, 0.76), xycoords="axes fraction", fontsize=9.5, color="#1a7d1a",
            fontweight="bold")
hs = [plt.Line2D([0], [0], marker=m, ls="", mfc="#999", mec="k", ms=9, label=s)
      for s, m in SEG_MARK.items()]
hs += [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=8, label=w)
       for w, c in WLC.items()]
ax.legend(handles=hs, fontsize=7.5, ncol=2, loc="lower right")
rho = spearman([r[M] for r in rows], [r["gain"] for r in rows])
ax.set_title(f"per SEGMENT (within workload)   pooled rho = {rho:+.2f}\n"
             "think→tool_call arrows now consistent",
             fontsize=10.5, fontweight="bold", color="#1a7d1a")

for ax in axes:
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.grid(alpha=0.2)
    ax.set_xlabel("warm4  =  suffix-copyable share  P(s ≥ 4)   (%)", fontsize=10)
axes[0].set_ylabel("MAT gain:  compose  −  SD-paper hybrid", fontsize=10.5)
fig.suptitle("The gap-explaining metric: suffix-copyable MASS (warm4), not boundary count — "
             "consistent at workload, subtask and segment level",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
fp = OUT / "WARM4_vs_gain.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
