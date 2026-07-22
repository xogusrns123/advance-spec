#!/usr/bin/env python3
"""THE one-figure summary for Dr.Lee and Kim: a single axis — warm coverage —
answers both of their questions. English-only labels (team figure convention).

  x  : warm coverage  Θs = P(s(p) >= 4)   (the MEASURE / area of the warm set)
  y  : gain in accepted tokens/round, one series per person's own baseline
         Dr.Lee frame: compose - best single proposer     (ρ = +0.94)
         Kim    frame: compose - per-step binary switch   (ρ = +0.87)
  21 task units; thin connectors join the two answers of the same unit.
  Gray band = the empirical decision boundary (Kim frame separates 21/21).

  PYTHONPATH=/workspace python3 scripts/plot_final_coverage.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
D = os.environ.get("INTERP_DIR", "results/interp_validation")
BLUE, PURPLE, GRAY, INK = "#4C78A8", "#B279A2", "#8a8a8a", "#333333"


def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[o[j]] == v[o[i]]:
                j += 1
            for k in range(i, j):
                r[o[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def iso_line(xs, ys):
    from sklearn.isotonic import IsotonicRegression
    import numpy as np
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(np.asarray(xs, float), np.asarray(ys, float))
    gx = np.linspace(min(xs), max(xs), 100)
    return gx, ir.predict(gx)


units = [u for u in json.load(open(f"{D}/units.json")) if u["task"] != "__all__"]
units.sort(key=lambda u: u["warm"])
X = [100 * u["warm"] for u in units]
Y1 = [u["g_drlee"] for u in units]
Y2 = [u["g_kim"] for u in units]
r1, r2 = spearman(X, Y1), spearman(X, Y2)

neg = max(x for x, y in zip(X, Y2) if y <= 0)
pos = min(x for x, y in zip(X, Y2) if y > 0)

fig, ax = plt.subplots(figsize=(11.5, 7.0))
ax.set_axisbelow(True)
ax.grid(alpha=0.3)
ax.axhline(0, color="#888", lw=1.0)
ax.axvspan(neg, pos, color=GRAY, alpha=0.16, lw=0)

for x, y1, y2 in zip(X, Y1, Y2):
    ax.plot([x, x], [y1, y2], color=GRAY, lw=0.8, alpha=0.35, zorder=1)

gx, gy = iso_line(X, Y1)
ax.plot(gx, gy, color=BLUE, lw=1.6, alpha=0.45, zorder=2)
gx, gy = iso_line(X, Y2)
ax.plot(gx, gy, color=PURPLE, lw=1.6, alpha=0.45, zorder=2)

ax.scatter(X, Y1, s=64, marker="o", color=BLUE, edgecolor="white",
           linewidth=0.7, zorder=3,
           label=f"Dr.Lee's question:  compose - best single proposer   (ρ = {r1:+.2f})")
ax.scatter(X, Y2, s=64, marker="^", color=PURPLE, edgecolor="white",
           linewidth=0.7, zorder=3,
           label=f"Kim's question:  compose - per-step switch (SD baseline)   (ρ = {r2:+.2f})")

notes = {("bfcl", "memory_rec_sum"): (4, 6), ("swebench", "xarray"): (-2, 8),
         ("specbench", "translation"): (0, 8), ("specbench", "roleplay"): (0, -14),
         ("specbench", "math_reasoning"): (2, -13), ("spider", "spider_dbt"): (0, 9),
         ("specbench", "rag"): (10, -12)}
for u, x, y in zip(units, X, Y1):
    key = (u["wl"], u["task"])
    if key in notes:
        dx, dy = notes[key]
        t = u["task"] if u["task"] != "spider_dbt" else "spider"
        ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color="#666")

ax.annotate(f"decision boundary {neg:.0f}-{pos:.0f}%\n"
            "Kim frame separates 21/21\n(right of band = compose wins)",
            xy=((neg + pos) / 2, 1.45), fontsize=9, ha="center", color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaa"))

ax.set_xlabel("warm coverage  Θs = P( suffix can copy ≥ 4 tokens at the position )   "
              "—  the 'area' of the warm set  (%)", fontsize=11.5)
ax.set_ylabel("compose gain  (accepted tokens / round)", fontsize=11.5)
ax.set_title("One axis answers both questions — warm coverage\n"
             "slot count (ρ ≤ 0.56) and boundary rate (ρ ≤ 0.41) do not separate workloads:\n"
             "gains follow the AREA of the warm set, not its perimeter",
             fontsize=11.5, pad=14)
ax.text(0.5, -0.13,
        "21 task units (Spider2-DBT · SWE-bench · BFCLv4 · Spec-Bench), "
        "Qwen3.5-27B DFlash+Suffix offline replay; vertical connectors join the "
        "two answers of the same unit", transform=ax.transAxes, ha="center",
        fontsize=8.5, color="#777")
ax.legend(fontsize=10.5, loc="upper left", frameon=True, framealpha=0.9)
ax.set_xlim(5, 58)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(rect=[0, 0.02, 1, 1])
out = f"{D}/figures/final_warm_coverage.png"
fig.savefig(out, dpi=170)
plt.close(fig)
print(f"wrote {out}  (band {neg:.1f}-{pos:.1f}%, rho {r1:+.2f}/{r2:+.2f})")
