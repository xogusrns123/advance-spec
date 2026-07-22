#!/usr/bin/env python3
"""Online-calibration WINDOW × (head×tail) ablation figures + table.

Reads the 6×6 cross product produced by run_online_window_ablation.sh:
    mat_{ds}_4way_calib_onl_{head}_{tail}_w{W}_split.replay.txt
head ∈ {logistic, beta, linear}, tail ∈ {isotonic, linear},
window ∈ {no-window(1e9), 16000, 8000, 4000, 2000, 1000}.

Emits (readable_outputs/figures/online_ablation/):
  window_ablation_grid.png   per-workload + MEAN subplots; MAT vs window,
                             one line per head×tail combo (∞ = no-window).
  window_ablation_mean.png   MEAN-across-workloads panel, standalone.
and prints a text table per workload + mean.

  python3 scripts/plot/plot_online_window_ablation.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/simulation/Dr.Lee Solution")
RLOG = BASE / "readable_outputs" / "figures" / "replay_logs"
OUT = BASE / "readable_outputs" / "figures" / "online_ablation"

WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "tau2-bench"}
HEADS = ["logistic", "beta", "linear"]
TAILS = ["isotonic", "linear"]
COMBOS = [(h, t) for h in HEADS for t in TAILS]
# windows in ASCENDING order for the x-axis; 1e9 => no-window (plotted as "inf")
WINDOWS = [1000, 2000, 4000, 8000, 16000, 1000000000]
NOWIN = 1000000000
# x positions: finite windows on a log-ish integer ladder, no-window one tick past 16k
XPOS = {1000: 0, 2000: 1, 4000: 2, 8000: 3, 16000: 4, NOWIN: 5}
XLABEL = {1000: "1K", 2000: "2K", 4000: "4K", 8000: "8K", 16000: "16K", NOWIN: "no-win"}

H_COL = {"logistic": "#4C78A8", "beta": "#54A24B", "linear": "#E4A11B"}
T_STYLE = {"isotonic": "-", "linear": "--"}
T_MARK = {"isotonic": "o", "linear": "s"}


def read_k(ds, h, t, w):
    fp = RLOG / f"mat_{ds}_4way_calib_onl_{h}_{t}_w{w}_split.replay.txt"
    if not fp.exists():
        return None
    m = re.search(r"calib: K=([0-9.]+)", fp.read_text())
    return float(m.group(1)) if m else None


# K[ds][(h,t)][w]
K = {ds: {c: {w: read_k(ds, *c, w) for w in WINDOWS} for c in COMBOS} for ds in WLS}

# MEAN across workloads (only where all 5 present)
MEAN = {c: {} for c in COMBOS}
for c in COMBOS:
    for w in WINDOWS:
        vals = [K[ds][c][w] for ds in WLS if K[ds][c][w] is not None]
        MEAN[c][w] = float(np.mean(vals)) if len(vals) == len(WLS) else (
            float(np.mean(vals)) if vals else None)

xs = [XPOS[w] for w in WINDOWS]
xt = [XLABEL[w] for w in WINDOWS]


def draw_panel(ax, series, title):
    for (h, t) in COMBOS:
        ys = [series[(h, t)][w] for w in WINDOWS]
        if all(v is None for v in ys):
            continue
        xv = [XPOS[w] for w, v in zip(WINDOWS, ys) if v is not None]
        yv = [v for v in ys if v is not None]
        ax.plot(xv, yv, color=H_COL[h], ls=T_STYLE[t], marker=T_MARK[t], ms=5,
                lw=1.8, label=f"{h}+{t}")
    ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_xlabel("online window (head pairs; tail = /4)", fontsize=8)


# ---- grid: 5 workloads + mean ------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
axes = axes.ravel()
for i, ds in enumerate(WLS):
    draw_panel(axes[i], K[ds], WL_NAME[ds])
    axes[i].set_ylabel("compose MAT", fontsize=8)
draw_panel(axes[5], MEAN, "MEAN (5 workloads)")
axes[5].set_ylabel("compose MAT", fontsize=8)
axes[5].legend(fontsize=7, ncol=2, loc="best")
fig.suptitle("Online calibration — window size ablation across head x tail combos\n"
             "(deployable split, test half; x = sliding-window size, no-win = full "
             "accumulation)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "window_ablation_grid.png", dpi=150, bbox_inches="tight")
print("saved ->", OUT / "window_ablation_grid.png")

# ---- standalone mean panel ---------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 6))
draw_panel(ax2, MEAN, "Online calibration window ablation — MEAN across 5 workloads")
ax2.set_ylabel("compose MAT (mean accepted tokens / verify step)", fontsize=10)
ax2.legend(fontsize=9, ncol=2, loc="best", title="head + tail")
fig2.tight_layout()
fig2.savefig(OUT / "window_ablation_mean.png", dpi=150, bbox_inches="tight")
print("saved ->", OUT / "window_ablation_mean.png")

# ---- text tables -------------------------------------------------------------
def combo_lbl(c):
    return f"{c[0][:4]}+{c[1][:3]}"

for ds in WLS + ["MEAN"]:
    src = MEAN if ds == "MEAN" else K[ds]
    print(f"\n== {ds} ==")
    print("combo".ljust(12) + "".join(f"{XLABEL[w]:>9}" for w in WINDOWS)
          + "   spread")
    for c in COMBOS:
        row = [src[c][w] for w in WINDOWS]
        fin = [v for v in row if v is not None]
        spread = (max(fin) - min(fin)) if fin else 0.0
        print(combo_lbl(c).ljust(12)
              + "".join((f"{v:>9.3f}" if v is not None else f"{'--':>9}") for v in row)
              + f"   {spread:>6.3f}")
