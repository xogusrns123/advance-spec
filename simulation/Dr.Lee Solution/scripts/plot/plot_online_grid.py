#!/usr/bin/env python3
"""Online-calibration methodology grid (deployable split, test half, window 8k):
the 6 head×tail calibrator combinations — head ∈ {logistic, beta, linear},
tail ∈ {isotonic, linear} — as grouped MAT bars per workload, with the
SD-paper hybrid drawn as the purple bar-to-beat line. CALIB_effect format.

Reads mat_{ds}_4way_calib_onl_{head}_{tail}_split.replay.txt + the deployable
fallback sweep. Emits readable_outputs/figures/mat(deployable)/ONLINE_grid.png

  python3 scripts/plot/plot_online_grid.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/simulation/Dr.Lee Solution")
RLOG = BASE / "readable_outputs" / "figures" / "replay_logs"
SWEEP = Path("/workspace/simulation/results/pipeline_deployable/segments")
OUT = BASE / "readable_outputs" / "figures" / "mat(deployable)"

WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench\nVerified",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
HEADS = ["logistic", "beta", "linear"]
TAILS = ["isotonic", "linear"]
# head -> color, tail -> hatch/shade (light=isotonic, dark=linear)
H_COL = {"logistic": "#4C78A8", "beta": "#54A24B", "linear": "#E4A11B"}
def shade(hexc, f):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    c = c * f + (1 - f)                       # blend toward white
    return tuple(c)
C_HYB = "#9467BD"


def ck(ds, h, t):
    fp = RLOG / f"mat_{ds}_4way_calib_onl_{h}_{t}_split.replay.txt"
    return float(re.search(r"calib: K=([0-9.]+)", fp.read_text()).group(1))


def hybrid_k(ds):
    d = next(iter(json.load(open(SWEEP / f"fallback_sweep_fresh_{ds}.json")).values()))
    if set(d) == {"calib", "test"}:
        best = max(d["calib"], key=lambda t: d["calib"][t]["K"])
        return d["test"][best]["K"]
    return max(v["K"] for v in d.values())


combos = [(h, t) for h in HEADS for t in TAILS]
K = {ds: {(h, t): ck(ds, h, t) for h, t in combos} for ds in WLS}
hyb = {ds: hybrid_k(ds) for ds in WLS}

x = np.arange(len(WLS))
n = len(combos)
w = 0.8 / n
half = (n - 1) / 2.0
fig, ax = plt.subplots(figsize=(15.5, 6.8))
for i, (h, t) in enumerate(combos):
    col = H_COL[h] if t == "linear" else shade(H_COL[h], 0.55)
    vals = [K[ds][(h, t)] for ds in WLS]
    b = ax.bar(x + (i - half) * w, vals, w, color=col, edgecolor="k",
               linewidth=0.5, zorder=3,
               label=f"{h} head + {t} tail")
    for bar in b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold", rotation=90)

for i, ds in enumerate(WLS):
    ax.hlines(hyb[ds], i - (half + 0.5) * w, i + (half + 0.5) * w, color=C_HYB,
              lw=2.4, zorder=4)
    ax.text(i + (half + 0.5) * w + 0.02, hyb[ds], f"{hyb[ds]:.2f}", color=C_HYB,
            fontsize=8.5, va="center", fontweight="bold")

ax.hlines([], [], [], color=C_HYB, lw=2.4, label="SD-paper hybrid (best τ)")
ax.set_xticks(x)
ax.set_xticklabels([WL_NAME[ds] for ds in WLS], fontsize=11)
ax.set_ylabel("compose MAT  (mean accepted tokens / verify step)", fontsize=11)
ax.set_title("Online calibration — head × tail calibrator grid (window 8k)\n"
             "(deployable split: fit on the calib half, MAT on the disjoint test "
             "half; purple line = SD-paper hybrid baseline)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right", frameon=True, ncol=2)
ax.grid(axis="y", alpha=0.2, zorder=0)
ax.set_ylim(0, max(max(max(K[ds].values()) for ds in WLS), max(hyb.values())) * 1.16)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fp = OUT / "ONLINE_grid.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
hdr = "wl".ljust(10) + "".join(f"{h[:3]}+{t[:3]:<4}" for h, t in combos) + "  hybrid"
print(hdr)
for ds in WLS:
    print(f"{ds:<10}" + "".join(f"{K[ds][c]:>8.2f}" for c in combos) + f"{hyb[ds]:>8.2f}")
print(f"{'MEAN':<10}" + "".join(
    f"{np.mean([K[ds][c] for ds in WLS]):>8.2f}" for c in combos)
    + f"{np.mean(list(hyb.values())):>8.2f}")
