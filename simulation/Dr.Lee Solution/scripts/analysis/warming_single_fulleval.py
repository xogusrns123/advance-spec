#!/usr/bin/env python3
"""Single-condition MAT (no held-out warm corpus, no pre-warm): the global suffix
tree starts empty and self-warms only from the eval stream. This is exactly the
'warming OFF' definition, so values are lifted from warming_4way.json['off'].
One plot, grouped by workload; four arms (suffix / SD-hybrid / compose / oracle).

NOTE: evaluable set = the recorded eval requests. Held-out warm conversations
have no dflash capture, so hybrid/compose/oracle cannot include them without GPU
re-collection; only suffix could be extended offline.

-> readable_outputs/figures/mat/warming/WARMING_single_fulleval.png
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
import numpy as np

OUT = Path("readable_outputs/figures/mat/warming")
WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
ARMS = ["suffix", "fallback", "calib", "oracle"]
ARM_NAME = {"suffix": "Suffix (single)", "fallback": "SD-paper hybrid",
            "calib": "Compose", "oracle": "Oracle (best handoff)"}
ARM_COL = {"suffix": "#F58518", "fallback": "#9467BD", "calib": "#54A24B",
           "oracle": "#E45756"}

data = json.load(open(OUT / "warming_4way.json"))
x = np.arange(len(WLS))
w = 0.20
fig, ax = plt.subplots(figsize=(12.5, 6.4))
vals = {arm: [data[wl]["arms"][arm]["off"] for wl in WLS] for arm in ARMS}
for j, arm in enumerate(ARMS):
    b = ax.bar(x + (j - 1.5) * w, vals[arm], w, color=ARM_COL[arm],
               edgecolor="k", linewidth=0.6, label=ARM_NAME[arm], zorder=3)
    for bar in b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")

# compose - hybrid gap per workload
for i, wl in enumerate(WLS):
    g = vals["calib"][i] - vals["fallback"][i]
    top = max(vals[a][i] for a in ARMS)
    ax.annotate(f"C−H {g:+.2f}", (i, top + 0.22), ha="center", fontsize=8.5,
                fontweight="bold", color="#1a7d1a" if g >= 0 else "#c0392b")

ax.set_xticks(x)
ax.set_xticklabels([WL_NAME[wl] for wl in WLS], fontsize=11)
ax.set_ylabel("MAT  (mean accepted tokens / round)", fontsize=11)
ax.set_title("MAT per workload — no held-out warm corpus "
             "(tree self-warmed by the eval stream)", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.2, zorder=0)
ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.18)
ax.legend(fontsize=9.5, loc="upper right", ncol=4, framealpha=0.92)
fig.tight_layout()
fp = OUT / "WARMING_single_fulleval.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
