"""MAT bars: monotone calibration vs joint Bayes (same/more features) vs oracle.
Numbers from analyze_calib_bayes_mat.py (served-scale = recon + 0.0256)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "simulation/results/calib_why_analysis/figures/bayes_vs_calib_mat.png"
# (label, served MAT, over-raw, form)
rows = [
    ("raw\n(ep,sp, 1 thr)", 1.318, 0.000, "raw"),
    ("bayes joint\n(ep,sp)", 1.328, 0.010, "joint"),
    ("calib MONOTONE\n(ep,sp,depth)", 1.334, 0.016, "mono"),
    ("bayes JOINT\n(ep,sp,depth)", 1.350, 0.032, "joint"),
    ("bayes JOINT\n(all 6 feat)", 1.364, 0.046, "joint"),
    ("oracle (GT)", 1.787, 0.469, "oracle"),
]
col = {"raw": "#1f77b4", "mono": "#9467bd", "joint": "#17becf", "oracle": "#2ca02c"}
labels = [r[0] for r in rows]; mats = [r[1] for r in rows]
over = [r[2] for r in rows]; cols = [col[r[3]] for r in rows]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.4, 1]})

# left: absolute MAT
b = ax.bar(range(len(rows)), mats, color=cols)
for i, (m, o) in enumerate(zip(mats, over)):
    ax.text(i, m + 0.006, f"{m:.3f}\n(+{o:.3f})" if i else f"{m:.3f}", ha="center",
            fontsize=8.5)
ax.axhline(1.318, color="#1f77b4", ls=":", lw=1, alpha=0.7)
ax.axhline(1.787, color="#2ca02c", ls="--", lw=1, alpha=0.7)
ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("MAT (served-scale est.)"); ax.set_ylim(1.25, 1.83)
ax.set_title("Joint Bayes > monotone calibration (same features)\n"
             "but all feature-based selectors << oracle")
ax.grid(axis="y", alpha=0.3)
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=col["mono"], label="monotone calibration"),
                   mp.Patch(color=col["joint"], label="joint Bayes (learned)"),
                   mp.Patch(color=col["oracle"], label="oracle (GT)")], fontsize=8)

# right: decomposition of the +0.046 (over raw)
parts = [("raw\nbaseline", 0.000, "#1f77b4"),
         ("+ form\n(mono→joint)\n@same feats", 0.016, "#17becf"),
         ("+ extra feats\n(mlen,cnt,tot)", 0.014, "#0aa"),
         ("irreducible\n(needs GT)", 0.469 - 0.046, "#cccccc")]
bottom = 0.0
for name, val, c in parts:
    ax2.bar(0, val, bottom=bottom, color=c, width=0.5, edgecolor="w")
    if val > 0.005:
        ax2.text(0, bottom + val / 2, f"{name}\n+{val:.3f}", ha="center", va="center",
                 fontsize=8)
    bottom += val
ax2.text(0, 0.480, "= oracle gap 0.469", ha="center", fontsize=9, weight="bold")
ax2.scatter([0], [0.016], s=0)  # spacer
ax2.set_xlim(-0.6, 0.6); ax2.set_xticks([])
ax2.set_ylabel("MAT recovered over raw")
ax2.set_title("of the raw→oracle gap (+0.469):\njoint+features claws back +0.046 (≈10%),\n"
              "calibration alone only +0.016 (3%)")
ax2.set_ylim(0, 0.50); ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Bayes/joint vs calibration: where the MAT comes from "
             "(Qwen3-14B, alive-conditioned, OOF)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
