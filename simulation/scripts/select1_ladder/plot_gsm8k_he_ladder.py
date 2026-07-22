"""Realized select-1 ladder for GSM8K (math) + HumanEval (code), 2-way (MTP+suffix), Qwen3.5-27B.
Grouped bars per subtask + overall: raw / calib(logistic) / bayes(GBM) / oracle; native MTP as a
dashed reference line. Numbers are REALIZED served accept-length-mean (each arm builds its own draft
chain live; oracle replays the record). Run IN docker (figures dir is root-owned)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = "/workspace/simulation/results/chain_hybrid_perdepth/gsm8k_humaneval_2way"
OUT = f"{DIR}/figures"; os.makedirs(OUT, exist_ok=True)

# per-subtask (from analyze_specbench_subtask.py) + overall (from run*.json)
PER = {  # subtask: raw, calib, bayes, oracle
    "gsm8k\n(math)":     [5.109, 3.477, 5.842, 6.512],
    "humaneval\n(code)": [5.829, 3.888, 6.587, 7.338],
    "overall":           [5.329, 3.609, 6.072, 6.767],
}
NATIVE = {"gsm8k\n(math)": None, "humaneval\n(code)": None, "overall": 6.017}
LABELS = ["raw\n(prob argmax)", "calib\n(logistic)", "bayes\n(GBM)", "oracle\n(ceiling)"]
COLORS = ["#9e9e9e", "#e57373", "#42a5f5", "#2e7d32"]

groups = list(PER.keys())
x = np.arange(len(groups)); w = 0.2
fig, ax = plt.subplots(figsize=(10, 5.6))
for i, (lab, col) in enumerate(zip(LABELS, COLORS)):
    vals = [PER[g][i] for g in groups]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.04, f"{v:.2f}", ha="center",
                va="bottom", fontsize=8)
# native reference (overall only)
for gi, g in enumerate(groups):
    nv = NATIVE[g]
    if nv is None: continue
    ax.hlines(nv, x[gi] - 2*w, x[gi] + 2*w, colors="black", linestyles="--", lw=1.4)
    ax.text(x[gi] + 2*w, nv, f" native {nv:.2f}", va="center", fontsize=8.5)

ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("realized mean accept length (MAT)")
ax.set_title("GSM8K + HumanEval · select-1 realized ladder · Qwen3.5-27B (MTP + suffix, 2-way)\n"
             "bayes(GBM) recovers ~50-52% of raw→oracle gap & beats native; marginal calib hurts")
ax.legend(ncol=4, fontsize=8.5, loc="upper left")
ax.set_ylim(0, 8.0); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = f"{OUT}/gsm8k_humaneval_realized_ladder.png"
fig.savefig(p, dpi=140); print("wrote", p)

# second panel: bayes recovery% + suffix-win% per subtask
fig2, ax2 = plt.subplots(figsize=(7, 4.2))
subs = ["gsm8k\n(math)", "humaneval\n(code)"]
rec = [52, 50]; sfx = [9.4, 9.5]
xx = np.arange(len(subs))
ax2.bar(xx - 0.2, rec, 0.4, label="bayes recovery % of raw→oracle", color="#42a5f5")
ax2.bar(xx + 0.2, sfx, 0.4, label="suffix-win % (decisive picks)", color="#ff9800")
for i, (rr, ss) in enumerate(zip(rec, sfx)):
    ax2.text(i - 0.2, rr + 0.8, f"{rr}%", ha="center", fontsize=9)
    ax2.text(i + 0.2, ss + 0.8, f"{ss}%", ha="center", fontsize=9)
ax2.set_xticks(xx); ax2.set_xticklabels(subs); ax2.set_ylabel("%")
ax2.set_title("Domain-invariant: recovery ~50% & suffix-win ~9.5% for both math & code")
ax2.legend(fontsize=8.5); ax2.set_ylim(0, 60); ax2.grid(axis="y", alpha=0.3)
fig2.tight_layout()
p2 = f"{OUT}/gsm8k_humaneval_recovery_suffixwin.png"
fig2.savefig(p2, dpi=140); print("wrote", p2)
