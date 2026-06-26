#!/usr/bin/env python3
"""Two-panel summary of the per-depth select-1 ceiling investigation (14B, pinned).

Left  : MAT ladder across every approach tried (model-only ... ORACLE).
Right : decisive-selection ACCURACY ladder — why MAT can't move much: even the
        joint Bayes ceiling on (eagle_p, suffix_p, count, total, match_len) is
        ~0.73, so calibration (any flavor) is near its information limit.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1
         else "simulation/results/chain_hybrid_perdepth/qwen3_14b_online")


def mat(p):
    v = []
    try:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("phase") and r["phase"] != "decode":
                continue
            a = r.get("accept_lengths")
            if isinstance(a, list):
                v += [int(x) for x in a]
            elif isinstance(a, (int, float)):
                v.append(int(a))
    except FileNotFoundError:
        return None
    return float(np.mean(v)) if v else None


refs = json.load(open(d / "run_refs.json"))["arms"]
mat_rows = [
    ("model-only", refs["baseline"]["accept_length_mean"], "#7f7f7f"),
    ("suffix-only", mat(d / "timing_suffix.jsonl"), "#d62728"),
    ("raw select1", refs["select1"]["accept_length_mean"], "#1f77b4"),
    ("online\n(best)", mat(d / "timing_online_beta_accept_rate_w1024.jsonl"), "#2ca02c"),
    ("multifeat\n(accept)", mat(d / "timing_multifeat_accept_rate.jsonl"), "#9467bd"),
    ("multifeat\n(target_p)", mat(d / "timing_multifeat_target_p.jsonl"), "#8c564b"),
    ("ORACLE", refs["select1_oracle"]["accept_length_mean"], "#e0b400"),
]
# decisive-selection accuracy ladder (held-out eval decisive, from analyses)
acc_rows = [
    ("base\n(majority)", 0.540, "#cccccc"),
    ("raw\n(sp>ep)", 0.695, "#1f77b4"),
    ("1-D calib", 0.708, "#2ca02c"),
    ("multifeat\n(+c,n,ml)", 0.719, "#9467bd"),
    ("Bayes ceiling\n(all feats)", 0.73, "#e0b400"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for ax, rows, ylab, title, ytop in (
        (ax1, mat_rows, "MAT (mean accepted draft tokens)",
         "MAT ladder — all approaches (14B, pinned eval)", None),
        (ax2, acc_rows, "decisive selection accuracy",
         "Decisive-pick accuracy — the information ceiling", 0.8)):
    labs = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    xs = np.arange(len(labs))
    bars = ax.bar(xs, vals, color=cols, width=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (ytop or max(vals)) * 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
    ax.set_ylim(0, ytop or max(vals) * 1.15); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = d / "figures" / "ceiling_summary.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"wrote {out}")
