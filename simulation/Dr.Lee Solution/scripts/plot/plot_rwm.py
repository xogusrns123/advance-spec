#!/usr/bin/env python3
"""The proposed COMMON metric — RWM (Reachable Warm Mass) — against both
parties' gains, with the empirical decision threshold.

  RWM/100tok = sum of warm-run lengths whose entry gap <= W(15) / tokens x 100
    Dr.Lee reading: template mass whose interrupting SLOT is short enough to bridge
    Kim reading:    warm mass unlocked when the BOUNDARY is packed into one step

  PYTHONPATH=/workspace python3 scripts/plot_rwm.py --dir results/interp_validation
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WL_MARK = {"spider": "o", "swebench": "s", "bfcl": "^", "specbench": "D"}
WL_COLOR = {"spider": "#4C78A8", "swebench": "#F58518", "bfcl": "#54A24B",
            "specbench": "#E45756"}


def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            for k in range(i, j):
                r[order[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    args = ap.parse_args()
    rows = json.load(open(os.path.join(args.dir, "metric_candidates.json")))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    panels = [("g_drlee", "compose − best single  (Dr.Lee frame)"),
              ("g_kim", "compose − binary switch  (Kim frame)")]
    # threshold band from the observed clean margin on the Kim frame
    neg = [r["EWS"] for r in rows if r["g_kim"] <= 0]
    pos = [r["EWS"] for r in rows if r["g_kim"] > 0]
    lo, hi = max(neg), min(pos)
    for ax, (g, lab) in zip(axes, panels):
        ax.set_axisbelow(True)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="#999", lw=0.8)
        ax.axvspan(lo, hi, color="#666", alpha=0.10, lw=0)
        for wl in WL_MARK:
            pts = [(r["EWS"], r[g]) for r in rows if r["wl"] == wl]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                           marker=WL_MARK[wl], s=58, color=WL_COLOR[wl], label=wl,
                           alpha=0.9, edgecolor="white", linewidth=0.6)
        rho = spearman([r["EWS"] for r in rows], [r[g] for r in rows])
        ax.set_title(f"{lab}   (ρ = {rho:+.2f})", fontsize=10.5)
        ax.set_xlabel("RWM — reachable warm mass / 100 tok\n"
                      "(warm runs whose entry gap ≤ 15 tok)", fontsize=9.5)
    axes[0].set_ylabel("gain (accepted tokens / round)", fontsize=10)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[1].annotate(f"decision band: RWM ≈ {lo:.0f}–{hi:.0f}\n"
                     "right of band → compose wins vs switch (21/21)",
                     xy=(hi, 0), xytext=(hi + 6, -0.22), fontsize=8.5,
                     arrowprops=dict(arrowstyle="->", color="#555"), color="#333")
    fig.suptitle("Proposed common metric: RWM = slot/boundary events weighted by "
                 "the warm mass they unlock", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(args.dir, "figures", "rwm_vs_gains.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (band {lo:.1f}-{hi:.1f})")


if __name__ == "__main__":
    main()
