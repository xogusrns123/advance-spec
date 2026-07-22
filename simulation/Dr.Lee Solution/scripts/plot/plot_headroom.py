#!/usr/bin/env python3
"""Figures for the grafting-headroom quantity (compute_headroom.py output).

  A: predicted C_graft (parameter-free curve functional) vs the MEASURED
     structural gap (live handoff_oracle - switch_oracle) — identity check.
  B: capacity decomposition per workload: stacked C_sel + C_graft (what the
     dual profile offers) with the realized compose-best_single overlaid
     (what the deployed controller captures).

  PYTHONPATH=/workspace python3 scripts/plot_headroom.py --dir results/interp_validation
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
C_SEL, C_GRAFT = "#B279A2", "#54A24B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    args = ap.parse_args()
    rows = json.load(open(os.path.join(args.dir, "headroom.json")))
    tu = [r for r in rows if r["task"] != "__all__"
          and r["meas_struct"] == r["meas_struct"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))

    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)
    lim = 1.05 * max(max(r["c_graft"] for r in tu),
                     max(r["meas_struct"] for r in tu))
    ax.plot([0, lim], [0, lim], color="#999", lw=1.0, ls="--", zorder=1)
    ax.text(lim * 0.97, lim * 0.90, "y = x", fontsize=9, color="#777",
            ha="right")
    for wl in WL_MARK:
        pts = [(r["c_graft"], r["meas_struct"]) for r in tu if r["wl"] == wl]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       marker=WL_MARK[wl], s=62, color=WL_COLOR[wl], label=wl,
                       alpha=0.9, edgecolor="white", linewidth=0.6, zorder=2)
    ax.set_xlabel("C_graft predicted from the dual profile (a(p), s(p))\n"
                  "E[ max_k≤a(p) (k + s(p+k)) − max(a(p), s(p)) ]  (tok/round)",
                  fontsize=9.5)
    ax.set_ylabel("measured structural gap:\nhandoff oracle − switch oracle (live replay)",
                  fontsize=9.5)
    ax.set_title("A — the formula IS the mechanism\n(ρ = +0.97, magnitude ratio 1.08)",
                 fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax = axes[1]
    wls = ["spider", "swebench", "bfcl", "specbench"]
    alls = {r["wl"]: r for r in rows if r["task"] == "__all__"}
    xs = range(len(wls))
    sel = [alls[w]["c_sel"] for w in wls]
    grf = [alls[w]["c_graft"] for w in wls]
    real = [alls[w]["g_drlee"] for w in wls]
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.bar(xs, sel, width=0.55, color=C_SEL,
           label="C_sel — selection capacity (complementarity)")
    ax.bar(xs, grf, width=0.55, bottom=sel, color=C_GRAFT,
           label="C_graft — grafting capacity (onramps within reach)")
    ax.scatter(xs, real, marker="D", s=70, color="#333", zorder=3,
               label="realized compose − best single")
    for i, (a_, b_, c_) in enumerate(zip(sel, grf, real)):
        ax.text(i, a_ + b_ + 0.05, f"{a_ + b_:.2f}", ha="center", fontsize=9)
        ax.text(i + 0.3, c_, f"{c_:+.2f}", fontsize=8.5, va="center", color="#333")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(wls, fontsize=10.5)
    ax.set_ylabel("tokens / round", fontsize=10)
    ax.set_title("B — capacity (stacked) vs what the controller captures (◆)",
                 fontsize=10.5)
    ax.set_ylim(min(-0.6, min(real) - 0.2), max(a_ + b_ for a_, b_ in zip(sel, grf)) + 1.1)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Grafting headroom: the parameter-free functional of the dual "
                 "profile that quantifies where composition works", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(args.dir, "figures", "headroom_validation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
