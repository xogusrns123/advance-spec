#!/usr/bin/env python3
"""Compare the two calibration OBJECTIVES on 0.5.12-native served MAT:
  target_p  (continuous q_target regression)   -> qwen3_14b_tp/run.json
  token_gt  (binary token==gt accept)          -> qwen3_14b_def/run.json
Both runs are within-run-fair (each carries its own raw/baseline/oracle on the
same 20 test tasks; raw drifts run-to-run via live web_search so only WITHIN-run
calib-raw deltas are comparable across objectives).

Two figures (out-dir):
  mat_compare.png   grouped MAT bars per calib method (raw | target_p | token_gt)
                    with baseline + oracle reference lines per run.
  delta_compare.png within-run (calib - raw) MAT delta per method, target_p vs
                    token_gt side by side (the apples-to-apples objective test).

Usage:
  python3 simulation/scripts/plot_target_p_vs_token_gt.py \
    --tp-run simulation/results/o4_perdepth/qwen3_14b_tp/run.json \
    --tg-run simulation/results/o4_perdepth/qwen3_14b_def/run.json \
    --out-dir simulation/results/comparison_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["histogram", "isotonic", "logistic", "beta"]
TP_COLOR, TG_COLOR, RAW_COLOR = "#2ca02c", "#1f77b4", "#7f7f7f"


def mat(run):
    a = run.get("arms", {})
    def g(k):
        v = a.get(k)
        return v.get("accept_length_mean") if isinstance(v, dict) else None
    return {
        "raw": g("select1"), "baseline": g("baseline"),
        "oracle": g("select1_oracle"),
        **{m: g(f"select1_calib_{m}") for m in METHODS},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tp-run", required=True)
    ap.add_argument("--tg-run", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    tp = mat(json.load(open(args.tp_run)))
    tg = mat(json.load(open(args.tg_run)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(METHODS))
    w = 0.38

    # --- Figure 1: absolute MAT bars ---
    fig, ax = plt.subplots(figsize=(9, 5.2))
    tp_vals = [tp[m] or np.nan for m in METHODS]
    tg_vals = [tg[m] or np.nan for m in METHODS]
    b1 = ax.bar(x - w/2, tp_vals, w, color=TP_COLOR, label="target_p calib")
    b2 = ax.bar(x + w/2, tg_vals, w, color=TG_COLOR, label="token_gt calib")
    for b in (b1, b2):
        for r in b:
            h = r.get_height()
            if not np.isnan(h):
                ax.text(r.get_x()+r.get_width()/2, h+0.005, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7)
    # raw / oracle / baseline reference lines (per run, dashed)
    for d, c, ls in ((tp, TP_COLOR, "--"), (tg, TG_COLOR, ":")):
        if d["raw"]:
            ax.axhline(d["raw"], color=c, lw=1.4, ls=ls, alpha=0.8)
        if d["oracle"]:
            ax.axhline(d["oracle"], color=c, lw=1.0, ls=ls, alpha=0.5)
    if tp["baseline"]:
        ax.axhline(tp["baseline"], color="#bbbbbb", lw=1.0, ls="-", alpha=0.7,
                   label=f"baseline {tp['baseline']:.3f}")
    ax.text(0.01, tp["raw"]+0.005, f"raw(tp) {tp['raw']:.3f}", color=TP_COLOR,
            fontsize=7, transform=ax.get_yaxis_transform())
    ax.text(0.01, tp["oracle"]+0.005, f"oracle(tp) {tp['oracle']:.3f} (ceiling)",
            color=TP_COLOR, fontsize=7, transform=ax.get_yaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(METHODS)
    ax.set_ylabel("MAT (accept_length_mean, served 0.5.12)")
    ax.set_title("Chain-hybrid calib MAT by objective — 14B bfcl web_search "
                 "(20 test, tail=64)\ntarget_p (continuous q_target) vs token_gt "
                 "(binary accept)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(tp["baseline"], tg["baseline"])-0.1,
                max(tp["oracle"], tg["oracle"])+0.1)
    fig.tight_layout(); fig.savefig(out/"mat_compare.png", dpi=150); plt.close(fig)

    # --- Figure 2: within-run (calib - raw) delta ---
    fig, ax = plt.subplots(figsize=(9, 5.2))
    tpd = [(tp[m]-tp["raw"]) if (tp[m] and tp["raw"]) else np.nan for m in METHODS]
    tgd = [(tg[m]-tg["raw"]) if (tg[m] and tg["raw"]) else np.nan for m in METHODS]
    b1 = ax.bar(x - w/2, tpd, w, color=TP_COLOR, label="target_p (calib - raw)")
    b2 = ax.bar(x + w/2, tgd, w, color=TG_COLOR, label="token_gt (calib - raw)")
    for b in (b1, b2):
        for r in b:
            h = r.get_height()
            if not np.isnan(h):
                ax.text(r.get_x()+r.get_width()/2,
                        h+(0.002 if h >= 0 else -0.004), f"{h:+.3f}",
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(METHODS)
    ax.set_ylabel("MAT improvement over raw (within-run)")
    ax.set_title("Calibration objective effect: within-run (calib - raw) MAT delta\n"
                 f"raw_tp={tp['raw']:.3f}  raw_tg={tg['raw']:.3f}  "
                 f"oracle~{tp['oracle']:.3f}  (>0 = calib helps)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out/"delta_compare.png", dpi=150); plt.close(fig)

    print("target_p:", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in tp.items()})
    print("token_gt:", {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in tg.items()})
    print("within-run (calib-raw):")
    for m in METHODS:
        print(f"  {m:9s} target_p {tpd[METHODS.index(m)]:+.4f}  "
              f"token_gt {tgd[METHODS.index(m)]:+.4f}")
    print(f"wrote {out}/mat_compare.png + delta_compare.png")


if __name__ == "__main__":
    main()
