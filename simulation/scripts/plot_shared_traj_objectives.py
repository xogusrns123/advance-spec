#!/usr/bin/env python3
"""RIGOROUS objective comparison on SHARED trajectories (no trajectory confound).

Each calib objective's maps are replayed (--replay-existing, deterministic) on the
SAME frozen trajectory, so raw/oracle are identical for both objectives and the
per-method calib MAT are DIRECTLY comparable in absolute terms (unlike the
cross-run delta comparison, which mixes objective effect with trajectory drift).

Two trajectories, two panels:
  T_tp  : target_p-calib (native qwen3_14b_tp)  vs  token_gt-calib (xtraj_def_on_tp)
  T_def : token_gt-calib (native qwen3_14b_def)  vs  target_p-calib (xtraj_tp_on_def)

Faithfulness check: the replayed run's raw'/oracle' must reproduce the native
run's raw/oracle (printed + annotated). If they diverge, the shared-trajectory
assumption is broken and the panel is flagged.

Usage:
  python3 simulation/scripts/plot_shared_traj_objectives.py \
    --tp-native   simulation/results/o4_perdepth/qwen3_14b_tp/run.json \
    --tg-on-tp    simulation/results/o4_perdepth/xtraj_def_on_tp/run.json \
    --tg-native   simulation/results/o4_perdepth/qwen3_14b_def/run.json \
    --tp-on-def   simulation/results/o4_perdepth/xtraj_tp_on_def/run.json \
    --out-dir     simulation/results/comparison_figures
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
TP_COLOR, TG_COLOR = "#2ca02c", "#1f77b4"


def mat(path):
    a = json.load(open(path)).get("arms", {})
    def g(k):
        v = a.get(k)
        return v.get("accept_length_mean") if isinstance(v, dict) else None
    return {
        "raw": g("select1"), "oracle": g("select1_oracle"),
        "baseline": g("baseline"),
        **{m: g(f"select1_calib_{m}") for m in METHODS},
    }


def panel(ax, traj_name, tp, tg, raw_ref, ora_ref):
    """tp/tg: dicts with per-method MAT measured on the SAME trajectory."""
    x = np.arange(len(METHODS)); w = 0.38
    tp_v = [tp[m] or np.nan for m in METHODS]
    tg_v = [tg[m] or np.nan for m in METHODS]
    b1 = ax.bar(x - w/2, tp_v, w, color=TP_COLOR, label="target_p calib")
    b2 = ax.bar(x + w/2, tg_v, w, color=TG_COLOR, label="token_gt calib")
    for b in (b1, b2):
        for r in b:
            h = r.get_height()
            if not np.isnan(h):
                ax.text(r.get_x()+r.get_width()/2, h+0.004, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7)
    if raw_ref:
        ax.axhline(raw_ref, color="#7f7f7f", lw=1.4, ls="--",
                   label=f"raw {raw_ref:.3f}")
    if ora_ref:
        ax.axhline(ora_ref, color="#d62728", lw=1.2, ls=":",
                   label=f"oracle {ora_ref:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(METHODS)
    ax.set_title(traj_name, fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7, loc="upper left")
    lo = min([v for v in (raw_ref, *tp_v, *tg_v) if v and not np.isnan(v)]) - 0.05
    hi = (ora_ref or max(tp_v+tg_v)) + 0.05
    ax.set_ylim(lo, hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tp-native", required=True, help="qwen3_14b_tp/run.json (target_p on T_tp)")
    ap.add_argument("--tg-on-tp", required=True, help="xtraj_def_on_tp/run.json (token_gt on T_tp)")
    ap.add_argument("--tg-native", required=True, help="qwen3_14b_def/run.json (token_gt on T_def)")
    ap.add_argument("--tp-on-def", required=True, help="xtraj_tp_on_def/run.json (target_p on T_def)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    tp_nat = mat(args.tp_native)   # target_p maps, T_tp
    tg_ott = mat(args.tg_on_tp)    # token_gt maps, T_tp
    tg_nat = mat(args.tg_native)   # token_gt maps, T_def
    tp_otd = mat(args.tp_on_def)   # target_p maps, T_def
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Faithfulness: replayed raw'/oracle' vs native raw/oracle on each trajectory.
    def chk(name, native, replay):
        for k in ("raw", "oracle"):
            n, r = native.get(k), replay.get(k)
            if n and r:
                d = abs(n - r)
                flag = "OK" if d <= 0.03 else "** DIVERGED **"
                print(f"  [{name}] {k}: native={n:.3f} replay={r:.3f} |Δ|={d:.3f} {flag}")

    print("Faithfulness (shared-trajectory replay reproduces raw/oracle):")
    print(" T_tp  (native target_p vs replayed token_gt):")
    chk("T_tp", tp_nat, tg_ott)
    print(" T_def (native token_gt vs replayed target_p):")
    chk("T_def", tg_nat, tp_otd)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))
    # On T_tp: raw/oracle from native target_p run (the canonical T_tp anchors).
    panel(axL, "Shared trajectory T_tp  (target_p's recorded run)",
          tp_nat, tg_ott, tp_nat["raw"], tp_nat["oracle"])
    # On T_def: raw/oracle from native token_gt run.
    panel(axR, "Shared trajectory T_def  (token_gt's recorded run)",
          tp_otd, tg_nat, tg_nat["raw"], tg_nat["oracle"])
    fig.suptitle("Calibration objective on a SHARED trajectory — 14B bfcl web_search "
                 "(20 test, tail=64)\nabsolute calib MAT directly comparable "
                 "(same decision points; --replay-existing)", fontsize=11)
    axL.set_ylabel("MAT (accept_length_mean, served 0.5.12)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out/"shared_traj_compare.png", dpi=150); plt.close(fig)

    # Within-shared-trajectory delta table (calib - raw, both objectives same raw).
    print("\nWithin-shared-trajectory (calib - raw) MAT, both objectives on same raw:")
    print(f"{'method':10s} | T_tp raw={tp_nat['raw']:.3f}            "
          f"| T_def raw={tg_nat['raw']:.3f}")
    print(f"{'':10s} |  target_p   token_gt   |  target_p   token_gt")
    for m in METHODS:
        ttp_tp = (tp_nat[m]-tp_nat['raw']) if tp_nat[m] else float('nan')
        ttg_tp = (tg_ott[m]-tp_nat['raw']) if tg_ott[m] else float('nan')
        ttp_df = (tp_otd[m]-tg_nat['raw']) if tp_otd[m] else float('nan')
        ttg_df = (tg_nat[m]-tg_nat['raw']) if tg_nat[m] else float('nan')
        print(f"{m:10s} |  {ttp_tp:+.4f}   {ttg_tp:+.4f}   "
              f"|  {ttp_df:+.4f}   {ttg_df:+.4f}")
    print(f"\nwrote {out}/shared_traj_compare.png")


if __name__ == "__main__":
    main()
