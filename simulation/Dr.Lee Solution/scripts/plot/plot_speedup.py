#!/usr/bin/env python3
"""ESTIMATED decode speedup (x vs vanilla AR) per workload, same 5 arms/colors as
the MAT figures. Speedup is derived from the simulated MAT and a per-round cost
factor c MEASURED on our own 27B DFlash+suffix serving:

    speedup_arm = (1 + MAT_arm) / c_arm

c = (round latency) / (vanilla per-token latency), anchored on the 27B
chain-handoff timing (chain_handoff_oracle/qwen35_27b_dflash): the DFlash-drafted
round (block head + verify) costs ~1.07 vanilla decode steps, a pure suffix-tree
round (no model draft) ~1.005. Fallback mixes them by its suffix round-share.

  dflash / compose / oracle : c = 1.07   (DFlash block draft + 1 verify forward)
  suffix                    : c = 1.005  (tree lookup, no model draft)
  fallback                  : c = share*1.005 + (1-share)*1.07

NOT a served speedup measurement — an estimate bridging the offline MAT with the
measured per-round cost. Numbers only (no vanilla-comparison is drawn as its own
reference series; vanilla = 1.0x baseline).

  python3 scripts/plot_speedup.py
"""
from __future__ import annotations
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

import plot_mat_4way as p4

OUT = p4.OUT
SEG = Path("/workspace/simulation/results/pipeline_4way/segments")
C_DRAFT, C_SUFFIX = 1.07, 1.005            # measured per-round cost factors (27B)


def fallback_share(ds):
    fp = SEG / f"fallback_sweep_fresh_{ds}.json"
    if not fp.exists():
        return 0.0
    taus = next(iter(json.load(open(fp)).values()))
    _, v = max(taus.items(), key=lambda kv: kv[1]["K"])
    return float(v.get("suffix_share", 0.0))


def cost(ds, arm):
    if arm == "suffix":
        return C_SUFFIX
    if arm == "fallback":
        s = fallback_share(ds)
        return s * C_SUFFIX + (1 - s) * C_DRAFT
    return C_DRAFT                          # dflash / calib / oracle


def main():
    p4.SWEEP_DIR = str(SEG)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-suffix", default="",
                    help="read mat_{ds}_4way_{grp}{sfx}.replay.txt (e.g. _raw)")
    args = ap.parse_args()
    p4.LOG_SUFFIX = args.log_suffix
    K, _ = p4.load()                        # K[ds][arm] = (MAT, rounds)
    OUT.mkdir(parents=True, exist_ok=True)
    wls = [ds for ds in p4.DS if K.get(ds)]

    # speedup[ds][arm] = (1 + MAT) / c
    S = {ds: {a: (1.0 + K[ds][a][0]) / cost(ds, a)
              for a in p4.PROPS if a in K[ds] and K[ds][a][0] > 0}
         for ds in wls}

    fig, ax = plt.subplots(figsize=(2.4 + 2.5 * len(wls), 5.4))
    n, g = len(p4.PROPS), 0.80
    bw = g / n
    ymax = 0.0
    for pi, p in enumerate(p4.PROPS):
        xs = [ci - g / 2 + bw * (pi + 0.5) for ci in range(len(wls))]
        ys = [S[ds].get(p, 0.0) for ds in wls]
        ymax = max(ymax, max(ys) if ys else 0)
        ax.bar(xs, ys, width=bw * 0.9, color=p4.COLORS[p], label=p4.LABELS[p])
        for x, y in zip(xs, ys):
            if y > 0:
                ax.text(x, y + 0.03, f"{y:.1f}×", ha="center", va="bottom",
                        fontsize=8.5, color="#555555")
    ax.axhline(1.0, color="#999999", lw=1, ls="--")
    ax.text(len(wls) - 0.5, 1.02, "vanilla AR = 1.0×", ha="right", va="bottom",
            fontsize=9, color="#777777")
    ax.set_ylim(0, ymax * 1.18)
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels([p4.DS_NAME[ds] for ds in wls], fontsize=8.5)
    ax.set_ylabel("estimated decode speedup  (× vs vanilla AR)", fontsize=11)
    ax.set_title("Dr.Lee extension — estimated decode speedup", fontsize=13)
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.99, 0.99,
            "estimate: speedup = (1+MAT) / c\n"
            "c measured on 27B (DFlash round ≈1.07, suffix-only ≈1.005\n"
            "vanilla decode steps); fallback mixes by its suffix share",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color="#666666")
    fig.tight_layout()
    fp = OUT / "SPEEDUP_per_workload.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")
    for ds in wls:
        print(f"  [{ds}] " + "  ".join(
            f"{p}={S[ds].get(p, 0):.2f}x(MAT {K[ds][p][0]:.2f})"
            for p in p4.PROPS if p in S[ds]))


if __name__ == "__main__":
    main()
