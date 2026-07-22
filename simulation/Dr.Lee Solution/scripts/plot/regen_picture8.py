#!/usr/bin/env python3
"""Regenerate Picture8 as one MAT-ladder bar chart PER WORKLOAD.

Synthetic Picture8 was a single pooled ladder. This produces one figure per
workload (Picture8_<workload>.png), same horizontal 4-bar format:

  DFlash only               = K(dflash)
  Suffix only               = K(suffix)
  Oracle per-round selection= K(switch_oracle)
  Oracle composition        = K(handoff_oracle)   ← "the gap this work targets"

All four are calibration-independent (curve-sim / oracle). Values are read from
the first report that carries handoff_oracle, searching the 0-warm alleval runs
(interp_validation, deck_curves). Workloads without a full ladder yet are
skipped (printed), to be added when their oracle replay finishes.

  python3 scripts/plot/regen_picture8.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"
LGRAY, MGRAY, ORANGE = "#C9C7BE", "#9A988F", "#D85A30"
# base bars are read from the 0-warm no-calib run (ground truth for the 3 calib-
# independent arms); handoff_oracle is borrowed from any full run whose
# switch_oracle MATCHES it (guarantees same 0-warm records / no stale mix).
BASE_DIR = "interp_validation_nocalib"
ORACLE_SEARCH = ["deck_curves", "interp_validation"]

MAIN = "spider"          # canonical single Picture8 (AgenticSQL) — fully real
WORKLOADS = [
    ("SWE-Bench", "swebench"),
    ("AgenticSQL", "spider"),
    ("BFCLv4", "bfcl"),
    ("τ²-bench", "tau2"),
    ("Spec-Bench", "specbench"),
]
BARS = [
    ("DFlash only", "dflash", LGRAY),
    ("Suffix only", "suffix", LGRAY),
    ("Oracle per-round\nselection", "switch_oracle", MGRAY),
    ("Oracle\ncomposition", "handoff_oracle", ORANGE),
]


def load_arms(d, wl):
    p = RESULTS / d / f"report_{wl}.json"
    return json.load(open(p))["arms"] if p.exists() else None


def find_arms(wl):
    """dflash/suffix/switch_oracle from the 0-warm no-calib run; handoff_oracle
    from a full run whose switch_oracle matches (same records)."""
    base = load_arms(BASE_DIR, wl)
    if base is None or "switch_oracle" not in base:
        return None, None
    so = base["switch_oracle"]["K"]
    for d in ORACLE_SEARCH:
        a = load_arms(d, wl)
        if a and "handoff_oracle" in a and "switch_oracle" in a \
                and abs(a["switch_oracle"]["K"] - so) < 0.05:
            merged = {k: base[k] for k in ("dflash", "suffix", "switch_oracle")}
            merged["handoff_oracle"] = a["handoff_oracle"]
            return merged, d
    return None, None


def main():
    done, skipped = [], []
    for disp, wl in WORKLOADS:
        arms, src = find_arms(wl)
        if arms is None:
            skipped.append(disp)
            continue
        vals = [arms[b[1]]["K"] for b in BARS]
        labels = [b[0] for b in BARS]
        colors = [b[2] for b in BARS]

        fig, ax = plt.subplots(figsize=(7.8, 4.2))
        ypos = list(range(len(BARS)))[::-1]      # top = first bar
        ax.barh(ypos, vals, color=colors, height=0.62)
        for y, v in zip(ypos, vals):
            ax.text(v + 0.12, y, f"{v:.1f}", va="center", ha="left",
                    fontsize=13, color="#333333")
        # gap arrow: oracle selection -> oracle composition
        sel, comp = arms["switch_oracle"]["K"], arms["handoff_oracle"]["K"]
        gap = 100.0 * (comp / sel - 1)
        yc = ypos[-1]
        ax.annotate("", xy=(comp, yc + 0.5), xytext=(sel, yc + 0.5),
                    arrowprops=dict(arrowstyle="<->", color="#1F2A44", lw=1.6))
        ax.text((sel + comp) / 2, yc + 0.62,
                f"the gap this work targets (+{gap:.0f}%)",
                ha="center", va="bottom", fontsize=11.5, color="#1F2A44")

        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=12.5)
        ax.set_xlabel("mean accepted tokens per verify (MAT)", fontsize=13)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(disp, fontsize=14, color="#1F2A44", loc="left")
        ax.tick_params(length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"Picture8_{wl}.png", dpi=150)
        if wl == MAIN:                       # canonical single Picture8
            fig.savefig(OUT / "Picture8.png", dpi=150)
        plt.close(fig)
        done.append(f"{disp} (src={src}): df={vals[0]:.1f} sf={vals[1]:.1f} "
                    f"sel={vals[2]:.1f} comp={vals[3]:.1f} gap=+{gap:.0f}%")

    print("RENDERED:")
    for d in done:
        print("  " + d)
    if skipped:
        print("SKIPPED (no full ladder yet):", ", ".join(skipped))


if __name__ == "__main__":
    main()
