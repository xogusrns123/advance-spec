#!/usr/bin/env python3
"""Picture6, NO-CALIBRATION version — MAT gain vs. boundary rate.

Same as regen_picture6.py but the composition uses RAW probabilities (no beta
hazard / isotonic tail): the arms come from a `validate_interpretations.py
--no-calib --skip-oracle` re-run stored in results/interp_validation_nocalib/.

    y = ΔMAT = K(compose_raw) − max(K(dflash), K(suffix))   [tokens / step]
    x = boundary rate (alternations / 100 tokens)

Everything (boundary rate + K ladder) is read from the SAME no-calib run, so it
is internally consistent and matches the current curves used elsewhere.

  python3 scripts/plot/regen_picture6_nocalib.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent
DIR = BASE / "results" / "interp_validation_nocalib"
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"
ORANGE, GRAY = "#D85A30", "#9A988F"

# (display label, workload file name, is_agentic)
ROWS = [
    ("SWE-Bench", "swebench", True),
    ("AgenticSQL", "spider", True),
    ("τ²-bench", "tau2", True),
    ("BFCLv4", "bfcl", True),
    ("Spec-Bench", "specbench", False),
]


def main():
    xs, ys, labs, cols = [], [], [], []
    for lab, w, ag in ROWS:
        r = json.load(open(DIR / f"report_{w}.json"))
        arms, st = r["arms"], r["structure"]
        df, sf, cp = arms["dflash"]["K"], arms["suffix"]["K"], arms["compose"]["K"]
        best = max(df, sf)
        xs.append(st["bound_per100"])
        ys.append(cp / best)                         # MAT ratio vs best single (×)
        labs.append(lab)
        cols.append(ORANGE if ag else GRAY)

    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    r = sxy / (sxx * syy) ** 0.5

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axhline(1.0, color="#B8B2A6", lw=1.4, zorder=0)
    xl = [min(xs) - 1.2, max(xs) + 1.2]
    ax.plot(xl, [slope * x + intercept for x in xl], ls="--", color="#6B7280",
            lw=1.8, zorder=1)
    ax.scatter(xs, ys, s=190, c=cols, zorder=3, edgecolor="white", lw=1.2)
    for x, y, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(10, 8),
                    fontsize=12.5, color="#222222")
    ax.text(0.97, 0.06, f"least-squares fit,  r = {r:.2f}  (n={n})",
            transform=ax.transAxes, ha="right", fontsize=11, color="#6B7280")
    ax.text(min(xs) - 1.0, 1.02, "compose = best single (no gain)",
            fontsize=10.5, color="#8A8378", va="bottom")

    ax.set_xlabel("boundary rate (alternations / 100 tokens)", fontsize=12.5)
    ax.set_ylabel("MAT vs. best single proposer  (×)\n"
                  "(compose / best)  [no calibration]", fontsize=12.5)
    ax.set_xlim(*xl)
    ax.tick_params(labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture6.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}   slope={slope:.3f} r={r:.2f}")
    for lab, x, y in zip(labs, xs, ys):
        print(f"  {lab:<12} boundary={x:5.2f}  ΔMAT(raw)={y:+.2f}")


if __name__ == "__main__":
    main()
