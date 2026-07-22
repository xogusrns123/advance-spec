#!/usr/bin/env python3
"""Regenerate Picture10 with REAL data (honest, no forced y=x).

G/R = canonical n-gram-repeat partition (_gr.rg_cover(s, 4); see regen_picture9).
Two panels, both from real numbers:
  Left   x = boundary rate (R<->G transitions / 100 tokens, from the curves)
         y = measured speedup vs best single = K.compose / max(K.dflash, K.suffix)
         least-squares fit + Pearson r.
  Right  x = predicted speedup from |G|,|R| ((E[|G|]+E[|R|]) / max)
         y = same measured speedup
         y = x reference line, real r + MAE.

Shown as-is (no forced fit): the model overpredicts absolute speedup but tracks
the ranking; Spec-Bench sits low (one proposer dominates -> little to compose).

Run inside the container (figures dir is root-owned):
  docker exec sglang-bench bash -lc \
    "cd '/workspace/simulation/Dr.Lee Solution' && python3 scripts/plot/regen_picture10.py"
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regen_picture9 import load_runs, load_K, predict_speedup, WL_SRC

BASE = Path(__file__).resolve().parent.parent.parent
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"
ORANGE, GRAY = "#D85A30", "#9A988F"


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")
    slope = sxy / sxx if sxx else 0.0
    return r, slope, my - slope * mx


def main():
    rows = []
    for disp, (wl, stem, ag) in WL_SRC.items():
        G, R, nb, ntok = load_runs(wl, stem)
        pred = predict_speedup(G)[0]          # rounds model, |G| distribution only
        k = load_K(wl)
        meas = k["compose"] / max(k["dflash"], k["suffix"])
        bound = 100.0 * nb / ntok                 # winner-flips / 100 tokens
        rows.append(dict(disp=disp, ag=ag, bound=bound, pred=pred, meas=meas))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    cols = [ORANGE if r["ag"] else GRAY for r in rows]

    # ---- Left: boundary rate vs measured speedup --------------------------
    xs = [r["bound"] for r in rows]
    ys = [r["meas"] for r in rows]
    rL, slope, icpt = pearson(xs, ys)
    axL.axhline(1.0, color="#B8B2A6", lw=1.4, zorder=0)
    xl = [min(xs) - 1, max(xs) + 1]
    axL.plot(xl, [slope * x + icpt for x in xl], ls="--", color="#6B7280",
             lw=1.7, zorder=1)
    axL.scatter(xs, ys, s=200, c=cols, edgecolor="white", lw=1.3, zorder=3)
    OFF_L = {"AgenticSQL": (-9, 8, "right"), "τ²-bench": (0, 11, "center"),
             "SWE-Bench": (10, -2, "left"), "BFCLv4": (-9, 6, "right"),
             "Spec-Bench": (0, 11, "center")}
    for r in rows:
        dx, dy, ha = OFF_L.get(r["disp"], (9, 7, "left"))
        axL.annotate(r["disp"], (r["bound"], r["meas"]), ha=ha,
                     textcoords="offset points", xytext=(dx, dy), fontsize=12)
    axL.text(0.97, 0.06, f"least-squares fit,  r = {rL:.2f}  (n={len(rows)})",
             transform=axL.transAxes, ha="right", fontsize=11, color="#6B7280")
    axL.set_xlabel("boundary rate (R↔G transitions / 100 tokens)", fontsize=12.5)
    axL.set_ylabel("measured speedup vs. best single (×)", fontsize=12.5)
    axL.set_title("Boundary rate vs. measured speedup",
                  fontsize=13.5, color="#1F2A44")

    # ---- Right: predicted vs measured, y = x ------------------------------
    xs = [r["pred"] for r in rows]
    ys = [r["meas"] for r in rows]
    rR, _, _ = pearson(xs, ys)
    mae = sum(abs(p - m) for p, m in zip(xs, ys)) / len(rows)
    lo = min(min(xs), min(ys)) - 0.10
    hi = max(max(xs), max(ys)) + 0.16
    axR.plot([lo, hi], [lo, hi], ls="--", color="#9AA0A6", lw=1.6, zorder=1)
    axR.text(lo + 0.05, lo + 0.02, "y = x", color="#9AA0A6", fontsize=12,
             rotation=45, va="bottom")
    axR.scatter(xs, ys, s=200, c=cols, edgecolor="white", lw=1.3, zorder=3)
    OFF_R = {"AgenticSQL": (-10, 9, "right"), "τ²-bench": (10, 2, "left"),
             "BFCLv4": (10, 2, "left"), "SWE-Bench": (-10, -12, "right"),
             "Spec-Bench": (10, -4, "left")}
    for r in rows:
        dx, dy, ha = OFF_R.get(r["disp"], (9, 7, "left"))
        axR.annotate(r["disp"], (r["pred"], r["meas"]), ha=ha,
                     textcoords="offset points", xytext=(dx, dy), fontsize=12)
    axR.text(0.03, 0.95,
             f"r = {rR:.2f},  MAE = {mae:.2f}  (n={len(rows)})",
             transform=axR.transAxes, ha="left", va="top", fontsize=11,
             color="#6B7280")
    axR.set_xlim(lo, hi)
    axR.set_ylim(lo, hi)
    axR.set_xlabel("predicted speedup from |G|,|R| distributions (×)",
                   fontsize=12.5)
    axR.set_ylabel("measured speedup (×)", fontsize=12.5)
    axR.set_title("Predicted vs. measured speedup",
                  fontsize=13.5, color="#1F2A44")

    for ax in (axL, axR):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture10.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {fp}   left r={rL:.2f}   right r={rR:.2f} MAE={mae:.2f}")
    for r in rows:
        print(f"  {r['disp']:<12} bound100={r['bound']:5.2f} "
              f"pred={r['pred']:.3f} measured={r['meas']:.3f}")


if __name__ == "__main__":
    main()
