#!/usr/bin/env python3
"""Fold the SD-hybrid mis-routing rate into the segment-level
boundary-density-vs-gain view WITHOUT going 3-D. Two 2-D encodings:

(1) BOUNDARY_density_vs_gain_segment_misroute.png
    keep the existing axes (x = winner-change boundary density /1K,
    y = compose - hybrid MAT gain) but map the hidden third variable
    (Type-A "wrong fallback" rate) onto marker COLOR (colorbar). Reveals
    that the points which break the density->gain trend are exactly the
    high-mis-route ones.

(2) SEGMENT_gain_vs_regret.png
    collapse "how often (density/flips)" x "how costly (severity)" into a
    single physical scalar -- the hybrid's routing REGRET (accepted tokens
    lost per round to mis-routing). x = regret/round, y = gain. gain is
    essentially the regret compose recovers, so this is a near y=x line.
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/workspace/simulation/results/pipeline_4way/"
MR = "readable_outputs/figures/mat/hybrid_misroute/hybrid_misroute_summary.json"
OUT = Path("readable_outputs/figures/mat/boundary_gap")
WLC = {"specbench": "#B279A2", "bfcl": "#54A24B", "swebench": "#F58518",
       "spider": "#4C78A8", "tau2": "#8c564b"}
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
SEG_MARK = {"think": "o", "tool_call": "s", "final": "^"}


def load_rows():
    den = json.load(open(BASE + "density_vs_hybridgain.json"))["segment"]
    mr = json.load(open(MR))
    rows = []
    for r in den:
        wl, seg = r["wl"], r["seg"]
        m = mr[wl]["segs"].get(seg)
        if not m:
            continue
        reg = m["regretA_tok"] + m["regretB_tok"]
        rows.append(dict(wl=wl, seg=seg, gain=r["gain"], dens=r["dens"],
                         A=m["typeA_wrong_fallback_pct"],
                         mis=m["total_misroute_pct"],
                         rpr=reg / m["rounds"]))
    return rows


def wl_legend(ax, present, extra=None, loc="upper left"):
    hs = [plt.Line2D([0], [0], marker="o", ls="", mfc=WLC[w], mec="k", ms=9,
                     label=WL_NAME[w]) for w in WLC if w in present]
    if extra:
        hs += extra
    ax.legend(handles=hs, fontsize=8.5, loc=loc, frameon=True)


def seg_handles():
    return [plt.Line2D([0], [0], marker=m, ls="", mfc="#bbb", mec="k", ms=9, label=s)
            for s, m in SEG_MARK.items()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    # ---- (1) existing axes, 3rd var = Type-A rate as color ----
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    A = np.array([r["A"] for r in rows])
    sc = None
    for r in rows:
        sc = ax.scatter(r["dens"], r["gain"], c=[r["A"]], cmap="YlOrRd",
                        vmin=0, vmax=A.max(), marker=SEG_MARK[r["seg"]],
                        s=150, edgecolor="k", linewidth=0.7, zorder=3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Type-A wrong-fallback rate  (% of hybrid rounds)", fontsize=10)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("winner-change boundary density  / 1K tok", fontsize=10.5)
    ax.set_ylabel("MAT gain:  compose − SD-paper hybrid", fontsize=10.5)
    ax.set_title("per segment — density vs gain, colored by hybrid mis-routing",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(handles=seg_handles(), fontsize=9, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "BOUNDARY_density_vs_gain_segment_misroute.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("saved -> BOUNDARY_density_vs_gain_segment_misroute.png")

    # ---- (2) collapse to a single scalar: hybrid regret/round ----
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    xs = [r["rpr"] for r in rows]
    ys = [r["gain"] for r in rows]
    for r in rows:
        ax.scatter(r["rpr"], r["gain"], c=WLC[r["wl"]], marker=SEG_MARK[r["seg"]],
                   s=130, edgecolor="k", linewidth=0.6, zorder=3)
    a, b = np.polyfit(np.array(xs), np.array(ys), 1)
    gx = np.linspace(min(xs), max(xs), 100)
    fit = ax.plot(gx, a * gx + b, color="#333", lw=1.8, ls="--", zorder=2,
                  label="linear fit")[0]
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("SD-hybrid routing regret  (accepted tokens lost / round)",
                  fontsize=10.5)
    ax.set_ylabel("MAT gain:  compose − SD-paper hybrid", fontsize=10.5)
    ax.set_title("per segment — gain vs hybrid routing regret (3-D collapsed to 2-D)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    wl_legend(ax, {r["wl"] for r in rows}, extra=seg_handles() + [fit],
              loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "SEGMENT_gain_vs_regret.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved -> SEGMENT_gain_vs_regret.png")


if __name__ == "__main__":
    main()
