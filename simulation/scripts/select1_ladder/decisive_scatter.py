"""Decisive-contested scatter: (model prob, suffix prob), colored by who is RIGHT.

2-way cells only. On decisive-contested positions (both proposers available, EXACTLY
one correct) plot one point per position:
  x = model (EAGLE3/MTP) prob, y = suffix prob
  BLUE  = model is the correct one
  RED   = suffix is the correct one
Dotted y=x = the selection boundary (rule picks suffix iff its prob > model's prob;
points above the line are routed to suffix, below to the model). A point is selected
CORRECTLY iff its color matches its side of the line.

Two separate figures per cell:
  decisive_scatter_raw_<cell>.png  -- raw probs
  decisive_scatter_cal_<cell>.png  -- full-pool OOF-isotonic-calibrated probs (the
                                      deployed monotone map applied to each axis)

Perfect separation (all blue below the line, all red above) => the boundary separates
the classes => selection works. Intermixing across the line => no monotone boundary
(no calibration) can separate them = the irreducible cap. Calibration only re-warps
the axes; it cannot un-mix points that overlap.

Run (host): python3 simulation/scripts/select1_ladder/decisive_scatter.py
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import ROOT, load_chains, loopy_rids, collect  # noqa: E402

OUTDIR = Path("simulation/results/calib_reliability/figures")
BLUE, RED = "#1f77b4", "#d62728"

CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", "EAGLE3", "eagle_token", "eagle_p"),
    "qwen35_27b_2way": ("qwen35_27b_ar", "MTP", "eagle_token", "eagle_p"),
}


def decisive_positions(chains, mtok, mpk):
    """accept-conditioned decisive-contested 2-way rows: model & suffix both
    available, exactly one correct. -> (model_p[], suffix_p[], model_correct[])."""
    mp, sp, mc = [], [], []
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            mt = r.get(mtok); mpv = r.get(mpk)
            st = r.get("suffix_token"); spv = r.get("suffix_p")
            avail = []
            if mt is not None and mpv is not None:
                avail.append("m")
            if st is not None and spv is not None:
                avail.append("s")
            hits = []
            if "m" in avail and gt is not None and mt == gt:
                hits.append("m")
            if "s" in avail and gt is not None and st == gt:
                hits.append("s")
            if len(avail) == 2 and len(hits) == 1:
                mp.append(float(mpv)); sp.append(float(spv))
                mc.append(1 if hits[0] == "m" else 0)
            if gt is not None and avail and len(hits) == 0:
                alive = False
    return np.array(mp), np.array(sp), np.array(mc, bool)


def scatter_fig(mx, sy, mcorr, mname, mode, tag, outpath):
    pick_suffix = sy > mx
    correct = np.where(pick_suffix, ~mcorr, mcorr)
    selacc = float(correct.mean())
    n_blue = int(mcorr.sum()); n_red = int((~mcorr).sum())

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    # light shading of the two routing regions
    ax.fill_between([0, 1], [0, 1], 1, color=RED, alpha=0.04, zorder=0)    # above: ->suffix
    ax.fill_between([0, 1], 0, [0, 1], color=BLUE, alpha=0.04, zorder=0)   # below: ->model
    order = np.random.default_rng(0).permutation(len(mx))  # interleave so neither hides
    cols = np.where(mcorr, BLUE, RED)
    ax.scatter(mx[order], sy[order], c=cols[order], s=7, alpha=0.18,
               linewidths=0, zorder=2)
    ax.plot([0, 1], [0, 1], "k:", lw=1.2, zorder=3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    px = "calibrated" if mode == "cal" else "raw"
    ax.set_xlabel(f"{mname} prob ({px})", fontsize=10)
    ax.set_ylabel(f"Suffix prob ({px})", fontsize=10)
    ax.set_title(f"Decisive-contested scatter — {tag} [{mode.upper()}]\n"
                 f"n={len(mx)}  boundary(y=x) selacc={selacc:.3f}  "
                 f"(blue {mname}-right={n_blue}, red suffix-right={n_red})",
                 fontsize=10)
    handles = [
        Line2D([0], [0], marker="o", ls="", color=BLUE, ms=7, label=f"{mname} is correct"),
        Line2D([0], [0], marker="o", ls="", color=RED, ms=7, label="Suffix is correct"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label="boundary suffix_p = model_p"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150); plt.close(fig)
    return selacc


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, (dirname, mname, mtok, mpk) in CELLS.items():
        d = f"{ROOT}/{dirname}"
        chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
        bad = loopy_rids(d)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        mx, sy, mcorr = decisive_positions(chains, mtok, mpk)
        print(f"\n=== {tag} (dir={dirname}) decisive n={len(mx)} "
              f"({mname}-right={int(mcorr.sum())}, suffix-right={int((~mcorr).sum())}) ===")
        # fit full-pool isotonic per proposer (the deployed monotone calib map)
        pool = collect(chains, [(mname, mtok, mpk, BLUE),
                                ("Suffix", "suffix_token", "suffix_p", RED)])
        iso = {}
        for nm in (mname, "Suffix"):
            p, y, g = pool[nm]
            iso[nm] = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        cmx = iso[mname].predict(mx); csy = iso["Suffix"].predict(sy)

        sa_raw = scatter_fig(mx, sy, mcorr, mname, "raw", tag,
                             OUTDIR / f"decisive_scatter_raw_{tag}.png")
        sa_cal = scatter_fig(cmx, csy, mcorr, mname, "cal", tag,
                             OUTDIR / f"decisive_scatter_cal_{tag}.png")
        print(f"  boundary selacc: raw={sa_raw:.3f} -> cal={sa_cal:.3f}  "
              f"(+{sa_cal-sa_raw:+.3f})")
        print(f"  wrote decisive_scatter_{{raw,cal}}_{tag}.png")


if __name__ == "__main__":
    main()
