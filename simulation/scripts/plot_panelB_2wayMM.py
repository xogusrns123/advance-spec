"""Panel-B DECISION BOUNDARY for the 2-way MODEL-vs-MODEL select-1, side by side:
  LEFT  = Qwen3-8B    DFlash(x) vs EAGLE3(y)
  RIGHT = Qwen3.5-27B MTP(x)    vs DFlash(y)

Each panel: (p_A, p_B) plane, background = EMPIRICAL mean(B==gt) per bin over
DECISIVE (exactly one of A,B hits gt) + alive-prefix rows.
  purple = A (x-axis) wins   ·   red = B (y-axis) wins   ·   white = tie region
Boundaries:
  black    raw   : p_B = p_A            (the prob-argmax rule -> pick B iff p_B>p_A)
  magenta  calib : cal_B(p_B)=cal_A(p_A) (per-proposer 1-D isotonic P(==gt); where a
                   calibrated argmax flips)  <-- this is the "does calibration move the cut?"
  lime     Bayes : empirical P(B==gt)=0.5 cell-edge staircase (the optimal pick)
  orange   mono  : monotone-GBM 0.5 contour (calibration ceiling)

WHY calibration barely helps: (1) the magenta calib boundary sits almost on top of the
black diagonal -> calibration hardly re-scales the two model probs relative to each other;
(2) even the lime/orange OPTIMAL boundary hugs the diagonal AND the background near it is
mixed (bins ~0.5) -> few decisive rows can be reclassified by ANY monotone cut. That
mixing is the per-position information ceiling.

Host-runnable (writes to a host-owned dir).
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage import distance_transform_edt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze_3way_ladder import loopy_rids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results" / "chain_hybrid_perdepth"
OUT = Path(__file__).resolve().parents[1] / "results" / "twoway_modelmodel" / "figures"

CELLS = [
    dict(model="Qwen3-8B", dir="qwen3_8b_dflash_e3_ceiling20",
         xtok="eagle_token", xp="eagle_p", xname="DFlash",
         ytok="e3_token", yp="e3_p", yname="EAGLE3", min_count=5, exclude_loopy=True),
    dict(model="Qwen3.5-27B", dir="qwen35_27b_3way_real_full",
         xtok="eagle_token", xp="eagle_p", xname="MTP",
         ytok="dflash_token", yp="dflash_p", yname="DFlash", min_count=8, exclude_loopy=False),
]
BINS = 20
cmap = LinearSegmentedColormap.from_list("A_B", ["#762a83", "#f7f7f7", "#b2182b"])
cmap.set_bad("#dddddd")


def load_points(cell):
    """(pA, pB, win_B) over decisive + alive-prefix rows."""
    dpath = ROOT / cell["dir"] / "decisions_select1_oracle.jsonl"
    bad = set()
    if cell.get("exclude_loopy"):
        bad = loopy_rids(str(ROOT / cell["dir"]), "decisions_select1_oracle.jsonl")
        print(f"{cell['model']}: excluding {len(bad)} loopy reqs")
    chains = defaultdict(list)
    for line in open(dpath):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("type") != "decision" or r.get("tail") or r.get("rid") in bad:
            continue
        chains[(r["rid"], r["decode_step"])].append(r)
    for k in chains:
        chains[k].sort(key=lambda r: r["depth"])
    xa, yb, win = [], [], []
    n_alive = n_both = n_dec = n_none = 0  # alive-row breakdown for the decisive rate
    for rs in chains.values():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            ta, tb = r.get(cell["xtok"]), r.get(cell["ytok"])
            pa, pb = r.get(cell["xp"]), r.get(cell["yp"])
            a_hit = gt is not None and ta == gt
            b_hit = gt is not None and tb == gt
            decisive = (a_hit != b_hit)  # exactly one of the two model proposers hit
            n_alive += 1
            n_both += int(a_hit and b_hit)
            n_dec += int(decisive)
            n_none += int(not (a_hit or b_hit))
            if decisive and pa is not None and pb is not None:
                xa.append(pa); yb.append(pb); win.append(1 if b_hit else 0)
            if not (a_hit or b_hit):
                alive = False
    stats = dict(n_alive=n_alive, n_both=n_both, n_dec=n_dec, n_none=n_none,
                 dec_rate=n_dec / max(n_alive, 1), agree_rate=n_both / max(n_alive, 1),
                 die_rate=n_none / max(n_alive, 1))
    return np.asarray(xa), np.asarray(yb), np.asarray(win), stats


def calib_boundary(pa, pb, win, grid):
    """per-proposer 1-D isotonic P(==gt); return pB(pA) locus where cal_B(pB)=cal_A(pA)."""
    ca = IsotonicRegression(out_of_bounds="clip").fit(pa, 1 - win)  # A wins == not-B
    cb = IsotonicRegression(out_of_bounds="clip").fit(pb, win)
    ca_g = ca.predict(grid); cb_g = cb.predict(grid)
    return np.array([grid[int(np.argmin(np.abs(cb_g - ca_g[i])))] for i in range(len(grid))])


fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.3))
for ax, cell in zip(axes, CELLS):
    pa, pb, win, st = load_points(cell)
    bwf = win.mean() if len(win) else float("nan")
    print(f"{cell['model']}: alive rows={st['n_alive']}  DECISIVE RATE={st['dec_rate']:.3f} "
          f"(agree={st['agree_rate']:.3f} both-miss={st['die_rate']:.3f})  "
          f"decisive pts w/ probs={len(pa)}  {cell['yname']}-win frac={bwf:.3f}  "
          f"{cell['xname']}_p med={np.median(pa):.3f}  {cell['yname']}_p med={np.median(pb):.3f}")

    s2, xe, ye = np.histogram2d(pa, pb, bins=BINS, range=[[0, 1], [0, 1]], weights=win)
    c2, _, _ = np.histogram2d(pa, pb, bins=BINS, range=[[0, 1], [0, 1]])
    with np.errstate(invalid="ignore"):
        Mm = s2 / c2
    Mm[c2 < cell["min_count"]] = np.nan
    im = ax.imshow(np.ma.masked_invalid(Mm.T), origin="lower", extent=[0, 1, 0, 1],
                   cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")

    # Bayes staircase (empirical 0.5 cell-edge, NN-filled for empty bins)
    Mfill = Mm.copy(); occ = ~np.isnan(Mfill)
    if not occ.all():
        idx = distance_transform_edt(~occ, return_distances=False, return_indices=True)
        Mfill = Mfill[tuple(idx)]
    B = (Mfill > 0.5).astype(int); segs = []; nx_, ny_ = B.shape
    for i in range(nx_):
        for j in range(ny_):
            if i + 1 < nx_ and B[i, j] != B[i + 1, j]:
                segs.append([(xe[i + 1], ye[j]), (xe[i + 1], ye[j + 1])])
            if j + 1 < ny_ and B[i, j] != B[i, j + 1]:
                segs.append([(xe[i], ye[j + 1]), (xe[i + 1], ye[j + 1])])
    ax.add_collection(LineCollection(segs, colors="lime", linewidths=2.6))
    ax.plot([], [], color="lime", lw=2.6, label="Bayes boundary (empirical P=0.5)")

    ax.plot([0, 1], [0, 1], "k-", lw=2.4, label="raw boundary  {p_B = p_A}")

    g = np.linspace(0, 1, 400)
    mono = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                          monotonic_cst=[-1, 1]).fit(np.c_[pa, pb], win)
    GX, GY = np.meshgrid(g, g)
    Z = mono.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
    ax.contour(GX, GY, Z, levels=[0.5], colors="darkorange", linewidths=2.8)
    ax.plot([], [], color="darkorange", lw=2.8, label="best-monotone (calib ceiling)")

    cb = calib_boundary(pa, pb, win, g)
    ax.plot(g, cb, color="magenta", lw=2.4, ls="--",
            label="calib boundary  {cal_B = cal_A}  (1-D isotonic)")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"{cell['xname']}_p  (x-axis proposer)")
    ax.set_ylabel(f"{cell['yname']}_p  (y-axis proposer)")
    ax.set_title(f"{cell['model']}  ·  {cell['xname']} vs {cell['yname']}\n"
                 f"decisive rate={100*st['dec_rate']:.0f}% of alive rows  "
                 f"(agree {100*st['agree_rate']:.0f}% · both-miss {100*st['die_rate']:.0f}%),  "
                 f"{cell['yname']} wins {100*bwf:.0f}% of decisive")
    ax.legend(fontsize=8.4, loc="upper center", framealpha=0.9)

fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label(
    "EMPIRICAL mean(B==gt) per bin | DECISIVE  (purple=x-proposer wins · red=y-proposer wins)")
fig.suptitle("Panel B decision boundary — 2-way MODEL-vs-MODEL, alive+decisive conditioned\n"
             "calib (magenta) ~ on top of raw diagonal (black) AND Bayes/mono hug it too "
             "= no monotone cut recovers much = per-position info ceiling", fontsize=11.5)
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "panelB_decision_boundary_2wayMM.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("wrote", p)
