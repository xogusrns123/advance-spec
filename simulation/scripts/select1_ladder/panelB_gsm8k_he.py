"""Panel-B DECISION BOUNDARY for the GSM8K+HumanEval select-1 ladder, 2-way, side by side:
  LEFT  = Qwen3-14B  (EAGLE3 + suffix)  -- BALANCED regime (raw > native)
  RIGHT = Qwen3.5-27B (MTP + suffix)    -- DOMINANT regime (raw < native)

Each panel: (model_p, suffix_p) space, background = EMPIRICAL mean(suffix==gt) per bin
(purple=model wins, white=tie, red=suffix wins), conditioned on DECISIVE (exactly one of
model/suffix hits gt) + alive prefix (chain not yet dead). Boundaries:
  black  raw    : sp = ep (the prob-argmax rule)
  lime   Bayes  : empirical P(suffix==gt)=0.5 cell-edge staircase (the optimal pick)
  orange best-mono: monotone-GBM 0.5 contour (the calibration ceiling)
The gap between the black diagonal and the lime/orange boundary = how mis-scaled raw
prob-argmax is = how much a learned selector (calib/bayes) can recover. 27B MTP_p piles
near 1.0 so the diagonal is badly mis-scaled (big gap -> bayes recovered +52%); 14B EAGLE3_p
is spread and suffix wins ~half the decisive cases, so the diagonal already sits near Bayes
(small gap -> bayes recovered only +12%). Run IN docker (figures dir root-owned)."""
from __future__ import annotations
import json, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.ndimage import distance_transform_edt

ROOT = "/workspace/simulation/results/chain_hybrid_perdepth"
CELLS = [
    {"dir": f"{ROOT}/gsm8k_humaneval_14b_2way", "model": "Qwen3-14B", "px": "EAGLE3_p",
     "regime": "BALANCED  (raw 1.28 > native 1.05)"},
    {"dir": f"{ROOT}/gsm8k_humaneval_2way", "model": "Qwen3.5-27B", "px": "MTP_p",
     "regime": "DOMINANT  (raw 5.33 < native 6.02)"},
]
ALIVE = {"eagle", "suffix", "both"}
BINS = 22; MIN_COUNT = 8; D0 = 0


def load_calib(path):
    """Return cal(grp, p, d): the served per-position calib map (logistic) interpolator."""
    M = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
             for d, v in dd.items()} for g, dd in json.load(open(path))["groups"].items()}

    def cal(grp, p, d):
        m = M[grp]
        if d in m:
            xs, ys = m[d]
        else:
            le = [k for k in m if k <= d]
            xs, ys = m[max(le)] if le else m[max(m)]
        return max(float(np.interp(p, xs, ys)), 1e-6)
    return cal

cmap = LinearSegmentedColormap.from_list("model_suffix", ["#762a83", "#f7f7f7", "#b2182b"])
cmap.set_bad("#dddddd")


def load_points(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("type") != "decision" or r.get("tail"):
            continue
        chains[(r["rid"], r["decode_step"])].append(r)
    for k in chains:
        chains[k].sort(key=lambda r: r["depth"])
    ep, sp, corr = [], [], []
    for rs in chains.values():
        alive = True
        for r in rs:
            if not alive:
                break
            h = r.get("oracle_hit")
            if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
                ep.append(r["eagle_p"]); sp.append(r["suffix_p"]); corr.append(1 if h == "suffix" else 0)
            if h not in ALIVE:
                alive = False
    return np.array(ep), np.array(sp), np.array(corr)


fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
for ax, cell in zip(axes, CELLS):
    ep, sp, corr = load_points(f"{cell['dir']}/decisions_select1_oracle.jsonl")
    swf = corr.mean() if len(corr) else float("nan")
    print(f"{cell['model']}: decisive pts={len(ep)}  suffix-win frac={swf:.3f}  "
          f"{cell['px']} median={np.median(ep):.3f}")

    s2, xe, ye = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]], weights=corr)
    c2, _, _ = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]])
    with np.errstate(invalid="ignore"):
        Mm = s2 / c2
    Mm[c2 < MIN_COUNT] = np.nan

    im = ax.imshow(np.ma.masked_invalid(Mm.T), origin="lower", extent=[0, 1, 0, 1],
                   cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")

    # Bayes staircase (empirical 0.5 cell-edge)
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

    ax.plot([0, 1], [0, 1], "k-", lw=2.4, label="raw boundary  {sp = ep}")

    g = np.linspace(0, 1, 400)
    mono = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                          monotonic_cst=[-1, 1]).fit(np.c_[ep, sp], corr)
    GX, GY = np.meshgrid(g, g)
    Z = mono.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
    ax.contour(GX, GY, Z, levels=[0.5], colors="darkorange", linewidths=2.8)
    ax.plot([], [], color="darkorange", lw=2.8, label="best-monotone (calib ceiling)")

    # calibration boundary: where calibrated suffix == calibrated eagle (depth 0).
    # Use the CORRECT objective (token==gt + accept-conditioned beta = the "cond-trained"
    # map), NOT the default survival label (which inflates suffix and collapses the boundary).
    cal = load_calib(f"{cell['dir']}/calib_cond-trained/calib_pp_beta.json")
    cs0 = np.array([cal("suffix", s, D0) for s in g])
    calib_b = np.array([g[int(np.argmin(np.abs(cs0 - cal("eagle", e, D0))))] for e in g])
    ax.plot(g, calib_b, color="magenta", lw=2.6, ls="--",
            label="calib boundary  {cal_s = cal_e}  (token_gt+acc-cond beta, d0)")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"{cell['px']}  (draft-token prob)"); ax.set_ylabel("suffix_p")
    ax.set_title(f"{cell['model']}  ·  {cell['regime']}\n"
                 f"decisive pts={len(ep)},  suffix wins {100*swf:.0f}% of decisive picks")
    ax.legend(fontsize=8.5, loc="upper center")

fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label(
    "EMPIRICAL mean(suffix==gt) per bin  |  DECISIVE  (purple=model wins · red=suffix wins)")
fig.suptitle("Panel B decision boundary — GSM8K+HumanEval select-1 (model vs suffix), "
             "alive+decisive conditioned\nraw diagonal far from Bayes (27B) = mis-scaled "
             "= recoverable;  near Bayes (14B) = already good", fontsize=12)
OUT = f"{CELLS[0]['dir']}/figures"; os.makedirs(OUT, exist_ok=True)
p = f"{OUT}/gsm8k_humaneval_panelB_decision_boundary.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("wrote", p)
