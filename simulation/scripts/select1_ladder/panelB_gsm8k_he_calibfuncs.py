"""Is it the calib FUNCTION or the marginal FRAMEWORK? Overlay ALL 4 per-proposer marginal calib
boundaries (histogram/isotonic/logistic/beta) against the JOINT 2-feature monotone-GBM boundary and
the empirical Bayes boundary, for GSM8K+HumanEval select-1 (14B EAGLE3 / 27B MTP).

Each marginal calib boundary = {sp : cal_suffix(sp,d0) = cal_eagle(ep,d0)}. The point: ALL FOUR
marginal functions land on the SAME wrong side (below the diagonal -> over-pick suffix), because at
equal raw prob the marginal accept-rate of suffix >> eagle (suffix-fires-precisely pools the 'both
correct' mass into suffix). Only the JOINT discriminator (fit on the comparative decisive label,
sees BOTH probs at once) tracks the empirical Bayes boundary. => the fix is joint, not a better 1-D
link function. Run IN docker (figures dir root-owned)."""
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
    {"dir": f"{ROOT}/gsm8k_humaneval_14b_2way", "model": "Qwen3-14B (EAGLE3)", "px": "EAGLE3_p",
     "regime": "BALANCED"},
    {"dir": f"{ROOT}/gsm8k_humaneval_2way", "model": "Qwen3.5-27B (MTP)", "px": "MTP_p",
     "regime": "DOMINANT"},
]
ALIVE = {"eagle", "suffix", "both"}
BINS = 22; MIN_COUNT = 8; D0 = 0
CALIB_FUNCS = [("histogram", "#ff80ab"), ("isotonic", "#f50057"),
               ("logistic", "#aa00ff"), ("beta", "#6a1b9a")]
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


def calib_boundary(path, g):
    M = {grp: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
               for d, v in dd.items()} for grp, dd in json.load(open(path))["groups"].items()}

    def cal(grp, p):
        m = M[grp]
        xs, ys = m[D0] if D0 in m else m[min(m)]
        return float(np.interp(p, xs, ys))
    cs = np.array([cal("suffix", s) for s in g])
    return np.array([g[int(np.argmin(np.abs(cs - cal("eagle", e))))] for e in g])


fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.4))
g = np.linspace(0, 1, 400)
for ax, cell in zip(axes, CELLS):
    ep, sp, corr = load_points(f"{cell['dir']}/decisions_select1_oracle.jsonl")
    s2, xe, ye = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]], weights=corr)
    c2, _, _ = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]])
    with np.errstate(invalid="ignore"):
        Mm = s2 / c2
    Mm[c2 < MIN_COUNT] = np.nan
    im = ax.imshow(np.ma.masked_invalid(Mm.T), origin="lower", extent=[0, 1, 0, 1],
                   cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")

    # empirical Bayes staircase
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
    ax.add_collection(LineCollection(segs, colors="lime", linewidths=3.0))
    ax.plot([], [], color="lime", lw=3.0, label="empirical Bayes (P=0.5)")
    ax.plot([0, 1], [0, 1], "k-", lw=2.2, label="raw {sp=ep}")

    # joint 2-feature monotone GBM (comparative decisive label)
    mono = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                          monotonic_cst=[-1, 1]).fit(np.c_[ep, sp], corr)
    GX, GY = np.meshgrid(g, g)
    Z = mono.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
    ax.contour(GX, GY, Z, levels=[0.5], colors="darkorange", linewidths=3.2)
    ax.plot([], [], color="darkorange", lw=3.2, label="JOINT GBM (comparative) ✓")

    # the 4 marginal calib functions, fit with the CORRECT objective (token==gt +
    # accept-conditioned = "cond-trained"), NOT the default survival label.
    for meth, col in CALIB_FUNCS:
        p = f"{cell['dir']}/calib_cond-trained/calib_pp_{meth}.json"
        if not os.path.exists(p):
            continue
        cb = calib_boundary(p, g)
        ax.plot(g, cb, color=col, lw=2.0, ls="--", label=f"marginal calib: {meth}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"{cell['px']}  (draft-token prob)"); ax.set_ylabel("suffix_p")
    ax.set_title(f"{cell['model']}  ·  {cell['regime']}\nsuffix wins {100*corr.mean():.0f}% of decisive picks")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)

fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label(
    "empirical mean(suffix==gt) per bin (purple=model · red=suffix)")
fig.suptitle("With the CORRECT objective (token==gt + accept-conditioned), all 4 marginal calib "
             "functions track the JOINT GBM / Bayes boundary.\n"
             "=> the earlier collapse was a wrong (survival) label, NOT the link function; "
             "calib works once the target is the per-token-correctness on the alive prefix.",
             fontsize=12)
OUT = f"{CELLS[0]['dir']}/figures"; os.makedirs(OUT, exist_ok=True)
pth = f"{OUT}/gsm8k_humaneval_panelB_calib_functions.png"
fig.savefig(pth, dpi=140, bbox_inches="tight")
print("wrote", pth)
