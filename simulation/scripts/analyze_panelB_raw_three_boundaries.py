"""Panel B, RAW style (empirical binned heatmap), with THREE boundaries overlaid:
  raw   : sp = ep (diagonal)
  Bayes : empirical P(suffix==gt | ep,sp, DECISIVE) = 0.5 contour (binned, no GBM)
  calib : cal_s(sp,d) = cal_e(ep,d)  (served accept_rate cond beta map, depth 0)
All empirical / non-parametric. Decisive, alive-conditioned."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
CALIB = "simulation/results/chain_hybrid_perdepth/qwen3_14b_ar/calib_cond-trained/calib_pp_beta.json"
OUT = "simulation/results/calib_why_analysis/figures/panelB_raw_three_boundaries.png"
ALIVE = {"eagle", "suffix", "both"}
BINS = 22; MIN_COUNT = 8; D0 = 0

_M = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
          for d, v in dd.items()} for g, dd in json.load(open(CALIB))["groups"].items()}
def cal(grp, p, d):
    m = _M[grp]
    if d in m: xs, ys = m[d]
    else:
        le = [k for k in m if k <= d]; xs, ys = m[max(le)] if le else m[max(m)]
    return max(float(np.interp(p, xs, ys)), 1e-6)

chains = defaultdict(list)
for line in open(f"{DIR}/decisions_select1_oracle.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    if r.get("type") != "decision" or r.get("tail"): continue
    chains[(r["rid"], r["decode_step"])].append(r)
for k in chains: chains[k].sort(key=lambda r: r["depth"])

ep, sp, corr = [], [], []
for rs in chains.values():
    alive = True
    for r in rs:
        if not alive: break
        h = r.get("oracle_hit")
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            ep.append(r["eagle_p"]); sp.append(r["suffix_p"]); corr.append(1 if h == "suffix" else 0)
        if h not in ALIVE: alive = False
ep = np.array(ep); sp = np.array(sp); corr = np.array(corr)

# empirical binned P(suffix==gt | bin) -- BACKGROUND (raw style, unchanged)
s2, xe, ye = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]], weights=corr)
c2, _, _ = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]])
with np.errstate(invalid="ignore"):
    Mm = s2 / c2
Mm[c2 < MIN_COUNT] = np.nan
cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#dddddd")
xc = (xe[:-1] + xe[1:]) / 2; yc = (ye[:-1] + ye[1:]) / 2

g = np.linspace(0, 1, 400)
# calib boundary at depth 0: sp = g(ep) solving cal_s(sp,0)=cal_e(ep,0)
cs0 = np.array([cal("suffix", s, D0) for s in g])
calib_b = np.array([g[np.argmin(np.abs(cs0 - cal("eagle", e, D0)))] for e in g])

# BEST-MONOTONE boundary = calibration framework's representational ceiling:
# the selection-optimal MONOTONE classifier (sp up, ep down). 0.5 contour on (ep,sp).
from sklearn.ensemble import HistGradientBoostingClassifier
mono = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                      monotonic_cst=[-1, 1]).fit(np.c_[ep, sp], corr)
GXm, GYm = np.meshgrid(g, g)
Zm = mono.predict_proba(np.c_[GXm.ravel(), GYm.ravel()])[:, 1].reshape(GXm.shape)

# Bayes boundary = STAIRCASE along the heatmap cells separating suffix-wins (>0.5)
# from eagle-wins (<0.5). Sparse cells filled by nearest-occupied so the region is
# contiguous; then we emit the cell-edge segments between opposite-class neighbours.
from scipy.ndimage import distance_transform_edt
Mfill = Mm.copy()                                  # Mm is (nx=ep, ny=sp)
occ = ~np.isnan(Mfill)
if not occ.all():
    idx = distance_transform_edt(~occ, return_distances=False, return_indices=True)
    Mfill = Mfill[tuple(idx)]
B = (Mfill > 0.5).astype(int)                      # 1 = suffix-wins cell
segs = []                                          # cell-edge segments (data coords)
nx_, ny_ = B.shape                                  # ep index, sp index
for i in range(nx_):
    for j in range(ny_):
        if i + 1 < nx_ and B[i, j] != B[i + 1, j]:   # vertical edge at ep=xe[i+1]
            segs.append([(xe[i + 1], ye[j]), (xe[i + 1], ye[j + 1])])
        if j + 1 < ny_ and B[i, j] != B[i, j + 1]:   # horizontal edge at sp=ye[j+1]
            segs.append([(xe[i], ye[j + 1]), (xe[i + 1], ye[j + 1])])

fig, ax = plt.subplots(figsize=(8.8, 7.4))
im = ax.imshow(np.ma.masked_invalid(Mm.T), origin="lower", extent=[0, 1, 0, 1],
               cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
fig.colorbar(im, ax=ax).set_label("EMPIRICAL mean(suffix==gt) per bin | DECISIVE")
# Bayes boundary = staircase along cell edges (suffix-wins vs eagle-wins)
from matplotlib.collections import LineCollection
ax.add_collection(LineCollection(segs, colors="lime", linewidths=2.6))
ax.plot([], [], color="lime", lw=2.6, label="Bayes boundary  (cell-edge staircase, P=0.5)")
ax.plot([0, 1], [0, 1], "k-", lw=2.4, label="raw boundary  {sp = ep}")
ax.contour(GXm, GYm, Zm, levels=[0.5], colors="darkorange", linewidths=2.8)
ax.plot([], [], color="darkorange", lw=2.8,
        label="best-monotone boundary (calibration ceiling)")
ax.plot(g, calib_b, color="magenta", lw=2.6, ls="--", label="calib boundary  {cal_s=cal_e} (d0)")
ax.set_xlabel("eagle_p"); ax.set_ylabel("suffix_p"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Panel B (raw/empirical heatmap) — raw vs Bayes vs calib boundary\n"
             "decisive, alive-conditioned (Qwen3-14B)")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
