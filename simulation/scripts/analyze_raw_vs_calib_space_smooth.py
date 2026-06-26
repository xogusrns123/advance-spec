"""raw_at_ceiling Panel B, in RAW vs CALIBRATED prob axes (side by side).

Same color = P(suffix==gt | ep, sp, DECISIVE). Left: raw axes (ep, sp), boundary
sp=ep (raw rule). Right: calibrated axes (cal_eagle, cal_suffix) via the cond-beta
map, boundary cal_s=cal_e (the calib rule). If calibration helped, its diagonal
should sit on the Bayes 0.5 contour better than raw's does.
"""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
ORACLE_LOG = f"{DIR}/decisions_select1_oracle.jsonl"
CALIB_MAP = f"{DIR}/calib_cond-trained/calib_pp_beta.json"
OUT = "simulation/results/calib_why_analysis/figures/raw_vs_calib_space_smooth.png"
ALIVE = {"eagle", "suffix", "both"}

# calib map applier (cond-trained beta)
_blob = json.load(open(CALIB_MAP))
_M = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
          for d, v in dd.items()} for g, dd in _blob["groups"].items()}
def cal(group, p, depth):
    m = _M[group]
    if depth in m:
        xs, ys = m[depth]
    else:
        le = [d for d in m if d <= depth]
        xs, ys = m[max(le)] if le else m[max(m)]
    return max(float(np.interp(p, xs, ys)), 1e-6)

chains = defaultdict(list)
for line in open(ORACLE_LOG):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("type") != "decision" or r.get("tail"):
        continue
    chains[(r["rid"], r["decode_step"])].append(r)
for k in chains:
    chains[k].sort(key=lambda r: r["depth"])

ep, sp, dep, corr = [], [], [], []
for rs in chains.values():
    alive = True
    for r in rs:
        if not alive:
            break
        h = r.get("oracle_hit")
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            ep.append(r["eagle_p"]); sp.append(r["suffix_p"]); dep.append(r["depth"])
            corr.append(1 if h == "suffix" else 0)
        if h not in ALIVE:
            alive = False
ep = np.array(ep); sp = np.array(sp); dep = np.array(dep, int); corr = np.array(corr)
ce = np.array([cal("eagle", ep[i], dep[i]) for i in range(len(ep))])
cs = np.array([cal("suffix", sp[i], dep[i]) for i in range(len(ep))])
raw_acc = (( sp > ep).astype(int) == corr).mean()
cal_acc = (((cs > ce).astype(int)) == corr).mean()
print(f"decisive n={len(corr)}  raw acc(sp>ep)={raw_acc:.3f}  calib acc(cs>ce)={cal_acc:.3f}")
print(f"cal_eagle range [{ce.min():.3f},{ce.max():.3f}]  cal_suffix range [{cs.min():.3f},{cs.max():.3f}]")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

def surf(X, lo, hi):
    # SMOOTH boundary: quadratic logistic (clean 0.5 contour for diagonal compare)
    m = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                      LogisticRegression(max_iter=4000))
    m.fit(X, corr)
    gx = np.linspace(lo[0], hi[0], 200); gy = np.linspace(lo[1], hi[1], 200)
    GX, GY = np.meshgrid(gx, gy)
    Z = m.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
    return GX, GY, Z

fig, (axR, axC) = plt.subplots(1, 2, figsize=(15, 6.2))

# RAW space
GX, GY, Z = surf(np.c_[ep, sp], (0, 0), (1, 1))
pc = axR.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.9)
fig.colorbar(pc, ax=axR).set_label("P(suffix==gt | DECISIVE)")
axR.hist2d(ep, sp, bins=40, range=[[0, 1], [0, 1]], cmin=8, cmap="Greys", alpha=0.3)
axR.plot([0, 1], [0, 1], "k-", lw=2.2, label="raw rule  sp = ep")
axR.contour(GX, GY, Z, levels=[0.5], colors="lime", linewidths=2.6)
axR.plot([], [], color="lime", lw=2.6, label="Bayes 0.5")
axR.set_xlabel("eagle_p (raw)"); axR.set_ylabel("suffix_p (raw)")
axR.set_title(f"RAW prob axes — raw rule acc={raw_acc:.3f}")
axR.legend(fontsize=8, loc="lower right"); axR.set_xlim(0, 1); axR.set_ylim(0, 1)

# CALIBRATED space
hi = (float(np.quantile(ce, 0.995)), float(np.quantile(cs, 0.995)))
GXc, GYc, Zc = surf(np.c_[ce, cs], (0, 0), hi)
pcc = axC.contourf(GXc, GYc, Zc, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.9)
fig.colorbar(pcc, ax=axC).set_label("P(suffix==gt | DECISIVE)")
axC.hist2d(ce, cs, bins=40, range=[[0, hi[0]], [0, hi[1]]], cmin=8, cmap="Greys", alpha=0.3)
dmax = min(hi)
axC.plot([0, dmax], [0, dmax], "k-", lw=2.2, label="calib rule  cal_s = cal_e")
axC.contour(GXc, GYc, Zc, levels=[0.5], colors="lime", linewidths=2.6)
axC.plot([], [], color="lime", lw=2.6, label="Bayes 0.5")
axC.set_xlabel("cal_eagle  =  cal('eagle', ep, depth)")
axC.set_ylabel("cal_suffix  =  cal('suffix', sp, depth)")
axC.set_title(f"CALIBRATED prob axes (cond beta) — calib rule acc={cal_acc:.3f}")
axC.legend(fontsize=8, loc="lower right"); axC.set_xlim(0, hi[0]); axC.set_ylim(0, hi[1])

fig.suptitle("Same decisive P(suffix==gt) surface in RAW vs CALIBRATED axes  "
             "(Qwen3-14B, alive decisive)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
