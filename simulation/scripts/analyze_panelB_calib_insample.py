"""IN-SAMPLE (overfit) decisive surface on CALIBRATED axes (cal_eagle_p x
cal_suffix_p), drawn GBM-style (left, high-capacity = memorizes in-sample) and
raw/empirical-binned style (right). Black diagonal = calib rule cal_s = cal_e."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
ORACLE_LOG = f"{DIR}/decisions_select1_oracle.jsonl"
CALIB_MAP = f"{DIR}/calib_cond-trained/calib_pp_beta.json"
OUT = "simulation/results/calib_why_analysis/figures/panelB_calib_insample.png"
ALIVE = {"eagle", "suffix", "both"}
BINS = 22
MIN_COUNT = 8

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
ep, sp, dep, corr, rids = [], [], [], [], []
for (rid, ds), rs in chains.items():
    alive = True
    for r in rs:
        if not alive:
            break
        h = r.get("oracle_hit")
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            ep.append(r["eagle_p"]); sp.append(r["suffix_p"]); dep.append(r["depth"])
            corr.append(1 if h == "suffix" else 0); rids.append(rid)
        if h not in ALIVE:
            alive = False
ep = np.array(ep); sp = np.array(sp); dep = np.array(dep, int)
corr = np.array(corr); rids = np.array(rids)
ce = np.array([cal("eagle", ep[i], dep[i]) for i in range(len(ep))])
cs = np.array([cal("suffix", sp[i], dep[i]) for i in range(len(ep))])
Xc = np.c_[ce, cs]

# in-sample OVERFIT GBM + its held-out CV (to label the overfit gap)
ov = HistGradientBoostingClassifier(max_depth=None, max_iter=600, learning_rate=0.25,
                                    min_samples_leaf=3, l2_regularization=0.0)
ov.fit(Xc, corr)
ins = ((ov.predict_proba(Xc)[:, 1] > 0.5).astype(int) == corr).mean()
accs = []
for tr, te in GroupKFold(5).split(Xc, corr, rids):
    mm = HistGradientBoostingClassifier(max_depth=None, max_iter=600, learning_rate=0.25,
                                        min_samples_leaf=3)
    mm.fit(Xc[tr], corr[tr])
    accs.append(((mm.predict_proba(Xc[te])[:, 1] > 0.5).astype(int) == corr[te]).mean())
cv = float(np.mean(accs))
print(f"calib-axes in-sample overfit GBM: in-sample={ins:.3f}  held-out={cv:.3f}")

fig, (axG, axR) = plt.subplots(1, 2, figsize=(15, 6.2))

# LEFT: GBM (in-sample overfit) surface
g = np.linspace(0, 1, 240); GX, GY = np.meshgrid(g, g)
Z = ov.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
pc = axG.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.9)
fig.colorbar(pc, ax=axG).set_label("P(suffix==gt | DECISIVE)")
axG.hist2d(ce, cs, bins=45, range=[[0, 1], [0, 1]], cmin=8, cmap="Greys", alpha=0.3)
axG.plot([0, 1], [0, 1], "k-", lw=2.2, label="calib rule  cal_s = cal_e")
axG.contour(GX, GY, Z, levels=[0.5], colors="lime", linewidths=2.0)
axG.plot([], [], color="lime", lw=2.0, label="P=0.5 boundary")
axG.set_xlabel("cal_eagle_p"); axG.set_ylabel("cal_suffix_p")
axG.set_title(f"GBM style — IN-SAMPLE overfit (memorizes)\n"
              f"in-sample {ins:.3f} / held-out {cv:.3f}")
axG.legend(fontsize=8, loc="lower right"); axG.set_xlim(0, 1); axG.set_ylim(0, 1)

# RIGHT: raw/empirical binned
s, _, _ = np.histogram2d(ce, cs, bins=BINS, range=[[0, 1], [0, 1]], weights=corr)
c, _, _ = np.histogram2d(ce, cs, bins=BINS, range=[[0, 1], [0, 1]])
with np.errstate(invalid="ignore"):
    Mm = s / c
Mm[c < MIN_COUNT] = np.nan
cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#dddddd")
im = axR.imshow(np.ma.masked_invalid(Mm.T), origin="lower", extent=[0, 1, 0, 1],
                cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
fig.colorbar(im, ax=axR).set_label("EMPIRICAL mean(suffix==gt) per bin | DECISIVE")
axR.plot([0, 1], [0, 1], "k-", lw=2.2, label="calib rule  cal_s = cal_e")
axR.set_xlabel("cal_eagle_p"); axR.set_ylabel("cal_suffix_p")
axR.set_title(f"raw/empirical style (in-sample data, bins={BINS}, white=0.5)")
axR.legend(fontsize=8, loc="lower right"); axR.set_xlim(0, 1); axR.set_ylim(0, 1)

fig.suptitle("IN-SAMPLE decisive surface on CALIBRATED axes (cal_eagle_p × cal_suffix_p): "
             "GBM (overfit, jagged) vs raw/empirical", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
