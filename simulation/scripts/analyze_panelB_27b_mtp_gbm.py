"""Panel B for 27B MTP, GBM style: smooth GBM P(suffix==gt | ep,sp, DECISIVE)
background (unconstrained HistGradientBoosting) + boundaries:
  raw       : sp = ep (diagonal)
  GBM-Bayes : unconstrained GBM 0.5 contour
  best-mono : monotone-constrained GBM 0.5 contour (calibration framework ceiling)
  calib     : cal_s(sp,d)=cal_e(ep,d), served cond-trained beta map, depth 0
Companion to analyze_panelB_27b_mtp.py (raw/empirical). ep = MTP draft prob."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

DIR = "simulation/results/chain_hybrid_perdepth/qwen35_27b_ar"
CALIB = f"{DIR}/calib_cond-trained/calib_pp_beta.json"
OUT = "simulation/results/calib_why_analysis/figures/panelB_27b_mtp_gbm.png"
ALIVE = {"eagle", "suffix", "both"}
D0 = 0

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
print(f"decisive points: {len(ep)}  suffix-wins frac={corr.mean():.3f}")

g = np.linspace(0, 1, 300)
GX, GY = np.meshgrid(g, g)
grid = np.c_[GX.ravel(), GY.ravel()]

# unconstrained GBM = Bayes-proxy smooth background
gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.06).fit(np.c_[ep, sp], corr)
Zb = gbm.predict_proba(grid)[:, 1].reshape(GX.shape)
# monotone-constrained GBM = best-monotone (calibration ceiling)
mono = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                      monotonic_cst=[-1, 1]).fit(np.c_[ep, sp], corr)
Zm = mono.predict_proba(grid)[:, 1].reshape(GX.shape)

cs0 = np.array([cal("suffix", s, D0) for s in g])
calib_b = np.array([g[np.argmin(np.abs(cs0 - cal("eagle", e, D0)))] for e in g])

fig, ax = plt.subplots(figsize=(8.8, 7.4))
im = ax.imshow(Zb, origin="lower", extent=[0, 1, 0, 1], cmap=plt.cm.RdBu_r,
               vmin=0, vmax=1, aspect="auto")
fig.colorbar(im, ax=ax).set_label("GBM P(suffix==gt | MTP_p, suffix_p, DECISIVE)")
ax.contour(GX, GY, Zb, levels=[0.5], colors="lime", linewidths=2.8)
ax.plot([], [], color="lime", lw=2.8, label="GBM-Bayes boundary (P=0.5)")
ax.plot([0, 1], [0, 1], "k-", lw=2.4, label="raw boundary {sp = ep}")
ax.contour(GX, GY, Zm, levels=[0.5], colors="darkorange", linewidths=2.8)
ax.plot([], [], color="darkorange", lw=2.8, label="best-monotone boundary (calib ceiling)")
ax.plot(g, calib_b, color="magenta", lw=2.6, ls="--", label="calib boundary {cal_s=cal_e} (d0)")
ax.set_xlabel("MTP_p  (draft-token prob, logged as eagle_p)"); ax.set_ylabel("suffix_p")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Panel B (GBM smooth) — raw vs GBM-Bayes vs best-mono vs calib\n"
             "decisive, alive-conditioned (Qwen3.5-27B MTP)")
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
