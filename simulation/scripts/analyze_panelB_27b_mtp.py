"""Panel B for 27B MTP, RAW style (empirical binned heatmap) + THREE boundaries:
  raw   : sp = ep (diagonal)
  Bayes : empirical P(suffix==gt | ep,sp, DECISIVE) = 0.5 cell-edge staircase
  best-mono : selection-optimal monotone classifier (calibration framework ceiling)
  calib : cal_s(sp,d) = cal_e(ep,d)  (served cond-trained beta map, depth 0)
ep here = MTP draft-token prob (logged as eagle_p). Decisive, alive-conditioned.

The 27B punchline vs 14B: MTP_p is concentrated near 1.0 (MTP is very confident),
so the raw diagonal sp=ep is badly MIS-SCALED — the Bayes boundary sits far from it,
demanding much higher suffix_p before picking suffix. That gap is what calibration
recovers (32% of the MAT gap) while it barely moved for 14B (5%)."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.ndimage import distance_transform_edt

import os as _os0
DIR = "simulation/results/chain_hybrid_perdepth/qwen35_27b_ar"
CALIB = f"{DIR}/calib_cond-trained/calib_pp_beta.json"
_sfx = "_clean" if _os0.environ.get("CLEAN", "1") == "1" else ""
OUT = f"simulation/results/calib_why_analysis/figures/panelB_27b_mtp_three_boundaries{_sfx}.png"
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

# Pre-pass: flag RUNAWAY rids (degenerate >5000-decision loopers = bfcl 42/44)
# whose repeated text inflates suffix-trie counts -> fake near-1.0 suffix_p.
import os as _os
EXCLUDE_RUNAWAY = _os.environ.get("CLEAN", "1") == "1"
_dec_n = defaultdict(int)
for line in open(f"{DIR}/decisions_select1_oracle.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    if r.get("type") == "decision" and not r.get("tail"):
        _dec_n[str(r["rid"])] += 1
RUNAWAY = {r for r, c in _dec_n.items() if c > 5000} if EXCLUDE_RUNAWAY else set()
print(f"excluding {len(RUNAWAY)} runaway rids" if RUNAWAY else "including all rids (contaminated)")

chains = defaultdict(list)
for line in open(f"{DIR}/decisions_select1_oracle.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    if r.get("type") != "decision" or r.get("tail"): continue
    if str(r["rid"]) in RUNAWAY: continue
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
print(f"decisive points: {len(ep)}  suffix-wins frac={corr.mean():.3f}  MTP_p median={np.median(ep):.3f}")

# empirical binned P(suffix==gt | bin) -- BACKGROUND
s2, xe, ye = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]], weights=corr)
c2, _, _ = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]])
with np.errstate(invalid="ignore"):
    Mm = s2 / c2
Mm[c2 < MIN_COUNT] = np.nan
# MTP-wins (low P_suffix) = PURPLE (MTP's scheme color), suffix-wins (high) = RED
# (suffix was red in the original RdBu_r), white at the 0.5 tie.
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
cmap = LinearSegmentedColormap.from_list("mtp_suffix", ["#762a83", "#f7f7f7", "#b2182b"])
cmap.set_bad("#dddddd")

g = np.linspace(0, 1, 400)
cs0 = np.array([cal("suffix", s, D0) for s in g])
calib_b = np.array([g[np.argmin(np.abs(cs0 - cal("eagle", e, D0)))] for e in g])

mono = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                      monotonic_cst=[-1, 1]).fit(np.c_[ep, sp], corr)
GXm, GYm = np.meshgrid(g, g)
Zm = mono.predict_proba(np.c_[GXm.ravel(), GYm.ravel()])[:, 1].reshape(GXm.shape)

# Bayes boundary = cell-edge staircase between suffix-wins (>0.5) and eagle-wins
Mfill = Mm.copy(); occ = ~np.isnan(Mfill)
if not occ.all():
    idx = distance_transform_edt(~occ, return_distances=False, return_indices=True)
    Mfill = Mfill[tuple(idx)]
B = (Mfill > 0.5).astype(int)
segs = []; nx_, ny_ = B.shape
for i in range(nx_):
    for j in range(ny_):
        if i + 1 < nx_ and B[i, j] != B[i + 1, j]:
            segs.append([(xe[i + 1], ye[j]), (xe[i + 1], ye[j + 1])])
        if j + 1 < ny_ and B[i, j] != B[i, j + 1]:
            segs.append([(xe[i], ye[j + 1]), (xe[i + 1], ye[j + 1])])

fig, ax = plt.subplots(figsize=(8.8, 7.4))
im = ax.imshow(np.ma.masked_invalid(Mm.T), origin="lower", extent=[0, 1, 0, 1],
               cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
fig.colorbar(im, ax=ax).set_label("EMPIRICAL mean(suffix==gt) per bin | DECISIVE")
ax.add_collection(LineCollection(segs, colors="lime", linewidths=2.6))
ax.plot([], [], color="lime", lw=2.6, label="Bayes boundary (cell-edge staircase, P=0.5)")
ax.plot([0, 1], [0, 1], "k-", lw=2.4, label="raw boundary {sp = ep}")
ax.contour(GXm, GYm, Zm, levels=[0.5], colors="darkorange", linewidths=2.8)
ax.plot([], [], color="darkorange", lw=2.8, label="best-monotone boundary (calib ceiling)")
ax.plot(g, calib_b, color="magenta", lw=2.6, ls="--", label="calib boundary {cal_s=cal_e} (d0)")
ax.set_xlabel("MTP_p  (draft-token prob, logged as eagle_p)"); ax.set_ylabel("suffix_p")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Panel B (raw/empirical heatmap) — raw vs Bayes vs calib boundary\n"
             "decisive, alive-conditioned (Qwen3.5-27B MTP"
             + (", RUNAWAY 42/44 EXCLUDED)" if _os.environ.get("CLEAN", "1") == "1" else ", all 20 tasks)"))
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
