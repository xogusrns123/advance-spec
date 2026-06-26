"""raw_at_ceiling Panel B in RAW vs CALIBRATED axes, but EMPIRICAL binned values
(no model fit). Color = mean(suffix==gt) over decisive rows in each 2D bin;
RdBu_r centered at 0.5 so the white band IS the empirical decision boundary. Black
diagonal = the rule (sp=ep raw / cal_s=cal_e calib). Bins with <MIN_COUNT samples
are blanked (grey)."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
ORACLE_LOG = f"{DIR}/decisions_select1_oracle.jsonl"
CALIB_MAP = f"{DIR}/calib_cond-trained/calib_pp_beta.json"
OUT = "simulation/results/calib_why_analysis/figures/raw_vs_calib_space.png"
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
ep = np.array(ep); sp = np.array(sp); dep = np.array(dep, int); corr = np.array(corr, float)
ce = np.array([cal("eagle", ep[i], dep[i]) for i in range(len(ep))])
cs = np.array([cal("suffix", sp[i], dep[i]) for i in range(len(ep))])
print(f"decisive n={len(corr)}  raw acc={((sp>ep)==corr).mean():.3f}  "
      f"calib acc={((cs>ce)==corr).mean():.3f}")

def binned(x, y):
    rng = [[0, 1], [0, 1]]
    s, _, _ = np.histogram2d(x, y, bins=BINS, range=rng, weights=corr)
    c, xe, ye = np.histogram2d(x, y, bins=BINS, range=rng)
    with np.errstate(invalid="ignore"):
        M = s / c
    M[c < MIN_COUNT] = np.nan
    return np.ma.masked_invalid(M.T), xe, ye   # .T so axis0=y for imshow

cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#dddddd")
fig, (axR, axC) = plt.subplots(1, 2, figsize=(15, 6.2))

MR, xe, ye = binned(ep, sp)
imr = axR.imshow(MR, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=0, vmax=1,
                 aspect="auto", interpolation="nearest")
fig.colorbar(imr, ax=axR).set_label("EMPIRICAL mean(suffix==gt) per bin | DECISIVE")
axR.plot([0, 1], [0, 1], "k-", lw=2.4, label="raw rule  sp = ep")
axR.set_xlabel("eagle_p (raw)"); axR.set_ylabel("suffix_p (raw)")
axR.set_title("RAW axes — empirical (white band = boundary)")
axR.legend(fontsize=8, loc="lower right"); axR.set_xlim(0, 1); axR.set_ylim(0, 1)

MC, _, _ = binned(ce, cs)
imc = axC.imshow(MC, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=0, vmax=1,
                 aspect="auto", interpolation="nearest")
fig.colorbar(imc, ax=axC).set_label("EMPIRICAL mean(suffix==gt) per bin | DECISIVE")
axC.plot([0, 1], [0, 1], "k-", lw=2.4, label="calib rule  cal_s = cal_e")
axC.set_xlabel("cal_eagle = cal('eagle', ep, depth)")
axC.set_ylabel("cal_suffix = cal('suffix', sp, depth)")
axC.set_title("CALIBRATED axes — empirical")
axC.legend(fontsize=8, loc="lower right"); axC.set_xlim(0, 1); axC.set_ylim(0, 1)

fig.suptitle("Empirical P(suffix==gt | DECISIVE) per bin — RAW vs CALIBRATED axes "
             f"(no fit; bins={BINS}, min {MIN_COUNT}/bin)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
