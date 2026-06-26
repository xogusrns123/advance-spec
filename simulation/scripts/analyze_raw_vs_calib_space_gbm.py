"""raw_at_ceiling STYLE (GBM contourf + density overlay), RAW vs CALIBRATED axes.
Filled GBM surface (not over-smoothed, not sparse) of P(suffix==gt | DECISIVE);
grey hist2d shows where the data actually lives; black diagonal = the rule; lime =
Bayes 0.5 contour."""
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
OUT = "simulation/results/calib_why_analysis/figures/raw_vs_calib_space_gbm.png"
ALIVE = {"eagle", "suffix", "both"}

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
print(f"decisive n={len(corr)}  raw acc={((sp>ep)==corr).mean():.3f}  "
      f"calib acc={((cs>ce)==corr).mean():.3f}")

def surf(X):  # same as raw_at_ceiling Panel B
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05)
    m.fit(X, corr)
    g = np.linspace(0, 1, 200)
    GX, GY = np.meshgrid(g, g)
    Z = m.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
    return GX, GY, Z

fig, (axR, axC) = plt.subplots(1, 2, figsize=(15, 6.2))

def panel(ax, x, y, xlab, ylab, rule, title):
    GX, GY, Z = surf(np.c_[x, y])
    pc = ax.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.9)
    fig.colorbar(pc, ax=ax).set_label("P(suffix==gt | DECISIVE)")
    ax.hist2d(x, y, bins=45, range=[[0, 1], [0, 1]], cmin=8, cmap="Greys", alpha=0.32)
    ax.plot([0, 1], [0, 1], "k-", lw=2.4, label=rule)
    ax.contour(GX, GY, Z, levels=[0.5], colors="lime", linewidths=2.6)
    ax.plot([], [], color="lime", lw=2.6, label="Bayes 0.5 contour")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

panel(axR, ep, sp, "eagle_p (raw)", "suffix_p (raw)", "raw rule  sp = ep",
      f"RAW axes — raw acc={((sp>ep)==corr).mean():.3f}")
panel(axC, ce, cs, "cal_eagle = cal('eagle', ep, depth)",
      "cal_suffix = cal('suffix', sp, depth)", "calib rule  cal_s = cal_e",
      f"CALIBRATED axes (cond beta) — calib acc={((cs>ce)==corr).mean():.3f}")

fig.suptitle("raw_at_ceiling style (GBM fit + density): decisive P(suffix==gt) "
             "in RAW vs CALIBRATED axes  (Qwen3-14B)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
