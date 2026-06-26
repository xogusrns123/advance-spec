"""Marginal correctness surfaces over the (eagle_p, suffix_p) plane, side by side:
   P(eagle_token == gt | ep, sp)   and   P(suffix_token == gt | ep, sp)
over ALL alive positions where BOTH proposers proposed (both/eagle/suffix/none).
These are NOT complementary (both->both right, none->both wrong); on the decisive
subset alone P(eagle)=1-P(suffix), but here we keep the full alive population so
each proposer's own reliability landscape shows.
  eagle_right = oracle_hit in {both, eagle};  suffix_right = oracle_hit in {both, suffix}
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
OUT = "simulation/results/calib_why_analysis/figures/eagle_suffix_surfaces.png"
ALIVE = {"eagle", "suffix", "both"}

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

ep, sp, eR, sR = [], [], [], []
for rs in chains.values():
    alive = True
    for r in rs:
        if not alive:
            break
        h = r.get("oracle_hit")
        # keep alive positions where BOTH proposers proposed (ep & sp present)
        if h in ("both", "eagle", "suffix", "none") and r["eagle_p"] is not None \
                and r["suffix_p"] is not None:
            ep.append(r["eagle_p"]); sp.append(r["suffix_p"])
            eR.append(1 if h in ("both", "eagle") else 0)
            sR.append(1 if h in ("both", "suffix") else 0)
        if h not in ALIVE:
            alive = False
ep = np.array(ep); sp = np.array(sp); eR = np.array(eR); sR = np.array(sR)
n = len(ep)
print(f"alive both-proposed positions n={n}: "
      f"P(eagle right)={eR.mean():.3f}  P(suffix right)={sR.mean():.3f}  "
      f"both-right={(eR & sR).mean():.3f}  neither={((1-eR)&(1-sR)).mean():.3f}")

def surface(y):
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05)
    m.fit(np.c_[ep, sp], y)
    gx = np.linspace(0, 1, 200)
    GX, GY = np.meshgrid(gx, gx)
    Z = m.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
    return GX, GY, Z

GXe, GYe, Ze = surface(eR)
GXs, GYs, Zs = surface(sR)

fig, (axE, axS) = plt.subplots(1, 2, figsize=(15, 6.2))
for ax, GX, GY, Z, title, who in [
        (axE, GXe, GYe, Ze, f"P(EAGLE token == gt)  (marginal, mean={eR.mean():.2f})", "eagle"),
        (axS, GXs, GYs, Zs, f"P(SUFFIX token == gt)  (marginal, mean={sR.mean():.2f})", "suffix")]:
    pc = ax.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r")
    cb = fig.colorbar(pc, ax=ax); cb.set_label(f"P({who} is right | ep, sp)")
    ax.hist2d(ep, sp, bins=40, range=[[0, 1], [0, 1]], cmin=8, cmap="Greys", alpha=0.3)
    ax.plot([0, 1], [0, 1], "k-", lw=1.8, label="raw boundary (sp=ep)")
    cs = ax.contour(GX, GY, Z, levels=[0.5], colors="lime", linewidths=2.2)
    ax.plot([], [], color="lime", lw=2.2, label=f"P({who} right)=0.5")
    ax.set_xlabel("eagle_p"); ax.set_ylabel("suffix_p")
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")

fig.suptitle("Marginal correctness surfaces — eagle vs suffix over the (ep,sp) "
             "plane  (Qwen3-14B, alive, both-proposed)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
