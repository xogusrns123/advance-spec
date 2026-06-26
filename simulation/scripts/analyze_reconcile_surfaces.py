"""Reconcile the two figures. Both are right, they draw DIFFERENT curves:
  marginal own-0.5  : {P(eagle right)=0.5} (vertical-ish) and {P(suffix right)=0.5}
                      (horizontal-ish) -- each surface's own level.
  decision boundary : {P(suffix right) = P(eagle right)} = where the two surfaces
                      CROSS -> diagonal-ish == the green line in raw_at_ceiling.
We overlay the crossover on the marginals and show the decision surface Zs-Ze.
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
OUT = "simulation/results/calib_why_analysis/figures/reconcile_surfaces.png"
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

ep, sp, eR, sR, dec, corr = [], [], [], [], [], []
for rs in chains.values():
    alive = True
    for r in rs:
        if not alive:
            break
        h = r.get("oracle_hit")
        if h in ("both", "eagle", "suffix", "none") and r["eagle_p"] is not None \
                and r["suffix_p"] is not None:
            ep.append(r["eagle_p"]); sp.append(r["suffix_p"])
            eR.append(1 if h in ("both", "eagle") else 0)
            sR.append(1 if h in ("both", "suffix") else 0)
            dec.append(1 if h in ("eagle", "suffix") else 0)        # decisive?
            corr.append(1 if h == "suffix" else 0)                  # (decisive) suffix-right
        if h not in ALIVE:
            alive = False
ep = np.array(ep); sp = np.array(sp); eR = np.array(eR); sR = np.array(sR)
dec = np.array(dec, bool); corr = np.array(corr)

def fit(X, y):
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05)
    m.fit(X, y)
    return m
gx = np.linspace(0, 1, 200)
GX, GY = np.meshgrid(gx, gx)
grid = np.c_[GX.ravel(), GY.ravel()]
Ze = fit(np.c_[ep, sp], eR).predict_proba(grid)[:, 1].reshape(GX.shape)   # P(eagle right)
Zs = fit(np.c_[ep, sp], sR).predict_proba(grid)[:, 1].reshape(GX.shape)   # P(suffix right)
# decisive-conditional P(suffix right) -- exactly what raw_at_ceiling fit
Zd = fit(np.c_[ep[dec], sp[dec]], corr[dec]).predict_proba(grid)[:, 1].reshape(GX.shape)

fig, ax = plt.subplots(1, 3, figsize=(19, 6))

def panel(a, Z, title, who):
    pc = a.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r")
    fig.colorbar(pc, ax=a)
    a.plot([0, 1], [0, 1], "k-", lw=1.2, alpha=0.5)
    a.contour(GX, GY, Z, levels=[0.5], colors="lime", linewidths=2.4)
    # crossover = decision boundary {Zs = Ze}
    a.contour(GX, GY, Zs - Ze, levels=[0.0], colors="magenta", linewidths=2.6,
              linestyles="--")
    a.set_xlabel("eagle_p"); a.set_ylabel("suffix_p")
    a.set_title(title); a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.plot([], [], color="lime", lw=2.4, label=f"P({who} right)=0.5 (own level)")
    a.plot([], [], color="magenta", lw=2.4, ls="--", label="DECISION bdy {Zs=Ze}")
    a.legend(fontsize=8, loc="lower right")

panel(ax[0], Ze, "P(EAGLE right) marginal", "eagle")
panel(ax[1], Zs, "P(SUFFIX right) marginal", "suffix")

# panel 3: the decision surface itself (what raw_at_ceiling showed), decisive-fit
pc = ax[2].contourf(GX, GY, Zd, levels=np.linspace(0, 1, 21), cmap="RdBu_r")
fig.colorbar(pc, ax=ax[2])
ax[2].plot([0, 1], [0, 1], "k-", lw=1.2, alpha=0.5, label="raw (sp=ep)")
ax[2].contour(GX, GY, Zd, levels=[0.5], colors="lime", linewidths=2.4)
ax[2].contour(GX, GY, Zs - Ze, levels=[0.0], colors="magenta", linewidths=2.4, linestyles="--")
ax[2].plot([], [], color="lime", lw=2.4, label="raw_at_ceiling bdy: P(suffix|decisive)=0.5")
ax[2].plot([], [], color="magenta", lw=2.4, ls="--", label="crossover {Zs=Ze}")
ax[2].set_xlabel("eagle_p"); ax[2].set_ylabel("suffix_p")
ax[2].set_title("DECISION surface P(suffix right | DECISIVE)\n"
                "its 0.5 (lime) == crossover (magenta) == ~diagonal")
ax[2].legend(fontsize=8, loc="lower right"); ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1)

fig.suptitle("Same data, different curves: each surface's own 0.5 (lime, axis-aligned) "
             "vs the DECISION boundary where they cross (magenta, ~diagonal)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
# numeric check: how close is the crossover to the diagonal vs the own-0.5 lines
print(f"P(eagle right)={eR.mean():.3f} P(suffix right)={sR.mean():.3f} decisive frac={dec.mean():.3f}")
