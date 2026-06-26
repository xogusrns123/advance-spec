"""Panel B style (decisive P(suffix==gt|ep,sp) surface + raw diagonal + 0.5 contour),
REGULARIZED vs IN-SAMPLE-OVERFIT fit, side by side. Same (ep,sp) plane as
raw_at_ceiling Panel B. The overfit (deep) surface carves jagged memorized
islands -> high in-sample accuracy that does NOT generalize."""
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
OUT = "simulation/results/calib_why_analysis/figures/panelB_insample_vs_reg.png"
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
ep, sp, corr, rids = [], [], [], []
for (rid, ds), rs in chains.items():
    alive = True
    for r in rs:
        if not alive:
            break
        h = r.get("oracle_hit")
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            ep.append(r["eagle_p"]); sp.append(r["suffix_p"])
            corr.append(1 if h == "suffix" else 0); rids.append(rid)
        if h not in ALIVE:
            alive = False
ep = np.array(ep); sp = np.array(sp); corr = np.array(corr); rids = np.array(rids)
X = np.c_[ep, sp]
g = np.linspace(0, 1, 240); GX, GY = np.meshgrid(g, g)
grid = np.c_[GX.ravel(), GY.ravel()]

def fit_eval(model):
    model.fit(X, corr)
    ins = ((model.predict_proba(X)[:, 1] > 0.5).astype(int) == corr).mean()
    accs = []
    for tr, te in GroupKFold(5).split(X, corr, rids):
        m2 = model.__class__(**model.get_params()); m2.fit(X[tr], corr[tr])
        accs.append(((m2.predict_proba(X[te])[:, 1] > 0.5).astype(int) == corr[te]).mean())
    Z = model.predict_proba(grid)[:, 1].reshape(GX.shape)
    return Z, ins, float(np.mean(accs))

reg = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)
ov = HistGradientBoostingClassifier(max_depth=None, max_iter=600, learning_rate=0.25,
                                    min_samples_leaf=3, l2_regularization=0.0)
Zr, ins_r, cv_r = fit_eval(reg)
Zo, ins_o, cv_o = fit_eval(ov)
print(f"regularized (depth3): in-sample={ins_r:.3f} held-out={cv_r:.3f}")
print(f"overfit (deep):       in-sample={ins_o:.3f} held-out={cv_o:.3f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2))
def panel(ax, Z, title):
    pc = ax.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.9)
    fig.colorbar(pc, ax=ax).set_label("P(suffix==gt | ep, sp, DECISIVE)")
    ax.hist2d(ep, sp, bins=45, range=[[0, 1], [0, 1]], cmin=8, cmap="Greys", alpha=0.3)
    ax.plot([0, 1], [0, 1], "k-", lw=2.2, label="raw rule  sp = ep")
    ax.contour(GX, GY, Z, levels=[0.5], colors="lime", linewidths=2.2)
    ax.plot([], [], color="lime", lw=2.2, label="P=0.5 boundary")
    ax.set_xlabel("eagle_p"); ax.set_ylabel("suffix_p"); ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

panel(axL, Zr, f"REGULARIZED (depth 3) — generalizable\nin-sample {ins_r:.3f} / held-out {cv_r:.3f}")
panel(axR, Zo, f"IN-SAMPLE OVERFIT (deep) — memorizes\nin-sample {ins_o:.3f} / held-out {cv_o:.3f}")

fig.suptitle("Panel-B surface, REGULARIZED vs IN-SAMPLE fit (decisive, ep×sp):  "
             "overfit carves jagged memorized islands (high in-sample) that DON'T "
             "generalize", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
