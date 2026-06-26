"""Calibration's THEORETICAL ceiling, from its math: family = all monotone
boundaries {sp >= g_d(ep)}. Two ceilings:
  perfect-calib  = 1[ P(suf|sp,d) > P(eag|ep,d) ]   (reliability objective, true marginals)
  best-monotone  = selection-optimal monotone classifier (mono sp:+1, ep:-1)
vs raw, served calib, full Bayes (unconstrained), oracle. Decompose:
  realized (raw->servedcalib) | fit-gap (served->perfect) |
  OBJECTIVE (perfect->best-monotone) | MONOTONICITY (best-mono->full Bayes) |
  IRREDUCIBLE (full Bayes->oracle). Reports decisive accuracy AND reconstructed MAT.
"""
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
CALIB = "simulation/results/chain_hybrid_perdepth/qwen3_14b_ar/calib_cond-trained/calib_pp_beta.json"
OUT = "simulation/results/calib_why_analysis/figures/calib_ceiling_theory.png"
ALIVE = {"eagle", "suffix", "both"}
_M = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
          for d, v in dd.items()} for g, dd in json.load(open(CALIB))["groups"].items()}
def cal(grp, p, d):
    m = _M[grp]
    if d in m: xs, ys = m[d]
    else:
        le = [k for k in m if k <= d]; xs, ys = m[max(le)] if le else m[max(m)]
    return max(float(np.interp(p, xs, ys)), 1e-6)

chains = defaultdict(list); step_acc = {}
for line in open(f"{DIR}/decisions_select1_oracle.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line); t = r.get("type")
    if t == "decision" and not r.get("tail"): chains[(r["rid"], r["decode_step"])].append(r)
    elif t == "step": step_acc[(r["rid"], r["decode_step"])] = r.get("accept_len")
for k in chains: chains[k].sort(key=lambda r: r["depth"])

# populations
dec_meta, dec_f, dec_y, dec_rid = [], [], [], []        # decisive: (ep,sp,depth) -> suffix-right
sX, sy, srid, eX, ey, erid = [], [], [], [], [], []     # marginal accept populations
for (rid, ds), rs in chains.items():
    alive = True
    for r in rs:
        if not alive: break
        h = r.get("oracle_hit"); d = r["depth"]
        if r.get("eagle_p") is not None:
            eX.append([r["eagle_p"], d]); ey.append(1 if h in ("both", "eagle") else 0); erid.append(rid)
        if r.get("suffix_p") is not None:
            sX.append([r["suffix_p"], d]); sy.append(1 if h in ("both", "suffix") else 0); srid.append(rid)
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            dec_meta.append((rid, ds, d)); dec_f.append([r["eagle_p"], r["suffix_p"], d])
            dec_y.append(1 if h == "suffix" else 0); dec_rid.append(rid)
        if h not in ALIVE:
            alive = False                      # accept-condition: stop at first dead depth
dec_f = np.array(dec_f); dec_y = np.array(dec_y); dec_rid = np.array(dec_rid)
sX = np.array(sX); sy = np.array(sy); srid = np.array(srid)
eX = np.array(eX); ey = np.array(ey); erid = np.array(erid)
ep, sp, dp = dec_f[:, 0], dec_f[:, 1], dec_f[:, 2]
N = len(dec_y)
gkf = GroupKFold(5)

def gbm(mono=None):
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.06,
                                          monotonic_cst=mono)

# OOF picks (1=suffix) for each rule on the decisive set
def oof_pick(predict_fold):
    pick = np.empty(N, int)
    for tr, te in gkf.split(dec_f, dec_y, dec_rid):
        pick[te] = predict_fold(tr, te)
    return pick

def pick_perfect(tr, te):
    # fit true marginals on the FULL proposing populations restricted to train rids
    tr_rids = set(dec_rid[tr].tolist())
    sm = np.array([r in tr_rids for r in srid]); em = np.array([r in tr_rids for r in erid])
    ms = gbm([1, 0]).fit(sX[sm], sy[sm])      # P(suf|sp,depth), monotone up in sp
    me = gbm([1, 0]).fit(eX[em], ey[em])      # P(eag|ep,depth), monotone up in ep
    Ps = ms.predict_proba(np.c_[sp[te], dp[te]])[:, 1]
    Pe = me.predict_proba(np.c_[ep[te], dp[te]])[:, 1]
    return (Ps > Pe).astype(int)

def pick_bestmono(tr, te):
    m = gbm([-1, 1, 0]).fit(dec_f[tr], dec_y[tr])   # mono: ep down, sp up, depth free
    return (m.predict_proba(dec_f[te])[:, 1] > 0.5).astype(int)

def pick_fullbayes(tr, te):
    m = gbm(None).fit(dec_f[tr], dec_y[tr])
    return (m.predict_proba(dec_f[te])[:, 1] > 0.5).astype(int)

rules = {
    "raw": (sp > ep).astype(int),
    "served calib": np.array([1 if cal("suffix", sp[i], int(dp[i])) > cal("eagle", ep[i], int(dp[i])) else 0
                              for i in range(N)]),
    "perfect calib": oof_pick(pick_perfect),
    "best monotone": oof_pick(pick_bestmono),
    "full Bayes": oof_pick(pick_fullbayes),
    "oracle": dec_y,   # by construction picks correctly
}
acc = {k: float((v == dec_y).mean()) for k, v in rules.items()}
print("decisive accuracy:")
for k in rules: print(f"  {k:14s} {acc[k]:.3f}")

# ---- MAT via reconstruction (map each rule's decisive picks back to chains) -----
pickmap = {k: {dec_meta[i]: ("suffix" if rules[k][i] else "eagle") for i in range(N)} for k in rules}
def is_correct(pk, hit): return hit == "both" or (hit == "eagle" and pk == "eagle") or (hit == "suffix" and pk == "suffix")
def recon(key, rs, pm):
    served = step_acc.get(key)
    if served is None: return None
    for r in rs:
        hit = r.get("oracle_hit")
        if hit not in ALIVE: return r["depth"]
        pk = pm.get((key[0], key[1], r["depth"]), "eagle") if hit in ("eagle", "suffix") else "eagle"
        if not is_correct(pk, hit): return r["depth"]
    return served
MAT = {}
for k in rules:
    Ls = [recon(key, rs, pickmap[k]) for key, rs in chains.items() if step_acc.get(key) is not None]
    MAT[k] = float(np.mean(Ls))
mo = MAT["oracle"]
print("MAT / loss vs oracle:")
for k in rules: print(f"  {k:14s} MAT={MAT[k]:.4f}  loss={mo-MAT[k]:.4f}")

# ================================ FIGURE =========================================
order = ["raw", "served calib", "perfect calib", "best monotone", "full Bayes", "oracle"]
cols = ["#1f77b4", "#9467bd", "#c44fa0", "#17becf", "#8c564b", "#2ca02c"]
fig, (axA, axM) = plt.subplots(1, 2, figsize=(13.5, 6.2))
# accuracy
av = [acc[k] for k in order]
axA.bar(range(6), av, color=cols)
for i, v in enumerate(av): axA.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
axA.set_xticks(range(6)); axA.set_xticklabels(order, fontsize=8, rotation=12)
axA.set_ylabel("decisive selection accuracy"); axA.set_ylim(0.45, 1.05)
axA.set_title("[A] selection accuracy: calibration's two ceilings\n"
              "perfect-calib (objective) vs best-monotone (representational)")
axA.grid(axis="y", alpha=0.3)
# MAT loss
ml = [mo - MAT[k] for k in order]
axM.bar(range(6), ml, color=cols)
for i, v in enumerate(ml): axM.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
axM.set_xticks(range(6)); axM.set_xticklabels(order, fontsize=8, rotation=12)
axM.set_ylabel("MAT loss vs oracle (recon, tokens)"); axM.set_ylim(0, 0.52)
axM.set_title("[M] MAT loss vs oracle"); axM.grid(axis="y", alpha=0.3)
# annotate the decomposition between key rungs (accuracy panel)
def gap(ax, vals, i, j, nm, dy=0.03):
    ax.annotate("", xy=(j, vals[j]), xytext=(i, vals[i]), arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text((i + j) / 2, max(vals[i], vals[j]) + dy, f"{nm}\n{vals[j]-vals[i]:+.3f}",
            ha="center", fontsize=7, color="#444")
for (i, j, nm) in [(0,1,"realized"),(1,2,"fit"),(2,3,"OBJECTIVE"),(3,4,"MONOTONICITY"),(4,5,"IRREDUCIBLE")]:
    gap(axA, av, i, j, nm)
fig.suptitle("Calibration's theoretical ceiling from its math (monotone-boundary family)  "
             "(Qwen3-14B, accept_rate)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=130)
print(f"\nwrote {OUT}")
