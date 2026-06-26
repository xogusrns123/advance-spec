"""What calibration mathematically changes, and why it barely moves selection.

Ladder (decisive accuracy): raw -> calib(marginal) -> joint Bayes(ep,sp,depth) ->
joint Bayes(all feat) -> oracle, decomposed into:
  scale-fix (raw->calib)      : rescaling sp,ep to comparable accept-probs
  cross-dependence (calib->joint same feats): calib's marginal limitation
  extra features              : signals calib ignores
  irreducible (joint->oracle) : informational ceiling
Plus the cross-independence test (the math condition for calib=Bayes) and the
calibration maps + induced boundary shift.
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
from sklearn.metrics import roc_auc_score

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
CALIB_MAP = "simulation/results/chain_hybrid_perdepth/qwen3_14b_ar/calib_cond-trained/calib_pp_beta.json"  # accept_rate
OUT = "simulation/results/calib_why_analysis/figures/calib_math.png"
ALIVE = {"eagle", "suffix", "both"}

_blob = json.load(open(CALIB_MAP))
_M = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
          for d, v in dd.items()} for g, dd in _blob["groups"].items()}
def cal(group, p, depth):
    m = _M[group]
    if depth in m: xs, ys = m[depth]
    else:
        le = [d for d in m if d <= depth]; xs, ys = m[max(le)] if le else m[max(m)]
    return max(float(np.interp(p, xs, ys)), 1e-6)

chains = defaultdict(list)
for line in open(f"{DIR}/decisions_select1_oracle.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    if r.get("type") != "decision" or r.get("tail"): continue
    chains[(r["rid"], r["decode_step"])].append(r)
for k in chains: chains[k].sort(key=lambda r: r["depth"])

# decisive set (for the ladder) + all-alive both-proposed (for cross-independence)
ep_d, sp_d, dp_d, corr, ml_d, cn_d, tt_d, rid_d = [], [], [], [], [], [], [], []
ep_a, sp_a, A_a, B_a = [], [], [], []
for rs in chains.values():
    alive = True
    for r in rs:
        if not alive: break
        h = r.get("oracle_hit")
        if r["eagle_p"] is not None and r["suffix_p"] is not None and h in ("both","eagle","suffix","none"):
            ep_a.append(r["eagle_p"]); sp_a.append(r["suffix_p"])
            A_a.append(1 if h in ("both","eagle") else 0); B_a.append(1 if h in ("both","suffix") else 0)
        if h in ("eagle","suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            ep_d.append(r["eagle_p"]); sp_d.append(r["suffix_p"]); dp_d.append(r["depth"])
            ml_d.append(r["match_len"] or 0); cn_d.append(r["suffix_count"] or 0); tt_d.append(r["suffix_total"] or 0)
            corr.append(1 if h == "suffix" else 0); rid_d.append(r["rid"])
        if h not in ALIVE: alive = False
ep = np.array(ep_d); sp = np.array(sp_d); dp = np.array(dp_d, int); corr = np.array(corr)
ml = np.array(ml_d, float); cn = np.array(cn_d, float); tt = np.array(tt_d, float); rid = np.array(rid_d)
epa = np.array(ep_a); spa = np.array(sp_a); A = np.array(A_a); B = np.array(B_a)

# ladder
raw_acc = ((sp > ep).astype(int) == corr).mean()
cs = np.array([cal("suffix", sp[i], dp[i]) for i in range(len(sp))])
ce = np.array([cal("eagle", ep[i], dp[i]) for i in range(len(ep))])
calib_acc = ((cs > ce).astype(int) == corr).mean()
def cv_acc(X):
    accs = []
    for tr, te in GroupKFold(5).split(X, corr, rid):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)
        m.fit(X[tr], corr[tr]); accs.append(((m.predict_proba(X[te])[:,1] > 0.5).astype(int) == corr[te]).mean())
    return float(np.mean(accs))
joint_epspd = cv_acc(np.c_[ep, sp, dp])
print(f"LADDER decisive accuracy:")
print(f"  raw(sp>ep)              {raw_acc:.3f}")
print(f"  calib(marginal,depth)   {calib_acc:.3f}   scale-fix  +{calib_acc-raw_acc:.3f}")
print(f"  joint Bayes(ep,sp,depth){joint_epspd:.3f}   cross-dep  +{joint_epspd-calib_acc:.3f}")
print(f"  oracle(GT)              1.000   irreducible +{1-joint_epspd:.3f}")

# cross-independence test: within own-score bins, does the OTHER score predict correctness?
def within_auc(stratify, score, y, nb=5):
    qs = np.quantile(stratify, np.linspace(0, 1, nb + 1)[1:-1])
    binid = np.digitize(stratify, qs)
    aa, ww = [], []
    for b in np.unique(binid):
        msk = binid == b
        if msk.sum() < 50 or len(set(y[msk].tolist())) < 2: continue
        a = roc_auc_score(y[msk], score[msk]); aa.append(max(a, 1 - a)); ww.append(msk.sum())
    aa, ww = np.array(aa), np.array(ww, float)
    return float((aa * ww).sum() / ww.sum())
# B=suffix-right: within sp-bins, does ep predict B? (violation of suf-corr ⟂ ep | sp)
auc_ep_given_sp = within_auc(spa, epa, B)
auc_sp_given_ep = within_auc(epa, spa, A)   # eagle-corr ⟂ sp | ep
# marginal (for contrast)
m_ep_B = max(roc_auc_score(B, epa), 1 - roc_auc_score(B, epa))
m_sp_A = max(roc_auc_score(A, spa), 1 - roc_auc_score(A, spa))
print(f"\nCROSS-INDEPENDENCE (condition for calib=Bayes):")
print(f"  AUC(ep -> suffix-right) marginal={m_ep_B:.3f}  within-sp-bins={auc_ep_given_sp:.3f}")
print(f"  AUC(sp -> eagle-right)  marginal={m_sp_A:.3f}  within-ep-bins={auc_sp_given_ep:.3f}")
print("  (within-bin AUC near 0.5 => cross-independence holds => calib ~ Bayes)")

# ================================ FIGURE =========================================
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# A: maps
a0 = ax[0, 0]
xs = np.linspace(0, 1, 200)
a0.plot(xs, [cal("eagle", x, 0) for x in xs], color="#1f77b4", lw=2, label="cal_eagle(ep, d0)")
a0.plot(xs, [cal("suffix", x, 0) for x in xs], color="#ff7f0e", lw=2, label="cal_suffix(sp, d0)")
a0.plot([0, 1], [0, 1], "k:", lw=1, label="identity (raw uses this)")
a0.set_xlabel("raw prob"); a0.set_ylabel("calibrated accept-prob")
a0.set_title("[A] what calibration does to each score\n(monotone reparametrization, per proposer)")
a0.legend(fontsize=8); a0.grid(alpha=0.3); a0.set_xlim(0, 1); a0.set_ylim(0, 1)

# B: boundary shift (decisive Bayes surface + raw diagonal + calib boundary @ d0)
a1 = ax[0, 1]
m2 = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05).fit(np.c_[ep, sp], corr)
g = np.linspace(0, 1, 200); GX, GY = np.meshgrid(g, g)
Z = m2.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
pc = a1.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.85)
fig.colorbar(pc, ax=a1).set_label("P(suffix right | DECISIVE)")
a1.plot([0, 1], [0, 1], "k-", lw=2, label="raw boundary {sp=ep}")
a1.contour(GX, GY, Z, levels=[0.5], colors="lime", linewidths=2.4)
a1.plot([], [], color="lime", lw=2.4, label="Bayes boundary {P=0.5}")
calib_b = np.array([ce_ for ce_ in g])  # boundary sp = g(ep): cal_s(sp,0)=cal_e(ep,0)
sp_grid = np.linspace(0, 1, 400)
cs0 = np.array([cal("suffix", s, 0) for s in sp_grid])
bnd = [sp_grid[np.argmin(np.abs(cs0 - cal("eagle", e, 0)))] for e in g]
a1.plot(g, bnd, color="magenta", lw=2.2, ls="--", label="calib boundary {cal_s=cal_e}, d0")
a1.set_xlabel("eagle_p"); a1.set_ylabel("suffix_p"); a1.set_xlim(0, 1); a1.set_ylim(0, 1)
a1.set_title("[B] calibration only BENDS the boundary\n(diagonal -> monotone curve, toward Bayes)")
a1.legend(fontsize=8, loc="lower right")

# C: ladder with gaps
a2 = ax[1, 0]
names = ["raw", "calib\n(marginal)", "joint Bayes\n(ep,sp,d)", "oracle"]
vals = [raw_acc, calib_acc, joint_epspd, 1.0]
cols = ["#1f77b4", "#9467bd", "#17becf", "#2ca02c"]
a2.bar(range(4), vals, color=cols)
for i, v in enumerate(vals): a2.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
gaps = [("scale-fix", 0, 1), ("cross-dep", 1, 2), ("IRREDUCIBLE", 2, 3)]
for nm, i, j in gaps:
    a2.annotate("", xy=(j, vals[j]), xytext=(i, vals[i]),
                arrowprops=dict(arrowstyle="->", color="#444"))
    a2.text((i + j) / 2, max(vals[i], vals[j]) + 0.03,
            f"{nm}\n+{vals[j]-vals[i]:.3f}", ha="center", fontsize=7.5, color="#444")
a2.set_xticks(range(4)); a2.set_xticklabels(names, fontsize=8); a2.set_ylim(0.45, 1.08)
a2.set_title("[C] decisive-accuracy ladder: where each gap comes from")
a2.grid(axis="y", alpha=0.3)

# D: MAT-loss ladder vs oracle (reconstructed; from mat_ladder.py), parallel to C
a3 = ax[1, 1]
mnames = ["raw", "calib\n(marginal)", "joint Bayes\n(ep,sp,d)", "oracle"]
mloss = [0.4687, 0.4503, 0.4372, 0.0]              # MAT loss vs oracle
mcols = ["#1f77b4", "#9467bd", "#17becf", "#2ca02c"]
a3.bar(range(4), mloss, color=mcols)
for i, vv in enumerate(mloss): a3.text(i, vv + 0.006, f"{vv:.3f}", ha="center", fontsize=9)
mrec = [("scale-fix", 0, 1), ("cross-dep", 1, 2), ("IRREDUCIBLE", 2, 3)]
for nm, i, j in mrec:
    a3.annotate("", xy=(j, mloss[j]), xytext=(i, mloss[i]),
                arrowprops=dict(arrowstyle="->", color="#444"))
    a3.text((i + j) / 2, (mloss[i] + mloss[j]) / 2 + 0.018,
            f"{nm}\n−{mloss[i]-mloss[j]:.3f}", ha="center", fontsize=7.5, color="#444")
a3.set_xticks(range(4)); a3.set_xticklabels(mnames, fontsize=8); a3.set_ylim(0, 0.52)
a3.set_ylabel("MAT loss vs oracle (recon, tokens)")
a3.set_title("[D] MAT-loss ladder (vs oracle): 93% is IRREDUCIBLE\n"
             "scale-fix −0.018 / cross-dep −0.013 only (ep,sp,d)")
a3.grid(axis="y", alpha=0.3)
a3.legend(fontsize=8)

fig.suptitle("What calibration mathematically changes (boundary bend) and why it barely "
             "moves selection  (Qwen3-14B, accept_rate)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=140)
print(f"\nwrote {OUT}")
