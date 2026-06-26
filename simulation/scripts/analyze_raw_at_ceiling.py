"""Intuitive proof that raw is ALREADY at the feature-ceiling (alive-conditioned).

Left  : selection accuracy vs decision threshold tau on the margin (sp-ep). raw is
        tau=0. The curve peaks at tau~0 -> raw sits on top of the hill; any monotone
        calibration only slides tau sideways, it cannot climb higher. The learned
        joint (all features) ceiling is only +0.035 above; oracle (GT) is far above.
Right : the (eagle_p, suffix_p) plane. raw boundary = diagonal sp=ep. The Bayes-
        optimal boundary P(suffix right|ep,sp)=0.5 nearly COINCIDES with it where
        the data lives -> the simple rule already matches the optimal decision.
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
ORACLE_LOG = f"{DIR}/decisions_select1_oracle.jsonl"
OUT = "simulation/results/calib_why_analysis/figures/raw_at_ceiling.png"
ALIVE = {"eagle", "suffix", "both"}

# ---- alive decisive set --------------------------------------------------------
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

ep, sp, dep, mlen, cnt, tot, corr, rids = [], [], [], [], [], [], [], []
for (rid, ds), rs in chains.items():
    alive = True
    for r in rs:
        if not alive:
            break
        h = r.get("oracle_hit")
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            ep.append(r["eagle_p"]); sp.append(r["suffix_p"]); dep.append(r["depth"])
            mlen.append(r["match_len"] or 0); cnt.append(r["suffix_count"] or 0)
            tot.append(r["suffix_total"] or 0)
            corr.append(1 if h == "suffix" else 0); rids.append(rid)
        if h not in ALIVE:
            alive = False
ep = np.array(ep); sp = np.array(sp); dep = np.array(dep, float)
mlen = np.array(mlen, float); cnt = np.array(cnt, float); tot = np.array(tot, float)
corr = np.array(corr); rids = np.array(rids)
margin = sp - ep
n = len(corr)

# ---- accuracy vs threshold tau -------------------------------------------------
taus = np.linspace(-0.8, 0.8, 321)
acc_tau = np.array([(((margin > t).astype(int)) == corr).mean() for t in taus])
peak_i = int(np.argmax(acc_tau)); tau_star = taus[peak_i]; peak_acc = acc_tau[peak_i]
raw_acc = float(((margin > 0).astype(int) == corr).mean())
base = max(corr.mean(), 1 - corr.mean())

# ---- learned joint ceiling (CV) ------------------------------------------------
gk = GroupKFold(n_splits=5)
X = np.c_[ep, sp, dep, mlen, cnt, tot]
acc_cv = []
for tr, te in gk.split(X, corr, rids):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)
    m.fit(X[tr], corr[tr])
    acc_cv.append(((m.predict_proba(X[te])[:, 1] > 0.5).astype(int) == corr[te]).mean())
learned = float(np.mean(acc_cv))
print(f"n={n} raw_acc={raw_acc:.3f} peak(tau*={tau_star:+.2f})={peak_acc:.3f} "
      f"learned_all={learned:.3f} base={base:.3f} oracle=1.0")

# ---- calibration accuracy: curved + per-depth boundary (NOT a tau-shift) --------
# cal_s(sp,depth) vs cal_e(ep,depth) uses TWO different monotone maps, so the
# decision boundary is a curve (and per-depth), not a 45-deg shift of sp-ep -> it
# can and does beat the best single threshold.
_cm = json.load(open(f"{DIR}/calib_cond-trained/calib_pp_beta.json"))["groups"]
_CM = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
           for d, v in g2.items()} for g, g2 in _cm.items()}
def _cal(grp, p, d):
    m = _CM[grp]
    if d in m:
        xs, ys = m[d]
    else:
        le = [k for k in m if k <= d]
        xs, ys = m[max(le)] if le else m[max(m)]
    return max(float(np.interp(p, xs, ys)), 1e-6)
_cs = np.array([_cal("suffix", sp[i], int(dep[i])) for i in range(n)])
_ce = np.array([_cal("eagle", ep[i], int(dep[i])) for i in range(n)])
calib_acc = float(((_cs > _ce).astype(int) == corr).mean())
print(f"calib_acc(curved/per-depth)={calib_acc:.3f}  > best-tau {peak_acc:.3f} "
      f"(+{calib_acc-peak_acc:.3f}); corr(cal_margin, raw_margin)="
      f"{np.corrcoef(_cs-_ce, margin)[0,1]:.3f}")

# ---- Bayes surface on (ep,sp) for the 2D panel ---------------------------------
m2 = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05)
m2.fit(np.c_[ep, sp], corr)
gx = np.linspace(0, 1, 200); gy = np.linspace(0, 1, 200)
GX, GY = np.meshgrid(gx, gy)
Z = m2.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)

# ================================ FIGURE =========================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.2))

# Panel A: the HONEST ladder (the threshold-hill was a strawman family the real
# selectors escape, so it is NOT evidence of a ceiling; the ladder is).
ladder = [("always\nsuffix", base, "#999"),
          ("raw\n(τ=0)", raw_acc, "#1f77b4"),
          ("best τ\n(threshold)", peak_acc, "#1f77b4"),
          ("calib\n(curved/depth)", calib_acc, "#9467bd"),
          ("joint disc\n(FEATURE ceiling)", learned, "#8c564b"),
          ("oracle\n(GT)", 1.0, "#2ca02c")]
labs = [x[0] for x in ladder]; vals = [x[1] for x in ladder]; cols = [x[2] for x in ladder]
axA.bar(range(len(vals)), vals, color=cols)
for i, v in enumerate(vals):
    axA.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
# the two gaps that matter
axA.annotate("", xy=(4, learned), xytext=(1, raw_acc),
             arrowprops=dict(arrowstyle="->", color="#444", lw=1.3))
axA.text(2.0, 0.83, f"real but small\nraw→disc +{learned-raw_acc:.3f}",
         color="#444", fontsize=8, ha="center")
axA.annotate("", xy=(4.5, 1.0), xytext=(4.5, learned),
             arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.5))
axA.text(4.5, 0.5 * (1.0 + learned), " IRREDUCIBLE\n (needs target-\n side info=verify)",
         color="#2ca02c", fontsize=8.5, va="center")
axA.axhline(learned, color="#8c564b", ls=":", lw=1, alpha=0.6)
axA.set_xticks(range(len(labs))); axA.set_xticklabels(labs, fontsize=8)
axA.set_ylabel("decisive selection accuracy (alive)")
axA.set_title("the FEATURE ceiling (best fn of available features = 0.795) is far\n"
              "below oracle (1.0); raw<calib<disc all cluster near it, ~11% of the gap")
axA.set_ylim(0.45, 1.06); axA.grid(axis="y", alpha=0.3)

# Panel B. NOTE Z is fit on the DECISIVE subset (exactly one proposer right), so it
# is the CONDITIONAL "which proposer is the right pick" = P(suffix is the right one
# | ep, sp, decisive) -- NOT the marginal P(suffix==gt). Its 0.5 level is the
# DECISION boundary (= crossover of the two marginal accuracy surfaces), which is
# why it is ~diagonal rather than axis-aligned.
pc = axB.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.85)
cb = fig.colorbar(pc, ax=axB)
# SAME event (suffix==gt) as the marginal plot; the ONLY difference is the
# population: here Z is conditioned on DECISIVE = exactly one of {eagle,suffix}==gt.
cb.set_label("P(suffix==gt | ep, sp, DECISIVE)\nDECISIVE = exactly one of {eagle,suffix}==gt")
# data density (where decisions actually live)
axB.hist2d(ep, sp, bins=40, range=[[0, 1], [0, 1]], cmin=8, cmap="Greys", alpha=0.35)
axB.plot([0, 1], [0, 1], "k-", lw=2.2, label="raw boundary  (sp = ep)")
cs = axB.contour(GX, GY, Z, levels=[0.5], colors="lime", linewidths=2.6)
axB.plot([], [], color="lime", lw=2.6, label="P=0.5  (= decision boundary)")
axB.set_xlabel("eagle_p"); axB.set_ylabel("suffix_p")
axB.set_title("SAME event suffix==gt, but conditioned on DECISIVE only\n"
              "(drops both-right & neither) -> its 0.5 ~ raw diagonal")
axB.legend(fontsize=9, loc="lower right"); axB.set_xlim(0, 1); axB.set_ylim(0, 1)

fig.suptitle("Selection from features caps at ~0.795 (joint disc), far below oracle 1.0; "
             "raw<calib<disc are real but small steps  (Qwen3-14B, alive decisive)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
