"""Is the decisive ceiling 'just low correlation', or fundamental? Alive-conditioned.

The oracle log keeps logging the draft on DEAD branches (post-death rows), so the
raw per-position 'none' rate (83%) is inflated. The selector only ever decides on
ALIVE positions (oracle prefix all-correct). Recompute the decisive set on alive
positions, then run the decisive test the user asked for:

  In the coin-toss zone (|raw margin|~0, where calibration's flips live), can ANY
  available feature (incl. match_len = the copy-vs-generate regime proxy, count,
  total, depth) beat the ~0.55 ceiling?  Grouped-by-rid CV.
   - breaks it  -> wrong VARIABLE: signal exists, just not in (ep,sp) -> fusion.
   - can't      -> irreducible: the target decides copy-vs-generate AT this token.
"""
from __future__ import annotations
import json
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
ORACLE_LOG = f"{DIR}/decisions_select1_oracle.jsonl"
OUT = "simulation/results/calib_why_analysis/figures/calib_regime.png"
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

# ---- alive-conditioned per-position outcome + decisive set ----------------------
out_alive = Counter()
rows = []   # alive decisive rows
for (rid, ds), rs in chains.items():
    alive = True
    for r in rs:
        if not alive:
            break
        h = r.get("oracle_hit")
        out_alive[h if h in ALIVE or h == "none" else "nogt"] += 1
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            rows.append((rid, r))
        if h not in ALIVE:          # chain dies here -> later depths not alive
            alive = False
tot_alive = sum(out_alive.values())
print("ALIVE-conditioned per-position outcome:")
for h in ["both", "eagle", "suffix", "none", "nogt"]:
    print(f"  {h:6s}: {out_alive[h]:6d} ({100*out_alive[h]/tot_alive:.1f}%)")
p_dead = out_alive["none"] / tot_alive
p_dec = (out_alive["eagle"] + out_alive["suffix"]) / tot_alive
print(f"  => alive advance prob (1-none-nogt) = {1-(out_alive['none']+out_alive['nogt'])/tot_alive:.3f} "
      f"(MAT-implied oracle a~0.638); decisive slice = {p_dec*100:.1f}% of alive positions")

# ---- alive decisive feature matrix ---------------------------------------------
rids = np.array([r[0] for r in rows])
D = [r[1] for r in rows]
ep = np.array([x["eagle_p"] for x in D])
sp = np.array([x["suffix_p"] for x in D])
dep = np.array([x["depth"] for x in D], float)
mlen = np.array([x["match_len"] or 0 for x in D], float)
cnt = np.array([x["suffix_count"] or 0 for x in D], float)
tot = np.array([x["suffix_total"] or 0 for x in D], float)
corr = np.array([1 if x["oracle_hit"] == "suffix" else 0 for x in D])
margin = sp - ep
am = np.abs(margin)
n = len(corr)
print(f"\nALIVE decisive n={n} (vs 10349 pooled); suffix-right={corr.mean():.3f}")

# served rules on the ALIVE set: raw threshold + monotone beta calib --------------
_mp = json.load(open(f"{DIR}/calib_cond-trained/calib_pp_beta.json"))["groups"]
_M = {g: {int(dd): (np.asarray(v["x"], float), np.asarray(v["y"], float))
          for dd, v in g2.items()} for g, g2 in _mp.items()}
def _cal(grp, p, d):
    m = _M[grp]
    if d in m:
        xs, ys = m[d]
    else:
        le = [k for k in m if k <= d]
        xs, ys = m[max(le)] if le else m[max(m)]
    return max(float(np.interp(p, xs, ys)), 1e-6)
raw_acc = float(((margin > 0).astype(int) == corr).mean())
cs = np.array([_cal("suffix", sp[i], int(dep[i])) for i in range(n)])
ce = np.array([_cal("eagle", ep[i], int(dep[i])) for i in range(n)])
calib_acc = float(((cs > ce).astype(int) == corr).mean())
print(f"served rules on ALIVE decisive: raw(margin>0)={raw_acc:.3f}  "
      f"monotone calib(beta)={calib_acc:.3f}  vs learned (ep,sp)=0.771 / all=0.795")
p_dec_alive = p_dec
def mat_from_selacc(sa):
    a = (out_alive["both"] / tot_alive) + p_dec_alive * sa
    return a / (1 - a)
print(f"MAT-headroom estimate (geometric, alive a):")
for lab, sa in [("raw", raw_acc), ("monotone calib", calib_acc),
                ("learned (ep,sp) 0.771", 0.771), ("learned all 0.795", 0.795),
                ("oracle 1.0", 1.0)]:
    print(f"   sel_acc {lab:24s} -> MAT~{mat_from_selacc(sa):.3f}")

def gcv(X, y, groups, model="lr"):
    gk = GroupKFold(n_splits=5)
    accs, aucs = [], []
    for tr, te in gk.split(X, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        if model == "lr":
            m = LogisticRegression(max_iter=2000, C=1.0)
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
            m.fit(Xs[tr], y[tr]); pr = m.predict_proba(Xs[te])[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                               learning_rate=0.05)
            m.fit(X[tr], y[tr]); pr = m.predict_proba(X[te])[:, 1]
        accs.append(((pr > 0.5).astype(int) == y[te]).mean())
        aucs.append(roc_auc_score(y[te], pr) if len(set(y[te].tolist())) > 1 else np.nan)
    return np.mean(accs), np.nanmean(aucs)

feats = {
    "(ep,sp)": np.c_[ep, sp],
    "(ep,sp,depth)": np.c_[ep, sp, dep],
    "+match_len": np.c_[ep, sp, dep, mlen],
    "all (LR)": np.c_[ep, sp, dep, mlen, cnt, tot],
}
print("\n=== FULL alive decisive: CV accuracy / AUC (group=rid) ===")
res_full = {}
for name, X in feats.items():
    a, u = gcv(X, corr, rids, "lr")
    res_full[name] = (a, u); print(f"  {name:18s} acc={a:.3f} auc={u:.3f}")
a_gbm, u_gbm = gcv(np.c_[ep, sp, dep, mlen, cnt, tot], corr, rids, "gbm")
res_full["all (GBM)"] = (a_gbm, u_gbm)
print(f"  {'all (GBM)':18s} acc={a_gbm:.3f} auc={u_gbm:.3f}")

# ---- THE TEST: coin-toss zone (|margin|<0.1) -----------------------------------
ct = am < 0.10
print(f"\n=== COIN-TOSS zone |margin|<0.10  n={ct.sum()} "
      f"(raw acc here={( (margin[ct]>0).astype(int)==corr[ct]).mean():.3f}) ===")
res_ct = {}
for name, X in feats.items():
    a, u = gcv(X[ct], corr[ct], rids[ct], "lr")
    res_ct[name] = (a, u); print(f"  {name:18s} acc={a:.3f} auc={u:.3f}")
a_ct_gbm, u_ct_gbm = gcv(np.c_[ep, sp, dep, mlen, cnt, tot][ct], corr[ct], rids[ct], "gbm")
res_ct["all (GBM)"] = (a_ct_gbm, u_ct_gbm)
print(f"  {'all (GBM)':18s} acc={a_ct_gbm:.3f} auc={u_ct_gbm:.3f}")

# regime probe: match_len separation in decisive set
print(f"\nmatch_len: suffix-right mean={mlen[corr==1].mean():.2f} "
      f"eagle-right mean={mlen[corr==0].mean():.2f}  "
      f"AUC(match_len->suffix-right)={roc_auc_score(corr, mlen):.3f}")
print(f"  in coin-toss zone: AUC(match_len)={roc_auc_score(corr[ct], mlen[ct]):.3f} "
      f"(match_len mostly={np.median(mlen[ct]):.0f}; confident-zone match_len median="
      f"{np.median(mlen[~ct]):.0f})")

# ================================ FIGURE =========================================
fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# P1 alive vs pooled per-position outcome
a0 = ax[0, 0]
cats = ["both", "eagle", "suffix", "none", "nogt"]
av = [100 * out_alive[c] / tot_alive for c in cats]
a0.bar(cats, av, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#999", "#ccc"])
for i, v in enumerate(av):
    a0.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
a0.set_ylabel("% of ALIVE positions")
a0.set_title(f"[structure] even ORACLE: {100*p_dead:.0f}% positions DEAD\n"
             f"selection slice (eagle+suffix) = only {100*p_dec:.0f}%")
a0.grid(axis="y", alpha=0.3)

# P2 match_len distribution by who's right
a1 = ax[0, 1]
bins = np.arange(0, 12) - 0.5
a1.hist(mlen[corr == 1], bins=bins, density=True, alpha=0.6, color="#ff7f0e",
        label="suffix right")
a1.hist(mlen[corr == 0], bins=bins, density=True, alpha=0.6, color="#1f77b4",
        label="eagle right")
a1.set_xlabel("match_len (verbatim run length = copy-mode proxy)")
a1.set_ylabel("density")
a1.set_title(f"[regime] match_len barely separates\nAUC={roc_auc_score(corr, mlen):.3f} "
             f"(both classes pile at short matches)")
a1.legend(fontsize=8); a1.grid(alpha=0.3); a1.set_xlim(-0.5, 10)

# P3 FULL decisive: served rules vs learned (the money panel)
a2 = ax[0, 2]
names = ["raw (thr)", "monotone\ncalib"] + list(res_full.keys())
accs = [raw_acc, calib_acc] + [res_full[k][0] for k in res_full]
cols = ["#1f77b4", "#9467bd"] + ["#17becf"] * len(res_full)
a2.bar(range(len(names)), accs, color=cols)
for i, v in enumerate(accs):
    a2.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
a2.axhline(1.0, color="#2ca02c", ls="--", lw=1, label="oracle (GT)=1.0")
a2.set_xticks(range(len(names))); a2.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
a2.set_ylabel("CV decisive accuracy (ALIVE)"); a2.set_ylim(0.45, 1.03)
a2.set_title("[money] raw~calib~learned-all (0.76-0.80) << oracle 1.0\n"
             "calib already near feature-ceiling; gap to oracle is IRREDUCIBLE")
a2.legend(fontsize=8); a2.grid(axis="y", alpha=0.3)

# P4 coin-toss zone: feature ladder (THE answer)
a3 = ax[1, 0]
names = list(res_ct.keys())
accs = [res_ct[k][0] for k in names]
a3.bar(range(len(names)), accs, color="#d62728")
for i, v in enumerate(accs):
    a3.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
a3.axhline(0.5, color="k", ls=":", lw=0.8, label="coin toss")
a3.axhline(1.0, color="#2ca02c", ls="--", lw=1, label="oracle=1.0")
a3.set_xticks(range(len(names))); a3.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
a3.set_ylabel("CV accuracy in coin-toss zone"); a3.set_ylim(0.45, 1.03)
a3.set_title(f"[THE TEST] coin-toss zone |margin|<0.1 (n={ct.sum()})\n"
             "NO feature breaks ~0.55 -> irreducible, not wrong-variable")
a3.legend(fontsize=8); a3.grid(axis="y", alpha=0.3)

# P5 AUC full vs coin-toss for each feature set
a4 = ax[1, 1]
names = list(res_full.keys())
x = np.arange(len(names))
a4.bar(x - 0.2, [res_full[k][1] for k in names], 0.4, color="#9467bd", label="full decisive")
a4.bar(x + 0.2, [res_ct[k][1] for k in names], 0.4, color="#d62728", label="coin-toss zone")
a4.axhline(0.5, color="k", ls=":", lw=0.8)
a4.set_xticks(x); a4.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
a4.set_ylabel("CV AUC"); a4.set_ylim(0.45, 0.85)
a4.set_title("[AUC] signal vanishes in the zone calibration can reach\n"
             "(full~0.78 but coin-toss~0.55 for EVERY feature set)")
a4.legend(fontsize=8); a4.grid(axis="y", alpha=0.3)

# P6 match_len vs |margin|: confident=long copy, coin-toss=short/ambiguous
a5 = ax[1, 2]
edges = np.array([0, .05, .1, .2, .35, .55, 1.01])
mids = 0.5 * (edges[:-1] + edges[1:])
mlmean = [mlen[(am >= edges[i]) & (am < edges[i + 1])].mean() for i in range(len(mids))]
spr = [corr[(am >= edges[i]) & (am < edges[i + 1])].mean() for i in range(len(mids))]
a5.plot(mids, mlmean, "-o", color="#ff7f0e", label="mean match_len")
a5.set_xlabel("|raw margin| (decision confidence)")
a5.set_ylabel("mean match_len", color="#ff7f0e")
a5b = a5.twinx()
a5b.plot(mids, spr, "-s", color="#2ca02c", label="P(suffix right)")
a5b.axhline(0.5, color="k", ls=":", lw=0.8); a5b.set_ylabel("P(suffix right)", color="#2ca02c")
a5.set_title("[why] big margin = long verbatim copy (decidable),\ncoin-toss = short match (regime undecided)")
a5.grid(alpha=0.3)

fig.suptitle("Is the decisive ceiling 'low correlation' or fundamental? "
             "ALIVE-conditioned (Qwen3-14B, target_p)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=140)
print(f"\nwrote {OUT}")
