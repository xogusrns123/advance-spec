"""WHY does calibration only flip ~10% of picks (=> ~5% of chains, +0.016 MAT)?

Decompose the flip mechanism on decisive decisions of the oracle log:
  calib_margin = cal_s(sp,d) - cal_e(ep,d) = raw_margin + push,
  push = (cal_s - sp) - (cal_e - ep)                 # calibration's net shove
  flip  <=>  sign(calib_margin) != sign(raw_margin)
A flip needs |push| to overcome |raw_margin| with opposite sign. So "why only ~10%"
= "why does push rarely beat raw_margin", and "why net~0" = "the few it beats sit
where P(correct)~0.5".  Tests: push magnitude vs raw_margin; flip-rate vs |margin|;
P(suffix-right) vs |margin|; flip outcome (right->wrong vs wrong->right) vs margin.
"""
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
OUT = "simulation/results/calib_why_analysis/figures/calib_flip_mechanism.png"

_blob = json.load(open(CALIB_MAP))
_MAPS = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
             for d, v in dd.items()} for g, dd in _blob["groups"].items()}

def cal(group, p, depth):
    m = _MAPS[group]
    if depth in m:
        xs, ys = m[depth]
    else:
        le = [d for d in m if d <= depth]
        xs, ys = m[max(le)] if le else m[max(m)]
    v = float(np.interp(p, xs, ys))
    return v if v > 1e-6 else 1e-6

# ---- decisive rows -------------------------------------------------------------
ep, sp, dep, corr = [], [], [], []
for line in open(ORACLE_LOG):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("type") != "decision" or r.get("tail"):
        continue
    if r.get("oracle_hit") not in ("eagle", "suffix"):
        continue
    if r["eagle_p"] is None or r["suffix_p"] is None:
        continue
    ep.append(r["eagle_p"]); sp.append(r["suffix_p"]); dep.append(r["depth"])
    corr.append(1 if r["oracle_hit"] == "suffix" else 0)
ep = np.array(ep); sp = np.array(sp); dep = np.array(dep, int); corr = np.array(corr)
n = len(corr)
cs = np.array([cal("suffix", sp[i], dep[i]) for i in range(n)])
ce = np.array([cal("eagle", ep[i], dep[i]) for i in range(n)])
raw_m = sp - ep
cal_m = cs - ce
push = cal_m - raw_m
raw_pick = (raw_m > 0).astype(int)        # 1=suffix
cal_pick = (cal_m > 0).astype(int)
flip = raw_pick != cal_pick

print(f"decisive n={n}; flip rate = {flip.mean()*100:.1f}%")
print(f"raw acc={ (raw_pick==corr).mean():.3f}  calib acc={(cal_pick==corr).mean():.3f}")

# H1 push magnitude vs raw margin
print(f"\n[push] median|push|={np.median(np.abs(push)):.3f}  "
      f"median|raw_margin|={np.median(np.abs(raw_m)):.3f}")
print(f"       |push|>|raw_margin| in {np.mean(np.abs(push)>np.abs(raw_m))*100:.1f}% of rows "
      f"(necessary condition for a flip)")

# H2 flip rate vs |raw margin|
edges = np.array([0, .02, .05, .1, .2, .35, .55, 1.01])
mid = 0.5 * (edges[:-1] + edges[1:])
am = np.abs(raw_m)
print("\n[flip vs |raw_margin|]   bin      n     flip%   P(suffix-right)  Pwin(majorityclass)")
flip_rate, psuf, n_bin = [], [], []
for i in range(len(edges) - 1):
    msk = (am >= edges[i]) & (am < edges[i + 1])
    nb = msk.sum()
    fr = flip[msk].mean() if nb else np.nan
    # in this |margin| band, how predictable is the label?
    ps = corr[msk].mean() if nb else np.nan
    flip_rate.append(fr); psuf.append(ps); n_bin.append(nb)
    print(f"   [{edges[i]:.2f},{edges[i+1]:.2f})  {nb:6d}  {fr*100:5.1f}   {ps:5.3f}")

# how separable is the label as a function of confidence |margin|?
# accuracy of raw rule within each |margin| band:
print("\n[raw rule accuracy by |raw_margin| band] (where can probs decide?)")
for i in range(len(edges) - 1):
    msk = (am >= edges[i]) & (am < edges[i + 1])
    if msk.sum():
        acc = (raw_pick[msk] == corr[msk]).mean()
        print(f"   [{edges[i]:.2f},{edges[i+1]:.2f})  acc={acc:.3f}  n={msk.sum()}")

# H3 flip outcomes
fm = flip
w2r = (fm & (cal_pick == corr) & (raw_pick != corr)).sum()
r2w = (fm & (raw_pick == corr) & (cal_pick != corr)).sum()
print(f"\n[flip outcomes] total flips={fm.sum()}  wrong->right={w2r}  right->wrong={r2w}  "
      f"net={w2r-r2w}  (=> +{(w2r-r2w)/n:.4f} accuracy)")

# ================================ FIGURE =========================================
fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# P1: the two calibration maps (what calibration DOES per group)
a0 = ax[0, 0]
xs = np.linspace(0, 1, 200)
for d, c in [(0, "#1f77b4"), (4, "#ff7f0e"), (10, "#2ca02c")]:
    a0.plot(xs, [cal("eagle", x, d) for x in xs], "-", color=c, lw=1.5)
    a0.plot(xs, [cal("suffix", x, d) for x in xs], "--", color=c, lw=1.5)
a0.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.6)
a0.plot([], [], "k-", label="eagle map"); a0.plot([], [], "k--", label="suffix map")
a0.text(0.05, 0.92, "colors = depth 0/4/10", fontsize=8)
a0.set_xlabel("raw prob"); a0.set_ylabel("calibrated P(correct)")
a0.set_title("[maps] both proposers map to a LOW, similar band\n(empirical accept ceiling ~0.5/0.26)")
a0.legend(fontsize=8); a0.grid(alpha=0.3); a0.set_xlim(0, 1); a0.set_ylim(0, 1)

# P2: raw_margin vs calib_margin (comparison-level)
a1 = ax[0, 1]
sub = np.random.default_rng(0).choice(n, size=min(n, 15000), replace=False)
sc = a1.scatter(raw_m[sub], cal_m[sub], c=flip[sub], cmap="coolwarm", s=4, alpha=0.4)
a1.axhline(0, color="k", lw=0.8); a1.axvline(0, color="k", lw=0.8)
a1.plot([-1, 1], [-1, 1], "g:", lw=1, label="calib==raw")
a1.set_xlabel("raw margin (sp-ep)"); a1.set_ylabel("calib margin (cal_s-cal_e)")
a1.set_title(f"[flip set] only points crossing an axis flip ({flip.mean()*100:.0f}%)\n"
             "blue=flip; flips hug the origin (small raw margin)")
a1.legend(fontsize=8); a1.grid(alpha=0.3); a1.set_xlim(-.8, .8); a1.set_ylim(-.5, .5)

# P3: THE answer — flip rate & label predictability vs |raw margin|
a2 = ax[0, 2]
a2.bar(range(len(mid)), [f * 100 for f in flip_rate], color="#9467bd", alpha=0.85,
       label="flip rate (%)")
a2.set_ylabel("flip rate (%)", color="#9467bd")
a2.set_xticks(range(len(mid)))
a2.set_xticklabels([f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(mid))],
                   rotation=40, ha="right", fontsize=7)
a2.set_xlabel("|raw margin|  (decision confidence)")
a2b = a2.twinx()
acc_band = []
for i in range(len(edges) - 1):
    msk = (am >= edges[i]) & (am < edges[i + 1])
    acc_band.append((raw_pick[msk] == corr[msk]).mean() if msk.sum() else np.nan)
a2b.plot(range(len(mid)), acc_band, "-o", color="#2ca02c", label="raw acc in band")
a2b.axhline(0.5, color="k", ls=":", lw=0.8)
a2b.set_ylabel("raw selection accuracy", color="#2ca02c"); a2b.set_ylim(0.3, 1.0)
a2.set_title("[WHY 10%] flips ONLY at |margin|~0 ...\n... which is exactly the coin-toss zone (acc~0.5)")
a2.legend(fontsize=8, loc="upper right")

# P4: push vs raw margin (why push rarely wins)
a3 = ax[1, 0]
a3.scatter(np.abs(raw_m[sub]), np.abs(push[sub]), s=4, alpha=0.25, color="#888")
a3.plot([0, 0.8], [0, 0.8], "r-", lw=1.2, label="|push|=|raw margin| (flip threshold)")
a3.set_xlabel("|raw margin|"); a3.set_ylabel("|calibration push|")
a3.set_title(f"[push] |push|>|margin| only {np.mean(np.abs(push)>np.abs(raw_m))*100:.0f}% "
             f"of rows\npush is too small to overturn a confident margin")
a3.legend(fontsize=8); a3.grid(alpha=0.3); a3.set_xlim(0, .8); a3.set_ylim(0, .8)

# P5: flip outcomes by margin band (net ~0 because coin-toss)
a4 = ax[1, 1]
w2r_b, r2w_b = [], []
for i in range(len(edges) - 1):
    msk = (am >= edges[i]) & (am < edges[i + 1]) & fm
    w2r_b.append((msk & (cal_pick == corr)).sum())
    r2w_b.append((msk & (raw_pick == corr)).sum())
x = np.arange(len(mid))
a4.bar(x - 0.2, w2r_b, 0.4, color="#2ca02c", label="wrong->right (good)")
a4.bar(x + 0.2, r2w_b, 0.4, color="#d62728", label="right->wrong (bad)")
a4.set_xticks(x)
a4.set_xticklabels([f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(mid))],
                   rotation=40, ha="right", fontsize=7)
a4.set_xlabel("|raw margin|"); a4.set_ylabel("# flipped picks")
a4.set_title(f"[WHY net~0] flips are ~coin-toss: {w2r} good vs {r2w} bad\n"
             f"net +{w2r-r2w} of {n} = +{(w2r-r2w)/n*100:.1f}pp accuracy only")
a4.legend(fontsize=8); a4.grid(axis="y", alpha=0.3)

# P6: where the decision mass sits vs |margin| (density)
a5 = ax[1, 2]
a5.hist(am, bins=np.linspace(0, 1, 41), color="#17becf", alpha=0.8)
a5.axvline(0.05, color="r", ls="--", lw=1)
frac_near = (am < 0.05).mean()
a5.text(0.07, a5.get_ylim()[1] * 0.8,
        f"{frac_near*100:.0f}% of decisions\nhave |margin|<0.05\n(the only flippable mass,\n& it's coin-toss)",
        fontsize=8.5)
a5.set_xlabel("|raw margin|"); a5.set_ylabel("# decisions")
a5.set_title("[mass] decisions pile up at small margin\nbut that mass is unwinnable (acc~0.5)")
a5.grid(axis="y", alpha=0.3)

fig.suptitle("WHY calibration changes only ~10% of picks and nets ~0 "
             "(Qwen3-14B, target_p, cond beta) — flip mechanism", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=140)
print(f"\nwrote {OUT}")
