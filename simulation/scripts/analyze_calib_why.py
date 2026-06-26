"""Why chain-hybrid select-1 calibration ~= raw and stays far from oracle.

BENEATH "signal is weak" -> structural causes, all reconstructed from the ONE
oracle decision log (GT-labeled; policy-independent at alive depths). KEY chain
model: selection only acts in the MAIN chain (depth 0..15). The suffix TAIL
(depth>=16) is suffix-only (no eagle, no decision) -> any policy that survives the
full main chain reaches the identical committed prefix and rides the SAME tail =
oracle's served accept_len. So a policy loses to oracle ONLY by mispicking inside
the main chain, and when it does it forfeits the ENTIRE downstream incl. a
possible long tail burst. That is the whole game.

  H0  metric geometry      MAT=a/(1-a) convex; raw->oracle lever is a few pp of a.
  H-A value concentration  oracle's gain is carried by a few steps that survive
                           into a long suffix tail; a main-chain mispick forfeits it.
  H-B difficulty confound  eagle_p & suffix_p track POSITION/run-length not token
                           correctness; margin AUC collapses within depth/difficulty.
  H-C flip leverage        calib changes many picks but ~all are downstream of a
                           dead chain (0 leverage); live ones cancel -> MAT flat.
"""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
ORACLE_LOG = f"{DIR}/decisions_select1_oracle.jsonl"
CALIB_MAP = f"{DIR}/calib_cond-trained/calib_pp_beta.json"
OUT = "simulation/results/calib_why_analysis"
SERVED = {"vanilla": 1.054, "raw": 1.318, "calib": 1.351, "oracle": 1.761}

# ---------- calib-map applier (matches _ServingIsoCalibrator) --------------------
_blob = json.load(open(CALIB_MAP))
_MAPS = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
             for d, v in dd.items()} for g, dd in _blob["groups"].items()}

def cal_predict(group, p, depth):
    m = _MAPS.get(group)
    if m is None:
        return p
    if depth in m:
        xs, ys = m[depth]
    else:
        le = [d for d in m if d <= depth]
        xs, ys = m[max(le)] if le else m[max(m)]
    v = float(np.interp(p, xs, ys))
    return v if v > 1e-6 else 1e-6

# ---------- load oracle log: main-chain decisions + served accept_len ------------
chains = defaultdict(list)
step_acc = {}
for line in open(ORACLE_LOG):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    t = r.get("type")
    if t == "decision":
        if r.get("tail"):
            continue                      # tail = suffix-only, no decision
        chains[(r["rid"], r["decode_step"])].append(r)
    elif t == "step":
        step_acc[(r["rid"], r["decode_step"])] = r.get("accept_len")
for k in chains:
    chains[k].sort(key=lambda r: r["depth"])
print(f"main-chain decisions in {len(chains)} steps; {len(step_acc)} served lengths")

ALIVE = {"eagle", "suffix", "both"}

def raw_pick(r):
    if r["suffix_p"] is None:
        return "eagle"
    if r["eagle_p"] is None:
        return "suffix"
    return "suffix" if r["suffix_p"] > r["eagle_p"] else "eagle"

def calib_pick(r):
    ep, sp, d = r["eagle_p"], r["suffix_p"], r["depth"]
    if sp is None:
        return "eagle"
    if ep is None:
        return "suffix"
    return "suffix" if cal_predict("suffix", sp, d) > cal_predict("eagle", ep, d) else "eagle"

def is_correct(pick, hit):
    if hit == "both":
        return True
    if hit == "eagle":
        return pick == "eagle"
    if hit == "suffix":
        return pick == "suffix"
    return False

def reconstruct(rows, pickfn, served):
    """Length under `pickfn`: die at first dead depth or first mispick in the main
    chain; if it survives all main-chain depths it rides the same tail = served."""
    for r in rows:
        hit = r.get("oracle_hit")
        if hit not in ALIVE:               # none/nogt -> everyone dead here
            return r["depth"], r
        if not is_correct(pickfn(r), hit):
            return r["depth"], r            # this policy dies, oracle continues
    return served, None                     # survived main chain -> tail

# ---------- reconstruct + validate ----------------------------------------------
Lo, Lr, Lc = [], [], []
val_match = val_total = 0
rescue_suffix = defaultdict(float)
rescue_eagle = defaultdict(float)
gain_steps = []                # (dL, L_oracle, died_in_main_bool)
for k, rows in chains.items():
    served = step_acc.get(k)
    if served is None:
        continue
    lo = served                                  # oracle = served accept_len
    # sanity: if oracle dies in main chain, served must equal that depth
    odie = None
    for r in rows:
        if r.get("oracle_hit") not in ALIVE:
            odie = r["depth"]; break
    val_total += 1
    if odie is None or odie == served:
        val_match += 1
    lr, rrow = reconstruct(rows, raw_pick, served)
    lc, _ = reconstruct(rows, calib_pick, served)
    Lo.append(lo); Lr.append(lr); Lc.append(lc)
    dL = lo - lr
    gain_steps.append((dL, lo))
    if dL > 0 and rrow is not None:              # raw died decisive in main chain
        hit = rrow["oracle_hit"]
        if hit == "suffix":
            rescue_suffix[rrow["depth"]] += dL
        elif hit == "eagle":
            rescue_eagle[rrow["depth"]] += dL

Lo = np.array(Lo, float); Lr = np.array(Lr, float); Lc = np.array(Lc, float)
print(f"\nvalidation (oracle main-death depth == served accept_len): "
      f"{val_match}/{val_total} = {100*val_match/max(val_total,1):.2f}%")
print(f"reconstructed MAT  oracle={Lo.mean():.4f}  raw={Lr.mean():.4f}  "
      f"calib={Lc.mean():.4f}")
print(f"served (ref)       oracle={SERVED['oracle']}  raw={SERVED['raw']}  "
      f"calib={SERVED['calib']}")

a = lambda L: L.mean() / (1 + L.mean())
a_raw, a_cal, a_orc = a(Lr), a(Lc), a(Lo)
a_van = SERVED["vanilla"] / (1 + SERVED["vanilla"])
print(f"\n[H0] advance prob a: raw={a_raw:.4f} calib={a_cal:.4f} oracle={a_orc:.4f}")
print(f"     raw->oracle lever in a = {a_orc-a_raw:+.4f}; calib captured "
      f"{(a_cal-a_raw)/(a_orc-a_raw)*100:.1f}%")
print(f"     MAT slope 1/(1-a)^2: raw={1/(1-a_raw)**2:.2f} oracle={1/(1-a_orc)**2:.2f}")

# ---------- H-A Lorenz + concentration ------------------------------------------
dLv = np.array([max(g[0], 0) for g in gain_steps], float)
order = np.argsort(dLv)[::-1]
cum = np.cumsum(dLv[order]); cum = cum / cum[-1]
frac = np.arange(1, len(cum) + 1) / len(cum)
gini = 2 * np.trapz(cum, frac) - 1   # desc-sorted -> curve above diagonal
top5 = cum[int(0.05 * len(cum))]; top10 = cum[int(0.10 * len(cum))]
pos = (dLv > 0).mean()
print(f"\n[H-A] {pos*100:.1f}% of steps gain anything; top5%={top5*100:.0f}% "
      f"top10%={top10*100:.0f}% of total gain (Gini={gini:.3f})")
tot_s = sum(rescue_suffix.values()); tot_e = sum(rescue_eagle.values())
print(f"      main-chain rescue: suffix={tot_s:.0f} ({100*tot_s/(tot_s+tot_e):.1f}%) "
      f"eagle={tot_e:.0f} ({100*tot_e/(tot_s+tot_e):.1f}%)")
# gain by oracle-length bucket
buck = [(0, 1, "die@1"), (1, 4, "2-4"), (4, 9, "5-9"), (9, 17, "10-16"),
        (17, 999, "17+ (tail burst)")]
gain_by_buck = []
for lo_, hi_, lab in buck:
    g = sum(max(d, 0) for d, L in gain_steps if lo_ < L <= hi_) if lo_ > 0 else \
        sum(max(d, 0) for d, L in gain_steps if L <= hi_)
    gain_by_buck.append((lab, g))
tot_gain = sum(g for _, g in gain_by_buck)
print("      gain share by oracle chain length:")
for lab, g in gain_by_buck:
    print(f"        {lab:18s} {100*g/tot_gain:5.1f}%")

# ---------- H-B confound (decisive rows) ----------------------------------------
ep, sp, dep, corr, ml = [], [], [], [], []
for rows in chains.values():
    for r in rows:
        if r.get("oracle_hit") not in ("eagle", "suffix"):
            continue
        if r["eagle_p"] is None or r["suffix_p"] is None:
            continue
        ep.append(r["eagle_p"]); sp.append(r["suffix_p"]); dep.append(r["depth"])
        corr.append(1 if r["oracle_hit"] == "suffix" else 0)
        ml.append(r["match_len"] or 0)
ep = np.array(ep); sp = np.array(sp); dep = np.array(dep, int)
corr = np.array(corr, int); ml = np.array(ml, float)
margin = sp - ep; easi = (ep + sp) / 2
def sauc(y, s):
    return roc_auc_score(y, s) if len(set(y.tolist())) > 1 else np.nan
auc_m = sauc(corr, margin); auc_e = sauc(corr, easi)
print(f"\n[H-B] decisive n={len(corr)} suffix-right={corr.mean():.3f}")
print(f"      AUC margin={auc_m:.3f} easiness={auc_e:.3f}")
print(f"      corr(sp,depth)={np.corrcoef(sp,dep)[0,1]:+.3f} "
      f"corr(sp,match_len)={np.corrcoef(sp,ml)[0,1]:+.3f} "
      f"corr(ep,depth)={np.corrcoef(ep,dep)[0,1]:+.3f}")
def within(strat):
    aa, ww = [], []
    for s in np.unique(strat):
        m = strat == s
        if m.sum() < 30:
            continue
        v = sauc(corr[m], margin[m])
        if v == v:
            aa.append(v); ww.append(m.sum())
    aa, ww = np.array(aa), np.array(ww, float)
    return (aa * ww).sum() / ww.sum()
wd = within(dep)
wq = within(np.digitize(easi, np.quantile(easi, [.2, .4, .6, .8])))
print(f"      margin AUC: marginal={auc_m:.3f} within-depth={wd:.3f} "
      f"within-difficulty={wq:.3f}  (robust -> NOT a confound; signal is real but capped)")

# decisive selection ACCURACY ladder (comparable to memory's ~0.71 ceiling) -------
raw_acc = float(((margin > 0).astype(int) == corr).mean())
# best single monotone threshold on the margin (upper bound for ANY monotone rule)
ths = np.unique(np.quantile(margin, np.linspace(0.01, 0.99, 199)))
best_acc = max(float((((margin > t).astype(int)) == corr).mean()) for t in ths)
# calib pick accuracy on the same decisive rows
cs = np.array([cal_predict("suffix", sp[i], dep[i]) for i in range(len(sp))])
ce = np.array([cal_predict("eagle", ep[i], dep[i]) for i in range(len(ep))])
calib_acc = float(((cs > ce).astype(int) == corr).mean())
acc_ladder = [("always-eagle", float((corr == 0).mean())),
              ("always-suffix", float((corr == 1).mean())),
              ("raw (margin>0)", raw_acc),
              ("best monotone thr", best_acc),
              ("calib (beta)", calib_acc),
              ("oracle (GT)", 1.0)]
print("      decisive selection ACCURACY ladder:")
for lab, v in acc_ladder:
    print(f"        {lab:20s} {v:.3f}")
print(f"      -> raw ~= best-threshold ~= calib (monotone re-threshold is saturated); "
      f"gap to oracle = AUC-capped overlap")

# ---------- H-C flip leverage ---------------------------------------------------
n_flip = n_live = help_ = hurt_ = 0
lev_h = lev_u = 0.0
for k, rows in chains.items():
    served = step_acc.get(k)
    if served is None:
        continue
    lr, _ = reconstruct(rows, raw_pick, served)
    lo = served
    for r in rows:
        if r.get("oracle_hit") not in ("eagle", "suffix"):
            continue
        rp, cp = raw_pick(r), calib_pick(r)
        if rp == cp:
            continue
        n_flip += 1
        if r["depth"] <= lr:
            n_live += 1
            lev = max(lo - r["depth"], 1)
            rc, cc = is_correct(rp, r["oracle_hit"]), is_correct(cp, r["oracle_hit"])
            if cc and not rc:
                help_ += 1; lev_h += lev
            elif rc and not cc:
                hurt_ += 1; lev_u += lev
print(f"\n[H-C] decisive pick-flips raw->calib: {n_flip}; live={n_live} "
      f"({100*n_live/max(n_flip,1):.1f}%) dead/downstream={100*(1-n_live/max(n_flip,1)):.1f}%")
print(f"      live: help={help_} hurt={hurt_} net={help_-hurt_:+d}; "
      f"leverage net={lev_h-lev_u:+.0f} tok; MAT(calib-raw)={Lc.mean()-Lr.mean():+.4f}")

# ================================ FIGURE =========================================
fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# P1 H0 convex map
a0 = ax[0, 0]
ag = np.linspace(0.45, 0.70, 300)
a0.plot(ag, ag / (1 - ag), "k-", lw=1.4, alpha=0.7)
for av, name, c in [(a_van, "vanilla", "#888"), (a_raw, "raw", "#1f77b4"),
                    (a_cal, "calib", "#9467bd"), (a_orc, "oracle", "#2ca02c")]:
    a0.scatter([av], [av / (1 - av)], s=90, color=c, zorder=5)
    a0.annotate(f"{name}={av/(1-av):.2f}", (av, av / (1 - av)),
                textcoords="offset points", xytext=(7, -3), fontsize=9, color=c)
a0.annotate("", xy=(a_orc, a_orc / (1 - a_orc)), xytext=(a_raw, a_raw / (1 - a_raw)),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.2))
a0.text(a_raw - 0.005, a_orc / (1 - a_orc),
        f"whole lever = {a_orc-a_raw:+.3f} in a\ncalib gets "
        f"{(a_cal-a_raw)/(a_orc-a_raw)*100:.0f}%", color="red", fontsize=8.5, va="center")
a0.set_xlabel("per-position advance prob  a"); a0.set_ylabel("MAT = a/(1-a)")
a0.set_title("[H0] MAT is a CONVEX blow-up of a tiny a-lever"); a0.grid(alpha=0.3)

# P2 H-A Lorenz
a1 = ax[0, 1]
a1.plot(np.r_[0, frac], np.r_[0, cum], color="#d62728", lw=2)
a1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="uniform")
a1.fill_between(frac, cum, frac, color="#d62728", alpha=0.12)
a1.scatter([0.05, 0.10], [top5, top10], color="#d62728", zorder=5)
a1.text(0.11, top10, f"top10%->{top10*100:.0f}%", fontsize=9, va="top")
a1.set_xlabel("fraction of decode-steps (sorted by gain)")
a1.set_ylabel("cumulative share of oracle-raw gain")
a1.set_title(f"[H-A] gain is CONCENTRATED (Gini={gini:.2f}); "
             f"{pos*100:.0f}% of steps gain >0"); a1.legend(fontsize=8); a1.grid(alpha=0.3)

# P3 H-A gain by oracle length
a2 = ax[0, 2]
labs = [b[0] for b in gain_by_buck]; gv = [100 * g / tot_gain for _, g in gain_by_buck]
cols = ["#c7c7c7", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
a2.bar(range(len(gv)), gv, color=cols)
for i, v in enumerate(gv):
    a2.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=9)
a2.set_xticks(range(len(labs))); a2.set_xticklabels(labs, fontsize=8, rotation=15)
a2.set_ylabel("share of total oracle-raw gain")
mid = gv[1] + gv[2]
a2.set_title(f"[H-A] gain lives in MID-length chains 2-9 ({mid:.0f}%)\n"
             f"a main-chain mispick forfeits it; rescue {100*tot_s/(tot_s+tot_e):.0f}% suffix/"
             f"{100*tot_e/(tot_s+tot_e):.0f}% eagle"); a2.grid(axis="y", alpha=0.3)

# P4 H-B' decisive selection-accuracy ladder (THE money panel)
a3 = ax[1, 0]
labL = [x[0] for x in acc_ladder]; valL = [x[1] for x in acc_ladder]
colL = ["#bbb", "#bbb", "#1f77b4", "#17becf", "#9467bd", "#2ca02c"]
a3.bar(range(len(valL)), valL, color=colL)
for i, v in enumerate(valL):
    a3.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
a3.axhline(acc_ladder[1][1], color="k", ls="--", lw=0.8, label="always-suffix base")
a3.set_xticks(range(len(labL))); a3.set_xticklabels(labL, rotation=30, ha="right", fontsize=8)
a3.set_ylabel("decisive selection accuracy vs GT"); a3.set_ylim(0, 1.04)
a3.set_title("[H-B] raw ~= best-threshold ~= calib  <<  oracle\n"
             "calibration is a MONOTONE re-threshold; raw's boundary is already optimal")
a3.legend(fontsize=8); a3.grid(axis="y", alpha=0.3)

# P5 H-B' WHY 0.72 is the ceiling: margin overlap + robustness (not a confound)
a4 = ax[1, 1]
bins = np.linspace(-1, 1, 41)
a4.hist(margin[corr == 1], bins=bins, density=True, alpha=0.55, color="#ff7f0e",
        label="suffix is right")
a4.hist(margin[corr == 0], bins=bins, density=True, alpha=0.55, color="#1f77b4",
        label="eagle is right")
a4.axvline(0, color="k", lw=1, ls=":")
a4.set_xlabel("decision margin  (suffix_p - eagle_p)"); a4.set_ylabel("density")
a4.set_title(f"[H-B] margin separates (AUC={auc_m:.2f}) but OVERLAPS irreducibly\n"
             f"within-depth AUC={wd:.2f}, within-difficulty={wq:.2f} -> NOT a confound")
a4.legend(fontsize=8); a4.grid(alpha=0.3)
a4.text(0.02, 0.97, "monotone calibration\ncannot raise this AUC,\nonly slide the 0-line",
        transform=a4.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="#caa"))

# P6 H-C flip leverage
a5 = ax[1, 2]
dL = (Lc - Lr).astype(int)
v_, c_ = np.unique(dL, return_counts=True)
a5.bar(v_, c_ / c_.sum(), color="#9467bd", width=0.8)
a5.set_xlim(-5, 5); a5.set_xlabel("per-chain  L_calib - L_raw")
a5.set_ylabel("fraction of chains")
a5.set_title(f"[H-C] {100*(dL==0).mean():.0f}% of chains UNCHANGED; net MAT "
             f"{Lc.mean()-Lr.mean():+.3f}")
a5.grid(axis="y", alpha=0.3)
a5.text(0.02, 0.97,
        f"pick-flips: {n_flip}\n live (can move MAT): {100*n_live/max(n_flip,1):.0f}%\n"
        f" dead/downstream: {100*(1-n_live/max(n_flip,1)):.0f}%\n"
        f"live help/hurt: {help_}/{hurt_}\nleverage net: {lev_h-lev_u:+.0f} tok",
        transform=a5.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="#f0f0f0", ec="#999"))

fig.suptitle("Why chain-hybrid select-1 calibration ~= raw, far from oracle  "
             "(Qwen3-14B, target_p, cond-trained beta)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{OUT}/figures/calib_why_consolidated.png"
fig.savefig(out, dpi=140)
print(f"\nwrote {out}")
