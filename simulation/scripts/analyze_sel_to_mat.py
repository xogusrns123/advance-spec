"""Selection accuracy -> MAT, formalized and measured per depth.

MAT_pi = sum_k prod_{d<k} a_d,  a_d = b_d + c_d s_d   (b=both,c=decisive,m=dead;
s_d = decisive selection accuracy at depth d). Oracle: s_d=1.
MAT_orc - MAT_pi = sum_j reach_j * c_j*(1-s_j) * (1 + E[oracle remaining | pass j]).

We (1) measure per-depth s_d^raw/s_d^calib, b_d,c_d,m_d, reach_j; (2) EXACTLY
decompose MAT_orc-MAT_raw and MAT_calib-... by the depth of the policy's FIRST
error (no independence assumption, via chain reconstruction); (3) verify the
closed-form a_d-product reproduces MAT; (4) report the leverage curve.
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
OUT = "simulation/results/calib_why_analysis/figures/sel_to_mat.png"
ALIVE = {"eagle", "suffix", "both"}
MAXD = 16

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

chains = defaultdict(list); step_acc = {}
for line in open(ORACLE_LOG):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    t = r.get("type")
    if t == "decision" and not r.get("tail"):
        chains[(r["rid"], r["decode_step"])].append(r)
    elif t == "step":
        step_acc[(r["rid"], r["decode_step"])] = r.get("accept_len")
for k in chains:
    chains[k].sort(key=lambda r: r["depth"])

def raw_pick(r):
    if r["suffix_p"] is None: return "eagle"
    if r["eagle_p"] is None: return "suffix"
    return "suffix" if r["suffix_p"] > r["eagle_p"] else "eagle"
def calib_pick(r):
    ep, sp, d = r["eagle_p"], r["suffix_p"], r["depth"]
    if sp is None: return "eagle"
    if ep is None: return "suffix"
    return "suffix" if cal("suffix", sp, d) > cal("eagle", ep, d) else "eagle"
def is_correct(pk, hit):
    return hit == "both" or (hit == "eagle" and pk == "eagle") or (hit == "suffix" and pk == "suffix")

# ---- per-depth structure + selection accuracy (alive-conditioned) ---------------
both = np.zeros(MAXD); dec = np.zeros(MAXD); dead = np.zeros(MAXD); reach = np.zeros(MAXD)
sc_raw = np.zeros(MAXD); sc_cal = np.zeros(MAXD); dec_n = np.zeros(MAXD)
# ---- exact MAT-gap decomposition by FIRST-error depth ---------------------------
loss_orc = np.zeros(MAXD); loss_cal = np.zeros(MAXD)   # MAT loss attributed to depth j
firsterr_orc = np.zeros(MAXD)                          # count of chains raw first-errs at j
Lorc_sum = Lraw_sum = Lcal_sum = 0.0; N = 0
lev_num = np.zeros(MAXD); lev_den = np.zeros(MAXD)     # E[L_orc - j | reach j, decisive]

for k, rs in chains.items():
    served = step_acc.get(k)
    if served is None: continue
    N += 1
    Lo = served
    Lr = Lc = None
    o_alive = True; raw_dead = cal_dead = False
    for r in rs:
        d = r["depth"]; hit = r.get("oracle_hit")
        # ---- per-depth STRUCTURE on the ORACLE-alive prefix only ----
        if o_alive:
            if hit == "both":
                reach[d] += 1; both[d] += 1
            elif hit in ("eagle", "suffix"):
                reach[d] += 1; dec[d] += 1; dec_n[d] += 1
                if is_correct(raw_pick(r), hit): sc_raw[d] += 1
                if is_correct(calib_pick(r), hit): sc_cal[d] += 1
                lev_num[d] += (Lo - d); lev_den[d] += 1     # oracle continuation, >=1
            elif hit == "none":
                reach[d] += 1; dead[d] += 1; o_alive = False  # oracle dies here
            else:                                              # nogt/boundary -> stop
                o_alive = False
        # ---- raw / calib chain reconstruction (find first-error depth) ----
        if not raw_dead:
            if hit not in ALIVE:
                Lr = d; raw_dead = True
            elif not is_correct(raw_pick(r), hit):
                Lr = d; raw_dead = True
                if hit in ("eagle", "suffix"):
                    loss_orc[d] += (Lo - d); firsterr_orc[d] += 1
        if not cal_dead:
            if hit not in ALIVE:
                Lc = d; cal_dead = True
            elif not is_correct(calib_pick(r), hit):
                Lc = d; cal_dead = True
                if hit in ("eagle", "suffix"):
                    loss_cal[d] += (Lo - d)
    if Lr is None: Lr = Lo
    if Lc is None: Lc = Lo
    Lorc_sum += Lo; Lraw_sum += Lr; Lcal_sum += Lc

MAT_orc = Lorc_sum / N; MAT_raw = Lraw_sum / N; MAT_cal = Lcal_sum / N
print(f"chains N={N}")
print(f"MAT  raw={MAT_raw:.4f}  calib={MAT_cal:.4f}  oracle={MAT_orc:.4f}")
print(f"gap  oracle-raw={MAT_orc-MAT_raw:.4f}  calib-raw={MAT_cal-MAT_raw:.4f}")
s_raw = np.where(dec_n > 0, sc_raw / np.maximum(dec_n, 1), np.nan)
s_cal = np.where(dec_n > 0, sc_cal / np.maximum(dec_n, 1), np.nan)
reach_frac = reach / N                       # P(reach depth d alive)
b_d = np.where(reach > 0, both / np.maximum(reach, 1), 0)
c_d = np.where(reach > 0, dec / np.maximum(reach, 1), 0)
lev = np.where(lev_den > 0, lev_num / np.maximum(lev_den, 1), 0)   # E[L_orc - j|decisive@j]

print("\n d | reach  both  dec   dead  | s_raw s_cal | lev(E[Lorc-d|dec]) | lossOrc lossCal")
for d in range(MAXD):
    m_d = 1 - b_d[d] - c_d[d]
    print(f"{d:2d} | {reach_frac[d]:.3f} {b_d[d]:.3f} {c_d[d]:.3f} {m_d:.3f} | "
          f"{s_raw[d]:.3f} {s_cal[d]:.3f} | {lev[d]:6.2f} | {loss_orc[d]/N:+.4f} {loss_cal[d]/N:+.4f}")

# closed-form: a_d = b_d + c_d s_d ; MAT = sum_{k=1..MAXD} prod_{d<k} a_d + survive*tail
def survprod(a):
    P = np.ones(MAXD + 1)
    for d in range(MAXD):
        P[d + 1] = P[d] * a[d]
    return P
def mat_formula(s, tail_mean):
    a = b_d + c_d * np.nan_to_num(s, nan=1.0)
    P = survprod(a)
    return float(P[1:MAXD + 1].sum() + P[MAXD] * tail_mean)
a_orc = b_d + c_d                       # = 1 - m_d on the alive prefix
Porc = survprod(a_orc)
tail_mean = max((MAT_orc - Porc[1:MAXD + 1].sum()) / max(Porc[MAXD], 1e-9), 0.0)
print(f"\n[formula check] a_d=b_d+c_d*s_d ; tail_mean(beyond d{MAXD})~{tail_mean:.3f}")
print(f"  MAT_orc formula={mat_formula(np.ones(MAXD), tail_mean):.4f} (meas {MAT_orc:.4f})")
print(f"  MAT_raw formula={mat_formula(s_raw, tail_mean):.4f} (meas {MAT_raw:.4f})")
print(f"  MAT_cal formula={mat_formula(s_cal, tail_mean):.4f} (meas {MAT_cal:.4f})")

# ================================ FIGURE =========================================
fig, ax = plt.subplots(2, 2, figsize=(16, 10))
dd = np.arange(MAXD)

a0 = ax[0, 0]
a0.plot(dd, s_raw, "-o", color="#1f77b4", ms=4, label="raw")
a0.plot(dd, s_cal, "-s", color="#9467bd", ms=4, label="calibration")
a0.axhline(1.0, color="#2ca02c", ls="--", lw=1.2, label="oracle = 1.0")
a0.axhline(0.5, color="#999", ls=":", lw=0.8)
a0.set_xlabel("draft depth d"); a0.set_ylabel("decisive selection accuracy s_d")
a0.set_title("[1] per-depth selection accuracy (raw vs calib vs oracle)")
a0.legend(fontsize=8); a0.grid(alpha=0.3); a0.set_ylim(0, 1.05)

a1 = ax[0, 1]
a1.plot(dd, reach_frac, "-o", color="#333", ms=4, label="reach_d = P(alive at d)")
a1.plot(dd, c_d, "-^", color="#ff7f0e", ms=4, label="decisive rate c_d (|alive)")
a1.plot(dd, b_d, "-v", color="#2ca02c", ms=4, label="both-match b_d (|alive)")
a1.set_xlabel("draft depth d"); a1.set_ylabel("probability")
a1.set_title("[2] structure: reach prob + decisive/both rates")
a1.legend(fontsize=8); a1.grid(alpha=0.3); a1.set_ylim(0, 1.05)

a2 = ax[1, 0]
# leverage = reach_j * (1 + E[oracle remaining | decisive@j]) ~ reach_j * (1+lev)? use Lorc-j directly
lever_weight = reach_frac * c_d * lev   # expected MAT at stake from a decisive error at d
a2.bar(dd - 0.2, reach_frac, 0.4, color="#333", alpha=0.6, label="reach_j")
a2b = a2.twinx()
a2b.plot(dd, lev, "-o", color="#d62728", label="E[L_orc - j | decisive@j] (oracle continuation)")
a2.set_xlabel("draft depth d"); a2.set_ylabel("reach_j", color="#333")
a2b.set_ylabel("oracle continuation length", color="#d62728")
a2.set_title("[3] leverage pieces: reach (falls) x oracle-continuation")
a2.legend(fontsize=8, loc="upper right"); a2b.legend(fontsize=8, loc="center right"); a2.grid(alpha=0.3)

a3 = ax[1, 1]
lo = loss_orc / N; lc = loss_cal / N
a3.bar(dd - 0.2, lo, 0.4, color="#2ca02c", alpha=0.8, label=f"oracle-raw loss/depth (Σ={lo.sum():.3f})")
a3.bar(dd + 0.2, lc, 0.4, color="#9467bd", alpha=0.8, label=f"calib-raw 'recovered'/depth")
a3.plot(dd, np.cumsum(lo), "-o", color="#1a7a1a", ms=3, label="cumulative oracle-raw gap")
a3.set_xlabel("depth j of raw's FIRST error"); a3.set_ylabel("MAT loss attributed to depth j")
a3.set_title("[4] EXACT MAT-gap decomposition by first-error depth\n"
             "(early errors dominate -> front-loaded)")
a3.legend(fontsize=8); a3.grid(alpha=0.3)

fig.suptitle("Selection accuracy -> MAT: per-depth s_d, leverage, and exact gap "
             "decomposition  (Qwen3-14B)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=140)
print(f"\nwrote {OUT}")
