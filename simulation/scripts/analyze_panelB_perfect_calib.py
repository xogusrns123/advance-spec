"""Panel B, fully EMPIRICAL: binned mean(suffix==gt|DECISIVE) background + the
PERFECT-calib boundary from the ASSUMPTION-FREE marginal conditional
P(token==gt|score) = E[Y|score] (empirical binning, no monotone/GBM assumption).
Shaded lens between y=x and the perfect-calib boundary = the picks calibration
flips = the recovered region. Annotated with recovered accuracy / MAT."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
OUT = "simulation/results/calib_why_analysis/figures/panelB_perfect_calib.png"
ALIVE = {"eagle", "suffix", "both"}
BINS = 20; MIN_COUNT = 8

chains = defaultdict(list); step_acc = {}
for line in open(f"{DIR}/decisions_select1_oracle.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line); t = r.get("type")
    if t == "decision" and not r.get("tail"): chains[(r["rid"], r["decode_step"])].append(r)
    elif t == "step": step_acc[(r["rid"], r["decode_step"])] = r.get("accept_len")
for k in chains: chains[k].sort(key=lambda r: r["depth"])

ep, sp, corr, dkey = [], [], [], []
spp, sy, epp, ey = [], [], [], []
for (rid, ds), rs in chains.items():
    alive = True
    for r in rs:
        if not alive: break
        h = r.get("oracle_hit")
        if r["suffix_p"] is not None: spp.append(r["suffix_p"]); sy.append(1 if h in ("both","suffix") else 0)
        if r["eagle_p"] is not None: epp.append(r["eagle_p"]); ey.append(1 if h in ("both","eagle") else 0)
        if h in ("eagle","suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            ep.append(r["eagle_p"]); sp.append(r["suffix_p"]); corr.append(1 if h == "suffix" else 0)
            dkey.append((rid, ds, r["depth"]))
        if h not in ALIVE: alive = False
ep = np.array(ep); sp = np.array(sp); corr = np.array(corr)
spp = np.array(spp); sy = np.array(sy); epp = np.array(epp); ey = np.array(ey)

# assumption-free marginal conditional P(token==gt|score) via empirical binning
def binned(x, y, nb=25):
    e = np.linspace(0, 1, nb + 1); bid = np.clip(np.digitize(x, e[1:-1]), 0, nb - 1)
    rate = np.full(nb, np.nan)
    for b in range(nb):
        m = bid == b
        if m.sum() >= 5: rate[b] = y[m].mean()
    xi = (e[:-1] + e[1:]) / 2; ok = ~np.isnan(rate)
    rate = np.interp(xi, xi[ok], rate[ok])
    return lambda q: np.interp(q, xi, rate)
Psuf = binned(spp, sy); Peag = binned(epp, ey)
g = np.linspace(0, 1, 400)
Psg = Psuf(g)
pb = np.array([g[np.argmin(np.abs(Psg - Peag([e])[0]))] for e in g])    # perfect-calib boundary

# recovered loss: raw vs perfect-calib
raw_pick = (sp > ep).astype(int); perf_pick = (Psuf(sp) > Peag(ep)).astype(int)
raw_acc = (raw_pick == corr).mean(); perf_acc = (perf_pick == corr).mean()
def is_correct(pk, hit): return hit == "both" or (hit == "eagle" and pk == "eagle") or (hit == "suffix" and pk == "suffix")
def recon(pm):
    Ls = []
    for (rid, ds), rs in chains.items():
        s = step_acc.get((rid, ds))
        if s is None: continue
        L = s
        for r in rs:
            hit = r.get("oracle_hit")
            if hit not in ALIVE: L = r["depth"]; break
            pk = pm.get((rid, ds, r["depth"]), "eagle") if hit in ("eagle", "suffix") else "eagle"
            if not is_correct(pk, hit): L = r["depth"]; break
        Ls.append(L)
    return float(np.mean(Ls))
mat_raw = recon({dkey[i]: ("suffix" if raw_pick[i] else "eagle") for i in range(len(corr))})
mat_perf = recon({dkey[i]: ("suffix" if perf_pick[i] else "eagle") for i in range(len(corr))})
print(f"raw acc={raw_acc:.3f} MAT={mat_raw:.4f} | perfect acc={perf_acc:.3f} MAT={mat_perf:.4f}")
print(f"recovered: ACC=+{perf_acc-raw_acc:.3f}  MAT=+{mat_perf-mat_raw:.4f}")

# empirical binned background
s2, _, _ = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]], weights=corr)
c2, _, _ = np.histogram2d(ep, sp, bins=BINS, range=[[0, 1], [0, 1]])
with np.errstate(invalid="ignore"):
    Mm = s2 / c2
Mm[c2 < MIN_COUNT] = np.nan
cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#dddddd")

fig, ax = plt.subplots(figsize=(8.8, 7.2))
im = ax.imshow(np.ma.masked_invalid(Mm.T), origin="lower", extent=[0, 1, 0, 1],
               cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
fig.colorbar(im, ax=ax).set_label("EMPIRICAL mean(suffix==gt) per bin | DECISIVE")
ax.fill_between(g, g, pb, color="yellow", alpha=0.35, label="flip region (raw↔perfect)")
ax.plot([0, 1], [0, 1], "k-", lw=2, label="raw boundary {sp=ep}")
ax.plot(g, pb, color="magenta", lw=2.8, label="PERFECT-calib {E[suf=gt|sp]=E[eag=gt|ep]}")
ax.set_xlabel("eagle_p"); ax.set_ylabel("suffix_p"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Perfect calibration = true conditional E[token=gt|score] (empirical, no GBM).\n"
             f"yellow lens = recovered region:  +{perf_acc-raw_acc:.3f} acc  /  +{mat_perf-mat_raw:.3f} MAT")
ax.legend(fontsize=8.5, loc="lower right")
ax.text(0.03, 0.95, f"raw acc {raw_acc:.3f} -> perfect {perf_acc:.3f}\n"
        f"raw MAT {mat_raw:.3f} -> perfect {mat_perf:.3f}", transform=ax.transAxes,
        fontsize=8.5, va="top", bbox=dict(boxstyle="round", fc="#fff", ec="#999", alpha=0.9))
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
