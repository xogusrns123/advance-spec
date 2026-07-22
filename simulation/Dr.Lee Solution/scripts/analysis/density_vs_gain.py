# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import os
#!/usr/bin/env python3
"""Does the team's boundary-DENSITY metric predict compose gain? Compare it against
boundary-VALUE metrics on the same 21 task units.

  density   traj_regions {s,d} boundaries per 1K tok  (team's chosen metric)
  C         sqrt(suffix_only * dflash_only)           (complementary coverage)
  valdens   density weighted by value unlocked per crossing:
            per {s,d} boundary, min(mean suffix-accept on s side,
            mean dflash-accept on d side); summed per 1K  (Kim-form, value-weighted)

Correlated (Spearman) with g_drlee / g_kim from units.json + emits a figure.
"""
import gzip, json, os, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_traj_warmcold import run_cover, segments_from_hits

RUN_LEN, GAP, MIN_SEG, TH = 4, 12, 16, 4
WL = {"spider": "perpos_spider_alleval/spider_4way", "swebench": "perpos_swebench_alleval/swebench_4way",
      "bfcl": "perpos_bfcl_full/bfcl_v4_full", "specbench": "perpos_specbench_full/specbench"}
WLC = {"spider": "#4C78A8", "swebench": "#F58518", "bfcl": "#54A24B", "specbench": "#B279A2"}
IDIR = os.environ.get("INTERP_DIR", "results/interp_validation")


def mask(segs, n):
    m = [False] * n
    for s, e in segs:
        for i in range(s, e):
            m[i] = True
    return m


def unit_stats(S, A):
    """boundaries, value-weighted boundaries, tokens, so, do for one trajectory."""
    n = len(S)
    seg_s = segments_from_hits(run_cover(S, RUN_LEN), GAP, MIN_SEG)
    seg_a = segments_from_hits(run_cover(A, RUN_LEN), GAP, MIN_SEG)
    m_s, m_a = mask(seg_s, n), mask(seg_a, n)
    cat = ["s" if ms else ("d" if ma else "n") for ms, ma in zip(m_s, m_a)]
    runs, st = [], 0
    for i in range(1, n + 1):
        if i == n or cat[i] != cat[st]:
            runs.append([st, i, cat[st]]); st = i
    nb = 0
    valb = 0.0
    for i in range(1, len(runs)):
        a_run, b_run = runs[i - 1], runs[i]
        if {a_run[2], b_run[2]} != {"s", "d"}:
            continue
        nb += 1
        s_run = a_run if a_run[2] == "s" else b_run
        d_run = a_run if a_run[2] == "d" else b_run
        s_depth = sum(S[j] for j in range(*s_run[:2])) / max(1, s_run[1] - s_run[0])
        d_depth = sum(A[j] for j in range(*d_run[:2])) / max(1, d_run[1] - d_run[0])
        valb += min(s_depth, d_depth)          # value a single crossing unlocks
    so = sum(1 for i in range(n) if S[i] >= TH and A[i] < TH)
    do = sum(1 for i in range(n) if A[i] >= TH and S[i] < TH)
    return nb, valb, n, so, do


def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        od = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j < n and v[od[j]] == v[od[i]]:
                j += 1
            for k in range(i, j):
                r[od[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


units = [u for u in json.load(open(f"{IDIR}/units.json")) if u["task"] != "__all__"]
rows = []
for wl, stem in WL.items():
    cur = {}
    with gzip.open(f"{IDIR}/curves_{wl}.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l); cur[r["rid"]] = r
    by_task = defaultdict(list)
    for rid, c in cur.items():
        by_task[c.get("task") or "all"].append(rid)
    for u in units:
        if u["wl"] != wl:
            continue
        rids = by_task.get(u["task"])
        if not rids:
            continue
        B = V = N = SO = DO = 0.0
        for rid in rids:
            c = cur[rid]
            A = [max(v, 0) for v in c["a"]]
            nb, vb, n, so, do = unit_stats(c["s"], A)
            B += nb; V += vb; N += n; SO += so; DO += do
        N = N or 1
        rows.append(dict(wl=wl, task=u["task"], g_drlee=u["g_drlee"], g_kim=u["g_kim"],
                         density=1000.0 * B / N, valdens=1000.0 * V / N,
                         C=((SO / N) * (DO / N)) ** 0.5))

METS = ["density", "valdens", "C"]
print(f"{'metric':<10}{'rho vs Lee':>12}{'rho vs Kim':>12}")
corr = {}
for m in METS:
    rl = spearman([r[m] for r in rows], [r["g_drlee"] for r in rows])
    rk = spearman([r[m] for r in rows], [r["g_kim"] for r in rows])
    corr[m] = (rl, rk)
    print(f"{m:<10}{rl:>12.2f}{rk:>12.2f}")

# figure: density vs gain (fails) beside C vs gain (works)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
LAB = {"density": "boundary density /1K (team metric)",
       "valdens": "value-weighted boundary density /1K",
       "C": "C = sqrt(suffix_only x dflash_only)"}
for ax, m in zip(axes, METS):
    for r in rows:
        ax.scatter(r[m], r["g_kim"], s=55, c=WLC[r["wl"]], edgecolor="k",
                   linewidth=0.4, alpha=0.9, zorder=3)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel(LAB[m], fontsize=9.5)
    ax.set_title(f"rho vs Kim gain = {corr[m][1]:+.2f}", fontsize=12, fontweight="bold",
                 color=("#1a7d1a" if abs(corr[m][1]) >= 0.6 else "#b00"))
    ax.grid(True, alpha=0.2)
axes[0].set_ylabel("Kim gain: compose - suffix hybrid", fontsize=10)
handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=8, label=w)
           for w, c in WLC.items()]
axes[0].legend(handles=handles, fontsize=8.5, loc="upper left")
fig.suptitle("Boundary DENSITY does not predict gain; boundary VALUE does  (21 task units)",
             fontsize=13, y=1.02, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.98])
fp = Path(IDIR) / "figures" / "density_vs_gain.png"
fig.savefig(fp, dpi=145, bbox_inches="tight")
print("saved ->", fp)
