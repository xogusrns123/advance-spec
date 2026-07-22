#!/usr/bin/env python3
"""Boundary-density bar chart across 5 workloads, styled to MATCH the MAT graph
(plot_mat_raw.py): same figsize / palette / bold value labels / grid.

Metric = DFlash<->Suffix winner-change boundaries per 1K tokens (threshold-free:
winner = argmax(dflash, suffix) accept; ties transparent; boundary = strict winner
flip). Aggregated per workload over conv-concatenated units.

  docker exec sglang-bench bash -lc \
    "cd '/workspace/simulation/Dr.Lee Solution' && python3 scripts/plot_boundary_density_bars.py"
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import gzip, json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IDIR = "results/interp_validation"
FIGDIR = Path(IDIR) / "figures"
# display order requested by the user
WL = [("specbench", "perpos_specbench_full/specbench"),
      ("bfcl", "perpos_bfcl_full/bfcl_v4_full"),
      ("swebench", "perpos_swebench_alleval/swebench_4way"),
      ("spider", "perpos_spider_alleval/spider_4way"),
      ("tau2", "perpos_tau2_alleval/tau2_4way")]
LABEL = {"specbench": "specbench", "bfcl": "bfclv4", "swebench": "swebench",
         "spider": "spider", "tau2": "tau2-bench"}
C_OURS = "#F58518"   # same orange as "Ours" in the MAT graph


def load_curves(p):
    o = {}
    with gzip.open(p, "rt") as f:
        for l in f:
            r = json.loads(l)
            o[r["rid"]] = r
    return o


def winner_flips(S, A):
    win = []
    for si, ai in zip(S, A):
        if ai > si:
            win.append("d")
        elif si > ai:
            win.append("s")
        # tie -> transparent
    return sum(1 for i in range(1, len(win)) if win[i] != win[i - 1]), len(S)


rows = []
for wl, stem in WL:
    cur = load_curves(f"{IDIR}/curves_{wl}.jsonl.gz")
    tr = json.load(open(f"results/{stem}.traces.json"))
    ev = {t["rid"]: t for t in tr["eval_traces"]}
    has_conv = any("conv" in t for t in tr["eval_traces"])
    units = {}
    for rid, cu in cur.items():
        t = ev.get(rid, {})
        uid = t.get("conv", rid) if has_conv else rid
        u = units.setdefault(uid, {"s": [], "a": []})
        u["s"].extend(cu["s"])
        u["a"].extend(max(v, 0) for v in cu["a"])
    B = N = 0
    for u in units.values():
        b, n = winner_flips(u["s"], u["a"])
        B += b; N += n
    dens = 1000.0 * B / (N or 1)
    rows.append((wl, dens))
    print(f"{wl:<10} boundaries={B:>6} tokens={N:>8} bound/1K={dens:6.2f}")

names = [LABEL[w] for w, _ in rows]
vals = [d for _, d in rows]

fig, ax = plt.subplots(figsize=(9.2, 5.6))
bars = ax.bar(names, vals, color=C_OURS, edgecolor="k", linewidth=0.5,
              width=0.62, zorder=3)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.015,
            f"{b.get_height():.1f}", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold")
ax.set_ylabel("DFlash ↔ Suffix winner-change boundaries / 1K tokens", fontsize=11)
ax.set_xlabel("workload", fontsize=11)
ax.set_title("Boundary density  —  winner = argmax(dflash, suffix) accept",
             fontsize=12.5, fontweight="bold")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=11)
ax.grid(axis="y", alpha=0.2, zorder=0)
ax.set_ylim(0, max(vals) * 1.18)
fig.tight_layout()
FIGDIR.mkdir(parents=True, exist_ok=True)
fp = FIGDIR / "boundary_density_5wl.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
