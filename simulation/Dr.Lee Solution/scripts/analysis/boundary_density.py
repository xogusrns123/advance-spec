#!/usr/bin/env python3
"""TEAM-DECIDED unified metric: DFlash<->Suffix boundary density per 1K tokens.

Ownership is defined EXACTLY as the 3rd subplot of figures/traj_regions (which
proposer is favored per position): smoothed regions from run_cover(run_len=4) +
segments_from_hits(gap=12, min_seg=16); cat = s (suffix, priority) / d (DFlash) /
n (neither). A boundary is a directly adjacent {s,d} pair (transitions through n
are NOT counted -- same as traj_regions' n_bound). Aggregated per workload over
the same conv-concatenated units traj_regions builds.

Emits results/interp_validation/figures/boundary_density.png (bar) + boundary_density.json.

  docker exec sglang-bench bash -lc \
    "cd '/workspace/simulation/Dr.Lee Solution' && python3 scripts/boundary_density.py"
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import gzip, json, os, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_traj_warmcold import run_cover, segments_from_hits

RUN_LEN, GAP, MIN_SEG = 4, 12, 16
WL = {  # display order -> (curve name, traces record stem)
    "spider":    "perpos_spider_alleval/spider_4way",
    "swebench":  "perpos_swebench_alleval/swebench_4way",
    "bfcl":      "perpos_bfcl_full/bfcl_v4_full",
    "specbench": "perpos_specbench_full/specbench",
}
WLC = {"spider": "#4C78A8", "swebench": "#F58518", "bfcl": "#54A24B", "specbench": "#B279A2"}
IDIR = "results/interp_validation"


def mask(segs, n):
    m = [False] * n
    for s, e in segs:
        for i in range(s, e):
            m[i] = True
    return m


def unit_boundaries(S, A):
    """threshold-free winner-flip: winner = argmax(a, s) per position; ties (a==s,
    incl. both=0) are transparent; a boundary is where the strict winner flips."""
    n = len(S)
    win = []
    for si, ai in zip(S, A):
        if ai > si:
            win.append("d")
        elif si > ai:
            win.append("s")
        # tie -> skip
    nb = sum(1 for i in range(1, len(win)) if win[i] != win[i - 1])
    return nb, n


def main():
    out = []
    for wl, stem in WL.items():
        cur = {}
        with gzip.open(f"{IDIR}/curves_{wl}.jsonl.gz", "rt") as f:
            for l in f:
                r = json.loads(l)
                cur[r["rid"]] = r
        tr = json.load(open(f"results/{stem}.traces.json"))
        ev = {t["rid"]: t for t in tr["eval_traces"]}
        has_conv = any("conv" in t for t in tr["eval_traces"])
        # conv-concatenated units (traj_regions rule)
        units = {}
        for rid, cu in cur.items():
            t = ev.get(rid, {})
            uid = t.get("conv", rid) if has_conv else rid
            u = units.setdefault(uid, {"s": [], "a": []})
            u["s"].extend(cu["s"])
            u["a"].extend(max(v, 0) for v in cu["a"])
        tot_b = tot_n = 0
        for u in units.values():
            nb, n = unit_boundaries(u["s"], u["a"])
            tot_b += nb
            tot_n += n
        dens = 1000.0 * tot_b / (tot_n or 1)
        out.append(dict(wl=wl, boundaries=tot_b, tokens=tot_n,
                        n_units=len(units), bound_per_1k=dens))
        print(f"{wl:<10} boundaries={tot_b:>6}  tokens={tot_n:>7}  "
              f"units={len(units):>4}  bound/1K={dens:6.2f}")

    # ---- bar figure
    order = list(WL)
    vals = [next(o["bound_per_1k"] for o in out if o["wl"] == w) for w in order]
    cols = [WLC[w] for w in order]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    bars = ax.bar(order, vals, color=cols, edgecolor="k", linewidth=0.5, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.015, f"{v:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("winner-change boundaries per 1K tokens", fontsize=11.5)
    ax.set_xlabel("workload", fontsize=11)
    ax.set_title("Unified metric: DFlash<->Suffix winner-change boundary density\n"
                 "(winner = argmax(dflash, suffix) accept; boundary = strict winner flips; threshold-free)",
                 fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout()
    figdir = Path(IDIR) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fp = figdir / "boundary_density.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    json.dump(out, open(figdir.parent / "boundary_density.json", "w"), indent=1)
    print("saved ->", fp)


if __name__ == "__main__":
    main()
