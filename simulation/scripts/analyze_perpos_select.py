#!/usr/bin/env python3
"""Per-position (depth) proposer-selection ratio for the 3-way select-1 study.

Restricted to DECISIVE positions: among the AVAILABLE proposers (token present),
some-but-not-all hit gt -- this excludes the trivial 'all correct' and 'all wrong'
rows where the pick does not matter. For every depth we report, over the decisive
rows at that depth:
  - raw pick share  : fraction where the raw argmax-prob selector picks proposer X
                      (one pick per row -> the three shares sum to 1)
  - oracle-hit share: fraction where proposer X hits gt (a valid pick; a row may
                      have >1 hitter, so these do NOT sum to 1) -- the "should pick"
                      ceiling. The raw-vs-oracle gap per position = selection head-
                      room the calibrator targets.
One panel per proposer; figure written to <run-dir>/figures/.

Reads the gt-forced ORACLE arm (decisions_select1_oracle.jsonl): the block is
teacher-forced to gt so every depth is a real gt-path decision (not truncated at
the first mismatch), which is exactly what a per-position distribution needs.

  python3 simulation/scripts/analyze_perpos_select.py            # both cells
  python3 simulation/scripts/analyze_perpos_select.py --cell 27b
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# run-dir (relative to --root) + proposer -> (token field, prob field) layout.
# 8B : DFlash is the served main (eagle_*), EAGLE3 is the in-process aux (e3_*).
# 27B: MTP is the served main (eagle_*), DFlash merged on the gt path (dflash_*).
CELLS = {
    "27b": {"dir": "qwen35_27b_3way_real_full", "model": "Qwen3.5-27B", "maxd": 16,
            "names": ("mtp", "dflash", "suffix"),
            "tok": {"mtp": "eagle_token", "dflash": "dflash_token", "suffix": "suffix_token"},
            "p":   {"mtp": "eagle_p",     "dflash": "dflash_p",     "suffix": "suffix_p"}},
    "8b":  {"dir": "qwen3_8b_dflash_e3_ceiling20", "model": "Qwen3-8B", "maxd": 15,
            "names": ("dflash", "e3", "suffix"),
            "tok": {"dflash": "eagle_token", "e3": "e3_token", "suffix": "suffix_token"},
            "p":   {"dflash": "eagle_p",     "e3": "e3_p",     "suffix": "suffix_p"}},
}
COLORS = {"mtp": "#1f77b4", "dflash": "#2ca02c", "e3": "#ff7f0e", "suffix": "#8c564b"}

# REALIZED (live self-rollout) arm: decisions_select1.jsonl. The committed
# trajectory follows the ACTUAL raw-argmax selection, so there is no clean gt for
# most positions (27B logs none; 8B only ~17% on the on-gt prefix) -> we cannot
# compute an "is correct / decisive" view. What IS clean is the realized SELECTION
# share from the `chosen` field (chosen==argmax prob in 97-100% of rows). Decisive's
# gt-free analog is "disagreement" (the emitted proposers do not all propose the
# same token -> the pick actually matters).
CELLS_REALIZED = {
    "27b": {"dir": "qwen35_27b_3way_real_full", "model": "Qwen3.5-27B", "maxd": 16,
            "names": ("mtp", "dflash", "suffix"),
            "tok": {"mtp": "eagle_token", "dflash": "dflash_token", "suffix": "suffix_token"},
            "chosen": {"eagle3": "mtp", "dflash": "dflash", "suffix": "suffix"}},
    "8b":  {"dir": "qwen3_8b_3way_realserve", "model": "Qwen3-8B", "maxd": 15,
            "names": ("dflash", "e3", "suffix"),
            "tok": {"dflash": "eagle_token", "e3": "e3_token", "suffix": "suffix_token"},
            "chosen": {"dflash": "dflash", "e3": "e3", "suffix": "suffix"}},
}


def analyze(path, cell):
    names, TOK, PK, maxd = cell["names"], cell["tok"], cell["p"], cell["maxd"]
    n_dec = np.zeros(maxd, dtype=int)
    raw = {p: np.zeros(maxd) for p in names}
    orc = {p: np.zeros(maxd) for p in names}
    total = 0
    for line in open(path):
        o = json.loads(line)
        if o.get("type") != "decision" or o.get("tail"):
            continue
        gt = o.get("gt_token")
        d = o.get("depth")
        if gt is None or d is None or d >= maxd:
            continue
        toks = {p: o.get(TOK[p]) for p in names}
        probs = {p: (o.get(PK[p]) if o.get(PK[p]) is not None else -1.0) for p in names}
        emitted = [p for p in names if toks[p] is not None]
        hits = [p for p in emitted if toks[p] == gt]
        # Decisive over the FULL set of proposers as fixed slots: a proposer that
        # did not emit a token here counts as not-correct AND not-selected (esp.
        # suffix). The position stays in the denominator; only 'all N correct'
        # (every proposer emitted and hit) and 'none correct' are excluded.
        if not (1 <= len(hits) <= len(names) - 1):     # decisive only
            continue
        n_dec[d] += 1; total += 1
        # raw argmax pick among proposers that actually emitted (absent can't win)
        raw[max(emitted, key=lambda p: probs[p])][d] += 1
        for p in hits:
            orc[p][d] += 1
    return n_dec, raw, orc, total


def plot(cell_key, root):
    cell = CELLS[cell_key]
    rundir = Path(root) / cell["dir"]
    n_dec, raw, orc, total = analyze(rundir / "decisions_select1_oracle.jsonl", cell)
    names, maxd = cell["names"], cell["maxd"]
    x = np.arange(maxd)
    denom = np.where(n_dec > 0, n_dec, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    share = {"oracle-hit share (is correct)": orc, "raw pick share (selected)": raw}
    for ax, (title, data) in zip(axes, share.items()):
        for p in names:
            y = data[p] / denom; y[n_dec == 0] = np.nan
            ax.plot(x, y, "-o", color=COLORS.get(p, "#333"), lw=2.2, ms=5, label=p)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("position (depth in block)")
        ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc="upper right", title="proposer")
    axes[0].set_ylabel("share of decisive positions")
    fig.suptitle(f"Per-position proposer selection ({'+'.join(names)}, {cell['model']}, "
                 f"decisive only)   ·  total decisive rows = {total:,}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figdir = rundir / "figures"; figdir.mkdir(exist_ok=True)
    out = figdir / f"perpos_select_3way_{cell_key}.png"
    fig.savefig(out, dpi=120); plt.close(fig)

    print(f"\n[{cell_key}] {cell['model']}  proposers={names}  total decisive={total}")
    print("depth :", " ".join(f"{i:4d}" for i in range(maxd)))
    print("n_dec :", " ".join(f"{int(v):4d}" for v in n_dec))
    for p in names:
        sh = raw[p] / denom; sh[n_dec == 0] = np.nan
        print(f"raw {p:7s}:", " ".join((f"{v:4.2f}" if not np.isnan(v) else "  - ") for v in sh))
    print(f"-> {out}")


def analyze_realized(path, cell):
    """Realized SELECTION share per depth from the live `chosen` field.
    Returns n_all/sel_all (over every drafted position) and n_dis/sel_dis
    (over 'disagreement' positions where the emitted proposers do not all agree
    -> the gt-free analog of 'the pick matters')."""
    names, TOK, CH, maxd = cell["names"], cell["tok"], cell["chosen"], cell["maxd"]
    n_all = np.zeros(maxd, dtype=int); n_dis = np.zeros(maxd, dtype=int)
    sel_all = {p: np.zeros(maxd) for p in names}
    sel_dis = {p: np.zeros(maxd) for p in names}
    tot_all = tot_dis = 0
    for line in open(path):
        o = json.loads(line)
        if o.get("type") != "decision" or o.get("tail"):
            continue
        d = o.get("depth")
        if d is None or d >= maxd:
            continue
        p = CH.get(o.get("chosen"))
        if p not in names:
            continue
        emitted_toks = [o.get(TOK[q]) for q in names if o.get(TOK[q]) is not None]
        disagree = len(set(emitted_toks)) > 1
        n_all[d] += 1; sel_all[p][d] += 1; tot_all += 1
        if disagree:
            n_dis[d] += 1; sel_dis[p][d] += 1; tot_dis += 1
    return names, n_all, sel_all, n_dis, sel_dis, tot_all, tot_dis


def plot_realized(cell_key, root):
    cell = CELLS_REALIZED[cell_key]
    rundir = Path(root) / cell["dir"]
    names, n_all, sel_all, n_dis, sel_dis, tot_all, tot_dis = analyze_realized(
        rundir / "decisions_select1.jsonl", cell)
    maxd = cell["maxd"]; x = np.arange(maxd)
    panels = [("selection share — all positions", n_all, sel_all),
              ("selection share — disagreement only (pick matters)", n_dis, sel_dis)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (title, nd, data) in zip(axes, panels):
        den = np.where(nd > 0, nd, 1)
        for p in names:
            y = data[p] / den; y[nd == 0] = np.nan
            ax.plot(x, y, "-o", color=COLORS.get(p, "#333"), lw=2.2, ms=5, label=p)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("position (depth in block)")
        ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc="upper right", title="proposer")
    axes[0].set_ylabel("realized selection share (chosen)")
    fig.suptitle(f"Per-position proposer selection — REALIZED live self-rollout "
                 f"({'+'.join(names)}, {cell['model']})   ·  no gt in this arm, so "
                 f"selection share only (chosen==argmax)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figdir = rundir / "figures"; figdir.mkdir(exist_ok=True)
    out = figdir / f"perpos_select_3way_realized_{cell_key}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n[{cell_key} REALIZED] {cell['model']}  all={tot_all}  disagree={tot_dis}")
    print("depth       :", " ".join(f"{i:4d}" for i in range(maxd)))
    print("n_disagree  :", " ".join(f"{int(v):4d}" for v in n_dis))
    for p in names:
        sh = sel_dis[p] / np.where(n_dis > 0, n_dis, 1); sh[n_dis == 0] = np.nan
        print(f"sel {p:7s}:", " ".join((f"{v:4.2f}" if not np.isnan(v) else "  - ") for v in sh))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="simulation/results/chain_hybrid_perdepth")
    ap.add_argument("--cell", choices=["8b", "27b", "both"], default="both")
    ap.add_argument("--arm", choices=["oracle", "realized", "both"], default="oracle")
    args = ap.parse_args()
    cells = ["27b", "8b"] if args.cell == "both" else [args.cell]
    for k in cells:
        if args.arm in ("oracle", "both"):
            plot(k, args.root)
        if args.arm in ("realized", "both"):
            plot_realized(k, args.root)


if __name__ == "__main__":
    main()
