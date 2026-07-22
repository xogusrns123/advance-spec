#!/usr/bin/env python3
"""Kim's prediction, visualized: one representative sample per workload, the
trajectory decomposed three ways (traj_warmcold style, same segmentation
machinery), one figure per workload with 3 shared-x subplots:

  1. Suffix (Memorizer):  warm region (suffix tracks) vs cold region — s(p)
  2. DFlash (Predictor):  PREDICTABLE region vs unpredictable — a(p)
     (deliberately not named warm/cold: it is model predictability, not memory)
  3. Suffix + DFlash:     per-position ownership — Suffix region (memorized;
     tail rides deep) / DFlash region (predictable but not memorized; head
     bridges) / uncovered. Every Suffix|DFlash adjacency is a boundary that
     composition packs into ONE verify step — the events Kim's story monetizes.

Signals come from the arm-independent interp_validation curves (s(p) = realized
suffix copy depth WITH eval-time warming, a(p) = DFlash block leading-match
run), so the regions are exactly the structures the replay arms lived in.
Color semantics are global: orange=suffix, blue=DFlash, gray=neither
(the original traj_warmcold shaded cold blue; here blue is reserved for DFlash).

  PYTHONPATH=/workspace python3 scripts/plot_traj_regions.py \
      --dir results/interp_validation --record results/perpos_spider_alleval/spider_4way.jsonl \
      --name spider
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "legend.title_fontsize": "medium",
})

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from plot_traj_warmcold import run_cover, segments_from_hits  # noqa: E402

ORANGE, BLUE, GRAY, INK = "#F58518", "#4C78A8", "#9a9a9a", "#333333"


def mask_from_segs(segs, n):
    m = [False] * n
    for s, e in segs:
        for i in range(s, e):
            m[i] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--record", required=True, help="record path (for .traces.json conv/task)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--run-len", type=int, default=4)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--min-seg", type=int, default=16)
    ap.add_argument("--min-len", type=int, default=150)
    ap.add_argument("--window", type=int, default=500,
                    help="fixed x-axis length (tokens); every workload is cropped "
                         "to this so all figures share the same token density")
    ap.add_argument("--unit-id", type=int, default=None)
    args = ap.parse_args()

    traces = json.load(open(Path(args.record).with_suffix(".traces.json")))
    ev = {t["rid"]: t for t in traces["eval_traces"]}
    has_conv = any("conv" in t for t in traces["eval_traces"])

    curves = {}
    with gzip.open(os.path.join(args.dir, f"curves_{args.name}.jsonl.gz"), "rt") as f:
        for l in f:
            r = json.loads(l)
            curves[r["rid"]] = r

    # units: conversation-concatenated calls (temporal order) or single call
    units = {}
    for rid, cu in curves.items():
        t = ev.get(rid, {})
        uid = t.get("conv", rid) if has_conv else rid
        u = units.setdefault(uid, {"label": cu.get("task") or "all", "s": [],
                                   "a": [], "call_offsets": [], "rids": []})
        u["call_offsets"].append(len(u["s"]))
        u["s"].extend(cu["s"])
        u["a"].extend(max(v, 0) for v in cu["a"])
        u["rids"].append(rid)

    # representative pick: warm fraction closest to the median (traj_warmcold rule),
    # among units long enough to fill the fixed window (fallback: the longest unit).
    need = max(args.min_len, args.window)
    wf = {}
    for uid, u in units.items():
        if len(u["s"]) >= need:
            wf[uid] = sum(run_cover(u["s"], args.run_len)) / len(u["s"])
    if not wf:
        wf = {max(units, key=lambda x: len(units[x]["s"])): 0.0}
    if args.unit_id is not None and args.unit_id in units:
        pick = args.unit_id
    else:
        med = sorted(wf.values())[len(wf) // 2]
        pick = min(wf, key=lambda x: (abs(wf[x] - med), -len(units[x]["s"])))
    u = units[pick]
    # crop to a fixed WINDOW so every workload is shown at the SAME token density
    S, A = u["s"][:args.window], u["a"][:args.window]
    N = len(S)
    u["call_offsets"] = [o for o in u["call_offsets"] if o < N]

    segs_s = segments_from_hits(run_cover(S, args.run_len), args.gap, args.min_seg)
    segs_a = segments_from_hits(run_cover(A, args.run_len), args.gap, args.min_seg)
    m_s, m_a = mask_from_segs(segs_s, N), mask_from_segs(segs_a, N)
    # WINNER ownership (threshold-free): the winner is the proposer with the higher
    # accept length -- s>a suffix, a>s dflash. A tie (a==s: both fail=0, or both
    # succeed equally) has no distinct winner and does NOT flip ownership: it just
    # carries forward the previous winner's region (the incumbent rides through).
    raw = [("d" if ai > si else "s" if si > ai else None) for si, ai in zip(S, A)]
    first = next((w for w in raw if w), "s")
    cat, prev = [], first
    for w in raw:
        prev = w or prev
        cat.append(prev)
    runs, st = [], 0
    for i in range(1, N + 1):
        if i == N or cat[i] != cat[st]:
            runs.append((st, i, cat[st])); st = i
    n_bound = sum(1 for i in range(1, len(runs)) if runs[i][2] != runs[i - 1][2])
    cov = {c: sum(1 for x in cat if x == c) / N for c in "sd"}

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.8, 2.8, 1.15]))

    def panel(ax, series, segs, col, ylab, seg_label, off_label, rest_label):
        prev = 0
        for s, e in segs:
            if s > prev:
                ax.axvspan(prev, s, color=GRAY, alpha=0.10, lw=0, zorder=0)
            ax.axvspan(s, e, color=col, alpha=0.15, lw=0, zorder=0)
            ax.axvline(s, color=INK, lw=0.8, alpha=0.6, zorder=3)
            ax.axvline(e, color=INK, lw=0.8, alpha=0.6, zorder=3)
            prev = e
        if prev < N:
            ax.axvspan(prev, N, color=GRAY, alpha=0.10, lw=0, zorder=0)
        ax.fill_between(range(N), series, step="mid", color=col, alpha=0.5,
                        lw=0, zorder=1)
        for off in u["call_offsets"][1:]:
            ax.axvline(off, color="#999999", lw=0.8, ls="--", alpha=0.7, zorder=2)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.set_ylim(0, max(6, max(series) * 1.08))
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        share = sum(e - s for s, e in segs) / N
        handles = [Patch(facecolor=col, alpha=0.5, label=f"{off_label} accept len @pos"),
                   Patch(facecolor=col, alpha=0.18, label=f"{seg_label} {share:.0%}"),
                   Patch(facecolor=GRAY, alpha=0.14,
                         label=f"{rest_label} {1 - share:.0%}")]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right",
                  ncol=3)

    panel(axes[0], S, segs_s, ORANGE, "suffix accept len",
          "warm region (suffix tracks)", "suffix",
          "cold region (suffix misses)")
    panel(axes[1], A, segs_a, BLUE, "DFlash accept len",
          "predictable region (DFlash foresees)", "DFlash",
          "unpredictable region")

    ax = axes[2]
    colmap = {"s": (ORANGE, 0.7), "d": (BLUE, 0.7)}
    for s, e, c in runs:
        ax.axvspan(s, e, color=colmap[c][0], alpha=colmap[c][1], lw=0)
    for off in u["call_offsets"][1:]:
        ax.axvline(off, color="#666666", lw=0.8, ls="--", alpha=0.8)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, args.window)          # fixed scale -> same density across workloads
    ax.set_xlabel("trajectory position (committed output tokens)", fontsize=10)
    handles = [
        Patch(facecolor=ORANGE, alpha=0.7, label=f"suffix wins (s>a) {cov['s']:.0%}"),
        Patch(facecolor=BLUE, alpha=0.7, label=f"dflash wins (a>s) {cov['d']:.0%}"),
    ]
    ax.legend(handles=handles, fontsize=8.3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.55), ncol=2)
    ax.set_title(f"winner changes = {n_bound}", fontsize=11)

    uid_txt = f"conv {pick}" if has_conv else f"rid {pick}"
    axes[0].set_title(
        f"{args.name} — one representative sample ({uid_txt}, task={u['label']}, "
        f"{len(u['rids'])} calls, first {N} of {args.window}-tok window)   "
        f"[copy-run>={args.run_len}, gap={args.gap}, min={args.min_seg}]",
        fontsize=10.5)
    fig.tight_layout()
    out_dir = Path(args.dir) / "figures" / "traj_regions"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"traj_regions_{args.name}.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"[{args.name}] unit={pick} task={u['label']} calls={len(u['rids'])} "
          f"N={N} | suffix {cov['s']:.0%} dflash {cov['d']:.0%} "
          f"| winner-changes={n_bound} ({1000.0*n_bound/N:.1f}/1K) -> {fp}")


if __name__ == "__main__":
    main()
