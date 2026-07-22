#!/usr/bin/env python3
"""Warm/cold TRAJECTORY decomposition — the prerequisite for the real versions of
required_figures_synthesized_ver: "warm tree" / "cold tree" there are CONDITIONS on
trajectory REGIONS (where the suffix tree tracks vs misses), not tree-fill states.

For one representative sample per workload, plot the trajectory (x = committed
token position) with the per-position warm-suffix accept length
(suffix_match_warm = greedy accept length of the suffix continuation grafted at
that position, from the capture record), and mark the suffix-tracking regions:

  warm region ('run', default) = positions INSIDE a realized copy run of length
                >= --run-len: the union of [q, q+match_q) over positions q with
                suffix_match_warm >= run-len — i.e. where the tree would actually
                CARRY a multi-token tail, the regime the hand-off exploits.
                Alternatives: 'roll' (rolling-mean match >= --tau, sustained
                shallow tracking counts too) and 'raw' (match >= --m0).
                Gaps <= --gap are closed, islands < --min-seg dropped; cold = rest.

Vertical lines at every warm/cold boundary; warm regions shaded; call boundaries
(multi-call conversations) as gray dashed verticals. The chosen segmentation is
saved as JSON next to the figures so the conditional-accept / survival figures
can condition on EXACTLY the same regions.

Representative pick: the unit (conversation, or call when the record has no conv)
whose warm fraction is closest to the workload median, among units with
>= --min-len positions (ties -> longer unit).

  # bfcl: one figure per category            swe: pool all repos into one workload
  python3 scripts/plot_traj_warmcold.py --record results/perpos_bfcl_full/bfcl_v4_full.jsonl
  python3 scripts/plot_traj_warmcold.py --record results/perpos_swebench/swebench_quick.jsonl \
      --pool swebench
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
ORANGE, DARK_ORANGE, INK = "#F58518", "#B04A0F", "#333333"


def run_cover(series, L):
    """True at positions inside a realized copy run of length >= L: the union of
    [q, q+match_q) over q with match_q >= L."""
    cov = [False] * len(series)
    for q, v in enumerate(series):
        if v >= L:
            for i in range(q, min(len(series), q + v)):
                cov[i] = True
    return cov


def segments_from_hits(hits, gap, min_seg):
    """Contiguous True-runs; close gaps <= gap; drop runs < min_seg. [(s,e), ...) spans."""
    runs, s = [], None
    for i, h in enumerate(hits):
        if h and s is None:
            s = i
        elif not h and s is not None:
            runs.append((s, i)); s = None
    if s is not None:
        runs.append((s, len(hits)))
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_seg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--pool", default=None,
                    help="pool ALL units into ONE workload with this name "
                         "(default: one workload per task label)")
    ap.add_argument("--seg-signal", default="run", choices=["run", "roll", "raw"],
                    help="warm criterion: inside a >=run-len realized copy run (run), "
                         "rolling-mean >= tau (roll), or raw match >= m0 (raw)")
    ap.add_argument("--run-len", type=int, default=4,
                    help="min realized copy-run length for warm (seg-signal=run)")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="rolling-mean threshold for warm (seg-signal=roll)")
    ap.add_argument("--m0", type=int, default=1, help="hit threshold on raw match (seg-signal=raw)")
    ap.add_argument("--gap", type=int, default=12, help="close cold gaps <= this inside warm")
    ap.add_argument("--min-seg", type=int, default=16, help="drop warm islands shorter than this")
    ap.add_argument("--window", type=int, default=21, help="rolling-mean window")
    ap.add_argument("--min-len", type=int, default=150, help="min unit length for the pick")
    ap.add_argument("--unit-id", type=int, default=None,
                    help="force this unit (conv id, or rid when no conv) instead of the median pick")
    ap.add_argument("--prefix", default="traj_warmcold", help="figure filename prefix")
    args = ap.parse_args()

    rp = Path(args.record)
    traces = json.load(open(rp.with_suffix(".traces.json")))
    ev = traces["eval_traces"]
    has_conv = any("conv" in t for t in ev)

    match = defaultdict(dict)                      # rid -> {pos: suffix_match_warm}
    for l in open(rp):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        match[r["rid"]][r["pos"]] = int(r.get("suffix_match_warm", 0))

    # units: conversation (concatenated calls, temporal order) or single call
    units = {}                                     # uid -> {label, series, call_offsets, rids}
    for t in ev:
        rid = t["rid"]
        if rid not in match:
            continue
        uid = t.get("conv", rid) if has_conv else rid
        u = units.setdefault(uid, {"label": t.get("task", "all"), "series": [],
                                   "call_offsets": [], "rids": []})
        by_pos = match[rid]
        seq = [by_pos.get(p, 0) for p in range(1, max(by_pos) + 1)]
        u["call_offsets"].append(len(u["series"]))
        u["series"].extend(seq)
        u["rids"].append(rid)

    groups = defaultdict(list)                     # workload -> [uid]
    for uid, u in units.items():
        groups[args.pool or u["label"]].append(uid)

    fig_dir = BASE / "readable_outputs" / "figures" / "traj_warmcold"
    seg_dir = BASE / "results" / "warmcold_segments"
    fig_dir.mkdir(parents=True, exist_ok=True); seg_dir.mkdir(parents=True, exist_ok=True)

    for wl in sorted(groups):
        uids = groups[wl]
        def _hits(s):
            if args.seg_signal == "run":
                return run_cover(s, args.run_len)
            if args.seg_signal == "roll":
                half = args.window // 2
                return [sum(s[max(0, i - half): i + half + 1])
                        / len(s[max(0, i - half): i + half + 1]) >= args.tau
                        for i in range(len(s))]
            return [v >= args.m0 for v in s]

        wf = {}
        for uid in uids:
            s = units[uid]["series"]
            if len(s) >= args.min_len:
                wf[uid] = sum(_hits(s)) / len(s)
        if not wf:                                  # all units short: take the longest
            wf = {max(uids, key=lambda u: len(units[u]["series"])): 0.0}
        if args.unit_id is not None and args.unit_id in units:
            pick = args.unit_id
        else:
            med = sorted(wf.values())[len(wf) // 2]
            pick = min(wf, key=lambda u: (abs(wf[u] - med), -len(units[u]["series"])))
        u = units[pick]
        m = u["series"]
        N = len(m)
        segs = segments_from_hits(_hits(m), args.gap, args.min_seg)
        warm_share = sum(e - s for s, e in segs) / N if N else 0.0

        fig, ax = plt.subplots(figsize=(11.0, 3.4))
        prev = 0
        for s, e in segs:                           # warm orange / cold blue spans
            if s > prev:
                ax.axvspan(prev, s, color="#4C78A8", alpha=0.07, lw=0, zorder=0)
            ax.axvspan(s, e, color=ORANGE, alpha=0.14, lw=0, zorder=0)
            ax.axvline(s, color=INK, lw=0.9, alpha=0.75, zorder=3)
            ax.axvline(e, color=INK, lw=0.9, alpha=0.75, zorder=3)
            prev = e
        if prev < N:
            ax.axvspan(prev, N, color="#4C78A8", alpha=0.07, lw=0, zorder=0)
        for off in u["call_offsets"][1:]:
            ax.axvline(off, color="#999999", lw=0.8, ls="--", alpha=0.7, zorder=2)
        ax.fill_between(range(N), m, step="mid", color=ORANGE, alpha=0.45, lw=0,
                        zorder=1)
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=ORANGE, alpha=0.45, label="suffix accept len @pos (warm tree)"),
            Patch(facecolor=ORANGE, alpha=0.20, label="warm region (suffix tracks)"),
            Patch(facecolor="#4C78A8", alpha=0.12, label="cold region (suffix misses)"),
        ]
        ax.set_xlim(0, N)
        ax.set_ylim(0, max(4, min(traces.get("num_spec", 32), max(m) + 1)))
        ax.set_xlabel("trajectory position  (committed output tokens)", fontsize=9.5)
        ax.set_ylabel("suffix accept len", fontsize=9.5)
        uid_txt = f"conv {pick}" if has_conv else f"rid {pick}"
        crit = {"run": f"copy-run>={args.run_len}",
                "roll": f"roll(w={args.window})>={args.tau}",
                "raw": f"match>={args.m0}"}[args.seg_signal]
        ax.set_title(f"warm/cold decomposition — {wl}  ({uid_txt}, {len(u['rids'])} calls, "
                     f"{N} tok)   shaded = suffix-tracking (warm) {warm_share:.0%}, "
                     f"unshaded = cold   [{crit}, gap={args.gap}, min={args.min_seg}]",
                     fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right", ncol=2)
        fig.tight_layout()
        fp = fig_dir / f"{args.prefix}_{wl}.png"
        fig.savefig(fp, dpi=150); plt.close(fig)

        json.dump({"workload": wl, "record": str(rp), "unit": pick,
                   "rids": u["rids"], "call_offsets": u["call_offsets"], "n_pos": N,
                   "params": {"seg_signal": args.seg_signal, "run_len": args.run_len,
                              "tau": args.tau, "window": args.window, "m0": args.m0,
                              "gap": args.gap, "min_seg": args.min_seg},
                   "warm_share": warm_share, "warm_segments": segs},
                  open(seg_dir / f"{wl}.json", "w"))
        print(f"[{wl}] unit={pick} calls={len(u['rids'])} n_pos={N} "
              f"warm={warm_share:.0%} segs={len(segs)} -> {fp.name}")


if __name__ == "__main__":
    main()
