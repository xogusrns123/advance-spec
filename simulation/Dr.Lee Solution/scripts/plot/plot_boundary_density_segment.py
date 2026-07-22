#!/usr/bin/env python3
"""Per-SEGMENT DFlash<->Suffix boundary density (team metric), on the current
5-workload 4-way records. Boundary = position where the strict winner flips,
winner = argmax(dflash_accept, suffix_accept) per position (ties transparent),
exactly as scripts/boundary_density.py — but split by output segment
(think / tool_call / final) using the same tagger as the segment MAT figures.

Per position:  a = leading-1 run of dflash_match (DFlash accept length)
               s = suffix_match_warm            (Suffix accept length)
Flips are counted between consecutive NON-TIE positions inside the same
(rid, segment) run; density = 1000 * flips / positions-in-segment.

Output: figures/mat/BOUNDARY_density_by_segment.png  (grouped: x=segment, per workload)

  python3 scripts/plot_boundary_density_segment.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from replay_extension import _ad  # noqa: E402
from replay_segments_5way import tag, piece_offsets  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "readable_outputs" / "figures" / "mat" / "boundary_gap"

WL = {"specbench": "perpos_specbench_alleval/specbench_4way",
      "bfcl": "perpos_bfcl_alleval/bfcl_4way",
      "swebench": "perpos_swebench_alleval/swebench_4way",
      "spider": "perpos_spider_alleval/spider_4way",
      "tau2": "perpos_tau2_alleval/tau2_4way"}
WL_LABEL = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
            "spider": "Spider2-DBT", "tau2": "τ²-bench"}
WLC = {"specbench": "#B279A2", "bfcl": "#54A24B", "swebench": "#F58518",
       "spider": "#4C78A8", "tau2": "#E45756"}
SEG_ORDER = ["think", "tool_call", "final"]
SEG_LABEL = {"think": "reasoning\n(think)", "tool_call": "tool call",
             "final": "text: response\n(final)"}


def winner(a, s):
    return "d" if a > s else ("s" if s > a else None)   # tie -> None (transparent)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    OUT.mkdir(parents=True, exist_ok=True)

    # dens[wl][seg] = (flips, tokens)
    dens = {}
    for wl, stem in WL.items():
        rec_path = BASE / "results" / f"{stem}.jsonl"
        tr_path = BASE / "results" / f"{stem}.traces.json"
        if not rec_path.exists() or not tr_path.exists():
            print(f"[{wl}] record missing, skip")
            continue
        tr = json.load(open(tr_path))
        ev = {t["rid"]: t for t in tr["eval_traces"]}
        recs = defaultdict(dict)
        for l in open(rec_path):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            recs[r["rid"]][r["pos"]] = r
        agg = defaultdict(lambda: [0, 0])          # seg -> [flips, tokens]
        for rid, rby in recs.items():
            t = ev.get(rid)
            if t is None:
                continue
            offs, full = piece_offsets(t["output_ids"], tok)
            cats = tag(full, offs, wl)
            # winners per segment, in output-position order
            seg_win = defaultdict(list)
            for pos, r in sorted(rby.items()):
                gi = pos - 1                        # output index of this round root
                if not (0 <= gi < len(cats)):
                    continue
                a = _ad(r["dflash_match"])
                s = int(r.get("suffix_match_warm", 0) or 0)
                seg = cats[gi]
                agg[seg][1] += 1                    # a captured position in this seg
                w = winner(a, s)
                if w is not None:
                    seg_win[seg].append(w)
            for seg, ws in seg_win.items():
                agg[seg][0] += sum(1 for i in range(1, len(ws)) if ws[i] != ws[i - 1])
        dens[wl] = {seg: (f, n) for seg, (f, n) in agg.items()}
        print(f"[{wl}] " + "  ".join(
            f"{seg}={1000 * dens[wl][seg][0] / dens[wl][seg][1]:.1f}/1K(n={dens[wl][seg][1]})"
            for seg in SEG_ORDER if seg in dens[wl]))

    # ---- grouped bars: x = segment, groups = workload ----
    wls = [w for w in WL if w in dens]
    segs = [s for s in SEG_ORDER if any(s in dens[w] for w in wls)]
    fig, ax = plt.subplots(figsize=(2.6 + 2.2 * len(segs), 5.4))
    n, g = len(wls), 0.82
    bw = g / n
    ymax = 0.0
    for wi, wl in enumerate(wls):
        xs, ys = [], []
        for si, seg in enumerate(segs):
            f, tot = dens[wl].get(seg, (0, 0))
            xs.append(si - g / 2 + bw * (wi + 0.5))
            ys.append(1000.0 * f / tot if tot else 0.0)
        ymax = max(ymax, max(ys) if ys else 0)
        ax.bar(xs, ys, width=bw * 0.9, color=WLC[wl], label=WL_LABEL[wl])
        for x, y in zip(xs, ys):
            if y > 0:
                ax.text(x, y + ymax * 0.012, f"{y:.0f}", ha="center", va="bottom",
                        fontsize=8, color="#555555")
    ax.set_ylim(0, ymax * 1.16)
    ax.set_xticks(range(len(segs)))
    ax.set_xticklabels([SEG_LABEL[s] for s in segs], fontsize=12)
    ax.set_ylabel("DFlash↔Suffix winner-change\nboundaries per 1K tokens", fontsize=11)
    ax.set_title("Boundary density by output segment\n"
                 "(winner = argmax(DFlash, Suffix) accept; boundary = strict winner flip)",
                 fontsize=12)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fp = OUT / "BOUNDARY_density_by_segment.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")

    # ---- one standalone image per workload ----
    for wl in wls:
        wsegs = [s for s in SEG_ORDER if s in dens[wl]]
        vals = [1000.0 * dens[wl][s][0] / dens[wl][s][1] if dens[wl][s][1] else 0.0
                for s in wsegs]
        ns = [dens[wl][s][1] for s in wsegs]
        ymx = max(vals) if vals else 1.0
        fig, ax = plt.subplots(figsize=(max(5.5, 2.2 + 1.9 * len(wsegs)), 5.0))
        xs = list(range(len(wsegs)))
        ax.bar(xs, vals, width=0.6, color=WLC[wl], zorder=3)
        for x, v, nn in zip(xs, vals, ns):
            ax.text(x, v + ymx * 0.015, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
        ax.set_ylim(0, ymx * 1.18)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{SEG_LABEL[s]}\nn={dens[wl][s][1]}" for s in wsegs],
                           fontsize=12)
        ax.set_ylabel("DFlash↔Suffix winner-change\nboundaries per 1K tokens", fontsize=11)
        ax.set_title(f"{WL_LABEL[wl]} — boundary density by segment\n"
                     "(winner = argmax(DFlash, Suffix) accept; boundary = strict winner flip)",
                     fontsize=12)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fpw = OUT / f"BOUNDARY_density_{wl}.png"
        fig.savefig(fpw, dpi=150); plt.close(fig)
        print(f"saved {fpw}")


if __name__ == "__main__":
    main()
