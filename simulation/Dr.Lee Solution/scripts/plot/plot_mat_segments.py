#!/usr/bin/env python3
"""Per-segment MAT bars, ONE figure per workload. Columns = output segments
(standing 4-way taxonomy: think / tool_call / preamble / final, shown in the
user's order: reasoning, tool call, text:preamble, text:response); bars = the
same five arms as the fallback-comparison figure, same colors/format.

Inputs: seg_{ds}_{arm}{tag}.jsonl written by replay_segments_5way.py.

  python3 scripts/plot_mat_segments.py --tag _pre \
      --note "records: 2026-07-03 capture snapshots"
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]  # scripts/plot/ -> Dr.Lee Solution
SEG_DIR = Path("/workspace/simulation/results/pipeline_4way/segments")
OUT = BASE / "readable_outputs" / "figures" / "mat" / "mat_bars"

PROPS = ["dflash", "suffix", "fallback", "calib", "oracle"]
LABELS = {"dflash": "DFlash (single)", "suffix": "Suffix (single)",
          "fallback": "SD-paper hybrid (fallback)",
          "calib": "Compose", "oracle": "Oracle (best handoff)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "fallback": "#9467BD",
          "calib": "#54A24B", "oracle": "#E45756"}

DS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
DS_LABEL = {"specbench": "Spec-Bench", "bfcl": "BFCL v4",
            "swebench": "SWE-bench Verified", "spider": "Spider2-DBT",
            "tau2": "τ²-bench"}
# harness + task count subtitle (consistent across every MAT figure)
WL_INFO = {"specbench": "480 tasks (all) · 6 subtasks",
           "bfcl": "bfcl_eval (prompt-mode FC) · 753 tasks · 5 categories",
           "swebench": "mini-swe-agent · 60 tasks (250-step) · 12 repos",
           "spider": "spider-agent-dbt · 68 tasks (all) · 68 databases",
           "tau2": "tau2 official sim · 64 tasks · 3 domains"}
TAUS = {"specbench": 32, "bfcl": 32, "swebench": 4, "spider": 16, "tau2": 16}
TAU_GRID = [0.5, 1, 2, 4, 8, 16, 32]     # swept grid (overridden from sweep json)

SEG_ORDER = ["think", "tool_call", "preamble", "final"]
SEG_LABEL = {"think": "reasoning\n(think)", "tool_call": "tool call",
             "preamble": "text: preamble", "final": "text: response\n(final)"}


def load(ds, tag):
    """-> {seg: {arm: (K, rounds)}}"""
    out = defaultdict(dict)
    for arm in PROPS:
        fp = SEG_DIR / f"seg_{ds}_{arm}{tag}.jsonl"
        if not fp.exists():
            continue
        agg = defaultdict(lambda: [0, 0.0])
        for line in open(fp):
            r = json.loads(line)
            a = agg[r["seg"]]; a[0] += 1; a[1] += r["acc"]
        for seg, (n, s) in agg.items():
            out[seg][arm] = (s / n if n else 0.0, n)
    return out


def main():
    global SEG_DIR, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-suffix", default="", help="appended to output filename")
    ap.add_argument("--note", default="")
    ap.add_argument("--taus-json", default="",
                    help='{"ds": tau} — overrides the built-in pooled-best taus')
    ap.add_argument("--seg-dir", default="",
                    help="override dir with seg_{ds}_{arm}{tag}.jsonl + sweep jsons")
    ap.add_argument("--out-dir", default="",
                    help="override output dir (e.g. the mat(deployable) folder)")
    ap.add_argument("--tau-note", default="",
                    help="override the corner τ* note line (e.g. deployable split)")
    args = ap.parse_args()
    if args.seg_dir:
        SEG_DIR = Path(args.seg_dir)
    if args.out_dir:
        OUT = Path(args.out_dir)
    OSFX = args.out_suffix
    LABELS["calib"] = ("Compose (raw, no calib)" if args.tag == "_nocalib"
                       else "Compose (logistic head + isotonic tail)")
    if args.taus_json:
        TAUS.update({k: float(v) for k, v in json.load(open(args.taus_json)).items()})
    grid = set()
    for ds in DS:
        fp = SEG_DIR / f"fallback_sweep_fresh_{ds}.json"
        if fp.exists():
            taus = next(iter(json.load(open(fp)).values()))
            if set(taus) == {"calib", "test"}:      # deployable split sweep
                taus = taus["test"]
            grid.update(float(t) for t in taus)
    if grid:
        TAU_GRID[:] = sorted(grid)
    OUT.mkdir(parents=True, exist_ok=True)

    for ds in DS:
        K = load(ds, args.tag)
        segs = [s for s in SEG_ORDER if s in K]
        if not segs:
            print(f"[{ds}] no segment logs, skip")
            continue
        # round share per segment, from the dflash arm (representative)
        tot = sum(K[s].get("dflash", (0, 0))[1] for s in segs) or 1
        names = []
        for s in segs:
            n_s = K[s].get("dflash", (0, 0))[1]
            pct = 100 * n_s / tot
            share = (f"{pct:.0f}% of rounds" if pct >= 1
                     else f"{pct:.1f}% of rounds (n={n_s})")
            names.append(SEG_LABEL[s] + "\n" + share)

        fig, ax = plt.subplots(figsize=(max(10.2, 3.0 + 2.5 * len(segs)), 5.6))
        n, g = len(PROPS), 0.80
        bw = g / n
        ymax = 0.0
        fb_labels = []
        for pi, p in enumerate(PROPS):
            xs = [si - g / 2 + bw * (pi + 0.5) for si in range(len(segs))]
            ys = [K[s].get(p, (0.0, 0))[0] for s in segs]
            ymax = max(ymax, max(ys) if ys else 0)
            ax.bar(xs, ys, width=bw * 0.9, color=COLORS[p], label=LABELS[p])
            for x, y in zip(xs, ys):
                if y > 0:
                    ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom",
                            fontsize=10.5, color="#444444")
            if p == "fallback":
                fb_labels = [(x, y) for x, y in zip(xs, ys) if y > 0]
        # per-bar best tau for the SD-paper hybrid (fixed per workload here)
        for x, y in fb_labels:
            ax.text(x, y + ymax * 0.055, f"τ*={TAUS[ds]:g}", ha="center",
                    va="bottom", fontsize=10.5, fontweight="bold",
                    color=COLORS["fallback"])
        ax.set_ylim(0, ymax * 1.32)
        ax.set_xticks(range(len(segs)))
        ax.set_xticklabels(names, fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel("mean accept length  (tokens)", fontsize=14)
        ax.legend(fontsize=11.5, frameon=False, loc="upper left", ncol=2)
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{DS_LABEL[ds]} — MAT by output segment\n{WL_INFO[ds]}",
                     fontsize=14)
        ax.text(0.99, 0.99,
                "SD-paper hybrid τ swept: {" + ", ".join(f"{t:g}" for t in TAU_GRID)
                + "}\n" + (args.tau_note or "τ* = best per workload")
                + "  ·  round share: DFlash arm",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
                color="#555555")
        fig.tight_layout()
        fp = OUT / f"MAT_segments_{ds}{OSFX}.png"
        fig.savefig(fp, dpi=150); plt.close(fig)
        print(f"saved {fp}")
        for s in segs:
            print(f"  [{s}] " + "  ".join(
                f"{p}={K[s].get(p, (0, 0))[0]:.2f}({K[s].get(p, (0, 0))[1]})"
                for p in PROPS))

    # ---- pooled figure: ALL workloads combined, MAT by output segment ---------
    agg = {}                                   # seg -> arm -> [sum_acc, rounds]
    for ds in DS:
        for seg, arms in load(ds, args.tag).items():
            for arm, (k, n) in arms.items():
                a = agg.setdefault(seg, {}).setdefault(arm, [0.0, 0])
                a[0] += k * n; a[1] += n
    P = {seg: {arm: (s / n if n else 0.0, n) for arm, (s, n) in arms.items()}
         for seg, arms in agg.items()}
    segs = [s for s in SEG_ORDER if s in P]
    if segs:
        tot = sum(P[s].get("dflash", (0, 0))[1] for s in segs) or 1
        names = [SEG_LABEL[s] + f"\n{100 * P[s].get('dflash', (0, 0))[1] / tot:.0f}% of rounds"
                 for s in segs]
        fig, ax = plt.subplots(figsize=(max(10.2, 3.0 + 2.5 * len(segs)), 5.6))
        n, g = len(PROPS), 0.80
        bw = g / n
        ymax = 0.0
        for pi, p in enumerate(PROPS):
            xs = [si - g / 2 + bw * (pi + 0.5) for si in range(len(segs))]
            ys = [P[s].get(p, (0.0, 0))[0] for s in segs]
            ymax = max(ymax, max(ys) if ys else 0)
            ax.bar(xs, ys, width=bw * 0.9, color=COLORS[p], label=LABELS[p])
            for x, y in zip(xs, ys):
                if y > 0:
                    ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom",
                            fontsize=10.5, color="#444444")
        ax.set_ylim(0, ymax * 1.30)
        ax.set_xticks(range(len(segs)))
        ax.set_xticklabels(names, fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel("mean accept length  (tokens)", fontsize=14)
        ax.legend(fontsize=11.5, frameon=False, loc="upper left", ncol=2)
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        present = " + ".join(DS_LABEL[ds] for ds in DS
                             if any(ds in f.name for f in
                                    SEG_DIR.glob(f"seg_{ds}_dflash{args.tag}.jsonl")))
        ax.set_title("All workloads — MAT by output segment\n"
                     + (present or "rounds-pooled") + " (rounds-pooled)",
                     fontsize=13)
        ax.text(0.99, 0.99,
                "SD-paper hybrid: " + (args.tau_note or "each workload's best τ")
                + "\nround share: DFlash arm",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
                color="#555555")
        fig.tight_layout()
        fp = OUT / f"MAT_per_segment{OSFX}.png"
        fig.savefig(fp, dpi=150); plt.close(fig)
        print(f"saved {fp}")
        for s in segs:
            print(f"  [{s}] " + "  ".join(
                f"{p}={P[s].get(p, (0, 0))[0]:.2f}({P[s].get(p, (0, 0))[1]})"
                for p in PROPS))


if __name__ == "__main__":
    main()
