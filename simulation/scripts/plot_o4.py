#!/usr/bin/env python3
"""O4 per-depth selection — real-serving comparison figures (per model).

8 versions: model-only (baseline), suffix-only, raw select-1, calibrated select-1
x4 (histogram/isotonic/logistic/beta, per-position maps), oracle.

Inputs (out_dir):
  run.json                  per-arm summary incl. accept_length_mean (MAT)
  timing_<arm>.jsonl        per-step decode records with accept_lengths (survival)
  decisions_<arm>.jsonl     per-depth decision records with `chosen` (selection)

Outputs (out_dir/figures):
  o4_mat.png        MAT bar per version
  o4_survival.png   P(accept_len >= d) vs depth, one line per version
  o4_selection.png  suffix-chosen fraction vs depth, for the SELECTION versions
                    (raw / 4x calib / oracle)

Usage (container):
  python3 simulation/scripts/plot_o4.py --dir simulation/results/o4_perdepth/qwen3_14b
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# All 8 versions (record excluded — GT producer, not a measured version).
ARM_ORDER = ["baseline", "suffix", "select1",
             "select1_calib_histogram", "select1_calib_isotonic",
             "select1_calib_logistic", "select1_calib_beta", "select1_oracle"]
SELECTION_ARMS = ["select1", "select1_calib_histogram", "select1_calib_isotonic",
                  "select1_calib_logistic", "select1_calib_beta", "select1_oracle"]
LABELS = {
    "baseline": "model-only",
    "suffix": "suffix-only",
    "select1": "raw",
    "select1_calib_histogram": "calib\n(histogram)",
    "select1_calib_isotonic": "calib\n(isotonic)",
    "select1_calib_logistic": "calib\n(Platt)",
    "select1_calib_beta": "calib\n(beta)",
    "select1_oracle": "ORACLE",
}
COLORS = {
    "baseline": "#7f7f7f", "suffix": "#d62728", "select1": "#1f77b4",
    "select1_calib_histogram": "#ff7f0e", "select1_calib_isotonic": "#2ca02c",
    "select1_calib_logistic": "#9467bd", "select1_calib_beta": "#8c564b",
    "select1_oracle": "#e0b400",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return rows


def accept_lengths(path: Path) -> np.ndarray:
    vals = []
    for r in load_jsonl(path):
        if r.get("phase") and r["phase"] != "decode":
            continue
        a = r.get("accept_lengths")
        if isinstance(a, list):
            vals.extend(int(x) for x in a)
        elif isinstance(a, (int, float)):
            vals.append(int(a))
    return np.asarray(vals, dtype=np.int64)


def detect_calib_meta(out_dir: Path):
    """Read calibration objective + variant from a calib map's meta so every figure
    is labeled. meta.label token_gt -> accept_rate; accept_conditioned -> cond/all."""
    for m in ("logistic", "isotonic", "histogram", "beta"):
        p = out_dir / f"calib_pp_{m}.json"
        if not p.exists():
            continue
        try:
            meta = json.load(open(p)).get("meta", {})
        except Exception:
            continue
        lab = meta.get("label")
        obj = {"target_p": "target_p", "token_gt": "accept_rate"}.get(lab, lab or "?")
        var = "cond-trained" if meta.get("accept_conditioned") else "all-trained"
        return obj, var
    return "?", "?"


def suffix_chosen_by_depth(path: Path):
    """depth -> suffix-chosen fraction (over non-tail decision records)."""
    n = defaultdict(int)
    s = defaultdict(int)
    for r in load_jsonl(path):
        if r.get("type") != "decision" or r.get("tail"):
            continue
        d = r.get("depth")
        if d is None:
            continue
        n[d] += 1
        if r.get("chosen") == "suffix":
            s[d] += 1
    depths = sorted(n)
    return depths, [s[d] / n[d] for d in depths], [n[d] for d in depths]


def selection_breakdown_by_depth(path: Path):
    """depth -> 3-way SELECTION outcome fractions over decisions where a real
    choice existed (both proposers offered a candidate, i.e. suffix_token != None):
      eagle  = disagree & chosen==eagle3   (EAGLE3 uniquely selected)
      suffix = disagree & chosen==suffix   (suffix uniquely selected)
      tie    = agreement True (eagle_token == suffix_token; pick is moot)
    The three sum to 1.0 at each depth. Returns (depths, {cat: [frac]}, n_avail)."""
    n = defaultdict(int)
    eag = defaultdict(int)
    suf = defaultdict(int)
    tie = defaultdict(int)
    for r in load_jsonl(path):
        if r.get("type") != "decision" or r.get("tail"):
            continue
        d = r.get("depth")
        if d is None or r.get("suffix_token") is None:
            continue  # no suffix candidate -> no selection was made
        n[d] += 1
        if r.get("agreement") is True:
            tie[d] += 1
        elif r.get("chosen") == "suffix":
            suf[d] += 1
        else:
            eag[d] += 1
    depths = sorted(n)
    fr = {
        "eagle": [eag[d] / n[d] for d in depths],
        "suffix": [suf[d] / n[d] for d in depths],
        "tie": [tie[d] / n[d] for d in depths],
    }
    return depths, fr, [n[d] for d in depths]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--summary", default="run.json")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = json.load(open(out_dir / args.summary))
    arms = summary["arms"]
    model = summary.get("model", "?").split("/")[-1]
    base_label = "MTP" if "mtp" in str(summary.get("preset", "")) else "EAGLE3"
    LABELS["baseline"] = f"model-only\n({base_label})"
    obj, variant = detect_calib_meta(out_dir)
    sub = (f"{model}, objective={obj}, calib={variant}, {summary.get('workload')} "
           f"{summary.get('n_tasks')} test tasks, S={summary.get('steps')}, topk=1")
    present = [a for a in ARM_ORDER
               if a in arms and arms[a].get("accept_length_mean") is not None]

    # ---- 1) MAT bars ----
    mats = [arms[a]["accept_length_mean"] for a in present]
    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * len(present) + 1.5), 4.6))
    xs = np.arange(len(present))
    bars = ax.bar(xs, mats, color=[COLORS[a] for a in present], width=0.62)
    for b, v in zip(bars, mats):
        ax.text(b.get_x() + b.get_width() / 2, v + max(mats) * 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([LABELS[a] for a in present], fontsize=8)
    ax.set_ylabel("MAT (mean accepted draft tokens / step)")
    ax.set_title(f"O4 per-depth selection — MAT\n{sub}", fontsize=10)
    ax.set_ylim(0, max(mats) * 1.18); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "o4_mat.png", dpi=150); plt.close(fig)
    print(f"o4_mat.png: {dict(zip(present, [round(m,3) for m in mats]))}")

    # ---- 2) Survival per depth (two x-ranges; adaptive y-top) ----
    surv_data = {}
    max_d = 0
    max_surv = 0.0
    for a in present:
        acc = accept_lengths(out_dir / f"timing_{a}.jsonl")
        if acc.size == 0:
            print(f"  (no timing for {a})"); continue
        dmax = int(acc.max()); max_d = max(max_d, dmax)
        depths = np.arange(1, dmax + 2)
        surv = np.array([(acc >= d).mean() for d in depths])
        surv_data[a] = (depths, surv)
        max_surv = max(max_surv, float(surv.max()))
    # cap y at ~0.8, but use 1.0 if any curve reaches >= 0.8
    ytop = 1.0 if max_surv >= 0.8 else 0.8

    def _draw_survival(xmax, fname):
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        for a in present:
            if a not in surv_data:
                continue
            depths, surv = surv_data[a]
            ax.plot(depths, surv, marker="o", ms=3, lw=1.6, color=COLORS[a],
                    label=f"{LABELS[a].replace(chr(10),' ')} "
                          f"(MAT {arms[a]['accept_length_mean']:.2f})")
        ax.axvline(args.steps, color="black", lw=1, ls=":")
        ax.set_xlabel("depth d"); ax.set_ylabel("survival  P(accept_len >= d)")
        ax.set_ylim(0, ytop)
        ax.set_title(f"O4 chain survival per depth\n{sub}", fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
        ax.set_xlim(0.5, xmax)
        fig.tight_layout(); fig.savefig(fig_dir / fname, dpi=150); plt.close(fig)

    _draw_survival(max(max_d + 1, args.steps + 1), "o4_survival.png")  # full (to ~32)
    _draw_survival(args.steps, "o4_survival_d16.png")                  # capped at S=16
    print(f"o4_survival.png (full, max depth {max_d}) + o4_survival_d16.png "
          f"(<= {args.steps})  ytop={ytop}")

    # ---- 3) Per-position selection 3-way breakdown (eagle / suffix / tie) ----
    # CANONICAL metric, shared with plot_o4_objectives / m4_selection: over
    # CHOICE-AVAILABLE decisions; the 3 panels sum to 1 at each depth.
    sel_present = [a for a in SELECTION_ARMS
                   if a in arms and arms[a].get("accept_length_mean") is not None]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for a in sel_present:
        depths, fr, _ = selection_breakdown_by_depth(out_dir / f"decisions_{a}.jsonl")
        if not depths:
            continue
        for ax, cat in zip(axes, ("eagle", "suffix", "tie")):
            ax.plot(depths, fr[cat], marker="o", ms=3, lw=1.6, color=COLORS[a],
                    label=LABELS[a].replace(chr(10), " "))
    for ax, t in zip(axes, [f"{base_label}-win (uniquely selected)",
                            "suffix-win (uniquely selected)", "tie (agreement)"]):
        ax.set_xlabel("depth d"); ax.set_ylim(0, 1.02); ax.set_title(t, fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction (choice-available decisions)"); axes[0].legend(fontsize=8)
    fig.suptitle(f"O4 per-position selection (3-way: {base_label}/suffix/tie)\n{sub}", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "o4_selection.png", dpi=150); plt.close(fig)
    print("o4_selection.png")

    # ---- 3b) Per-position selection 3-way breakdown, one image per outcome ----
    # Among decisions where a real choice existed (suffix candidate present):
    #   eagle  = {base_label} uniquely selected, suffix = suffix uniquely selected,
    #   tie    = both proposers agreed (pick is moot). Each image overlays ALL
    #   selection arms (raw / 4x calib / oracle). The three fractions sum to 1.
    bd = {}
    for a in sel_present:
        depths, fr, navail = selection_breakdown_by_depth(
            out_dir / f"decisions_{a}.jsonl")
        if depths:
            bd[a] = (depths, fr, navail)
    cats = [
        ("eagle", f"{base_label} uniquely selected",
         "o4_selection_eagle.png", f"{base_label}-only select"),
        ("suffix", "suffix uniquely selected",
         "o4_selection_suffix.png", "suffix-only select"),
        ("tie", f"tie ({base_label} token == suffix token)",
         "o4_selection_tie.png", "tie (agree)"),
    ]
    for cat, desc, fname, ylab in cats:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        any_line = False
        for a in SELECTION_ARMS:
            if a not in bd:
                continue
            depths, fr, _ = bd[a]
            ax.plot(depths, fr[cat], marker="o", ms=3, lw=1.6, color=COLORS[a],
                    label=LABELS[a].replace(chr(10), " "))
            any_line = True
        ax.set_xlabel("depth d")
        ax.set_ylabel(f"{ylab}  (frac of choice-available decisions)")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"O4 selection outcome: {desc}\n{sub}", fontsize=10)
        ax.grid(alpha=0.3)
        if any_line:
            ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / fname, dpi=150); plt.close(fig)
    print("o4_selection_{eagle,suffix,tie}.png")


if __name__ == "__main__":
    main()
