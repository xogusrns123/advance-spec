#!/usr/bin/env python3
"""O4 per-depth real-serving comparison: OLD vs NEW (accept-conditioned) calib.

Overlays, per depth, the reference arms (model-only / suffix / raw / oracle) and
the 4 calibration methods fit two ways:
  OLD  = current fitter (pools dead-chain rows)            -> solid lines
  NEW  = --accept-conditioned fitter (alive-prefix only)   -> dashed lines

All arms come from a REAL-SERVING REPLAY run pinned to one standalone-eagle3
trajectory (measure_chain_hybrid.py --pin-trajectory), so depth curves are
comparable across arms (no FP-tie trajectory drift). The calibration OBJECTIVE
(target_p vs accept_rate) is a property of how the maps were fit; this script
just labels/segregates the figures by --objective. The comparison metric is the
same either way: chain survival and MAT.

Inputs (out_dir):
  run_<variant>.json                  per-arm summary incl. accept_length_mean
                                      (variants e.g. all-trained, cond-trained)
  timing_<arm>.jsonl                  reference arms (baseline/suffix/select1/oracle)
  timing_select1_calib_<m>_<v>.jsonl  calib method m, variant v (e.g. all-trained,
                                      cond-trained = accept-conditioned)
  decisions_<arm>.jsonl / _<v>.jsonl  per-depth `chosen` for selection panel
  (the runbook renames each pass's calib outputs with the _<variant> suffix)

Outputs (out_dir/figures/<objective>/):
  o4_mat.png            MAT bar, refs + 4 methods x variants
  o4_survival.png       P(accept_len>=d) vs depth, first-variant solid / rest dashed
  o4_survival_d<S>.png  same, capped at S
  o4_conditional.png    per-step conditional accept cond(d)=surv(d)/surv(d-1)
                        -- directly shows whether the conditional is flat
  o4_selection.png      model-chosen fraction vs depth, per variant

Usage (container, as root):
  python3 simulation/scripts/plot_o4_objectives.py \
      --dir simulation/results/chain_hybrid_perdepth/qwen3_14b_tp \
      --objective target_p --variants all-trained,cond-trained --steps 16
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

METHODS = ["histogram", "isotonic", "logistic", "beta"]
# Reference arms are variant-independent (taken from the OLD run, which serves
# them in pass 1); suffix comes from sim_suffix_only.py (timing only).
REF_ARMS = ["baseline", "suffix", "select1", "select1_oracle"]
REF_LABEL = {"baseline": "model-only", "suffix": "suffix-only",
             "select1": "raw", "select1_oracle": "ORACLE"}
REF_COLOR = {"baseline": "#7f7f7f", "suffix": "#d62728",
             "select1": "#1f77b4", "select1_oracle": "#e0b400"}
CALIB_COLOR = {"histogram": "#ff7f0e", "isotonic": "#2ca02c",
               "logistic": "#9467bd", "beta": "#8c564b"}
CALIB_LABEL = {"histogram": "histogram", "isotonic": "isotonic",
               "logistic": "Platt", "beta": "beta"}


def load_jsonl(path: Path) -> list:
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


def survival(acc: np.ndarray):
    """(depths>=1, surv) where surv(d) = P(accept_len >= d)."""
    if acc.size == 0:
        return np.array([]), np.array([])
    dmax = int(acc.max())
    depths = np.arange(1, dmax + 2)
    surv = np.array([(acc >= d).mean() for d in depths])
    return depths, surv


def conditional(acc: np.ndarray):
    """(depths>=1, cond) where cond(d) = P(accept>=d | accept>=d-1)
    = surv(d)/surv(d-1), surv(0)=1. NaN where the denominator is 0."""
    depths, surv = survival(acc)
    if depths.size == 0:
        return depths, surv
    prev = np.concatenate([[1.0], surv[:-1]])  # surv(d-1), surv(0)=1
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(prev > 0, surv / prev, np.nan)
    return depths, cond


def selection_breakdown_by_depth(path: Path):
    """CANONICAL 3-way SELECTION breakdown per depth over CHOICE-AVAILABLE decisions
    (suffix_token != None: both proposers offered a candidate). Shared definition
    with plot_o4 / plot_consolidated:
      eagle  = chosen==eagle3 & not agreement  (EAGLE3 uniquely selected)
      suffix = chosen==suffix                  (suffix uniquely selected)
      tie    = agreement True                  (tokens equal; pick is moot)
    The three fractions sum to 1.0 at each depth. Returns (depths, {cat: [frac]})."""
    n = defaultdict(int)
    eg = defaultdict(int)
    su = defaultdict(int)
    ti = defaultdict(int)
    for r in load_jsonl(path):
        if r.get("type") != "decision" or r.get("tail") or r.get("suffix_token") is None:
            continue
        d = r.get("depth")
        if d is None:
            continue
        n[d] += 1
        if r.get("agreement") is True:
            ti[d] += 1
        elif r.get("chosen") == "suffix":
            su[d] += 1
        else:
            eg[d] += 1
    depths = sorted(n)
    fr = {"eagle": [eg[d] / n[d] for d in depths],
          "suffix": [su[d] / n[d] for d in depths],
          "tie": [ti[d] / n[d] for d in depths]}
    return depths, fr


def build_series(out_dir: Path, summaries: dict, variants: list) -> list:
    """List of dicts: {label, color, ls, mat, timing, dec}. Reference arms first
    (solid), then calib method x variant (old solid, new dashed)."""
    series = []
    ref = summaries.get(variants[0], {}).get("arms", {})  # ref arms from first variant
    for a in REF_ARMS:
        timing = out_dir / f"timing_{a}.jsonl"
        dec = out_dir / f"decisions_{a}.jsonl"
        mat = (ref.get(a) or {}).get("accept_length_mean")
        if mat is None:  # suffix-only has no run.json row; derive from timing
            acc = accept_lengths(timing)
            mat = float(acc.mean()) if acc.size else None
        if mat is None and not timing.exists():
            continue
        series.append({"label": REF_LABEL[a], "color": REF_COLOR[a], "ls": "-",
                       "mat": mat, "timing": timing, "dec": dec})
    for v in variants:
        arms = summaries.get(v, {}).get("arms", {})
        for m in METHODS:
            key = f"select1_calib_{m}"
            timing = out_dir / f"timing_{key}_{v}.jsonl"
            dec = out_dir / f"decisions_{key}_{v}.jsonl"
            mat = (arms.get(key) or {}).get("accept_length_mean")
            if mat is None:
                acc = accept_lengths(timing)
                mat = float(acc.mean()) if acc.size else None
            if mat is None and not timing.exists():
                continue
            series.append({
                "label": f"{CALIB_LABEL[m]} [{v}]", "color": CALIB_COLOR[m],
                "ls": ("-" if v == variants[0] else "--"),
                "mat": mat, "timing": timing, "dec": dec})
    return series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--objective", choices=["target_p", "accept_rate"],
                    default="accept_rate")
    ap.add_argument("--variants", default="all-trained,cond-trained",
                    help="comma list of run_<variant>.json variants "
                         "(first = solid + reference arms)")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.dir)
    fig_dir = out_dir / "figures" / args.objective
    fig_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    summaries = {}
    for v in variants:
        p = out_dir / f"run_{v}.json"
        if p.exists():
            summaries[v] = json.load(open(p))
    if not summaries:
        raise SystemExit(f"no run_<variant>.json found in {out_dir} for {variants}")
    meta = next(iter(summaries.values()))
    model = str(meta.get("model", "?")).split("/")[-1]
    sub = (f"{model}, {args.objective}, {meta.get('workload')} "
           f"{meta.get('n_tasks')} test tasks, S={meta.get('steps', args.steps)}, "
           f"pinned replay")

    series = build_series(out_dir, summaries, variants)
    if not series:
        raise SystemExit("no plottable arms (missing timing/run files)")

    # ---- MAT bar ----
    lab = [s["label"] for s in series if s["mat"] is not None]
    mats = [s["mat"] for s in series if s["mat"] is not None]
    cols = [s["color"] for s in series if s["mat"] is not None]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(lab) + 2), 4.6))
    xs = np.arange(len(lab))
    bars = ax.bar(xs, mats, color=cols, width=0.7)
    for b, val in zip(bars, mats):
        ax.text(b.get_x() + b.get_width() / 2, val + max(mats) * 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("MAT (mean accepted draft tokens / step)")
    ax.set_title(f"O4 MAT — all-trained vs cond-trained (accept-conditioned) calib\n{sub}", fontsize=10)
    ax.set_ylim(0, max(mats) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "o4_mat.png", dpi=150)
    plt.close(fig)
    print(f"o4_mat.png ({len(lab)} arms)")

    # ---- precompute acc per series ----
    for s in series:
        s["acc"] = accept_lengths(s["timing"])

    # ---- survival ----
    def draw_survival(xmax, fname):
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        for s in series:
            depths, surv = survival(s["acc"])
            if depths.size == 0:
                continue
            mt = f" (MAT {s['mat']:.2f})" if s["mat"] is not None else ""
            ax.plot(depths, surv, marker="o", ms=3, lw=1.6,
                    color=s["color"], ls=s["ls"], label=s["label"] + mt)
        ax.axvline(args.steps, color="black", lw=1, ls=":")
        ax.set_xlabel("depth d")
        ax.set_ylabel("survival  P(accept_len >= d)")
        ax.set_xlim(0.5, xmax)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"O4 chain survival per depth\n{sub}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, handlelength=3.5, handletextpad=0.5)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    max_d = max((int(s["acc"].max()) for s in series if s["acc"].size), default=1)
    draw_survival(max(max_d + 1, args.steps + 1), "o4_survival.png")
    draw_survival(args.steps, f"o4_survival_d{args.steps}.png")
    print("o4_survival.png + capped")

    # ---- per-step conditional accept (the 'flat?' hypothesis) ----
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for s in series:
        depths, cond = conditional(s["acc"])
        if depths.size == 0:
            continue
        ax.plot(depths, cond, marker="o", ms=3, lw=1.6,
                color=s["color"], ls=s["ls"], label=s["label"])
    ax.axvline(args.steps, color="black", lw=1, ls=":")
    ax.set_xlabel("depth d")
    ax.set_ylabel("conditional accept  P(accept>=d | accept>=d-1)")
    ax.set_xlim(0.5, args.steps)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"O4 per-step conditional acceptance\n{sub}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, handlelength=3.5, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "o4_conditional.png", dpi=150)
    plt.close(fig)
    print("o4_conditional.png")

    # ---- selection: 3-way breakdown per depth (eagle / suffix / tie) ----
    # CANONICAL metric, shared with plot_o4 / m4_selection: over CHOICE-AVAILABLE
    # decisions (both proposers offered a candidate). raw + ORACLE references plus
    # every calib algorithm x variant overlaid; the 3 panels sum to 1 at each depth.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    any_line = False
    for s in series:
        dec = str(s["dec"])
        if dec.endswith("decisions_baseline.jsonl") or dec.endswith("decisions_suffix.jsonl"):
            continue  # degenerate: model-only has no suffix candidate, suffix-only all-suffix
        depths, fr = selection_breakdown_by_depth(s["dec"])
        if not depths:
            continue
        for ax, cat in zip(axes, ("eagle", "suffix", "tie")):
            ax.plot(depths, fr[cat], marker="o", ms=3, lw=1.5,
                    color=s["color"], ls=s["ls"], label=s["label"])
        any_line = True
    for ax, t in zip(axes, ["eagle-win (uniquely selected)",
                            "suffix-win (uniquely selected)", "tie (agreement)"]):
        ax.set_xlabel("depth d"); ax.set_ylim(0, 1.02); ax.set_title(t, fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction (choice-available decisions)")
    if any_line:
        axes[0].legend(fontsize=6, ncol=2, handlelength=3.5, handletextpad=0.5)
    fig.suptitle(f"O4 per-depth selection (3-way: eagle/suffix/tie) — objective={args.objective}\n{sub}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "o4_selection.png", dpi=150)
    plt.close(fig)
    print("o4_selection.png")


if __name__ == "__main__":
    main()
