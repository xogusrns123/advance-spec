#!/usr/bin/env python3
"""Render chain-hybrid S=16+tail result figures.

Inputs (out_dir = simulation/results/chain_hybrid_tail):
  run_s16_tail.json            arm headline metrics (accept_length_mean, ...)
  timing_<arm>.jsonl           per-step decode records with accept_lengths
  agent_trajectory_<arm>.json     agent step metrics (completion_tokens, latency_s)
  agent_results_vanilla.json   vanilla (no-spec) reference for speedup

Outputs (out_dir/figures):
  mat_bars.png                 MAT (mean accepted draft tokens/step) per arm
  speedup_bars.png             generation-throughput speedup vs vanilla
  survival_per_depth.png       P(accept_len >= d) per depth, tail region marked

Usage (inside container):
    python3 simulation/scripts/plot_chain_hybrid_tail.py \
        [--dir simulation/results/chain_hybrid_tail] [--steps 16]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARM_ORDER = ["baseline", "record", "suffix", "hybrid_e3",
             "select1", "select1_calib", "select1_calib_jeffreys",
             "select1_calib_offline", "select1_calib_jeffreys_offline",
             "select1_oracle"]
LABELS = {
    "baseline": "baseline\n(eagle3)",
    "record": "baseline\n(record/GT)",
    "suffix": "baseline\n(suffix dec.)",
    "hybrid_e3": "hybrid\n(score fallback)",
    "select1_oracle": "selection\nORACLE (ceiling)",
    "select1": "per-depth\nselect-1 (raw)",
    "select1_calib": "select-1\n+calib",
    "select1_calib_jeffreys": "select-1\n+calib (Jeffreys)",
    "select1_calib_offline": "+calib\noffline (in-sample)",
    "select1_calib_jeffreys_offline": "+calib (Jeffreys)\noffline (in-sample)",
    "vanilla": "vanilla\n(no spec)",
}
COLORS = {
    "baseline": "#7f7f7f",
    "record": "#c7c7c7",
    "suffix": "#d62728",
    "hybrid_e3": "#9467bd",
    "select1_oracle": "#e0b400",
    "select1": "#1f77b4",
    "select1_calib": "#ff7f0e",
    "select1_calib_jeffreys": "#2ca02c",
    "select1_calib_offline": "#ffbb78",   # lighter orange (in-sample twin)
    "select1_calib_jeffreys_offline": "#98df8a",  # lighter green
    "vanilla": "#c7c7c7",
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


def agent_throughput(path: Path) -> tuple[float, int, float]:
    """Generation tok/s from agent step metrics: sum(completion)/sum(latency).
    Prefill is included in latency (uniform across arms; negligible for the
    ~1-2k-token responses of this workload)."""
    d = json.load(open(path))
    tok = 0
    lat = 0.0
    for q in d.get("questions", []):
        for s in q["agent_metrics"]["steps"]:
            ct, ls = s.get("completion_tokens"), s.get("latency_s")
            if ct and ls:
                tok += ct
                lat += ls
    return (tok / lat if lat else float("nan")), tok, lat


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="simulation/results/chain_hybrid_tail")
    ap.add_argument("--summary", default="run_s16_tail.json")
    ap.add_argument("--steps", type=int, default=16,
                    help="Head chain length S (tail region starts here)")
    ap.add_argument("--speedup", action="store_true",
                    help="Also render speedup_bars.png (OFF by default — "
                         "vanilla-relative speedup framing is not part of "
                         "standard result reporting)")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = json.load(open(out_dir / args.summary))
    arms = summary["arms"]
    tail_cfg = summary.get("tail", {})
    model = summary.get("model", "?").split("/")[-1]
    sub = (f"{model}, {summary.get('workload')} "
           f"{summary.get('n_tasks')} test tasks, "
           f"S={summary.get('steps')}, topk=1, "
           f"tail T_max={tail_cfg.get('max_tokens')}")
    # The baseline drafter is EAGLE3 for the qwen3 presets, the native MTP
    # head for *_mtp presets.
    if "mtp" in str(summary.get("preset", "")):
        LABELS["baseline"] = "baseline\n(MTP)"
    # Plot whichever arms the run actually contains (canonical order), so
    # the in-sample *_offline arms appear when present.
    global ARMS
    ARMS = [a for a in ARM_ORDER
            if a in arms and arms[a].get("accept_length_mean") is not None]

    # ---- 1) MAT bars -----------------------------------------------------
    mats = [arms[a].get("accept_length_mean") for a in ARMS]
    fig, ax = plt.subplots(figsize=(max(7.2, 1.55 * len(ARMS) + 1.5), 4.4))
    xs = np.arange(len(ARMS))
    bars = ax.bar(xs, mats, color=[COLORS[a] for a in ARMS], width=0.62)
    for b, v in zip(bars, mats):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[a] for a in ARMS], fontsize=9)
    ax.set_ylabel("MAT (mean accepted draft tokens / step)")
    ax.set_title(f"Chain-hybrid MAT\n{sub}", fontsize=10)
    ax.set_ylim(0, max(mats) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "mat_bars.png", dpi=150)
    plt.close(fig)
    print(f"mat_bars.png: {dict(zip(ARMS, [round(m,3) for m in mats]))}")

    # ---- 2) Speedup vs vanilla (opt-in only) ------------------------------
    van_path = out_dir / "agent_results_vanilla.json"
    if args.speedup and van_path.exists():
        van_tps, van_tok, van_lat = agent_throughput(van_path)
        rows = []
        for a in ARMS:
            ares = out_dir / f"agent_trajectory_{a}.json"
            tps, tok, lat = agent_throughput(ares)
            rows.append((a, tps, tps / van_tps))
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        xs = np.arange(len(rows))
        sp = [r[2] for r in rows]
        bars = ax.bar(xs, sp, color=[COLORS[r[0]] for r in rows], width=0.62)
        for b, (a, tps, s) in zip(bars, rows):
            ax.text(b.get_x() + b.get_width() / 2, s + 0.012,
                    f"{s:.2f}x\n({tps:.1f} tok/s)", ha="center", va="bottom",
                    fontsize=9)
        ax.axhline(1.0, color="black", lw=1, ls="--",
                   label=f"vanilla = {van_tps:.1f} tok/s")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[r[0]] for r in rows], fontsize=9)
        ax.set_ylabel("generation throughput speedup vs vanilla")
        ax.set_title(f"Speedup vs vanilla (agent-level tok/s)\n{sub}",
                     fontsize=10)
        ax.set_ylim(0, max(sp) * 1.22)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "speedup_bars.png", dpi=150)
        plt.close(fig)
        print(f"speedup_bars.png: vanilla={van_tps:.2f} tok/s, "
              + ", ".join(f"{a}={s:.3f}x" for a, _, s in rows))
    elif van_path.exists():
        # Vanilla is measured as a reference NUMBER; the speedup chart itself
        # is opt-in (--speedup) and not part of standard reporting.
        van_tps, _, _ = agent_throughput(van_path)
        print(f"vanilla reference: {van_tps:.2f} tok/s "
              f"(speedup chart not rendered; use --speedup to opt in)")
    else:
        print("(no vanilla measurement found)")

    # ---- 3) Survival per depth -------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    max_d = 0
    for a in ARMS:
        acc = accept_lengths(out_dir / f"timing_{a}.jsonl")
        if acc.size == 0:
            print(f"  (no timing data for {a})")
            continue
        dmax = int(acc.max())
        max_d = max(max_d, dmax)
        depths = np.arange(1, dmax + 2)
        surv = [(acc >= d).mean() for d in depths]
        ax.plot(depths, surv, marker="o", ms=3, lw=1.6, color=COLORS[a],
                label=f"{LABELS[a].replace(chr(10), ' ')} "
                      f"(MAT {arms[a].get('accept_length_mean'):.3f})")
    ax.axvline(args.steps, color="black", lw=1, ls=":")
    ax.text(args.steps + 0.3, 0.5, f"S={args.steps}\n(tail region →)",
            fontsize=8, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("depth d")
    ax.set_ylabel("survival  P(accept_len ≥ d)")
    ax.set_title(f"Chain survival per depth\n{sub}", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, max(max_d + 1, args.steps + 2))
    fig.tight_layout()
    fig.savefig(fig_dir / "survival_per_depth.png", dpi=150)
    plt.close(fig)
    print(f"survival_per_depth.png (max accepted depth observed: {max_d})")


if __name__ == "__main__":
    main()
