#!/usr/bin/env python3
"""Confirmation plots: 3 methods x 3 charts per model.

Methods: baseline / select1 / select1_calib_offline
Charts : mat_bars_3.png        MAT per arm
         survival_3.png        P(accept_len >= d) per depth
         toks_3.png            absolute generation tok/s (agent-level;
                               NOT a vanilla-relative speedup chart)

Usage:
    python3 simulation/scripts/plot_confirm3.py \
        --dir simulation/results/chain_hybrid_v2_14b --summary run_merged.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CANDIDATES = ["baseline", "suffix", "select1", "select1_calib_offline"]
COLORS = {"baseline": "#7f7f7f", "suffix": "#d62728", "select1": "#1f77b4",
          "select1_calib_offline": "#ffbb78"}
ARMS: list = []  # resolved in main() from the summary


def label(arm: str, mtp: bool) -> str:
    return {
        "baseline": "baseline\n(MTP)" if mtp else "baseline\n(eagle3)",
        "suffix": "baseline\n(suffix decoding)",
        "select1": "per-depth\nselect-1",
        "select1_calib_offline": "select-1 calib\noffline (in-sample)",
    }[arm]


def load_jsonl(path: Path):
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
    return np.asarray(vals, dtype=np.int64)


def agent_tps(path: Path) -> float:
    d = json.load(open(path))
    tok = 0
    lat = 0.0
    for q in d.get("questions", []):
        for s in q["agent_metrics"]["steps"]:
            ct, ls = s.get("completion_tokens"), s.get("latency_s")
            if ct and ls:
                tok += ct
                lat += ls
    return tok / lat if lat else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--summary", default="run_merged.json")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.dir)
    summary = json.load(open(out_dir / args.summary))
    arms = summary["arms"]
    global ARMS
    ARMS = [a for a in CANDIDATES
            if a in arms and arms[a].get("accept_length_mean") is not None]
    mtp = "mtp" in str(summary.get("preset", ""))
    model = summary.get("model", "?").split("/")[-1]
    tail_cfg = summary.get("tail", {})
    sub = (f"{model}, {summary.get('workload')} 10 tasks, S={summary.get('steps')}, "
           f"topk=1, tail T_max={tail_cfg.get('max_tokens')}"
           f" — offline arm is IN-SAMPLE (train tasks)")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [label(a, mtp) for a in ARMS]
    colors = [COLORS[a] for a in ARMS]

    # 1) MAT bars
    mats = [arms[a].get("accept_length_mean") for a in ARMS]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(np.arange(len(ARMS)), mats, color=colors, width=0.55)
    for b, v in zip(bars, mats):
        ax.text(b.get_x() + b.get_width() / 2, v + max(mats) * 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(len(ARMS)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAT (accepted draft tokens / step)")
    ax.set_title(f"MAT — 3-method check\n{sub}", fontsize=9)
    ax.set_ylim(0, max(mats) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "mat_bars_3.png", dpi=150)
    plt.close(fig)

    # 2) survival per depth
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for a in ARMS:
        acc = accept_lengths(out_dir / f"timing_{a}.jsonl")
        if acc.size == 0:
            continue
        dmax = int(acc.max())
        depths = np.arange(1, dmax + 2)
        ax.plot(depths, [(acc >= d).mean() for d in depths], marker="o",
                ms=3, lw=1.6, color=COLORS[a],
                label=f"{label(a, mtp).replace(chr(10), ' ')} "
                      f"(MAT {arms[a].get('accept_length_mean'):.3f})")
    ax.axvline(args.steps, color="black", lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("depth d")
    ax.set_ylabel("P(accept_len ≥ d)")
    ax.set_title(f"Chain survival — 3-method check\n{sub}", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "survival_3.png", dpi=150)
    plt.close(fig)

    # 3) absolute generation throughput (agent-level)
    tps = [agent_tps(out_dir / f"agent_results_{a}.json") for a in ARMS]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(np.arange(len(ARMS)), tps, color=colors, width=0.55)
    for b, v in zip(bars, tps):
        ax.text(b.get_x() + b.get_width() / 2, v + max(tps) * 0.012,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(len(ARMS)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("generation throughput (tok/s, agent-level)")
    ax.set_title(f"Absolute tok/s — 3-method check\n{sub}", fontsize=9)
    ax.set_ylim(0, max(tps) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "toks_3.png", dpi=150)
    plt.close(fig)

    print(f"{model}: MAT={dict(zip(ARMS, [round(m,3) for m in mats]))} "
          f"tok/s={dict(zip(ARMS, [round(t,1) for t in tps]))}")
    print(f"wrote {fig_dir}/{{mat_bars_3,survival_3,toks_3}}.png")


if __name__ == "__main__":
    main()
