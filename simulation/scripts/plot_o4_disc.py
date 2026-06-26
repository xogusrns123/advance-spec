#!/usr/bin/env python3
"""O4 joint-discriminator figures — does logistic/beta discriminator beat raw &
the per-proposer calibrators, approaching oracle?

The discriminator eval run (`--dir`) holds raw + oracle + disc_logistic +
disc_beta, all under --replay-all so they decode the SAME trajectory (fair MAT).
The per-proposer calibration arms live in a sibling replay run (`--replay-dir`,
same model/test-slice/flags); we merge them into one comparison. raw + oracle
appear in BOTH runs — we print both so the cross-run consistency is visible
before trusting the merge.

Figures (<dir>/figures):
  o4_disc_mat.png        MAT bar: baseline, raw, calib x4, disc x2, oracle
  o4_disc_survival.png   P(accept_len >= d): raw vs disc x2 vs oracle
  o4_disc_selection.png  suffix-chosen fraction by depth: raw vs disc x2 vs oracle

Usage (container):
  python3 simulation/scripts/plot_o4_disc.py \
      --dir        simulation/results/o4_perdepth/qwen3_14b_disc \
      --replay-dir simulation/results/o4_perdepth/qwen3_14b_replay
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_o4 import accept_lengths, suffix_chosen_by_depth  # noqa: E402

# MAT comparison order. calib x4 + baseline come from the replay dir; raw,
# oracle, disc x2 come from the disc dir (within-run-fair).
MAT_ORDER = ["baseline", "select1",
             "select1_calib_histogram", "select1_calib_isotonic",
             "select1_calib_logistic", "select1_calib_beta",
             "select1_disc_logistic", "select1_disc_beta", "select1_oracle"]
DISC_DIR_ARMS = {"select1", "select1_oracle",
                 "select1_disc_logistic", "select1_disc_beta"}
SURV_SEL_ARMS = ["select1", "select1_disc_logistic", "select1_disc_beta",
                 "select1_oracle"]
LABELS = {
    "baseline": "model-only", "select1": "raw",
    "select1_calib_histogram": "calib\n(hist)",
    "select1_calib_isotonic": "calib\n(iso)",
    "select1_calib_logistic": "calib\n(Platt)",
    "select1_calib_beta": "calib\n(beta)",
    "select1_disc_logistic": "DISC\n(logistic)",
    "select1_disc_beta": "DISC\n(beta)",
    "select1_oracle": "ORACLE",
}
COLORS = {
    "baseline": "#7f7f7f", "select1": "#1f77b4",
    "select1_calib_histogram": "#ffbb78", "select1_calib_isotonic": "#98df8a",
    "select1_calib_logistic": "#c5b0d5", "select1_calib_beta": "#c49c94",
    "select1_disc_logistic": "#2ca02c", "select1_disc_beta": "#17becf",
    "select1_oracle": "#e0b400",
}


def mat_of(summary, arm):
    a = summary.get("arms", {}).get(arm)
    if a is None:
        return None
    return a.get("accept_length_mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="discriminator eval run dir")
    ap.add_argument("--replay-dir", default=None,
                    help="sibling replay run with calib arms (same flags)")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    D = Path(args.dir)
    fig_dir = D / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    disc = json.load(open(D / "run.json"))
    rep = json.load(open(Path(args.replay_dir) / "run.json")) if args.replay_dir else None
    model = disc.get("model", "?").split("/")[-1]
    sub = (f"{model}, {disc.get('workload')} {disc.get('n_tasks')} test tasks, "
           f"S={disc.get('steps')}, topk=1, replay-all")

    # cross-run consistency: raw + oracle must agree between the two runs
    if rep is not None:
        for a in ("select1", "select1_oracle"):
            print(f"consistency {a}: disc={mat_of(disc,a)} replay={mat_of(rep,a)}")

    def pick_mat(arm):
        if arm in DISC_DIR_ARMS:
            return mat_of(disc, arm)
        return mat_of(rep, arm) if rep is not None else None

    present = [a for a in MAT_ORDER if pick_mat(a) is not None]
    mats = [pick_mat(a) for a in present]

    # ---- 1) combined MAT bars ----
    fig, ax = plt.subplots(figsize=(max(9, 1.25 * len(present) + 1.5), 4.8))
    xs = np.arange(len(present))
    bars = ax.bar(xs, mats, color=[COLORS[a] for a in present], width=0.66)
    for b, v in zip(bars, mats):
        ax.text(b.get_x() + b.get_width() / 2, v + max(mats) * 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    raw = pick_mat("select1"); orc = pick_mat("select1_oracle")
    if raw is not None:
        ax.axhline(raw, color="#1f77b4", ls="--", lw=0.9, label=f"raw={raw:.3f}")
    if orc is not None:
        ax.axhline(orc, color="#e0b400", ls="--", lw=0.9, label=f"oracle={orc:.3f}")
    ax.set_xticks(xs); ax.set_xticklabels([LABELS[a] for a in present], fontsize=8)
    ax.set_ylabel("MAT (mean accepted draft tokens / step)")
    ax.set_title(f"O4 joint discriminator vs raw / calib / oracle — MAT\n{sub}",
                 fontsize=10)
    ax.set_ylim(0, max(mats) * 1.18); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "o4_disc_mat.png", dpi=150)
    plt.close(fig)
    print("o4_disc_mat.png:", {a: round(m, 3) for a, m in zip(present, mats)})

    # ---- 2) survival ----
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    max_d = 0
    for a in SURV_SEL_ARMS:
        acc = accept_lengths(D / f"timing_{a}.jsonl")
        if acc.size == 0:
            continue
        dmax = int(acc.max()); max_d = max(max_d, dmax)
        depths = np.arange(1, dmax + 2)
        surv = [(acc >= d).mean() for d in depths]
        ax.plot(depths, surv, marker="o", ms=3, lw=1.6, color=COLORS[a],
                label=f"{LABELS[a].replace(chr(10),' ')} "
                      f"(MAT {mat_of(disc,a):.2f})")
    ax.axvline(args.steps, color="black", lw=1, ls=":")
    ax.set_yscale("log"); ax.set_xlabel("depth d")
    ax.set_ylabel("survival  P(accept_len >= d)")
    ax.set_title(f"O4 discriminator chain survival\n{sub}", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_xlim(0.5, max(max_d + 1, args.steps + 2))
    fig.tight_layout(); fig.savefig(fig_dir / "o4_disc_survival.png", dpi=150)
    plt.close(fig)
    print(f"o4_disc_survival.png (max depth {max_d})")

    # ---- 3) selection ratio by depth ----
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for a in SURV_SEL_ARMS:
        depths, frac, _ = suffix_chosen_by_depth(D / f"decisions_{a}.jsonl")
        if not depths:
            continue
        ax.plot(depths, frac, marker="o", ms=3, lw=1.6, color=COLORS[a],
                label=LABELS[a].replace(chr(10), " "))
    ax.set_xlabel("depth d"); ax.set_ylabel("suffix-chosen fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"O4 discriminator per-position selection\n{sub}", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "o4_disc_selection.png", dpi=150)
    plt.close(fig)
    print("o4_disc_selection.png")


if __name__ == "__main__":
    main()
