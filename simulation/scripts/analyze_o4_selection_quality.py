"""Why calibrated selection ~= raw (both << oracle): selection ACCURACY analysis.

On the oracle arm's decision log every decision carries eagle_p, suffix_p, the
draft depth, and the GROUND TRUTH (oracle_hit = which proposer matched GT). On
the DECISIVE decisions (exactly one proposer matches GT, oracle_hit in
{eagle,suffix}) the correct pick is known. We apply each selection rule OFFLINE
to those same inputs and measure how often it picks the GT proposer:

  always-eagle / always-suffix  prob-blind baselines
  raw                           pick suffix iff suffix_p > eagle_p
  calib_<method>                pick suffix iff cal_suffix(suffix_p,depth) >
                                cal_eagle(eagle_p,depth)   (per-position maps)
  oracle                        100% by definition

Also reports AUC of each rule's decision margin for separating suffix-right from
eagle-right — if AUC ~ 0.5, the proposer probabilities simply don't DISCRIMINATE
which one is right, so no monotone calibration can help (the ceiling is GT-only).

Figure (out_dir/figures/o4_selquality.png): accuracy bars + the margin
distribution overlap (suffix-right vs eagle-right) that limits every prob rule.

Usage:
  python3 simulation/scripts/analyze_o4_selection_quality.py \
      --dir simulation/results/o4_perdepth/qwen3_14b_replay --model-label EAGLE3
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation/oracle"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
sys.path.insert(0, "/workspace/simulation/oracle")
from chain_hybrid_patch import _ServingIsoCalibrator  # noqa: E402

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

METHODS = ["histogram", "isotonic", "logistic", "beta"]


def load_decisive(path):
    ep, sp, dep, corr = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail"):
                continue
            hit = r.get("oracle_hit")
            if hit not in ("eagle", "suffix"):
                continue  # decisive only: exactly one proposer matches GT
            if r.get("eagle_p") is None or r.get("suffix_p") is None:
                continue
            ep.append(float(r["eagle_p"])); sp.append(float(r["suffix_p"]))
            dep.append(int(r["depth"])); corr.append(1 if hit == "suffix" else 0)
    return (np.asarray(ep), np.asarray(sp), np.asarray(dep, int),
            np.asarray(corr, int))


def auc(y, score):
    if roc_auc_score is None or len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    args = ap.parse_args()
    D = Path(args.dir)
    ep, sp, dep, corr = load_decisive(D / "decisions_select1_oracle.jsonl")
    n = len(corr)
    fsuf = float(corr.mean())
    rows = []  # (label, accuracy, auc)
    rows.append(("always-%s" % args.model_label, float(np.mean(0 == corr)), float("nan")))
    rows.append(("always-suffix", float(np.mean(1 == corr)), float("nan")))
    raw_pick = (sp > ep).astype(int)
    rows.append(("raw", float(np.mean(raw_pick == corr)), auc(corr, sp - ep)))
    cal_margins = {}
    for m in METHODS:
        cal = _ServingIsoCalibrator.load(str(D / ("calib_pp_%s.json" % m)))
        cs = np.array([cal.predict("suffix", sp[i], sp[i], int(dep[i])) for i in range(n)])
        ce = np.array([cal.predict("eagle", ep[i], ep[i], int(dep[i])) for i in range(n)])
        pick = (cs > ce).astype(int)
        cal_margins[m] = cs - ce
        rows.append(("calib_%s" % m, float(np.mean(pick == corr)), auc(corr, cs - ce)))
    rows.append(("oracle", 1.0, 1.0))

    print(f"decisive decisions n={n}  (suffix-right={corr.sum()} / "
          f"{args.model_label}-right={n - corr.sum()}; suffix-right frac={fsuf:.3f})")
    print(f"  {'rule':18s} {'sel.acc':>8s} {'AUC':>7s}")
    for lab, a, u in rows:
        print(f"  {lab:18s} {a:8.3f} {u:7.3f}" if u == u else
              f"  {lab:18s} {a:8.3f} {'   -':>7s}")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    labs = [r[0] for r in rows]; accs = [r[1] for r in rows]
    colors = ["#999999", "#999999", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
              "#8c564b", "#e0b400"]
    ax1.bar(np.arange(len(labs)), accs, color=colors[:len(labs)])
    for i, a in enumerate(accs):
        ax1.text(i, a + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.axhline(fsuf, color="k", ls="--", lw=0.8, label=f"always-suffix={fsuf:.3f}")
    ax1.set_xticks(np.arange(len(labs)))
    ax1.set_xticklabels(labs, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("selection accuracy vs GT (decisive depths)")
    ax1.set_ylim(0, 1.02); ax1.grid(axis="y", alpha=0.3); ax1.legend(fontsize=8)
    ax1.set_title("How often each rule picks the GT proposer")

    # margin distribution overlap (raw): suffix-right vs eagle-right
    margin = sp - ep
    bins = np.linspace(-1, 1, 41)
    ax2.hist(margin[corr == 1], bins=bins, density=True, alpha=0.55,
             color="#ff7f0e", label="suffix is right")
    ax2.hist(margin[corr == 0], bins=bins, density=True, alpha=0.55,
             color="#1f77b4", label=f"{args.model_label} is right")
    ax2.axvline(0, color="k", lw=1, ls=":")
    ax2.set_xlabel("raw decision margin  (suffix_p − eagle_p)")
    ax2.set_ylabel("density")
    ax2.set_title(f"Margin barely separates the two classes\n"
                  f"(raw AUC={auc(corr, sp-ep):.3f}) → probs don't discriminate")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle(f"O4 selection quality — {args.model_label} (fair/replay-all)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = D / "figures" / "o4_selquality.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
