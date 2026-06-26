"""Root cause of calib < raw selection: the suppression -> wrong-flip chain.

On the oracle log's DECISIVE decisions (exactly one proposer matches GT) we
mechanistically decompose why each calibration method picks worse than raw:

  (1) SUPPRESSION — the per-position maps map suffix DOWN more than eagle, so the
      calibrated margin (cal_suffix - cal_eagle) is shifted toward eagle vs the
      raw margin (suffix_p - eagle_p). Shown by the map curves + mean margins.
  (2) FLIPS — relative to raw, calib flips some picks. Decompose each flip into
      suffix->eagle vs eagle->suffix, and whether it turned a CORRECT pick into a
      wrong one (lost) or a wrong pick into correct (fixed).
  (3) Because suffix is the GT proposer MORE often than eagle on decisive depths,
      suppressing suffix loses more than it fixes -> net accuracy (=> MAT) drops.

Figure (out_dir/figures/o4_calib_why.png): map curves cal_eagle vs cal_suffix
(suffix sits below) + per-method flip decomposition (net correct lost/fixed).

Usage:
  python3 simulation/scripts/analyze_o4_calib_why.py \
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

sys.path.insert(0, "/workspace/simulation/oracle")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
from chain_hybrid_patch import _ServingIsoCalibrator  # noqa: E402

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
            if r.get("oracle_hit") not in ("eagle", "suffix"):
                continue
            if r.get("eagle_p") is None or r.get("suffix_p") is None:
                continue
            ep.append(float(r["eagle_p"])); sp.append(float(r["suffix_p"]))
            dep.append(int(r["depth"])); corr.append(1 if r["oracle_hit"] == "suffix" else 0)
    return (np.asarray(ep), np.asarray(sp), np.asarray(dep, int), np.asarray(corr, int))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    args = ap.parse_args()
    D = Path(args.dir)
    ep, sp, dep, corr = load_decisive(D / "decisions_select1_oracle.jsonl")
    n = len(corr)
    raw_pick = (sp > ep).astype(int)
    raw_acc = float(np.mean(raw_pick == corr))
    print(f"decisive n={n}  suffix-right={corr.sum()} ({corr.mean():.3f})  "
          f"raw picks suffix={raw_pick.mean():.3f}  raw acc={raw_acc:.3f}")

    cals = {m: _ServingIsoCalibrator.load(str(D / f"calib_pp_{m}.json")) for m in METHODS}
    flip_stats = {}
    for m in METHODS:
        cal = cals[m]
        cs = np.array([cal.predict("suffix", sp[i], sp[i], int(dep[i])) for i in range(n)])
        ce = np.array([cal.predict("eagle", ep[i], ep[i], int(dep[i])) for i in range(n)])
        cpick = (cs > ce).astype(int)
        acc = float(np.mean(cpick == corr))
        s2e = (raw_pick == 1) & (cpick == 0)   # raw chose suffix, calib flips to eagle
        e2s = (raw_pick == 0) & (cpick == 1)
        # a flip is a LOSS if raw was correct & calib wrong; FIX if raw wrong & calib correct
        s2e_lost = int(np.sum(s2e & (corr == 1)))   # suffix was right -> calib now wrong
        s2e_fix = int(np.sum(s2e & (corr == 0)))    # eagle was right -> calib now correct
        e2s_lost = int(np.sum(e2s & (corr == 0)))
        e2s_fix = int(np.sum(e2s & (corr == 1)))
        flip_stats[m] = dict(acc=acc, pick_suffix=float(cpick.mean()),
                             s2e=int(s2e.sum()), e2s=int(e2s.sum()),
                             s2e_lost=s2e_lost, s2e_fix=s2e_fix,
                             e2s_lost=e2s_lost, e2s_fix=e2s_fix,
                             margin_shift=float(np.mean((cs - ce)) - np.mean((sp - ep))))
        net = (s2e_fix + e2s_fix) - (s2e_lost + e2s_lost)
        print(f"  {m:9s} acc={acc:.3f} picks_suffix={cpick.mean():.3f}  "
              f"flips: suffix->eagle={int(s2e.sum())} (lost {s2e_lost}, fix {s2e_fix}) | "
              f"eagle->suffix={int(e2s.sum())} (lost {e2s_lost}, fix {e2s_fix}) | "
              f"net correct {net:+d}")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    # Panel A: map curves at a representative depth (suffix below eagle = suppression)
    grid = np.linspace(0, 1, 101)
    d0 = 2
    cmap = {"histogram": "#ff7f0e", "isotonic": "#2ca02c", "logistic": "#9467bd", "beta": "#8c564b"}
    for m in METHODS:
        cal = cals[m]
        ce = [cal.predict("eagle", x, x, d0) for x in grid]
        cs = [cal.predict("suffix", x, x, d0) for x in grid]
        ax1.plot(grid, ce, color=cmap[m], lw=1.6, ls="-",
                 label=f"{m}: cal_{args.model_label.lower()}")
        ax1.plot(grid, cs, color=cmap[m], lw=1.6, ls="--",
                 label=f"{m}: cal_suffix")
    ax1.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.6)
    ax1.set_xlabel("raw probability p"); ax1.set_ylabel(f"calibrated P(accept) @depth {d0}")
    ax1.set_title(f"Maps suppress suffix below {args.model_label}\n"
                  "(dashed=suffix sits under solid=model → flips to model)", fontsize=10)
    ax1.legend(fontsize=6, ncol=2); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1.02)

    # Panel B: flip decomposition (net correct lost vs fixed) per method
    x = np.arange(len(METHODS)); w = 0.38
    lost = [flip_stats[m]["s2e_lost"] + flip_stats[m]["e2s_lost"] for m in METHODS]
    fix = [flip_stats[m]["s2e_fix"] + flip_stats[m]["e2s_fix"] for m in METHODS]
    ax2.bar(x - w / 2, lost, w, color="#d62728", label="correct → wrong (lost)")
    ax2.bar(x + w / 2, fix, w, color="#2ca02c", label="wrong → correct (fixed)")
    for i, m in enumerate(METHODS):
        ax2.text(i, max(lost[i], fix[i]) + 5,
                 f"net {fix[i]-lost[i]:+d}", ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(METHODS, fontsize=9)
    ax2.set_ylabel("decisions flipped vs raw")
    ax2.set_title("Calib flips lose more correct picks than it fixes\n"
                  "(suppressing suffix, which is right more often)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Why calib < raw — {args.model_label} (fair/replay-all)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = D / "figures" / "o4_calib_why.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
