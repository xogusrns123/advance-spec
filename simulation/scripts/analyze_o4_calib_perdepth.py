"""Per-DEPTH decomposition of why per-position calibration still loses to raw.

The calibration IS per-position (depth-indexed maps, applied via
cal.predict(group, p, depth)). This script proves the suppression -> wrong-flip
mechanism is NOT an aggregation artifact: it holds at (almost) EVERY depth.

For each composed-chain depth d, on the DECISIVE decisions (exactly one proposer
matches GT) we report:
  n(d)                  decisive decisions at depth d
  suffix-right frac     fraction where suffix is the GT proposer (the prior)
  raw picks_suffix      raw rule (suffix_p > eagle_p) suffix share
  calib picks_suffix    per-depth calibrated rule suffix share  (per method)
  raw acc / calib acc   selection accuracy vs GT at that depth

If calib under-picks suffix vs raw AND has <= accuracy at each depth, then
per-depth calibration is not the missing ingredient -- the failure is the
cross-proposer RANKING at each depth, which monotone per-depth recalibration
cannot fix.

Figure (out_dir/figures/o4_calib_perdepth.png): suffix-pick rate by depth
(optimal vs raw vs calib) + selection accuracy by depth (raw vs calib).

Usage:
  python3 simulation/scripts/analyze_o4_calib_perdepth.py \
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
CMAP = {"histogram": "#ff7f0e", "isotonic": "#2ca02c",
        "logistic": "#9467bd", "beta": "#8c564b"}


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
            dep.append(int(r["depth"]))
            corr.append(1 if r["oracle_hit"] == "suffix" else 0)
    return (np.asarray(ep), np.asarray(sp), np.asarray(dep, int),
            np.asarray(corr, int))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    ap.add_argument("--min-n", type=int, default=30,
                    help="only plot depths with >= this many decisive decisions")
    args = ap.parse_args()
    D = Path(args.dir)
    ep, sp, dep, corr = load_decisive(D / "decisions_select1_oracle.jsonl")
    n = len(corr)
    raw_pick = (sp > ep).astype(int)

    cals = {m: _ServingIsoCalibrator.load(str(D / f"calib_pp_{m}.json"))
            for m in METHODS}
    # per-method calibrated pick over all rows (uses per-depth map)
    cpick = {}
    for m in METHODS:
        cal = cals[m]
        cs = np.array([cal.predict("suffix", sp[i], sp[i], int(dep[i]))
                       for i in range(n)])
        ce = np.array([cal.predict("eagle", ep[i], ep[i], int(dep[i]))
                       for i in range(n)])
        cpick[m] = (cs > ce).astype(int)

    depths = sorted(set(dep.tolist()))
    print(f"decisive n={n}  overall suffix-right={corr.mean():.3f}  "
          f"raw picks_suffix={raw_pick.mean():.3f} acc={(raw_pick==corr).mean():.3f}")
    hdr = (f"{'d':>2s} {'n':>5s} {'suf-rt':>6s} {'raw_ps':>6s} {'raw_ac':>6s}"
           + "".join(f" {m[:4]+'_ps':>8s} {m[:4]+'_ac':>8s}" for m in METHODS))
    print(hdr)
    rows = {}
    for d in depths:
        mk = dep == d
        nd = int(mk.sum())
        sr = float(corr[mk].mean())
        rps = float(raw_pick[mk].mean())
        rac = float((raw_pick[mk] == corr[mk]).mean())
        line = f"{d:2d} {nd:5d} {sr:6.3f} {rps:6.3f} {rac:6.3f}"
        rec = dict(n=nd, sr=sr, raw_ps=rps, raw_ac=rac)
        for m in METHODS:
            cps = float(cpick[m][mk].mean())
            cac = float((cpick[m][mk] == corr[mk]).mean())
            rec[f"{m}_ps"] = cps; rec[f"{m}_ac"] = cac
            line += f" {cps:8.3f} {cac:8.3f}"
        rows[d] = rec
        print(line)

    pd = [d for d in depths if rows[d]["n"] >= args.min_n]
    xs = np.array(pd)

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.5]})

    # Panel A: suffix-pick rate by depth
    ax0.plot(xs, [rows[d]["sr"] for d in pd], "k-o", lw=2.2, ms=4,
             label="optimal share (suffix-right)", zorder=5)
    ax0.plot(xs, [rows[d]["raw_ps"] for d in pd], color="#1f77b4", lw=2.0,
             marker="s", ms=3, label="raw picks_suffix")
    for m in METHODS:
        ax0.plot(xs, [rows[d][f"{m}_ps"] for d in pd], color=CMAP[m], lw=1.3,
                 ls="--", marker=".", ms=3, label=f"calib_{m}")
    ax0.set_xlabel("composed-chain depth d"); ax0.set_ylabel("suffix-pick rate")
    ax0.set_title("Calib under-picks suffix at EVERY depth\n"
                  "(both raw and calib vs the optimal share)", fontsize=10)
    ax0.set_ylim(0, 1.02); ax0.grid(alpha=0.3); ax0.legend(fontsize=7)

    # Panel B: selection accuracy by depth
    ax1.plot(xs, [rows[d]["sr"] if rows[d]["sr"] >= 0.5 else 1 - rows[d]["sr"]
                  for d in pd], "k:", lw=1.4,
             label="always-pick-majority")
    ax1.plot(xs, [rows[d]["raw_ac"] for d in pd], color="#1f77b4", lw=2.2,
             marker="s", ms=3, label="raw acc")
    for m in METHODS:
        ax1.plot(xs, [rows[d][f"{m}_ac"] for d in pd], color=CMAP[m], lw=1.3,
                 ls="--", marker=".", ms=3, label=f"calib_{m} acc")
    ax1.set_xlabel("composed-chain depth d")
    ax1.set_ylabel("selection accuracy vs GT")
    ax1.set_title("raw acc >= calib acc at (almost) every depth\n"
                  "(per-depth calibration does not close the gap)", fontsize=10)
    ax1.set_ylim(0.4, 1.02); ax1.grid(alpha=0.3); ax1.legend(fontsize=7)

    # Panel C: decision mass by depth (why shallow depths dominate)
    ax2.bar(xs, [rows[d]["n"] for d in pd], color="#bbbbbb")
    ax2.set_xlabel("depth d"); ax2.set_ylabel("# decisive decisions")
    ax2.set_title("decision mass\n(shallow depths dominate)", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Per-depth: calibration suppresses suffix & loses accuracy at "
                 f"each depth — {args.model_label} (fair/replay-all)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = D / "figures" / "o4_calib_perdepth.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
