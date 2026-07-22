#!/usr/bin/env python3
"""Post-calibration reliability diagrams: CALIBRATED score (x) vs realized
conditional accept (y), per workload x per method (independent figures).

  head (DFlash) : x = cal_h(conf) = P(match) after LOGISTIC head calibration;
                  y = realized P(match) in that calibrated-prob bin.
                  Well-calibrated -> points on the y=x diagonal (prob space).
  tail (Suffix) : x = cal_t(arctic score) = E[accept] after ISOTONIC tail
                  calibration; y = realized accept length in that bin.
                  Well-calibrated -> points on y=x (token space).

Reuses the extracted (raw_score, realized) pairs in the reliability cache
(scripts/plot/calib_reliability_boxplot.py --extract).

-> readable_outputs/figures/mat/calibration/calib_reliability/
     CALIBREL_<wl>_head.png , CALIBREL_<wl>_tail.png
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calib_reliability_boxplot import dflash_fit, suffix_fit  # same fitters

CACHE = Path("/workspace/simulation/results/pipeline_4way/calib_pairs")
OUT = Path("readable_outputs/figures/mat/calibration/calib_reliability")
WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
# method: (proposer, calibrator name, deployed label, color, unit)
METHODS = {
    "head": ("dflash", "logistic", "DFlash head — logistic calibration",
             "#4C78A8", "P(match)"),
    "tail": ("suffix", "isotonic", "Suffix tail — isotonic calibration",
             "#F58518", "accept length (tokens)"),
}


def load_pairs(wl, proposer):
    fp = CACHE / f"{wl}_{proposer}.json.gz"
    if not fp.exists():
        return None, None
    with gzip.open(fp, "rt") as f:
        data = json.load(f)
    xs = np.array([p[0] for v in data.values() for p in v], dtype=float)
    ys = np.array([p[1] for v in data.values() for p in v], dtype=float)
    return xs, ys


def apply_cal(cal, xs):
    if hasattr(cal, "predict"):
        return np.asarray(cal.predict(xs), dtype=float)
    return np.array([cal(float(x)) for x in xs], dtype=float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for wl in WLS:
        for mkey, (prop, cname, label, col, unit) in METHODS.items():
            xs, ys = load_pairs(wl, prop)
            if xs is None or len(xs) < 50:
                print(f"[{wl}/{mkey}] no/low cache, skip", flush=True); continue
            fitter = dflash_fit if prop == "dflash" else suffix_fit
            cal = fitter(cname, xs.tolist(), ys.tolist())
            cx = apply_cal(cal, xs)                      # calibrated score
            hi = 1.0 if prop == "dflash" else float(np.percentile(cx, 99.5))
            hi = max(hi, 1e-6)
            edges = np.linspace(0, hi, 13)
            mids, my, cnt = [], [], []
            for lo, up in zip(edges[:-1], edges[1:]):
                m = (cx >= lo) & (cx < up)
                if m.sum() >= 20:
                    mids.append((lo + up) / 2)
                    my.append(float(ys[m].mean()))
                    cnt.append(int(m.sum()))
            if not mids:
                print(f"[{wl}/{mkey}] no populated bins, skip", flush=True); continue
            mids, my, cnt = np.array(mids), np.array(my), np.array(cnt)
            # ECE: |calibrated - realized| weighted by bin count
            ece = float((np.abs(mids - my) * cnt).sum() / cnt.sum())

            fig, ax = plt.subplots(figsize=(6.2, 6.0))
            lim = hi
            ax.plot([0, lim], [0, lim], ls="--", color="#888", lw=1.4,
                    zorder=1, label="perfect calibration (y=x)")
            sizes = 30 + 300 * cnt / cnt.max()
            ax.scatter(mids, my, s=sizes, c=col, edgecolor="k", linewidth=0.6,
                       alpha=0.85, zorder=3, label="calibrated bins (size ∝ n)")
            ax.plot(mids, my, color=col, lw=1.2, alpha=0.6, zorder=2)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_aspect("equal", "box")
            ax.set_xlabel(f"calibrated score  ({unit})", fontsize=11)
            ax.set_ylabel(f"realized {unit}", fontsize=11)
            ax.set_title(f"{WL_NAME[wl]} — {label}\nreliability (ECE={ece:.3f})",
                         fontsize=12, fontweight="bold")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9, loc="upper left", frameon=True)
            fig.tight_layout()
            fp = OUT / f"CALIBREL_{wl}_{mkey}.png"
            fig.savefig(fp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved -> {fp}  (ECE={ece:.3f}, bins={len(mids)})", flush=True)


if __name__ == "__main__":
    main()
