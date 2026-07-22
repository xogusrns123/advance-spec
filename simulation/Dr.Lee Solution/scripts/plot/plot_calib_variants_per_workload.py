#!/usr/bin/env python3
"""Per-workload calibration-variant comparison (4-way), one INDEPENDENT figure
per workload. Bars: compose with
  raw            (no calibration)
  logistic head  (head=logistic, tail=raw)
  isotonic tail  (head=raw, tail=isotonic)
  log head + iso tail  (head=logistic, tail=isotonic)
K read from the all-eval deck replay logs mat_<ds>_4way_calib_{raw,H-logistic,
isotail,logiso}.replay.txt.

-> readable_outputs/figures/mat/calibration/CALIB_variants_<ds>.png
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RLOG = Path("readable_outputs/figures/replay_logs")
OUT = Path("readable_outputs/figures/mat/calibration")
WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
# (log suffix, label, color)  green gradient raw->fully calibrated
VARIANTS = [("_raw", "no calib", "#cfe8c8"),
            ("_H-logistic", "head calib.\n(logistic)", "#93cd8a"),
            ("_isotail", "tail calib.\n(isotonic)", "#57a457"),
            ("_logiso", "head&tail calib.", "#2f7a2f")]


def compose_k(ds, sfx):
    txt = (RLOG / f"mat_{ds}_4way_calib{sfx}.replay.txt").read_text()
    m = re.search(r"calib:\s*K=([\d.]+)", txt)
    return float(m.group(1)) if m else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in WLS:
        ks = [compose_k(ds, sfx) for sfx, _, _ in VARIANTS]
        if any(k is None for k in ks):
            print(f"[{ds}] missing log, skip", flush=True); continue
        labels = [lab for _, lab, _ in VARIANTS]
        cols = [c for _, _, c in VARIANTS]
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        b = ax.bar(range(len(ks)), ks, width=0.62, color=cols,
                   edgecolor="k", linewidth=0.6, zorder=3)
        for bar, k in zip(b, ks):
            ax.text(bar.get_x() + bar.get_width() / 2, k + 0.03, f"{k:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("MAT  (mean accepted tokens / verify step)", fontsize=11)
        ax.set_title(f"{WL_NAME[ds]} — compose calibration variants (all-eval)",
                     fontsize=12.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_ylim(0, max(ks) * 1.15)
        fig.tight_layout()
        fp = OUT / f"CALIB_variants_{ds}.png"
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {fp}  ({'/'.join(f'{k:.2f}' for k in ks)})", flush=True)


if __name__ == "__main__":
    main()
