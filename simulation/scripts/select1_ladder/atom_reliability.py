"""Suffix reliability drawn with the IDENTICAL reliability_count.py panel (so it matches
the other reliability graphs exactly): x = raw prob | calibrated prob, accept-rate o-line
+/-SE, dotted y=x, total+accepted COUNT bars on the twin axis (the count bar at the 0.5
bin IS the suffix prob-atom mass), shared count axis, same legend + ECE in title.

2 cells x {raw, calibrated} = a 4-panel figure, suffix only (the proposer whose prob
piles at atoms). Reuses reliability_count.panel/bin_stats/oof_isotonic verbatim.

Run (host): python3 simulation/scripts/select1_ladder/atom_reliability.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reliability_count as rc  # noqa: E402

OUTDIR = Path("simulation/results/calib_reliability/figures")
SUF = "#d62728"
CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", "EAGLE3", "eagle_token", "eagle_p"),
    "qwen35_27b_2way": ("qwen35_27b_ar", "MTP", "eagle_token", "eagle_p"),
}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CELLS), 2, figsize=(11, 4.4 * len(CELLS)), squeeze=False)
    for r, (tag, (dirname, mname, mtok, mpk)) in enumerate(CELLS.items()):
        d = f"{rc.ROOT}/{dirname}"
        chains = rc.load_chains(f"{d}/decisions_select1_oracle.jsonl")
        bad = rc.loopy_rids(d)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        pool = rc.collect(chains, [(mname, mtok, mpk, "#1f77b4"),
                                   ("Suffix", "suffix_token", "suffix_p", SUF)])
        p, y, g = pool["Suffix"]
        cal = rc.oof_isotonic(p, y, g)
        cymax = max(rc.bin_stats(p, y)[0].max(), rc.bin_stats(cal, y)[0].max())
        e_raw = rc.panel(axes[r][0], p, y, SUF,
                         f"Suffix [{tag}]: RAW prob  (n={len(y)}, base={y.mean():.3f})",
                         count_ymax=cymax)
        e_cal = rc.panel(axes[r][1], cal, y, SUF,
                         f"Suffix [{tag}]: CALIBRATED prob (OOF isotonic)",
                         count_ymax=cymax)
        axes[r][0].set_xlabel("raw prob", fontsize=8)
        axes[r][1].set_xlabel("calibrated prob", fontsize=8)
        print(f"{tag}: suffix n={len(y)} ECE raw={e_raw:.3f} -> cal={e_cal:.3f}")
    fig.suptitle("Suffix reliability (identical reliability_count format) — the count bar "
                 "at the 0.5 bin = the prob-atom mass\n"
                 "(accept-conditioned, label=token==gt; dotted=perfect calibration y=x)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outp = OUTDIR / "atom_reliability.png"
    fig.savefig(outp, dpi=140); plt.close(fig)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
