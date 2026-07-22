#!/usr/bin/env python3
r"""fig10: DFlash-as-prior tail (--tail-cal dfprior, raw head). MAT vs the prior
concentration s, per workload, against the succession (uniform prior) and fixed-
scalar baselines. Shows (a) dfprior beats succession at small s, (b) loses to the
fixed scalar, (c) opposite s-response by domain (text-heavy rises with s, copy-
heavy falls) — the DFlash-veto helps only where the suffix is unreliable.

  docker exec sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace python3 scripts/analysis/plot_dfprior_sweep.py'
"""
from __future__ import annotations
import re
from pathlib import Path

RLOG = Path("readable_outputs/figures/replay_logs")
OUT = Path("results/headweight_tailsmooth")
DSS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
K_RE = re.compile(r"^\s*calib:\s*K=([0-9.]+)")
SVALS = [("025", 0.25), ("05", 0.5), ("1", 1), ("2", 2), ("4", 4),
         ("8", 8), ("16", 16), ("32", 32)]
# raw-head baselines (from earlier sweeps)
SUCC = {"specbench": 3.55, "bfcl": 3.61, "swebench": 3.53, "spider": 4.13, "tau2": 3.37}
FIXED = {"specbench": 4.05, "bfcl": 3.94, "swebench": 3.60, "spider": 4.26, "tau2": 3.46}  # w=0.075
COL = {"specbench": "#1f77b4", "bfcl": "#ff7f0e", "swebench": "#2ca02c",
       "spider": "#d62728", "tau2": "#9467bd"}


def getk(p):
    if not p.exists():
        return None
    for ln in p.read_text().splitlines():
        m = K_RE.match(ln)
        if m:
            return float(m.group(1))
    return None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = {ds: [] for ds in DSS}
    for tag, sv in SVALS:
        for ds in DSS:
            k = getk(RLOG / f"mat_{ds}_hwts_rawhead_dfprior_s{tag}_split.replay.txt")
            curves[ds].append((sv, k))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    means = []
    for tag_i, (tag, sv) in enumerate(SVALS):
        vals = [getk(RLOG / f"mat_{ds}_hwts_rawhead_dfprior_s{tag}_split.replay.txt") for ds in DSS]
        if all(v is not None for v in vals):
            means.append((sv, sum(vals) / len(vals)))
    for ds in DSS:
        pts = [(s, k) for s, k in curves[ds] if k is not None]
        if pts:
            ax.plot([s for s, _ in pts], [k for _, k in pts], "-o", ms=4,
                    color=COL[ds], label=f"{ds}")
    if means:
        ax.plot([s for s, _ in means], [m for _, m in means], "-s", color="black",
                lw=2.5, ms=6, label="MEAN (5 wl)")
    # per-workload optimal-s mean (adaptive-s upper bound)
    adapt = 0.0
    for ds in DSS:
        best = max((k for _, k in curves[ds] if k is not None), default=0)
        adapt += best
    adapt /= len(DSS)
    ax.axhline(sum(SUCC.values()) / 5, ls="--", color="#8c564b", lw=1.3,
               label=f"succession (uniform prior)  {sum(SUCC.values())/5:.3f}")
    ax.axhline(sum(FIXED.values()) / 5, ls="--", color="#17becf", lw=1.3,
               label=f"fixed scalar (champion)  {sum(FIXED.values())/5:.3f}")
    ax.axhline(adapt, ls=":", color="black", lw=1.3,
               label=f"dfprior, per-wl-optimal s (adaptive UB)  {adapt:.3f}")
    ax.set_xscale("log"); ax.set_xlabel("DFlash prior concentration s  (log)")
    ax.set_ylabel("MAT (accepted tokens / round)")
    ax.set_title("fig10: DFlash-as-prior tail (k+q·s)/(n+s), raw head — MAT vs prior strength s")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(OUT / "fig10_dfprior_sweep.png", dpi=150)
    print("fig10 ->", OUT / "fig10_dfprior_sweep.png")
    print(f"MEAN-optimal:  {max(means, key=lambda t: t[1]) if means else None}")
    print(f"per-wl-optimal-s adaptive mean (upper bound): {adapt:.3f}")
    for ds in DSS:
        print(f"  {ds:10s} " + "  ".join(f"s{sv}={k}" for (sv, k) in
              [(s, kk) for s, kk in curves[ds]]))


if __name__ == "__main__":
    main()
