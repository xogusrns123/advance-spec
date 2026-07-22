#!/usr/bin/env python3
r"""Per-workload adaptation trace figures for the adaptive-scalar arms.

Panel 1: acting w_t vs global round (task-change boundaries shaded, per-wl
best fixed w* dashed). Panel 2: rolling-window MAT of the adaptive arm vs the
degenerate fixed baseline (deg0075) aligned by round index. English labels.

  docker exec sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace python3 \
    scripts/analysis/plot_adaptive_trace.py --arms cftl_w8k ftlw_inf ratio_g03 succratio_raw'
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ST = Path("results/adaptive_scalar")
DSS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
# per-wl best fixed w (raw head) from the existing fix sweep (E0 argbest)
WSTAR = {"specbench": 0.05, "bfcl": 0.05, "swebench": 0.10,
         "spider": 0.075, "tau2": 0.10}


def rolling(x, win):
    if len(x) < win:
        return np.array([]), np.array([])
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    m = (c[win:] - c[:-win]) / win
    return np.arange(win, len(x) + 1), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--baseline", default="deg0075")
    ap.add_argument("--roll", type=int, default=2000)
    ap.add_argument("--dss", nargs="+", default=DSS)
    args = ap.parse_args()

    for ds in args.dss:
        traces = {}
        for arm in args.arms + [args.baseline]:
            p = ST / f"trace_{ds}_{arm}.json"
            if p.exists():
                traces[arm] = json.load(open(p))
        if not traces or args.baseline not in traces:
            print(f"[{ds}] missing traces, skip")
            continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        # task boundaries from the baseline's rid_index
        ridx = traces[args.baseline].get("rid_index", [])
        prev_task, bounds = None, []
        for rid, task, start in ridx:
            if task != prev_task:
                bounds.append((start, task)); prev_task = task
        for arm in args.arms:
            tr = traces.get(arm)
            if tr is None:
                continue
            ax1.plot(tr["w"], lw=0.9, label=arm)
            xs, ms = rolling(tr["acc"], args.roll)
            if len(xs):
                ax2.plot(xs, ms, lw=1.0, label=arm)
        xs, ms = rolling(traces[args.baseline]["acc"], args.roll)
        if len(xs):
            ax2.plot(xs, ms, lw=1.0, ls="--", color="#7f7f7f",
                     label=f"{args.baseline} (fixed)")
        if ds in WSTAR:
            ax1.axhline(WSTAR[ds], ls=":", color="#d62728", lw=1,
                        label=f"per-wl best fixed w*={WSTAR[ds]}")
        for start, task in bounds[1:]:
            ax1.axvline(start, color="#cccccc", lw=0.5)
        ax1.set_ylabel("acting tail weight w_t")
        ax1.set_title(f"{ds}: adaptive scalar trace (test half)")
        ax1.legend(fontsize=7, ncol=2)
        ax2.set_ylabel(f"rolling-{args.roll} MAT")
        ax2.set_xlabel("round (stream order)")
        ax2.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        out = ST / f"trace_{ds}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"figure -> {out}")


if __name__ == "__main__":
    main()
