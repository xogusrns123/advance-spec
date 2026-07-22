#!/usr/bin/env python3
"""Per-workload SUBTASK MAT figures — ONE image per workload (like the segment
figures, but columns are task labels instead of output segments). Same five
arms, colors and format as plot_mat_4way (reused directly):

  dflash / suffix / SD-paper hybrid (fallback, best τ) / Compose / Oracle

  specbench : 6 subtasks (8 MT-bench cats pooled into mt_bench)
  bfcl      : 5 categories
  swebench  : 11 repos (django, astropy, matplotlib, ...)
  spider    : single label in the record — pass --spider-by-db to split by the
              conv_map task_id (database); off by default (53 mostly-singleton
              databases -> noisy).

  python3 scripts/plot_mat_subtask.py [--workloads specbench bfcl swebench]
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_mat_4way as p4

# per-workload capture traces (for the per-subtask TASK counts = distinct eval convs)
TRACES = {"specbench": "results/perpos_specbench_alleval/specbench_4way.traces.json",
          "bfcl": "results/perpos_bfcl_alleval/bfcl_4way.traces.json",
          "swebench": "results/perpos_swebench_alleval/swebench_4way.traces.json",
          "spider": "results/perpos_spider_alleval/spider_4way.traces.json",
          "tau2": "results/perpos_tau2_alleval/tau2_4way.traces.json"}


def task_counts(ds):
    """{task_label: # distinct eval conversations (tasks)}; MT-bench cats pooled."""
    p = p4.BASE / TRACES.get(ds, "")
    if not p.exists():
        return {}
    tr = json.load(open(p))
    seen = {}
    for t in tr["eval_traces"]:
        seen.setdefault(t.get("task", "all"), set()).add(t.get("conv"))
    counts = {lab: len(s) for lab, s in seen.items()}
    mt = {lab: counts.pop(lab) for lab in list(counts) if lab in p4.MT_CATS}
    if mt:
        pooled = {t.get("conv") for t in tr["eval_traces"] if t.get("task") in p4.MT_CATS}
        counts["mt_bench"] = len(pooled)
    return counts

OUT = p4.OUT
DS_LABEL = {"specbench": "Spec-Bench", "bfcl": "BFCL v4",
            "swebench": "SWE-bench Verified", "spider": "Spider2-DBT",
            "tau2": "τ²-bench"}
# harness + task count subtitle (consistent across every MAT figure)
WL_INFO = {"specbench": "480 tasks (all) · 6 subtasks",
           "bfcl": "bfcl_eval (prompt-mode FC) · 753 tasks · 5 categories",
           "swebench": "mini-swe-agent · 19 of 500 tasks (self-terminated, 250 steps) · 7 repos",
           "spider": "spider-agent-dbt · 68 tasks (all) · 68 databases",
           "tau2": "tau2 official sim · 64 tasks · 3 domains"}


def order_tasks(ds, T):
    """subtask column order: curated for specbench/bfcl, size-desc for the rest."""
    if ds in p4.SUB_ORDER:
        subs = [t for t in p4.SUB_ORDER[ds] if t in T[ds]]
        subs += [t for t in sorted(T[ds]) if t not in subs]
        return subs
    return sorted(T[ds], key=lambda t: -T[ds][t].get("dflash", (0, 0))[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-suffix", default="")
    ap.add_argument("--compose-suffix", default="",
                    help="override ONLY compose(calib) group, e.g. _raw (no calib)")
    ap.add_argument("--out-suffix", default="", help="appended to filename, e.g. _nocalib")
    ap.add_argument("--note", default="")
    ap.add_argument("--sweep-dir",
                    default="/workspace/simulation/results/pipeline_4way/segments")
    ap.add_argument("--workloads", nargs="+",
                    default=["specbench", "bfcl", "swebench", "spider", "tau2"])
    args = ap.parse_args()

    p4.LOG_SUFFIX = args.log_suffix
    p4.COMPOSE_SUFFIX = args.compose_suffix
    if args.compose_suffix == "_raw":
        p4.LABELS["calib"] = "Compose (raw, no calib)"
    OUT_SUFFIX = args.out_suffix
    p4.SWEEP_DIR = args.sweep_dir
    K, T = p4.load()
    OUT.mkdir(parents=True, exist_ok=True)
    corner = ("τ swept: {" + ", ".join(f"{t:g}" for t in p4.FB_GRID) + "}\n"
              "τ* on each SD-paper hybrid bar = best per workload") if p4.FB_GRID else ""

    for ds in args.workloads:
        # per-task breakdown only when the workload has >1 real task label; else a
        # single pooled column from the OVERALL K (spider has no subtask-type axis —
        # 68 one-off databases — and its per-task dict only carries the fallback arm).
        tc = task_counts(ds)
        if T.get(ds) and len(T[ds]) > 1:
            tasks = order_tasks(ds, T)
            vals = {t: T[ds][t] for t in tasks}
            names = [f"{t}\ntasks={tc.get(t, '?')}" for t in tasks]
        elif K.get(ds):
            tasks = ["all databases"]
            vals = {tasks[0]: K[ds]}
            names = [f"all databases\ntasks={sum(tc.values()) or '?'}"]
        else:
            print(f"[{ds}] no data, skip")
            continue

        narrow = len(tasks) <= 3
        fig, ax = plt.subplots(figsize=(max(7.5, 2.6 + 1.9 * len(tasks)), 5.6))
        ymax = p4.bars(ax, tasks, lambda t: vals[t],
                       tau_of=lambda t: p4.FB_TAUS.get(ds), fs_val=8.5)
        if narrow:                       # extra top room so the legend clears the bars
            ax.set_ylim(0, ymax * 1.45)
        ax.set_xticks(range(len(tasks)))
        rot = 20 if len(tasks) > 6 else 0
        ax.set_xticklabels(names, fontsize=11 if len(tasks) <= 6 else 9.5,
                           rotation=rot, ha="center" if rot == 0 else "right")
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel("mean accept length  (tokens)", fontsize=13)
        ax.set_title(f"{DS_LABEL[ds]} — MAT per subtask\n{WL_INFO[ds]}",
                     fontsize=13)
        ax.legend(fontsize=10.5, frameon=False, loc="upper left",
                  ncol=1 if narrow else 2)
        if corner:
            ax.text(0.99, 0.99, corner, transform=ax.transAxes, ha="right",
                    va="top", fontsize=9.5, color="#555555")
        fig.tight_layout()
        fp = OUT / f"MAT_subtask_{ds}{OUT_SUFFIX}.png"
        fig.savefig(fp, dpi=150); plt.close(fig)
        print(f"saved {fp}")
        for t in tasks:
            print(f"  [{t.splitlines()[0]}] " + "  ".join(
                f"{p}={vals[t].get(p, (0, 0))[0]:.2f}({vals[t].get(p, (0, 0))[1]})"
                for p in p4.PROPS))


if __name__ == "__main__":
    main()
