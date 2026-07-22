#!/usr/bin/env python3
"""Per-depth COMPETITION (first-crossing hand-off) vs COMPOSITION (argmax
hand-off) across the trimmed trend workloads, raw and calibrated signals for
both controllers — 6 bars per workload:

  DFlash only | Suffix only | Select raw | Select calib | Comp raw | Comp calib

  select raw   = extend head while affine(conf_j) > T/(1+T), T = raw arctic score
  select calib = SAME first-crossing rule, logistic hazard + isotonic T (clean
                 signal ablation at fixed controller)
  comp raw     = argmax_k[1+G_k+S_k*T] with affine hazard + scalar raw T
  comp calib   = argmax with logistic hazard + per-k isotonic T_k

Sources (same 3-way disjoint test splits as all MAT figures):
  mat_{bfcl,specbench}_select.replay.txt   by-task select/select_calib (+rounds)
  mat_{swebench,spider}_select.replay.txt  overall select/select_calib
  mat_bfcl_wl / mat_specbench_wl / mat_swebench / mat_spider_wl .replay.txt
                                           dflash/suffix/chain/calib

  python3 scripts/plot_select_per_workload.py
"""
from __future__ import annotations
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

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "readable_outputs" / "figures" / "replay_logs"    # replay-log inputs
OUT = BASE / "readable_outputs" / "figures" / "mat"            # figure outputs
PROPS = ["dflash", "suffix", "select", "select_calib", "chain", "calib"]
LABELS = {"dflash": "DFlash only", "suffix": "Suffix only",
          "select": "Select (raw)", "select_calib": "Select (calibrated)",
          "chain": "Composition (raw)", "calib": "Composition (calibrated)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "select": "#E45756",
          "select_calib": "#B94A8C", "chain": "#54A24B", "calib": "#2CA0A0"}
ORDER = ["qa", "math_reasoning", "web_search", "memory", "swebench", "spider_dbt"]
POOL = {"memory": ["memory_kv", "memory_rec_sum", "memory_vector"]}
DISPLAY = {"qa": "QA\n(Spec-Bench)", "math_reasoning": "Math\n(Spec-Bench)",
           "web_search": "Web Search\n(BFCL v4)", "memory": "Memory\n(BFCL v4)",
           "swebench": "SWE-Bench\n(Verified)", "spider_dbt": "Agentic SQL\n(Spider2-DBT)"}


def parse_by_task(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s+\[(.+?)\]\s+(.*)", line)
        if m:
            vals = {p: (float(v), int(r) if r else None)
                    for p, v, r in re.findall(r"(\w+)=([0-9.]+)(?:\((\d+)\))?",
                                              m.group(2))}
            if set(vals) & set(PROPS):
                out.setdefault(m.group(1), {}).update(vals)
    return out


def parse_overall(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s+(\w+): K=([0-9.]+)\s+\(rounds=(\d+)\)", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), int(m.group(3)))
    return out


def main():
    by_task = {}
    for f in ("mat_specbench_wl", "mat_specbench_select",
              "mat_bfcl_wl", "mat_bfcl_select"):
        for t, vals in parse_by_task((FIG / f"{f}.replay.txt").read_text()).items():
            by_task.setdefault(t, {}).update(vals)
    K = {t: {p: v[0] for p, v in d.items()} for t, d in by_task.items()}
    for wl, files in (("swebench", ("mat_swebench", "mat_swebench_select")),
                      ("spider_dbt", ("mat_spider_wl", "mat_spider_select"))):
        K[wl] = {}
        for f in files:
            K[wl].update({p: v for p, (v, _) in
                          parse_overall((FIG / f"{f}.replay.txt").read_text()).items()})
    for w, members in POOL.items():                 # round-weighted pool
        K[w] = {}
        for p in PROPS:
            ks = [by_task[t][p] for t in members if p in by_task.get(t, {})]
            if len(ks) == len(members) and all(r is not None for _, r in ks):
                K[w][p] = sum(k * r for k, r in ks) / sum(r for _, r in ks)
    wls = [w for w in ORDER if w in K and K[w]]

    n, g = len(PROPS), 0.82
    bw = g / n
    fig, ax = plt.subplots(figsize=(2.2 + 2.1 * len(wls), 5.0))
    for pi, p in enumerate(PROPS):
        xs = [ti - g / 2 + bw * (pi + 0.5) for ti in range(len(wls))]
        ys = [K[w].get(p, 0.0) for w in wls]
        ax.bar(xs, ys, width=bw * 0.90, color=COLORS[p], label=LABELS[p])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=7, color="#555555")
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels([DISPLAY.get(w, w) for w in wls], fontsize=10)
    ax.set_ylabel("mean accept length / round  (tokens)", fontsize=10)
    ax.set_title("Per-depth competition (first-crossing) vs composition (argmax) — "
                 "raw vs calibrated signals   [measured]\n"
                 "Qwen3.5-27B + DFlash + Suffix — 3-way disjoint split, "
                 "full-round replay", fontsize=10)
    ymax = max(K[w].get(p, 0) for w in wls for p in PROPS)
    ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "select_vs_composition_per_workload.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")
    for w in wls:
        print(f"  [{w}] " + "  ".join(f"{p}={K[w].get(p, 0):.2f}" for p in PROPS))


if __name__ == "__main__":
    main()
