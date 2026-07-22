#!/usr/bin/env python3
"""Warming effect, 4-way. Warming fills the suffix GLOBAL tree from the warm set;
DFlash uses no suffix so it is warming-invariant and dropped. The four
suffix-touching arms of the 4-way deck are shown:
  suffix (single) / SD-paper hybrid (fallback @ pooled-best tau) /
  compose (beta head + isotonic tail) / oracle (best handoff).

  warming ON  : suffix.fit(warm set)  -> cross-trajectory memory present
  warming OFF : suffix.fit([])         -> only within-eval add_response growth
Calibrators (compose) are fit once and held fixed across on/off, so each bar
isolates the GLOBAL-TREE warming effect for that arm. Tested on the eval set.

-> readable_outputs/figures/mat/warming/WARMING_effect_4way.png
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from replay_extension import _ad, _fit_beta, _fit_tail_iso   # noqa: E402
from measure_k_fusion import ArcticSuffix                    # noqa: E402
from replay_segments_5way import replay_arm                  # noqa: E402

WL = {"specbench": "results/perpos_specbench_alleval/specbench_4way",
      "bfcl": "results/perpos_bfcl_alleval/bfcl_4way",
      "swebench": "results/perpos_swebench_alleval/swebench_4way",
      "spider": "results/perpos_spider_alleval/spider_4way",
      "tau2": "results/perpos_tau2_alleval/tau2_4way"}
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
TAU = {"specbench": 32, "bfcl": 16, "swebench": 16, "spider": 16, "tau2": 16}
ARMS = ["suffix", "fallback", "calib", "oracle"]
ARM_NAME = {"suffix": "Suffix (single)", "fallback": "SD-paper hybrid",
            "calib": "Compose", "oracle": "Oracle (best handoff)"}
ARM_COL = {"suffix": "#F58518", "fallback": "#9467BD", "calib": "#54A24B",
           "oracle": "#E45756"}
MAXR = 4096
OUT = Path("readable_outputs/figures/mat/warming")


def arm_mat(recs, eval_traces, warm_traces, arm, num_spec, calib, tau, warm_on):
    suffix = ArcticSuffix(); suffix.fit(warm_traces if warm_on else [])
    tot, n = 0.0, 0
    for rid, rby in recs.items():
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        for m, acc in replay_arm(rby, tr["output_ids"], tr["prompt_ids"],
                                 suffix, arm, num_spec, MAXR, calib, tau):
            tot += acc; n += 1
    return tot / n if n else 0.0


def build_calib(recs, warm_traces, eval_traces, num_spec):
    pairs = []
    for rby in recs.values():
        for r in rby.values():
            conf, match = r["dflash_conf"], r["dflash_match"]
            ad = _ad(match)
            for d in range(min(ad + 1, len(conf))):
                pairs.append((float(conf[d]), int(match[d])))
    hx = [x for x, _ in pairs]; hy = [y for _, y in pairs]
    cal_h = _fit_beta(hx, hy)
    cal_t = _fit_tail_iso(warm_traces, recs, eval_traces, set(recs), num_spec, MAXR)
    return (cal_h, cal_t)


CELLDIR = OUT / "cells"

# already-computed values from the serial run (avoid recompute)
SEED = {
    ("specbench", "suffix"): (1.17, 1.13), ("specbench", "fallback"): (4.00, 4.02),
    ("specbench", "calib"): (3.84, 3.86), ("specbench", "oracle"): (4.75, 4.71),
    ("bfcl", "suffix"): (2.08, 1.70), ("bfcl", "fallback"): (3.65, 3.46),
    ("bfcl", "calib"): (3.90, 3.62), ("bfcl", "oracle"): (5.14, 4.71),
    ("swebench", "suffix"): (2.48, 2.32), ("swebench", "fallback"): (2.66, 2.59),
}


def compute_cell(wl, arm):
    stem = WL[wl]
    traces = json.load(open(stem + ".traces.json"))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)
    recs = defaultdict(dict)
    for l in open(stem + ".jsonl"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r
    calib = build_calib(recs, warm_traces, eval_traces, num_spec) if arm == "calib" else None
    on = arm_mat(recs, eval_traces, warm_traces, arm, num_spec, calib, TAU[wl], True)
    off = arm_mat(recs, eval_traces, warm_traces, arm, num_spec, calib, TAU[wl], False)
    CELLDIR.mkdir(parents=True, exist_ok=True)
    json.dump({"wl": wl, "arm": arm, "on": on, "off": off},
              open(CELLDIR / f"cell_{wl}_{arm}.json", "w"))
    print(f"{wl:<10} {arm:<9} on={on:.2f} off={off:.2f} gain={on-off:+.2f}", flush=True)


def seed_cells():
    CELLDIR.mkdir(parents=True, exist_ok=True)
    for (wl, arm), (on, off) in SEED.items():
        json.dump({"wl": wl, "arm": arm, "on": on, "off": off},
                  open(CELLDIR / f"cell_{wl}_{arm}.json", "w"))
    print(f"seeded {len(SEED)} cells", flush=True)


def do_plot():
    data = {wl: {"arms": {}} for wl in WL}
    for wl in WL:
        for arm in ARMS:
            fp = CELLDIR / f"cell_{wl}_{arm}.json"
            if not fp.exists():
                print(f"MISSING cell {wl}/{arm}", flush=True)
                continue
            c = json.load(open(fp))
            data[wl]["arms"][arm] = {"on": c["on"], "off": c["off"]}
    json.dump(data, open(OUT / "warming_4way.json", "w"), indent=1)

    # ---- 2x2 grid, one panel per arm; 5 workloads x (OFF grey, ON color) ----
    wls = list(WL.keys())
    x = np.arange(len(wls)); w = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, arm in zip(axes.ravel(), ARMS):
        off = [data[wl]["arms"][arm]["off"] for wl in wls]
        on = [data[wl]["arms"][arm]["on"] for wl in wls]
        b1 = ax.bar(x - w / 2, off, w, label="warming OFF (empty tree)",
                    color="#bdbdbd", edgecolor="k", linewidth=0.5, zorder=3)
        b2 = ax.bar(x + w / 2, on, w, label="warming ON (warm-set tree)",
                    color=ARM_COL[arm], edgecolor="k", linewidth=0.5, zorder=3)
        top = max(max(on), max(off))
        dv = top * 0.014                       # bar-value label offset
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dv,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        for i in range(len(wls)):
            g = on[i] - off[i]
            ax.annotate(f"{g:+.2f}", (i, max(on[i], off[i]) + top * 0.07),
                        ha="center", fontsize=9.5, fontweight="bold",
                        color="#1a7d1a" if g >= 0 else "#c0392b")
        ax.set_title(ARM_NAME[arm], fontsize=12, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([WL_NAME[wl] for wl in wls], fontsize=9)
        ax.set_ylabel("MAT (mean accepted tokens / round)", fontsize=10)
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.set_ylim(0, top * 1.34)
        # narrow legend over the (consistently shortest) SWE-bench group so it
        # never overlaps the tall-group gain annotations
        ax.legend(fontsize=8, loc="upper center", ncol=1, framealpha=0.92)
    fig.suptitle("Warming effect (4-way) — suffix global tree warmed by the warm set; "
                 "tested on eval set", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fp = OUT / "WARMING_effect_4way.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print("saved ->", fp, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl", choices=list(WL))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.seed:
        seed_cells()
    elif a.plot:
        do_plot()
    elif a.wl and a.arm:
        compute_cell(a.wl, a.arm)
    else:
        ap.error("need --wl+--arm, or --seed, or --plot")
