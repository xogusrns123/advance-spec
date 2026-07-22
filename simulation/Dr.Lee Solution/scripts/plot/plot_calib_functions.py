#!/usr/bin/env python3
"""Visualize the raw vs calibrated ESTIMATOR FUNCTIONS behind the composition
controller k* = argmax_k [1 + G_k + S_k*T_k], per dataset, with the calibration
data behind them:

  left  (head hazard a_d):  raw = affine 0.69*conf+0.29  vs
                            calibrated = logistic P(match|conf)
                            fit on the SAME accept-conditioned calibrate-split
                            pairs the replay uses; histogram of conf + binned
                            empirical P(match) behind.
  right (tail expectation): raw = identity T=score  vs
                            calibrated = isotonic score->E[realized accept]
                            (pairs from _fit_tail_iso's chain-policy replay,
                            budget-aware); histogram of scores + binned
                            empirical E[accept] behind.

  python3 scripts/plot_calib_functions.py --record results/perpos_bfcl_full/bfcl_v4_sub3.jsonl \
      --group-mode conv --name bfcl
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from replay_extension import _ad, _fit_logistic, _fit_beta, _fit_tail_iso  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RAW_C, CAL_C, BETA_C, INK, HIST_C = "#54A24B", "#B94A8C", "#E45756", "#333333", "#9AA7B4"


def load_split(record, group_mode):
    """Replicate replay_extension's calibrate/test split exactly."""
    rp = Path(record)
    traces = json.load(open(rp.with_suffix(".traces.json")))
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    recs = defaultdict(dict)
    for l in open(rp):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r
    raw = {rid: eval_traces[rid].get("conv", rid) for rid in eval_traces}
    order = {c: i for i, c in enumerate(sorted(set(raw.values())))}
    if group_mode == "convlabel":
        lab_of = {}
        for rid in eval_traces:
            lab_of.setdefault(raw[rid], eval_traces[rid].get("task", ""))
        rank, par = {}, {}
        for c in sorted(set(raw.values())):
            l = lab_of[c]
            par[c] = rank.get(l, 0) % 2
            rank[l] = rank.get(l, 0) + 1
        split_par = {rid: par[raw[rid]] for rid in raw}
    else:
        split_par = {rid: order[raw[rid]] % 2 for rid in raw}
    calib_rids = {rid for rid in recs if split_par.get(rid, 0) == 0}
    test_rids = {rid for rid in recs if split_par.get(rid, 0) == 1}
    return traces, eval_traces, recs, calib_rids, test_rids


def hazard_pairs(recs, rids):
    """Accept-conditioned (conf[d], match[d]) for d <= leading run — the same
    collection the replay's hazard calibrator uses."""
    xs, ys = [], []
    for rid in sorted(rids):
        for r in recs[rid].values():
            conf, match = r["dflash_conf"], r["dflash_match"]
            ad = _ad(match)
            for d in range(min(ad + 1, len(conf))):
                xs.append(float(conf[d])); ys.append(int(match[d]))
    return xs, ys


def nll(fn, xs, ys):
    import math
    s = 0.0
    for x, y in zip(xs, ys):
        p = min(max(fn(x), 1e-4), 1 - 1e-4)
        s += -(math.log(p) if y else math.log(1 - p))
    return s / max(len(xs), 1)


def binned(xs, ys, edges, min_n=25):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    cx, cy, cn = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (xs >= a) & (xs < b)
        if m.sum() >= min_n:
            cx.append(xs[m].mean()); cy.append(ys[m].mean()); cn.append(int(m.sum()))
    return cx, cy, cn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--group-mode", default="conv", choices=["conv", "convlabel"])
    ap.add_argument("--name", required=True, help="dataset tag for title/filename")
    ap.add_argument("--max-rounds", type=int, default=4096)
    args = ap.parse_args()

    traces, eval_traces, recs, calib_rids, test_rids = load_split(args.record,
                                                                  args.group_mode)
    num_spec = traces.get("num_spec", 32)

    hx, hy = hazard_pairs(recs, calib_rids)          # fit side
    cal_h = _fit_logistic(hx, hy)
    cal_b = _fit_beta(hx, hy)
    aff = lambda c: min(1.0, max(0.0, 0.69 * float(c) + 0.29))     # noqa: E731
    tx_h, ty_h = hazard_pairs(recs, test_rids)       # held-out side
    nlls = {"affine": nll(aff, tx_h, ty_h),
            "logistic": nll(cal_h, tx_h, ty_h),
            "beta": nll(cal_b, tx_h, ty_h)}
    cal_t, tx, ty = _fit_tail_iso(traces["warm_traces"], recs, eval_traces,
                                  calib_rids, num_spec, args.max_rounds,
                                  return_pairs=True)
    print(f"[{args.name}] hazard pairs={len(hx)} (test {len(tx_h)})  "
          f"tail pairs={len(tx)}  calib calls={len(calib_rids)}  "
          f"held-out NLL: affine {nlls['affine']:.4f} / logistic "
          f"{nlls['logistic']:.4f} / beta {nlls['beta']:.4f}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.4))

    # -- head hazard ----------------------------------------------------------
    g = np.linspace(0, 1, 201)
    ax1.plot(g, np.clip(0.69 * g + 0.29, 0, 1), color=RAW_C, lw=2.0,
             label=f"raw:  a = 0.69·conf + 0.29   (NLL {nlls['affine']:.3f})")
    ax1.plot(g, [cal_h(v) for v in g], color=CAL_C, lw=2.0,
             label=f"logistic P(match | conf)   (NLL {nlls['logistic']:.3f})")
    ax1.plot(g, [cal_b(v) for v in g], color=BETA_C, lw=2.0, ls="-.",
             label=f"beta calibration (Kull'17)   (NLL {nlls['beta']:.3f})")
    bx, by, bn = binned(hx, hy, np.linspace(0, 1, 21))
    ax1.scatter(bx, by, s=22, color=INK, zorder=5, label="empirical (binned)")
    axh = ax1.twinx()
    axh.hist(hx, bins=40, color=HIST_C, alpha=0.30, density=True)
    axh.set_yticks([]); axh.set_ylabel("")
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False); axh.spines[sp].set_visible(False)
    ax1.set_xlabel("DFlash conf (softmax max-prob at depth d)", fontsize=9.5)
    ax1.set_ylabel("a_d = P(match at depth d | survived < d)", fontsize=9.5)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.set_title(f"head hazard estimator — n={len(hx)} accept-cond. pairs "
                  f"(NLL = held-out, {len(tx_h)} test pairs)", fontsize=9.5)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.grid(alpha=0.3)

    # -- tail expectation ------------------------------------------------------
    tx_a = np.asarray(tx, float)
    xmax = float(min(max(tx_a.max(), 1.0), np.percentile(tx_a, 99.5) + 2))
    gs = np.linspace(0, xmax, 300)
    ax2.plot(gs, gs, color=RAW_C, lw=2.0, ls="--", label="raw:  T = score (identity)")
    ax2.plot(gs, [cal_t(v) for v in gs], color=CAL_C, lw=2.0,
             label="calibrated:  isotonic E[accept | score]")
    qedges = np.unique(np.quantile(tx_a, np.linspace(0, 1, 16)))
    bx, by, bn = binned(tx, ty, qedges, min_n=20)
    ax2.scatter(bx, by, s=22, color=INK, zorder=5, label="empirical (binned)")
    axh2 = ax2.twinx()
    axh2.hist(np.clip(tx_a, 0, xmax), bins=40, color=HIST_C, alpha=0.30, density=True)
    axh2.set_yticks([]); axh2.set_ylabel("")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False); axh2.spines[sp].set_visible(False)
    ax2.set_xlabel("suffix probe score (budget-aware, at candidate hand-off)",
                   fontsize=9.5)
    ax2.set_ylabel("T = E[realized tail accept]  (tokens)", fontsize=9.5)
    ax2.set_xlim(0, xmax)
    ax2.set_ylim(0, max(xmax, max(ty) if ty else 1) * 1.05)
    ax2.set_title(f"tail expectation estimator — n={len(tx)} chain-policy pairs",
                  fontsize=9.5)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"raw vs calibrated estimator functions — {args.name}  "
                 f"(fit on the calibrate split, {len(calib_rids)} calls; "
                 f"hist = signal distribution)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fp = BASE / "readable_outputs" / "figures" / "calib" / f"calib_functions_{args.name}.png"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")


if __name__ == "__main__":
    main()
