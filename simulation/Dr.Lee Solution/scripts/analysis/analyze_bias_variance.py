#!/usr/bin/env python3
"""Bias/variance of the two online-calibration targets, per workload and per window.

Consumes results/bias_variance/pairs_{ds}.npz (extract_bias_variance_pairs.py).

(A) RAW-SIGNAL bias/variance  ->  WHY online calibration is needed
    HEAD  reliability curve  E[accept | DFlash conf-bin]  vs  y=x.
          signed bias  = sum_b w_b (mean_conf_b - accept_rate_b)   (>0 = over-confident)
          |bias| (ECE) = sum_b w_b |mean_conf_b - accept_rate_b|
          irreducible  = sum_b w_b p_b(1-p_b)   (Bernoulli noise no map can remove)
    TAIL  E[accept_len | score-bin] vs y=x. arctic over-estimates ~3-6x.
          bias = mean_score - mean_acc; ratio = mean_score / mean_acc.

(B) WINDOW estimator bias/variance  ->  WHY window is a weak lever w/ a 4-8K optimum
    Slide the online calibrator over the serving-order stream at STEP-pair
    checkpoints; head window = W, tail window = W/4 (OnlineCalib coupling).
    Reference "truth" = full-data fit (head logistic, tail isotonic).
      bias(W)  = RMS_x[ mean_t hat_y_t(x) - y_true(x) ]     (density-weighted)
      std(W)   = sqrt( mean_x Var_t[ hat_y_t(x) ] )         (density-weighted)
    both in output units (accept-rate for head, accept-length for tail).

Emits results/bias_variance/figures/{raw_reliability,window_bias_variance}.png +
a text table.

  python3 scripts/analysis/analyze_bias_variance.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from replay_extension import _fit_logistic  # noqa: E402

BASE = Path("/workspace/simulation/Dr.Lee Solution")
BV = BASE / "results" / "bias_variance"
OUT = BV / "figures"
WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "tau2-bench"}
WINDOWS = [1000, 2000, 4000, 8000, 16000, 10**9]      # 1e9 = no-window
WLABEL = {1000: "1K", 2000: "2K", 4000: "4K", 8000: "8K", 16000: "16K", 10**9: "no-win"}
STEP = 500                                            # checkpoint cadence (pairs)
MIN_FIT = 100                                         # min pairs before a map is trusted
NGRID = 12                                            # density-weighted quantile grid
HEAD_WARM = 16000                                     # skip cold-start: mature = t >= max finite window
TAIL_WARM = 4000


def qgrid(x, n=NGRID):
    """n interior quantile points of x (density-representative eval grid)."""
    qs = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    return np.quantile(x, qs)


# ---------- (A) raw reliability curves ----------------------------------------
def head_reliability(conf, match, bw=0.05):
    edges = np.arange(0.0, 1.0 + bw, bw)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    rows = []
    N = len(conf)
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n < 20:
            continue
        p = float(match[m].mean()); c = float(conf[m].mean())
        rows.append((c, p, n))
    if not rows:
        return None
    c = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
    n = np.array([r[2] for r in rows]); w = n / n.sum()
    signed = float((w * (c - p)).sum())
    ece = float((w * np.abs(c - p)).sum())
    irred = float((w * p * (1 - p)).sum())
    return dict(c=c, p=p, n=n, signed=signed, ece=ece, irred=irred, N=int(N))


def tail_reliability(score, acc, nb=14):
    edges = np.quantile(score, np.linspace(0, 1, nb + 1))
    edges = np.unique(edges)
    idx = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n < 20:
            continue
        rows.append((float(score[m].mean()), float(acc[m].mean()),
                     float(acc[m].std()), n))
    s = np.array([r[0] for r in rows]); a = np.array([r[1] for r in rows])
    sd = np.array([r[2] for r in rows]); n = np.array([r[3] for r in rows])
    w = n / n.sum()
    return dict(s=s, a=a, sd=sd, n=n,
                bias=float((score - acc).mean()),
                ratio=float(score.mean() / max(1e-9, acc.mean())),
                irred=float((w * sd ** 2).sum()), N=int(len(score)))


# ---------- (B) windowed estimator bias/variance ------------------------------
def fit_head(x, y):
    if len(x) < MIN_FIT or len(set(y.tolist())) < 2:
        return None
    try:
        f = _fit_logistic(x.tolist(), y.tolist())
        return f
    except Exception:
        return None


def fit_tail(x, y):
    if len(x) < MIN_FIT or len(np.unique(x)) < 3:
        return None
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(x, y)
    return lambda v: ir.predict(np.maximum(0.0, v))


def window_bv(x, y, kind, windows):
    """Return {W: (bias, std)} density-weighted over a quantile grid. Estimator =
    the online windowed calibrator fit on the last W pairs, refit every STEP pairs;
    target = the full-data (grand) map. Aggregate only over MATURE checkpoints
    (t >= max finite window) so the finite-sample cold-start warmup — identical
    'identity until warm' for every window — does not contaminate the steady-state
    comparison. bias = ||mean_t pred - grand|| (staleness/drift; ~0 if stationary);
    std = sqrt(mean_x Var_t[pred]) (finite-window sampling noise, falls with W)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    grid = qgrid(x)
    w = np.ones_like(grid) / len(grid)             # equal-mass quantiles => density weights
    if kind == "head":
        fit, truth_f, warm = fit_head, fit_head(x, y), HEAD_WARM
    else:
        fit, truth_f, warm = fit_tail, fit_tail(x, y), TAIL_WARM
    if truth_f is None:
        return {W: (np.nan, np.nan) for W in windows}
    y_true = np.asarray([truth_f(np.array([g]))[0] if kind == "tail" else truth_f(g)
                         for g in grid], float)
    N = len(x)
    ckpts = [t for t in range(STEP, N + 1, STEP) if t >= min(warm, N)]
    if len(ckpts) < 3:                             # too few mature: use all >= MIN_FIT
        ckpts = [t for t in range(STEP, N + 1, STEP) if t >= MIN_FIT]
    res = {}
    for W in windows:
        preds = []
        for t in ckpts:
            lo = 0 if W >= N else max(0, t - W)
            f = fit(x[lo:t], y[lo:t])
            if f is None:
                continue
            gy = (f(grid) if kind == "tail" else np.asarray([f(g) for g in grid], float))
            preds.append(np.asarray(gy, float))
        if len(preds) < 2:
            res[W] = (np.nan, np.nan); continue
        P = np.vstack(preds)                       # (checkpoints, grid)
        mean_pred = P.mean(0)
        bias = float(np.sqrt((w * (mean_pred - y_true) ** 2).sum()))
        var = float((w * P.var(0)).sum())
        res[W] = (bias, float(np.sqrt(var)))
    return res


def main():
    data = {}
    for ds in WLS:
        fp = BV / f"pairs_{ds}.npz"
        if not fp.exists():
            print(f"[skip] {fp} missing"); continue
        data[ds] = np.load(fp)
    if not data:
        print("no pair files found"); return
    OUT.mkdir(parents=True, exist_ok=True)

    HR = {ds: head_reliability(d["head_conf"], d["head_match"]) for ds, d in data.items()}
    TR = {ds: tail_reliability(d["tail_score"], d["tail_acc"]) for ds, d in data.items()}
    HB = {ds: window_bv(d["head_conf"], d["head_match"], "head",
                        [w if w < 10**9 else 10**9 for w in WINDOWS]) for ds, d in data.items()}
    TB = {ds: window_bv(d["tail_score"], d["tail_acc"], "tail",
                        [max(1, w // 4) if w < 10**9 else 10**9 for w in WINDOWS])
          for ds, d in data.items()}

    # ---------- Figure 1: raw reliability (why calibrate) ----------
    ncol = len(data)
    fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 7.2))
    if ncol == 1:
        axes = axes.reshape(2, 1)
    for j, ds in enumerate(data):
        ax = axes[0, j]; h = HR[ds]
        if h:
            ax.plot([0, 1], [0, 1], "k:", lw=0.9, alpha=0.7)
            ax.plot(h["c"], h["p"], "-o", color="#4C78A8", ms=4, lw=1.6)
            ax.fill_between(h["c"], h["c"], h["p"], color="#4C78A8", alpha=0.15)
            ax.set_title(f"{WL_NAME[ds]}\nhead: bias={h['signed']:+.3f} "
                         f"ECE={h['ece']:.3f}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("DFlash prob", fontsize=8)
        if j == 0:
            ax.set_ylabel("conditional accept rate", fontsize=8)
        ax.grid(alpha=0.25)
        ax2 = axes[1, j]; t = TR[ds]
        mx = max(t["s"].max(), t["a"].max()) * 1.05
        ax2.plot([0, mx], [0, mx], "k:", lw=0.9, alpha=0.7)
        ax2.plot(t["s"], t["a"], "-s", color="#E4572E", ms=4, lw=1.6)
        ax2.fill_between(t["s"], t["s"], t["a"], color="#E4572E", alpha=0.15)
        ax2.set_title(f"tail: score/acc ratio={t['ratio']:.1f}x  "
                      f"bias={t['bias']:+.2f}", fontsize=9)
        ax2.set_xlabel("Suffix score", fontsize=8)
        if j == 0:
            ax2.set_ylabel("realized accept length", fontsize=8)
        ax2.grid(alpha=0.25)
    fig.suptitle("Raw-signal miscalibration = the bias online calibration removes\n"
                 "top: DFlash prob vs conditional accept rate (gap = head bias);  "
                 "bottom: Suffix score vs accept length (gap = tail over-estimate)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "raw_reliability.png", dpi=150, bbox_inches="tight")
    print("saved ->", OUT / "raw_reliability.png")

    # ---------- Figure 2: window bias/variance ----------
    xs = list(range(len(WINDOWS)))
    xt = [WLABEL[w] for w in WINDOWS]
    fig2, axes2 = plt.subplots(2, ncol, figsize=(3.4 * ncol, 7.2), sharex=True)
    if ncol == 1:
        axes2 = axes2.reshape(2, 1)
    for j, ds in enumerate(data):
        for row, (BVd, name, unit) in enumerate(
                [(HB[ds], "head", "accept rate"), (TB[ds], "tail", "accept length")]):
            ax = axes2[row, j]
            keys = ([w if w < 10**9 else 10**9 for w in WINDOWS] if name == "head"
                    else [max(1, w // 4) if w < 10**9 else 10**9 for w in WINDOWS])
            bias = [BVd[k][0] for k in keys]
            std = [BVd[k][1] for k in keys]
            ax.plot(xs, bias, "-o", color="#B02418", ms=4, label="bias (vs full-data)")
            ax.plot(xs, std, "-s", color="#2C7FB8", ms=4, label="std (over time)")
            ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=7)
            if row == 0:
                ax.set_title(WL_NAME[ds], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{name}: err ({unit})", fontsize=8)
            ax.grid(alpha=0.25)
            if row == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")
    fig2.suptitle("Online estimator bias-variance vs window (head=W, tail=W/4)\n"
                  "both bias & variance fall monotonically with window; TAIL (bottom, "
                  "~accept-length tokens) dwarfs HEAD (top, ~accept-rate) by ~50x -- "
                  "all estimation difficulty is on the tail", fontsize=12, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(OUT / "window_bias_variance.png", dpi=150, bbox_inches="tight")
    print("saved ->", OUT / "window_bias_variance.png")

    # ---------- text table ----------
    print("\n== RAW-SIGNAL bias (why calibrate) ==")
    print("wl".ljust(11) + "head_signed head_ECE head_irr  tail_ratio tail_bias tail_irr")
    for ds in data:
        h, t = HR[ds], TR[ds]
        print(f"{ds:<11}{h['signed']:>+10.3f}{h['ece']:>9.3f}{h['irred']:>9.3f}"
              f"{t['ratio']:>11.2f}{t['bias']:>+10.2f}{t['irred']:>9.2f}")
    print("\n== WINDOW estimator bias / std (head) ==")
    print("wl".ljust(11) + "".join(f"{WLABEL[w]:>15}" for w in WINDOWS))
    for ds in data:
        keys = [w if w < 10**9 else 10**9 for w in WINDOWS]
        print(f"{ds:<11}" + "".join(
            f"{HB[ds][k][0]:>6.3f}/{HB[ds][k][1]:<8.3f}" for k in keys))
    print("\n== WINDOW estimator bias / std (tail) ==")
    print("wl".ljust(11) + "".join(f"{WLABEL[w]:>15}" for w in WINDOWS))
    for ds in data:
        keys = [max(1, w // 4) if w < 10**9 else 10**9 for w in WINDOWS]
        print(f"{ds:<11}" + "".join(
            f"{TB[ds][k][0]:>6.3f}/{TB[ds][k][1]:<8.3f}" for k in keys))


if __name__ == "__main__":
    main()
