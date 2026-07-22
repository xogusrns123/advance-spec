#!/usr/bin/env python3
"""Regenerate the 0716_TODO_Synthesized deck from REAL data, same format, no
annotations (drops the 'placeholder — synthetic data' watermark and the
interpretive call-outs). Representative workload = SPECBENCH (its raw head ECE
0.084 and raw tail MAE 8.01 are exactly the placeholder seeds); fig7 drift is the
deck's stated SpecBench -> BFCL v4 switch. fig5 is a data-free schematic and is
NOT regenerated.

Data sources (all real):
  head/tail pairs   results/bias_variance/pairs_{specbench,bfcl}.npz
  MAT by combo/win  readable_outputs/figures/replay_logs/mat_specbench_*_split.replay.txt
  raw (no-calib)    mat_specbench_4way_calib_raw_split.replay.txt

  python3 scripts/plot/regen_todo_figs.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from replay_extension import _fit_logistic, _fit_beta  # noqa: E402

BASE = Path("/workspace/simulation/Dr.Lee Solution")
BV = BASE / "results" / "bias_variance"
RLOG = BASE / "readable_outputs" / "figures" / "replay_logs"
OUT = BASE / "readable_outputs" / "figures" / "0716_regenerated"
OUT.mkdir(parents=True, exist_ok=True)

WL = "specbench"
C_LIN, C_LOG, C_BETA = "#808080", "#d62728", "#2ca02c"
C_ISO = "#c0392b"
C_BAR_HEAD, C_BAR_TAIL = "#4C78A8", "#E8912A"
C_RAWBAR = "#cccccc"
EE = np.arange(0, 1.0001, 0.1)                    # head ECE bins

d = np.load(BV / f"pairs_{WL}.npz")
hc, hm = d["head_conf"].astype(float), d["head_match"].astype(float)
ts, ta = d["tail_score"].astype(float), d["tail_acc"].astype(float)


def eval_on(fscalar, arr, ng=201, lo=0.0, hi=1.0):
    """Vectorize a scalar monotone map via a grid + linear interpolation."""
    g = np.linspace(lo, hi, ng)
    gy = np.array([fscalar(v) for v in g], float)
    return np.interp(np.asarray(arr, float), g, gy)


def lin_fit(x, y):
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


# ---------- calibrator fits on real specbench pairs --------------------------
ha, hb = lin_fit(hc, hm)
head_lin = lambda v: np.clip(ha * np.asarray(v, float) + hb, 0, 1)
_head_log = _fit_logistic(hc.tolist(), hm.tolist())
_head_beta = _fit_beta(hc.tolist(), hm.tolist())
head_log = lambda v: eval_on(_head_log, v)
head_beta = lambda v: eval_on(_head_beta, v)
ta_a, ta_b = lin_fit(ts, ta)
tail_lin = lambda v: np.maximum(0.0, ta_a * np.asarray(v, float) + ta_b)
_ir = IsotonicRegression(out_of_bounds="clip"); _ir.fit(ts, ta)
tail_iso = lambda v: _ir.predict(np.maximum(0.0, np.asarray(v, float)))


def parse_k(fp):
    m = re.search(r"calib: K=([0-9.]+)", Path(fp).read_text())
    return float(m.group(1)) if m else np.nan


# MAT = mean accepted DRAFT tokens per verify step, EXCLUDING the bonus token
# (project-standard "crossover K"; user convention). raw-cals 2.75, online
# logi+iso ~4.20, oracle ~4.79 on specbench.
def mat_combo(h, t, w=8000):
    return parse_k(RLOG / f"mat_{WL}_4way_calib_onl_{h}_{t}_w{w}_split.replay.txt")


RAW_MAT = parse_k(RLOG / f"mat_{WL}_4way_calib_raw_split.replay.txt")


def ece_binned(p, y, edges=EE):
    """ECE of calibrated values p vs outcomes y, binned on p."""
    p = np.asarray(p, float)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    N = len(p); e = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            e += (m.sum() / N) * abs(p[m].mean() - y[m].mean())
    return e


def bin_means(x, y, edges):
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    return np.array([y[idx == b].mean() if (idx == b).sum() else np.nan
                     for b in range(len(edges) - 1)])


# =========================== FIG 1 ===========================================
def fig1():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
    xs = np.linspace(0, 1, 200)
    a1.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1.4, label="y = x (target)")
    a1.plot(xs, head_lin(xs), color=C_LIN, lw=2.6, label="linear")
    a1.plot(xs, head_log(xs), color=C_LOG, lw=2.6, label="logistic")
    a1.plot(xs, head_beta(xs), color=C_BETA, lw=2.6, label="beta")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    a1.set_xlabel("raw DFlash probability"); a1.set_ylabel("calibrated probability")
    a1.set_title("Head candidates: linear / logistic / beta", fontweight="bold")
    a1.legend(loc="upper left"); a1.grid(alpha=0.3, ls=":")
    xt = np.linspace(0, 30, 300)
    a2.plot(xt, tail_lin(xt), color=C_LIN, lw=2.6, label="linear")
    a2.plot(xt, tail_iso(xt), color=C_ISO, lw=2.6, label="isotonic")
    a2.set_xlim(0, 30)
    a2.set_xlabel("raw suffix score"); a2.set_ylabel("calibrated accept length")
    a2.set_title("Tail candidates: linear / isotonic", fontweight="bold")
    a2.legend(loc="upper left"); a2.grid(alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / "fig1.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# =========================== FIG 2 ===========================================
def fig2():
    edges = np.arange(0, 1.0001, 0.1)
    centers = (edges[:-1] + edges[1:]) / 2
    emp = bin_means(hc, hm, edges)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
    xs = np.linspace(0, 1, 200)
    a1.bar(centers, emp, width=0.09, color=C_BAR_HEAD, label="empirical accept rate")
    a1.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1.2, label="y = x (target)")
    a1.plot(xs, head_lin(xs), color=C_LIN, lw=2.4, label="linear fit")
    a1.plot(xs, head_log(xs), color=C_LOG, lw=2.4, label="logistic fit")
    a1.plot(xs, head_beta(xs), color=C_BETA, lw=2.4, ls="-.", label="beta fit")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1.02)
    a1.set_xlabel("raw DFlash probability (conf)"); a1.set_ylabel("conditional accept rate (mean)")
    a1.set_title("Fitted calibration functions vs empirical accept rate", fontweight="bold")
    a1.legend(loc="upper left"); a1.grid(alpha=0.3, ls=":")
    vals = [ece_binned(hc, hm), ece_binned(head_lin(hc), hm),
            ece_binned(head_log(hc), hm), ece_binned(head_beta(hc), hm)]
    names = ["raw", "linear", "logistic", "beta"]; cols = [C_RAWBAR, C_LIN, C_LOG, C_BETA]
    b = a2.bar(names, vals, color=cols, edgecolor="k", linewidth=0.5)
    for bar, v in zip(b, vals):
        a2.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11)
    a2.set_ylabel("head ECE"); a2.set_ylim(0, max(vals) * 1.18)
    a2.set_title("Head ECE by calibration function", fontweight="bold")
    a2.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / "fig2.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# =========================== FIG 3 ===========================================
def fig3():
    edges = np.arange(0, 31, 2.0)
    centers = (edges[:-1] + edges[1:]) / 2
    emp = bin_means(ts, ta, edges)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
    xt = np.linspace(0, 30, 300)
    a1.bar(centers, emp, width=1.8, color=C_BAR_TAIL, label="realized accept length (mean)")
    a1.plot(xt, tail_lin(xt), color=C_LIN, lw=2.4, label="linear fit")
    a1.plot(xt, tail_iso(xt), color=C_ISO, lw=2.4, label="isotonic fit")
    a1.set_xlim(0, 30)
    a1.set_xlabel("raw suffix score"); a1.set_ylabel("realized accept length (mean)")
    a1.set_title("Fitted regressions vs realized accept length", fontweight="bold")
    a1.legend(loc="upper left"); a1.grid(alpha=0.3, ls=":")
    vals = [float(np.mean(np.abs(ts - ta))),
            float(np.mean(np.abs(tail_lin(ts) - ta))),
            float(np.mean(np.abs(tail_iso(ts) - ta)))]
    names = ["raw", "linear", "isotonic"]; cols = [C_RAWBAR, C_LIN, C_ISO]
    b = a2.bar(names, vals, color=cols, edgecolor="k", linewidth=0.5)
    for bar, v in zip(b, vals):
        a2.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11)
    a2.set_ylabel("tail MAE (tokens)"); a2.set_ylim(0, max(vals) * 1.15)
    a2.set_title("Tail calibration error by function", fontweight="bold")
    a2.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / "fig3.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# =========================== FIG 4 ===========================================
def fig4():
    heads = ["linear", "logistic", "beta"]
    lin_t = [mat_combo(h, "linear") for h in heads]
    iso_t = [mat_combo(h, "isotonic") for h in heads]
    x = np.arange(len(heads)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w / 2, lin_t, w, color=C_LIN, label="linear tail")
    b2 = ax.bar(x + w / 2, iso_t, w, color=C_BAR_TAIL, label="isotonic tail")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=11)
    ax.axhline(RAW_MAT, ls="--", color="#888888", lw=1.6,
               label=f"raw (no calibration) = {RAW_MAT:.2f}")
    ax.set_xticks(x); ax.set_xticklabels([f"{h} head" for h in heads])
    ax.set_ylabel("MAT (tokens/step)")
    ax.set_ylim(min(RAW_MAT, min(lin_t + iso_t)) - 0.3, max(lin_t + iso_t) + 0.35)
    ax.set_title("End-to-end MAT by head × tail calibration function", fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5)); ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / "fig4.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# =========================== FIG 6 ===========================================
WINDOWS6 = [512, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
WLAB6 = ["512", "1K", "2K", "4K", "8K", "16K", "32K", "64K"]


def steady_ece_vs_window(x, y, windows, step=500, fwd=2000):
    """Predictive windowed-logistic head ECE, averaged over mature checkpoints
    (t >= max window). At t: fit on [t-W:t], score ECE on forward block [t:t+fwd]."""
    N = len(x)
    warm = min(max(windows), N // 2)
    ckpts = [t for t in range(step, N - fwd, step) if t >= warm]
    out = {}
    for W in windows:
        vals = []
        for t in ckpts:
            f = _fit_logistic(x[max(0, t - W):t].tolist(), y[max(0, t - W):t].tolist())
            p = eval_on(f, x[t:t + fwd])
            vals.append(ece_binned(p, y[t:t + fwd]))
        out[W] = float(np.mean(vals)) if vals else np.nan
    return out


def fig6():
    ece_w = steady_ece_vs_window(hc, hm, WINDOWS6)
    mat_w = {w: parse_k(RLOG / f"mat_{WL}_4way_calib_onl_logistic_isotonic_w{w}_split.replay.txt")
             for w in WINDOWS6}
    xs = np.arange(len(WINDOWS6))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    a1.plot(xs, [ece_w[w] for w in WINDOWS6], "-o", color=C_BAR_HEAD, lw=2.2, ms=7)
    a1.set_xticks(xs); a1.set_xticklabels(WLAB6)
    a1.set_xlabel("window size (records)"); a1.set_ylabel("steady-state head ECE")
    a1.set_title("ECE vs window size", fontweight="bold"); a1.grid(alpha=0.3, ls=":")
    a2.plot(xs, [mat_w[w] for w in WINDOWS6], "-s", color=C_BETA, lw=2.2, ms=7)
    a2.axhline(RAW_MAT, ls="--", color="#888888", lw=1.6, label=f"raw = {RAW_MAT:.2f}")
    a2.set_xticks(xs); a2.set_xticklabels(WLAB6)
    a2.set_xlabel("window size (records)"); a2.set_ylabel("MAT (tokens/step)")
    a2.set_title("MAT vs window size", fontweight="bold")
    a2.legend(loc="lower right"); a2.grid(alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / "fig6.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# =========================== FIG 7 ===========================================
def fig7():
    db = np.load(BV / "pairs_bfcl.npz")
    xb, yb = db["head_conf"].astype(float), db["head_match"].astype(float)
    X = np.concatenate([hc, xb]); Y = np.concatenate([hm, yb])
    switch = len(hc)
    step, fwd = 200, 2000
    lo = max(0, switch - 10000); hi = min(len(X) - fwd, switch + 40000)
    ts_ax = list(range(lo, hi, step))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for W, c, lab in [(1000, C_LOG, "window = 1K"), (8000, C_BETA, "window = 8K"),
                      (64000, C_BAR_HEAD, "window = 64K")]:
        ys = []
        for t in ts_ax:
            f = _fit_logistic(X[max(0, t - W):t].tolist(), Y[max(0, t - W):t].tolist())
            p = eval_on(f, X[t:t + fwd])
            ys.append(ece_binned(p, Y[t:t + fwd]))
        ax.plot([(t - switch) / 1000 for t in ts_ax], ys, color=c, lw=1.6, label=lab)
    ax.axvline(0, ls="--", color="#999999", lw=1.6)
    ax.set_xlabel("decoding steps after switch (×1K)")
    ax.set_ylabel("head ECE (running)")
    ax.set_ylim(0, 0.10)
    ax.set_title("Calibration recovery after workload drift", fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / "fig7.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 done")
    fig2(); print("fig2 done")
    fig3(); print("fig3 done")
    fig4(); print("fig4 done")
    fig6(); print("fig6 done")
    fig7(); print("fig7 done")
    print("all ->", OUT)
