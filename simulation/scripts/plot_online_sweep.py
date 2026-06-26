#!/usr/bin/env python3
"""Online windowed-calibration sweep figures.

Two figure families, all from a pinned real-serving sweep (qwen3_14b_online):

(1) RELIABILITY (the histogram + red fitted-curve style, cf. plot_calib_methods):
    per (group in {eagle,suffix}, method, label, window) draw the empirical
    reliability bars (per raw-prob bin) + the online-calibrated value as a RED
    line + the y=x diagonal. Empirical target:
      target_p   -> mean q_target in the bin (continuous)
      accept_rate-> mean(drafted token == target committed token) in the bin
    Source: online_pairs_online_<method>_<label>_w<window>.jsonl.

(2) OUTCOME: MAT bar + survival + per-step conditional-accept overlays, online
    methods vs the shared refs (model-only / raw / suffix / oracle), per
    (label, window). Source: run_*.json + timing_*.jsonl.

NAMING: under target_p the "logistic"/"beta" methods are LinearRegression
analogs (sklearn LogisticRegression needs 0/1 labels), so they are labeled
'linear(p)' / 'linear(logit)' — NOT 'logistic'/'beta'. accept_rate keeps the
real logistic/beta names.

Usage (container, as root):
  python3 simulation/scripts/plot_online_sweep.py \
      --dir simulation/results/chain_hybrid_perdepth/qwen3_14b_online
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BIN_W = 0.05
RED = "crimson"
METHODS = ["histogram", "isotonic", "logistic", "beta"]
LABELS = ["target_p", "accept_rate"]
GROUPS = [("eagle", "EAGLE-3", "#1f77b4", "#6baed6"),
          ("suffix", "suffix decoding", "#ff7f0e", "#fdae6b")]
REF = [("baseline", "model-only", "#7f7f7f"), ("select1", "raw", "#1f77b4"),
       ("suffix", "suffix-only", "#d62728"), ("select1_oracle", "ORACLE", "#e0b400")]
ONLINE_COLOR = {"histogram": "#ff7f0e", "isotonic": "#2ca02c",
                "logistic": "#9467bd", "beta": "#8c564b"}


def disp_method(method: str, label: str) -> str:
    if label == "target_p":
        return {"histogram": "histogram", "isotonic": "isotonic",
                "logistic": "linear(p)", "beta": "linear(logit)"}[method]
    return method  # accept_rate: real logistic/beta


def load_jsonl(path: Path) -> list:
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return rows


def binned(p, y):
    """centers, mean(y) per fixed-width bin, counts."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    edges = np.arange(0.0, 1.0 + BIN_W, BIN_W)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2
    rates = np.full(nb, np.nan); counts = np.zeros(nb, int)
    for b in range(nb):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b]:
            rates[b] = float(y[m].mean())
    return centers, rates, counts


def accept_lengths(path: Path) -> np.ndarray:
    vals = []
    for r in load_jsonl(path):
        if r.get("phase") and r["phase"] != "decode":
            continue
        a = r.get("accept_lengths")
        if isinstance(a, list):
            vals.extend(int(x) for x in a)
        elif isinstance(a, (int, float)):
            vals.append(int(a))
    return np.asarray(vals, dtype=np.int64)


def survival(acc):
    if acc.size == 0:
        return np.array([]), np.array([])
    d = np.arange(1, int(acc.max()) + 2)
    return d, np.array([(acc >= k).mean() for k in d])


def reliability_fig(out, group_label, bar_c, edge_c, method_label, label,
                    centers, rates, cal_centers, cal_vals, n, base):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ylab = "mean q_target" if label == "target_p" else "P(accept)"
    ax.bar(centers, rates, width=BIN_W * 0.9, color=bar_c, edgecolor=edge_c,
           label=f"empirical ({ylab})")
    ax.plot(cal_centers, cal_vals, color=RED, lw=2, marker="o", ms=3,
            label="online-calibrated (red)")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", alpha=0.7, label="y=x")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("draft probability p")
    ax.set_ylabel(ylab)
    ax.set_title(f"{method_label} — {group_label}  [{label}]\n"
                 f"(n={n}, base={base:.3f})", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--windows", default="256,1024")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    d = Path(args.dir)
    windows = [int(w) for w in args.windows.split(",")]
    fig_root = d / "figures" / "online"

    # ---------- (1) reliability ----------
    for label in LABELS:
        for w in windows:
            for method in METHODS:
                pp = d / f"online_pairs_online_{method}_{label}_w{w}.jsonl"
                rows = load_jsonl(pp)
                if not rows:
                    continue
                for gk, glabel, bc, ec in GROUPS:
                    praw, emp, cal_p, cal_v = [], [], [], []
                    for r in rows:
                        rp = r.get(f"{gk}_p")
                        if rp is None:
                            continue
                        if label == "target_p":
                            e = r.get(f"q_{gk}")
                        else:
                            tok = r.get(f"{gk}_token"); com = r.get("committed_token")
                            e = (1.0 if (tok is not None and com is not None
                                         and int(tok) == int(com)) else 0.0)
                        if e is None:
                            continue
                        praw.append(rp); emp.append(e)
                        cv = r.get(f"{gk}_p_online")
                        if cv is not None:
                            cal_p.append(rp); cal_v.append(cv)
                    if len(praw) < 50:
                        continue
                    centers, rates, _ = binned(praw, emp)
                    cc, cvals, _ = binned(cal_p, cal_v) if cal_p else (centers, np.full_like(centers, np.nan), None)
                    m = ~np.isnan(rates)
                    mc = ~np.isnan(cvals)
                    out = (fig_root / "reliability" / f"{label}_w{w}"
                           / f"calib_{gk}_{method}.png")
                    reliability_fig(out, glabel, bc, ec,
                                    disp_method(method, label), label,
                                    centers[m], rates[m], cc[mc], cvals[mc],
                                    len(praw), float(np.mean(emp)))
    print("reliability figures done")

    # ---------- (2) outcome: MAT / survival / conditional ----------
    refs = {}
    rj = d / "run_refs.json"
    refarms = json.load(open(rj))["arms"] if rj.exists() else {}
    for label in LABELS:
        for w in windows:
            roj = d / f"run_online_{label}_w{w}.json"
            online_arms = json.load(open(roj))["arms"] if roj.exists() else {}
            # MAT bar
            names, mats, cols = [], [], []
            for a, lab, c in REF:
                v = (refarms.get(a) or {}).get("accept_length_mean")
                if v is None:
                    acc = accept_lengths(d / f"timing_{a}.jsonl")
                    v = float(acc.mean()) if acc.size else None
                if v is not None:
                    names.append(lab); mats.append(v); cols.append(c)
            for method in METHODS:
                v = (online_arms.get(f"select1_online_{method}") or {}).get("accept_length_mean")
                if v is None:
                    acc = accept_lengths(d / f"timing_online_{method}_{label}_w{w}.jsonl")
                    v = float(acc.mean()) if acc.size else None
                if v is not None:
                    names.append(f"online\n{disp_method(method,label)}"); mats.append(v)
                    cols.append(ONLINE_COLOR[method])
            if mats:
                fig, ax = plt.subplots(figsize=(max(8, 0.9*len(names)+2), 4.6))
                xs = np.arange(len(names))
                bars = ax.bar(xs, mats, color=cols, width=0.7)
                for b, v in zip(bars, mats):
                    ax.text(b.get_x()+b.get_width()/2, v+max(mats)*0.01, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)
                ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7)
                ax.set_ylabel("MAT"); ax.set_ylim(0, max(mats)*1.18); ax.grid(axis="y", alpha=0.3)
                ax.set_title(f"Online {label} w={w} — MAT vs refs (pinned 14B)", fontsize=10)
                fig.tight_layout(); (fig_root).mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_root / f"mat_{label}_w{w}.png", dpi=150); plt.close(fig)
            # survival + conditional overlay
            series = []
            for a, lab, c in REF:
                series.append((lab, c, "-", accept_lengths(d / f"timing_{a}.jsonl")))
            for method in METHODS:
                series.append((f"online {disp_method(method,label)}",
                               ONLINE_COLOR[method], "--",
                               accept_lengths(d / f"timing_online_{method}_{label}_w{w}.jsonl")))
            for kind in ("survival", "conditional"):
                fig, ax = plt.subplots(figsize=(8.4, 5.0))
                drew = False
                for lab, c, ls, acc in series:
                    dd, sv = survival(acc)
                    if dd.size == 0:
                        continue
                    if kind == "conditional":
                        prev = np.concatenate([[1.0], sv[:-1]])
                        with np.errstate(divide="ignore", invalid="ignore"):
                            sv = np.where(prev > 0, sv/prev, np.nan)
                    ax.plot(dd, sv, color=c, ls=ls, lw=1.5, marker="o", ms=2.5, label=lab)
                    drew = True
                ax.axvline(args.steps, color="black", lw=1, ls=":")
                ax.set_xlim(0.5, args.steps); ax.set_ylim(0, 1.0)
                ax.set_xlabel("depth d")
                ax.set_ylabel("survival P(accept>=d)" if kind == "survival"
                              else "conditional P(accept>=d | >=d-1)")
                ax.set_title(f"Online {label} w={w} — {kind} (pinned 14B)", fontsize=10)
                ax.grid(alpha=0.3)
                if drew:
                    ax.legend(fontsize=7, ncol=2)
                fig.tight_layout()
                fig.savefig(fig_root / f"{kind}_{label}_w{w}.png", dpi=150); plt.close(fig)
    print(f"outcome figures done -> {fig_root}")


if __name__ == "__main__":
    main()
