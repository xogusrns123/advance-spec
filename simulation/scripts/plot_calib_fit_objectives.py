#!/usr/bin/env python3
"""Per-(group, fit/test, algorithm) calibration figures for BOTH objectives, in the
calib_verify style: empirical bars + the fitted calibrator as a RED line.

One figure per (objective, group∈{eagle3,suffix}, split∈{fit,test}, method∈
{histogram,isotonic,logistic,beta}) — i.e. type / train-test / algorithm.

  calib_fit_<group>_<method>.png   bars = empirical target per raw-prob bin (fit set),
                                   RED line = fitted calibrator, gray dotted y=x.
  calib_test_<group>_<method>.png  bars = empirical target per CALIBRATED-prob bin
                                   (test set), RED dashed y=x; title ECE/Brier(MSE).

Target per objective (the y the map regresses to):
  token_gt -> accept = 1[token==gt_token]   (binary)  — def lineage
  target_p -> q_target                       (continuous) — tp lineage
Reuses plot_calib_methods fit_{histogram,isotonic} (work for continuous y too);
logistic/beta get continuous LinearRegression variants for target_p (binary
LogisticRegression for token_gt, matching the deployed fitter). Pooled across depths
for a single clean calibrator curve (like calib_verify); deployed maps are per-depth.

Usage (run inside sglang-bench as root — figure dirs are root-owned):
  python3 simulation/scripts/plot_calib_fit_objectives.py \
    --base simulation/results/o4_perdepth \
    --out-dir simulation/results/o4_perdepth/objective_compare/calibration
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_calib_methods import (  # noqa: E402
    BIN_W, EPS, RED, _grid, fit_histogram, fit_isotonic, fit_logistic, fit_beta,
)
from plot_calib_reliability import binned  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

METHODS = [("histogram", "Histogram binning"), ("isotonic", "Isotonic regression"),
           ("logistic", "Logistic regression"), ("beta", "Beta calibration")]
# file key, decision-log key, pretty, bar face, bar edge
GROUPS = [("eagle3", "eagle", "EAGLE-3", "#1f77b4", "#6baed6"),
          ("suffix", "suffix", "suffix decoding", "#ff7f0e", "#fdae6b")]


def fit_logistic_cont(p, y):
    lr = LinearRegression().fit(p.reshape(-1, 1), y)
    predict = lambda x: np.clip(lr.predict(np.asarray(x, float).reshape(-1, 1)), 0, 1)
    g = _grid(p); return g, predict(g), "line", predict


def fit_beta_cont(p, y):
    pc = np.clip(p, EPS, 1 - EPS)
    lr = LinearRegression().fit(np.column_stack([np.log(pc), np.log(1 - pc)]), y)
    def predict(x):
        xc = np.clip(np.asarray(x, float), EPS, 1 - EPS)
        return np.clip(lr.predict(np.column_stack([np.log(xc), np.log(1 - xc)])), 0, 1)
    g = _grid(p); return g, predict(g), "line", predict


def fit_spec(method, p, y, centers, rates, continuous):
    if method == "histogram":
        return fit_histogram(p, y, centers, rates)
    if method == "isotonic":
        return fit_isotonic(p, y)
    if method == "logistic":
        return fit_logistic_cont(p, y) if continuous else fit_logistic(p, y)
    return fit_beta_cont(p, y) if continuous else fit_beta(p, y)


def qmap(path):
    m = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        m[(r["rid"], r["decode_step"], r["depth"])] = (r.get("q_eagle"), r.get("q_suffix"))
    return m


def collect(dec_path, dkey, *, target, qfile=None):
    """Return (p_array, y_array) for one group, pooled over depths."""
    q = qmap(qfile) if qfile else None
    P, Y = [], []
    tok_key = f"{dkey}_token"; p_key = f"{dkey}_p"
    for l in open(dec_path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("type") != "decision" or r.get("tail"):
            continue
        rawp, tok = r.get(p_key), r.get(tok_key)
        if rawp is None or tok is None:
            continue
        if target == "accept":
            gt = r.get("gt_token")
            if gt is None:
                continue
            y = 1.0 if tok == gt else 0.0
        else:
            key = (r["rid"], r["decode_step"], r["depth"])
            if q is None or key not in q:
                continue
            qe, qs = q[key]
            y = qe if dkey == "eagle" else qs
            if y is None:
                continue
        P.append(float(rawp)); Y.append(float(y))
    return np.asarray(P), np.asarray(Y)


def fig_fit(out, gkey, label, bar_c, edge_c, mkey, mlabel, spec, p, y, centers, rates, ylab):
    cx, cy, kind, _ = spec
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(centers, rates, width=BIN_W * 0.9, color=bar_c, edgecolor=edge_c,
           label=f"empirical {ylab} (fit bin)")
    (ax.step(cx, cy, where="mid", color=RED, lw=2, label="fitted calibrator")
     if kind == "step" else ax.plot(cx, cy, color=RED, lw=2, label="fitted calibrator"))
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{label} draft probability p"); ax.set_ylabel(ylab)
    ax.set_title(f"{mlabel} — {label} (fit)\n(fit n={len(p)}, base={float(y.mean()):.3f})",
                 fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best"); fig.tight_layout()
    fig.savefig(out / f"calib_fit_{gkey}_{mkey}.png", dpi=150); plt.close(fig)


def fig_test(out, gkey, label, bar_c, edge_c, mkey, mlabel, spec, p, y, ylab):
    predict = spec[3]
    cal = np.clip(predict(p), 0.0, 1.0)
    centers, rates, counts = binned(cal, y)
    m = ~np.isnan(rates); w = counts[m] / max(1, counts[m].sum())
    ece = float(np.sum(w * np.abs(centers[m] - rates[m]))) if m.any() else float("nan")
    brier = float(np.mean((cal - y) ** 2)) if len(y) else float("nan")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(centers, rates, width=BIN_W * 0.9, color=bar_c, edgecolor=edge_c,
           label=f"empirical {ylab} (test bin)")
    ax.plot([0, 1], [0, 1], color=RED, lw=2, ls="--", label="y = x (target)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"calibrated probability ({mlabel})"); ax.set_ylabel(f"empirical {ylab} (test)")
    ax.set_title(f"{mlabel} — {label} (test, offline)\n"
                 f"test n={len(p)}, ECE={ece:.3f}, MSE={brier:.3f}", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best"); fig.tight_layout()
    fig.savefig(out / f"calib_test_{gkey}_{mkey}.png", dpi=150); plt.close(fig)
    return ece, brier


def per_bin_stats(x, y):
    """Per fixed-width prob bin: (mean_x, mean_y, count, SE_of_mean). Empty dropped.
    SE = std(y)/sqrt(n) — for binary y this equals binomial sqrt(r(1-r)/n)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    edges = np.arange(0.0, 1.0 + BIN_W, BIN_W)
    xs, rs, ns, ses = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) if hi < 1.0 else (x >= lo) & (x <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        yb = y[m]; r = float(yb.mean())
        xs.append(float(x[m].mean())); rs.append(r); ns.append(n)
        ses.append(float(yb.std()) / math.sqrt(n))
    return (np.asarray(xs), np.asarray(rs), np.asarray(ns, float), np.asarray(ses))


def _bxp_stats(rs, ses):
    """Box per bin: box = mean ± SE, whisker = ± 2 SE, median line = mean."""
    return [dict(med=r, q1=max(0.0, r - se), q3=min(1.0, r + se),
                 whislo=max(0.0, r - 2 * se), whishi=min(1.0, r + 2 * se), fliers=[])
            for r, se in zip(rs, ses)]


def scat_fit(out, gkey, label, color, edge, mkey, mlabel, spec, p, y, ylab):
    cx, cy, kind, _ = spec
    xs, rs, ns, _se = per_bin_stats(p, y)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if len(xs):
        ax.scatter(xs, rs, s=15 + 285 * (ns / ns.max()), c=color, edgecolors=edge,
                   lw=0.6, alpha=0.85, zorder=3, label=f"empirical {ylab} (bin, size∝n)")
    (ax.step(cx, cy, where="mid", color=RED, lw=2, label="fitted calibrator")
     if kind == "step" else ax.plot(cx, cy, color=RED, lw=2, label="fitted calibrator"))
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{label} draft probability p"); ax.set_ylabel(ylab)
    ax.set_title(f"{mlabel} — {label} (raw)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best"); fig.tight_layout()
    fig.savefig(out / f"calib_fit_{gkey}_{mkey}.png", dpi=150); plt.close(fig)


def scat_test(out, gkey, label, color, edge, mkey, mlabel, spec, p, y, ylab):
    cal = np.clip(spec[3](p), 0.0, 1.0)
    xs, rs, ns, _se = per_bin_stats(cal, y)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if len(xs):
        ax.scatter(xs, rs, s=15 + 285 * (ns / ns.max()), c=color, edgecolors=edge,
                   lw=0.6, alpha=0.85, zorder=3, label=f"empirical {ylab} (bin, size∝n)")
    ax.plot([0, 1], [0, 1], color=RED, lw=2, ls="--", label="y = x (target)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"calibrated probability ({mlabel})"); ax.set_ylabel(f"empirical {ylab} (test)")
    ax.set_title(f"{mlabel} — {label} (test, offline)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best"); fig.tight_layout()
    fig.savefig(out / f"calib_test_{gkey}_{mkey}.png", dpi=150); plt.close(fig)


def _box(ax, xs, rs, ses, color, edge):
    bp = ax.bxp(_bxp_stats(rs, ses), positions=xs, widths=BIN_W * 0.6,
                manage_ticks=False, patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor=color, edgecolor=edge, alpha=0.8)
    for med in bp["medians"]:
        med.set(color=RED, lw=1.4)
    for wk in bp["whiskers"] + bp["caps"]:
        wk.set(color=edge, lw=0.9)


def box_fit(out, gkey, label, color, edge, mkey, mlabel, spec, p, y, ylab):
    cx, cy, kind, _ = spec
    xs, rs, ns, ses = per_bin_stats(p, y)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if len(xs):
        _box(ax, xs, rs, ses, color, edge)
    (ax.step(cx, cy, where="mid", color=RED, lw=2, label="fitted calibrator")
     if kind == "step" else ax.plot(cx, cy, color=RED, lw=2, label="fitted calibrator"))
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xlabel(f"{label} draft probability p"); ax.set_ylabel(ylab)
    ax.set_title(f"{mlabel} — {label} (raw)\nbox = {ylab} mean ± SE, whisker ± 2 SE",
                 fontsize=9.5)
    ax.grid(alpha=0.25)
    proxy = Patch(facecolor=color, edgecolor=edge, alpha=0.8, label=f"{ylab} (mean ± SE)")
    h, _l = ax.get_legend_handles_labels(); ax.legend(handles=[proxy] + h, fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out / f"calib_fit_{gkey}_{mkey}.png", dpi=150); plt.close(fig)


def box_test(out, gkey, label, color, edge, mkey, mlabel, spec, p, y, ylab):
    cal = np.clip(spec[3](p), 0.0, 1.0)
    xs, rs, ns, ses = per_bin_stats(cal, y)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if len(xs):
        _box(ax, xs, rs, ses, color, edge)
    ax.plot([0, 1], [0, 1], color=RED, lw=2, ls="--", label="y = x (target)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xlabel(f"calibrated probability ({mlabel})"); ax.set_ylabel(f"empirical {ylab} (test)")
    ax.set_title(f"{mlabel} — {label} (test, offline)\nbox = {ylab} mean ± SE, whisker ± 2 SE",
                 fontsize=9.5)
    ax.grid(alpha=0.25)
    proxy = Patch(facecolor=color, edgecolor=edge, alpha=0.8, label=f"{ylab} (mean ± SE)")
    h, _l = ax.get_legend_handles_labels(); ax.legend(handles=[proxy] + h, fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out / f"calib_test_{gkey}_{mkey}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="simulation/results/o4_perdepth")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    B = Path(a.base)
    OBJ = {
        "token_gt": dict(fit_dec=B / "qwen3_14b_def_train/decisions_select1_oracle.jsonl",
                         fit_q=None, test_dec=B / "qwen3_14b_def/decisions_select1_oracle.jsonl",
                         test_q=None, target="accept", continuous=False, ylab="P(accept)"),
        "target_p": dict(fit_dec=B / "qwen3_14b_tp_train/decisions_select1_oracle.jsonl",
                         fit_q=B / "qwen3_14b_tp_train/target_probs.jsonl",
                         test_dec=B / "qwen3_14b_tp/decisions_select1_oracle.jsonl",
                         test_q=B / "qwen3_14b_tp/target_probs_test.jsonl",
                         target="q", continuous=True, ylab="q_target"),
    }
    for obj, cfg in OBJ.items():
        hist_dir = Path(a.out_dir) / obj / "histogram"; hist_dir.mkdir(parents=True, exist_ok=True)
        sc_dir = Path(a.out_dir) / obj / "scatter"; sc_dir.mkdir(parents=True, exist_ok=True)
        bx_dir = Path(a.out_dir) / obj / "boxplot"; bx_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {obj} (continuous={cfg['continuous']}) -> {obj}/{{histogram,scatter,boxplot}}")
        for gkey, dkey, label, bar_c, edge_c in GROUPS:
            p_fit, y_fit = collect(cfg["fit_dec"], dkey, target=cfg["target"], qfile=cfg["fit_q"])
            p_te, y_te = collect(cfg["test_dec"], dkey, target=cfg["target"], qfile=cfg["test_q"])
            if p_fit.size == 0 or p_te.size == 0:
                print(f"   {gkey}: empty (fit={p_fit.size} test={p_te.size}) — skip"); continue
            centers, rates, _ = binned(p_fit, y_fit)
            for mkey, mlabel in METHODS:
                spec = fit_spec(mkey, p_fit, y_fit, centers, rates, cfg["continuous"])
                # histogram (bar) style
                fig_fit(hist_dir, gkey, label, bar_c, edge_c, mkey, mlabel, spec,
                        p_fit, y_fit, centers, rates, cfg["ylab"])
                ece, mse = fig_test(hist_dir, gkey, label, bar_c, edge_c, mkey, mlabel, spec,
                                    p_te, y_te, cfg["ylab"])
                # scatter style (bin markers, size∝n)
                scat_fit(sc_dir, gkey, label, bar_c, edge_c, mkey, mlabel, spec,
                         p_fit, y_fit, cfg["ylab"])
                scat_test(sc_dir, gkey, label, bar_c, edge_c, mkey, mlabel, spec,
                          p_te, y_te, cfg["ylab"])
                # boxplot style (bin mean ± SE)
                box_fit(bx_dir, gkey, label, bar_c, edge_c, mkey, mlabel, spec,
                        p_fit, y_fit, cfg["ylab"])
                box_test(bx_dir, gkey, label, bar_c, edge_c, mkey, mlabel, spec,
                         p_te, y_te, cfg["ylab"])
                print(f"   {gkey:6s} {mkey:9s} fit_n={p_fit.size} test_n={p_te.size} "
                      f"ECE={ece:.3f} MSE={mse:.3f}")
    print(f"wrote histogram+scatter+boxplot calib figures under "
          f"{a.out_dir}/<objective>/{{histogram,scatter,boxplot}}/")


if __name__ == "__main__":
    main()
