"""Per-position (per-depth) calibration — fit graphs, two layouts.

An INDEPENDENT calibrator is fit for each draft depth k. Samples are split
70:30 (sample-level, not by task) into train/test; the 4 calibrators
(histogram / isotonic / Platt-logistic / beta) are fit on the 70% train of each
depth separately; per-depth test ECE (on the 30%) annotates the subplots.

Per draft (model-based EAGLE3/MTP, model-free suffix) x method, FIT-side
(raw draft prob -> accept rate) two images:
  pp_<draft>_<m>_subplots.png — grid, one subplot per depth k: train prob->accept
                   histogram (draft color) + red fitted calibrator + faint y=x.
  pp_<draft>_<m>_overlay.png  — all depths' fitted curves on one axes, viridis-
                   colored by depth (colorbar) + red dashed y=x — shows the
                   calibration function shifting with depth.

...and TEST-side (CALIBRATED prob -> accept rate, on the held-out 30% of each
depth — each depth's frozen calibrator applied to its own test slice) two more:
  pp_test_<draft>_<m>_subplots.png  — grid, one subplot per depth k: histogram of
                   the calibrated probability vs empirical accept rate + red
                   dashed y=x (perfect-calibration target); per-depth test ECE.
  pp_test_<draft>_<m>_overlay.png   — all depths' calibrated-prob reliability
                   curves on ONE axes, viridis-colored by depth (colorbar) + red
                   dashed y=x — per-position test analog of pp_*_overlay.png.
  pp_test_<draft>_<m>_aggregate.png — all depths' calibrated test probabilities
                   POOLED into one reliability histogram vs y=x; pooled test
                   ECE/Brier. Direct per-depth analog of the parent dir's
                   calib_test_* (which uses a single global calibrator instead).

=> 2 drafts x 4 methods x (2 fit + 3 test) layouts = 40 images per model.

Input: the (prob,accept) pairs from extract_calib_pairs.py, where the list index
in each row's "m"/"s" array IS the depth k.

Usage:
  python3 simulation/scripts/calib_perposition.py \
      --pairs simulation/results/calib_verify/pairs_14b.jsonl.gz \
      --out-dir simulation/results/calib_verify/qwen3_14b \
      --model-label EAGLE3 --model-color "#1f77b4" --max-positions 16
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_calib_methods import (  # noqa: E402
    BIN_W, RED, binned, fit_beta, fit_histogram, fit_isotonic, fit_logistic,
)

METHODS = [
    ("histogram", "Histogram binning"),
    ("isotonic", "Isotonic regression"),
    ("logistic", "Logistic regression (Platt scaling)"),
    ("beta", "Beta calibration"),
]
SUFFIX_COLOR, SUFFIX_EDGE = "#ff7f0e", "#e8810b"


def load_per_depth(pairs_path: str, max_pos: int):
    md = defaultdict(lambda: ([], []))
    sd = defaultdict(lambda: ([], []))
    with gzip.open(pairs_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for k, (p, y) in enumerate(r.get("m", [])):
                if k < max_pos:
                    md[k][0].append(p); md[k][1].append(y)
            for k, (p, y) in enumerate(r.get("s", [])):
                if k < max_pos:
                    sd[k][0].append(p); sd[k][1].append(y)

    def fin(d):
        return {k: (np.asarray(v[0], np.float64), np.asarray(v[1], np.float64))
                for k, v in d.items()}
    return {"model": fin(md), "suffix": fin(sd)}


def ece(p, y, n_bins=15):
    if len(y) == 0:
        return float("nan")
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    y = np.asarray(y, float)
    N = len(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.any():
            e += (m.sum() / N) * abs(float(p[m].mean()) - float(y[m].mean()))
    return e


def _cont_spec(method, p, y):
    """Continuous-label (target_p) replacement for the binary fit_logistic /
    fit_beta (sklearn LogisticRegression rejects continuous y). Returns the same
    (cx, cy, draw_kind, predict) spec tuple, using the LinearRegression variants
    from fit_chain_hybrid_calib_perpos._fit_continuous."""
    from fit_chain_hybrid_calib_perpos import _fit_continuous  # noqa: E402
    predict = _fit_continuous(method, p, y)
    lo = max(1e-6, float(np.min(p)))
    hi = min(1.0 - 1e-6, float(np.max(p)))
    g = np.linspace(lo, hi, 400) if hi > lo else np.asarray([lo])
    return (g, predict(g), "line", predict)


def fit_depth(p, y, frac, rng):
    n = len(y)
    idx = rng.permutation(n)
    ntr = max(1, int(round(n * frac)))
    tr, te = idx[:ntr], idx[ntr:]
    p_tr, y_tr, p_te, y_te = p[tr], y[tr], p[te], y[te]
    centers, rates, _ = binned(p_tr, y_tr)
    # Continuous objective (target_p): y is q_target in [0,1], not binary {0,1}.
    # histogram/isotonic handle it natively; logistic/beta need the regression
    # variants (the binary sklearn classifiers raise on continuous labels).
    uniq = np.unique(y_tr)
    is_cont = uniq.size > 2 or not np.all(np.isin(uniq, (0.0, 1.0)))
    specs = {
        "histogram": fit_histogram(p_tr, y_tr, centers, rates),
        "isotonic": fit_isotonic(p_tr, y_tr),
        "logistic": (_cont_spec("logistic", p_tr, y_tr) if is_cont
                     else fit_logistic(p_tr, y_tr)),
        "beta": (_cont_spec("beta", p_tr, y_tr) if is_cont
                 else fit_beta(p_tr, y_tr)),
    }
    eces = {m: ece(specs[m][3](p_te), y_te) if len(y_te) else float("nan")
            for m in specs}
    return {"centers": centers, "rates": rates, "specs": specs, "ece": eces,
            "n": int(ntr), "acc": float(y_tr.mean()),
            "p_tr": p_tr, "y_tr": y_tr,  # raw train slice (for scatter/box views)
            "p_te": p_te, "y_te": y_te, "n_te": int(len(y_te))}


def reliability_stats(cal, y):
    """Bin calibrated prob vs empirical accept rate (BIN_W bins) + ECE/Brier.

    ECE/Brier match calib_verify.group2_fig so the per-depth TEST figures here
    are directly comparable to the global calib_test_* in the parent dir:
      ECE = sum_bin (n_bin/N) * |bin_center - accept_rate|;  Brier = mean(cal-y)^2.
    """
    cal = np.clip(np.asarray(cal, float), 0.0, 1.0)
    y = np.asarray(y, float)
    centers, rates, counts = binned(cal, y)
    m = ~np.isnan(rates)
    if m.any():
        w = counts[m] / max(1, counts[m].sum())
        e = float(np.sum(w * np.abs(centers[m] - rates[m])))
    else:
        e = float("nan")
    b = float(np.mean((cal - y) ** 2)) if len(y) else float("nan")
    return centers, rates, counts, e, b


def draw_subplots(out, draft_key, label, bar_c, edge_c, mkey, mlabel, depths,
                  fitted, note=""):
    K = len(depths)
    cols = math.ceil(math.sqrt(K))
    rows = math.ceil(K / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6),
                             squeeze=False)
    for i, k in enumerate(depths):
        ax = axes[i // cols][i % cols]
        fd = fitted[k]
        cx, cy, kind, _ = fd["specs"][mkey]
        ax.bar(fd["centers"], fd["rates"], width=BIN_W * 0.9, color=bar_c,
               edgecolor=edge_c)
        if kind == "step":
            ax.step(cx, cy, where="mid", color=RED, lw=1.6)
        else:
            ax.plot(cx, cy, color=RED, lw=1.6)
        ax.plot([0, 1], [0, 1], color="gray", lw=0.6, ls=":", alpha=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"k={k}  n={fd['n']}  acc={fd['acc']:.2f}  "
                     f"ECE={fd['ece'][mkey]:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(K, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Per-position {mlabel} — {label}  (train hist + red fit, "
                 f"per-depth; ECE on 30% test){note}", fontsize=11)
    fig.supxlabel("draft probability p", fontsize=9)
    fig.supylabel("P(accept)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_p = out / f"pp_{draft_key}_{mkey}_subplots.png"
    fig.savefig(out_p, dpi=150); plt.close(fig)
    return out_p


def draw_overlay(out, draft_key, label, mkey, mlabel, depths, fitted, note=""):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    kmax = max(depths) if depths else 1
    norm = Normalize(vmin=0, vmax=kmax)
    cmap = plt.get_cmap("viridis")
    for k in depths:
        cx, cy, kind, _ = fitted[k]["specs"][mkey]
        col = cmap(norm(k))
        if kind == "step":
            ax.step(cx, cy, where="mid", color=col, lw=1.3, alpha=0.9)
        else:
            ax.plot(cx, cy, color=col, lw=1.3, alpha=0.9)
    ax.plot([0, 1], [0, 1], color=RED, lw=1.6, ls="--", label="y = x")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("draft probability p"); ax.set_ylabel("calibrated P(accept)")
    ax.set_title(f"Per-position {mlabel} — {label}  (fitted curve per depth){note}",
                 fontsize=10.5)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cb.set_label("depth k")
    fig.tight_layout()
    out_p = out / f"pp_{draft_key}_{mkey}_overlay.png"
    fig.savefig(out_p, dpi=150); plt.close(fig)
    return out_p


def draw_test_subplots(out, draft_key, label, bar_c, edge_c, mkey, mlabel,
                       depths, fitted, note=""):
    """Per-depth TEST reliability: each depth's frozen calibrator applied to its
    own 30% test slice -> histogram(calibrated prob) vs accept rate + y=x."""
    K = len(depths)
    cols = math.ceil(math.sqrt(K))
    rows = math.ceil(K / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6),
                             squeeze=False)
    for i, k in enumerate(depths):
        ax = axes[i // cols][i % cols]
        fd = fitted[k]
        predict = fd["specs"][mkey][3]
        p_te, y_te = fd["p_te"], fd["y_te"]
        if len(y_te):
            centers, rates, _c, e, _b = reliability_stats(predict(p_te), y_te)
            ax.bar(centers, rates, width=BIN_W * 0.9, color=bar_c,
                   edgecolor=edge_c)
        else:
            e = float("nan")
        ax.plot([0, 1], [0, 1], color=RED, lw=1.4, ls="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"k={k}  n={fd['n_te']}  ECE={e:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(K, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Per-position {mlabel} — {label}  (calibrated prob vs accept, "
                 f"30% test; red dashed y=x target){note}", fontsize=11)
    fig.supxlabel("calibrated probability", fontsize=9)
    fig.supylabel("empirical accept rate (test)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_p = out / f"pp_test_{draft_key}_{mkey}_subplots.png"
    fig.savefig(out_p, dpi=150); plt.close(fig)
    return out_p


def draw_test_aggregate(out, draft_key, label, bar_c, edge_c, mkey, mlabel,
                        depths, fitted, note=""):
    """Pool every depth's calibrated test probabilities into ONE reliability
    histogram vs y=x — the per-depth analog of the parent's calib_test_*."""
    cals, ys = [], []
    for k in depths:
        fd = fitted[k]
        if len(fd["y_te"]):
            cals.append(np.clip(fd["specs"][mkey][3](fd["p_te"]), 0.0, 1.0))
            ys.append(fd["y_te"])
    cal = np.concatenate(cals) if cals else np.asarray([], float)
    y = np.concatenate(ys) if ys else np.asarray([], float)
    centers, rates, _c, e, b = reliability_stats(cal, y)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(centers, rates, width=BIN_W * 0.9, color=bar_c, edgecolor=edge_c,
           label="empirical accept (test bin)")
    ax.plot([0, 1], [0, 1], color=RED, lw=2, ls="--", label="y = x (target)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"calibrated probability  ({mlabel})")
    ax.set_ylabel("empirical accept rate (test)")
    ax.set_title(f"Per-position {mlabel} — {label} (test, pooled over depths){note}\n"
                 f"test n={len(y)}, ECE={e:.3f}, Brier={b:.3f}", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_p = out / f"pp_test_{draft_key}_{mkey}_aggregate.png"
    fig.savefig(out_p, dpi=150); plt.close(fig)
    return out_p, e, b


def draw_test_overlay(out, draft_key, label, mkey, mlabel, depths, fitted, note=""):
    """Per-position TEST reliability on ONE axes (mirror of draw_overlay): for
    each depth, calibrated-prob bin centers -> empirical accept rate over its 30%
    test slice, viridis-colored by depth, vs red-dashed y=x target."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    kmax = max(depths) if depths else 1
    norm = Normalize(vmin=0, vmax=kmax)
    cmap = plt.get_cmap("viridis")
    for k in depths:
        fd = fitted[k]
        if not len(fd["y_te"]):
            continue
        cal = np.clip(fd["specs"][mkey][3](fd["p_te"]), 0.0, 1.0)
        centers, rates, _counts = binned(cal, fd["y_te"])
        m = ~np.isnan(rates)
        if m.sum() < 2:
            continue
        ax.plot(centers[m], rates[m], color=cmap(norm(k)), lw=1.3, alpha=0.9,
                marker="o", ms=2.5)
    ax.plot([0, 1], [0, 1], color=RED, lw=1.6, ls="--", label="y = x (target)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("calibrated probability")
    ax.set_ylabel("empirical accept rate (test)")
    ax.set_title(f"Per-position {mlabel} — {label}  (test reliability per depth){note}",
                 fontsize=10.5)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cb.set_label("depth k")
    fig.tight_layout()
    out_p = out / f"pp_test_{draft_key}_{mkey}_overlay.png"
    fig.savefig(out_p, dpi=150); plt.close(fig)
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    ap.add_argument("--model-color", default="#1f77b4")
    ap.add_argument("--max-positions", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir) / "perposition" / "histogram"  # bar reliability here
    out.mkdir(parents=True, exist_ok=True)
    data = load_per_depth(args.pairs, args.max_positions)
    rng = np.random.default_rng(args.seed)

    drafts = [("model", args.model_label, args.model_color, args.model_color),
              ("suffix", "Suffix", SUFFIX_COLOR, SUFFIX_EDGE)]
    written = []
    for key, label, bar_c, edge_c in drafts:
        per_depth = data[key]
        depths = sorted(per_depth)[:args.max_positions]
        if not depths:
            print(f"  WARNING: no {key} depths; skip", file=sys.stderr)
            continue
        fitted = {k: fit_depth(*per_depth[k], args.train_frac, rng)
                  for k in depths}
        draft_key = label.lower()
        print(f"== {label}: {len(depths)} depths "
              f"(k0 acc={fitted[depths[0]]['acc']:.3f} -> "
              f"k{depths[-1]} acc={fitted[depths[-1]]['acc']:.3f}) ==",
              file=sys.stderr)
        for mkey, mlabel in METHODS:
            written.append(draw_subplots(out, draft_key, label, bar_c, edge_c,
                                         mkey, mlabel, depths, fitted))
            written.append(draw_overlay(out, draft_key, label, mkey, mlabel,
                                        depths, fitted))
            written.append(draw_test_subplots(out, draft_key, label, bar_c,
                                              edge_c, mkey, mlabel, depths,
                                              fitted))
            written.append(draw_test_overlay(out, draft_key, label, mkey,
                                             mlabel, depths, fitted))
            g2, agg_ece, agg_brier = draw_test_aggregate(
                out, draft_key, label, bar_c, edge_c, mkey, mlabel, depths,
                fitted)
            written.append(g2)
            mean_ece = np.nanmean([fitted[k]["ece"][mkey] for k in depths])
            print(f"   {mkey:9s} mean per-depth test ECE={mean_ece:.4f}  "
                  f"pooled test ECE={agg_ece:.4f} Brier={agg_brier:.4f}",
                  file=sys.stderr)

    print(f"wrote {len(written)} figures to {out}")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
