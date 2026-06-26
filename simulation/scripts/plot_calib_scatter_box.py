"""Scatter + box-plot views of the calibration reliability data.

Companion to calib_verify.py (aggregate, 30/10 task split) and
calib_perposition.py (per-depth, 70/30 sample split). Those draw bar
"histograms" of probability -> accept rate; this draws two ALTERNATIVE views of
the SAME (probability, accept) data, organized into typed subfolders so the
figure set stays navigable:

  <out>/scatter/calib_{fit,test}_<draft>_<method>.png
      Per probability-bin SCATTER: one dot per bin, x = mean predicted prob,
      y = empirical accept rate, dot size ∝ #samples. fit = raw draft prob vs the
      fitted calibrator (red); test = calibrated prob vs y=x (target).
  <out>/boxplot/calib_{fit,test}_<draft>_<method>.png
      Per probability-bin BOX: box = accept-rate mean ± 1 SE, whisker ± 2 SE
      (SE = sqrt(r(1-r)/n)); the red center line marks the mean. Same reference.
  <out>/perposition/scatter/pp_{,test_}<draft>_<method>.png
      All depths' per-bin dots on ONE axes, viridis-colored by depth, vs y=x.
  <out>/perposition/boxplot/pp_{,test_}<draft>_<method>.png
      Grid, one box-per-bin panel per depth (fit = train slice + fitted curve;
      test = 30% held-out slice + y=x).

The bar histograms now live in <out>/histogram/ and <out>/perposition/histogram/
(calib_verify.py / calib_perposition.py write there directly).

Usage (inside sglang-bench container, from /workspace):
  python3 simulation/scripts/plot_calib_scatter_box.py \
      --pairs simulation/results/calib_verify/pairs_14b.jsonl.gz \
      --out-dir simulation/results/calib_verify/qwen3_14b \
      --model-label EAGLE3 --model-color "#1f77b4"
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_calib_methods import BIN_W, RED, binned  # noqa: E402
from calib_verify import load_split, _fit_specs  # noqa: E402
from calib_perposition import load_per_depth, fit_depth  # noqa: E402

METHODS = [
    ("histogram", "Histogram binning"),
    ("isotonic", "Isotonic regression"),
    ("logistic", "Logistic regression (Platt scaling)"),
    ("beta", "Beta calibration"),
]
SUFFIX_COLOR, SUFFIX_EDGE = "#ff7f0e", "#e8810b"


def per_bin_stats(x, y):
    """Per fixed-width prob bin over x: (mean_x, accept_rate, count, SE_of_rate).

    Empty bins are dropped. SE = sqrt(r(1-r)/n) is the standard error of the
    per-bin accept-rate estimate.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    edges = np.arange(0.0, 1.0 + BIN_W, BIN_W)
    xs, rs, ns, ses = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) if hi < 1.0 else (x >= lo) & (x <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        r = float(y[m].mean())
        xs.append(float(x[m].mean())); rs.append(r); ns.append(n)
        ses.append(math.sqrt(max(r * (1.0 - r), 0.0) / n))
    return (np.asarray(xs), np.asarray(rs), np.asarray(ns, float),
            np.asarray(ses))


def _bxp_stats(rs, ses):
    """Box per bin: box = mean ± SE, whisker = ± 2 SE, median line = mean."""
    return [dict(med=r, q1=max(0.0, r - se), q3=min(1.0, r + se),
                 whislo=max(0.0, r - 2 * se), whishi=min(1.0, r + 2 * se),
                 fliers=[]) for r, se in zip(rs, ses)]


def _draw_ref(ax, spec, stage, draft_label, mlabel):
    """fit -> fitted calibrator curve (red) + faint y=x; test -> red dashed y=x."""
    if stage == "fit":
        cx, cy, kind = spec
        if kind == "step":
            ax.step(cx, cy, where="mid", color=RED, lw=2, label="fitted calibrator")
        else:
            ax.plot(cx, cy, color=RED, lw=2, label="fitted calibrator")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel(f"{draft_label} draft probability p")
    else:
        ax.plot([0, 1], [0, 1], color=RED, lw=2, ls="--", label="y = x (target)")
        ax.set_xlabel(f"calibrated probability  ({mlabel})")
    ax.set_ylabel("empirical accept rate")


def _stage_word(stage):
    return "raw" if stage == "fit" else "calibrated"


# ----------------------------- aggregate ---------------------------------- #

def scatter_fig(path, draft_label, color, edge, mlabel, stage, x, y, spec):
    xs, rs, ns, _ = per_bin_stats(x, y)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if len(xs):
        s = 15.0 + 285.0 * (ns / ns.max())
        ax.scatter(xs, rs, s=s, c=color, edgecolors=edge, lw=0.6, alpha=0.85,
                   zorder=3, label="empirical accept (bin, size∝n)")
    _draw_ref(ax, spec, stage, draft_label, mlabel)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"{mlabel} — {draft_label} ({_stage_word(stage)})", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def box_fig(path, draft_label, color, edge, mlabel, stage, x, y, spec):
    xs, rs, ns, ses = per_bin_stats(x, y)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if len(xs):
        bp = ax.bxp(_bxp_stats(rs, ses), positions=xs, widths=BIN_W * 0.6,
                    manage_ticks=False, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=color, edgecolor=edge, alpha=0.8)
        for med in bp["medians"]:
            med.set(color=RED, lw=1.4)
        for wk in bp["whiskers"] + bp["caps"]:
            wk.set(color=edge, lw=0.9)
    _draw_ref(ax, spec, stage, draft_label, mlabel)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_title(f"{mlabel} — {draft_label} ({_stage_word(stage)})\n"
                 f"box = accept rate mean ± SE, whisker ± 2 SE", fontsize=9.5)
    ax.grid(alpha=0.25)
    proxy = Patch(facecolor=color, edgecolor=edge, alpha=0.8,
                  label="accept rate (mean ± SE)")
    h, _l = ax.get_legend_handles_labels()
    ax.legend(handles=[proxy] + h, fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------- perposition -------------------------------- #

def _depth_xy(fd, stage, mkey):
    """(x, y) for one depth: fit = raw train slice; test = calibrated 30% test."""
    if stage == "fit":
        return fd["p_tr"], fd["y_tr"]
    if not len(fd["y_te"]):
        return np.asarray([]), np.asarray([])
    return np.clip(fd["specs"][mkey][3](fd["p_te"]), 0.0, 1.0), fd["y_te"]


def pp_scatter_fig(path, draft_label, mlabel, stage, mkey, fitted, depths, note=""):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    norm = Normalize(vmin=0, vmax=max(depths) if depths else 1)
    cmap = plt.get_cmap("viridis")
    for k in depths:
        x, yy = _depth_xy(fitted[k], stage, mkey)
        if not len(yy):
            continue
        xs, rs, ns, _ = per_bin_stats(x, yy)
        if not len(xs):
            continue
        ax.scatter(xs, rs, s=12.0 + 120.0 * (ns / ns.max()),
                   color=cmap(norm(k)), edgecolors="none", alpha=0.8, zorder=3)
    ax.plot([0, 1], [0, 1], color=RED, lw=1.6, ls="--", label="y = x (target)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("calibrated probability" if stage == "test"
                  else f"{draft_label} draft probability p")
    ax.set_ylabel("empirical accept rate")
    ax.set_title(f"Per-position {mlabel} — {draft_label}  (per-depth dots){note}",
                 fontsize=10.5)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cb.set_label("depth k")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def pp_box_subplots(path, draft_label, color, edge, mlabel, stage, mkey, fitted,
                    depths, note=""):
    K = len(depths)
    cols = math.ceil(math.sqrt(K)); rows = math.ceil(K / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6),
                             squeeze=False)
    for i, k in enumerate(depths):
        ax = axes[i // cols][i % cols]
        fd = fitted[k]
        x, yy = _depth_xy(fd, stage, mkey)
        xs, rs, ns, ses = per_bin_stats(x, yy) if len(yy) else (np.asarray([]),) * 4
        if len(xs):
            bp = ax.bxp(_bxp_stats(rs, ses), positions=xs, widths=BIN_W * 0.6,
                        manage_ticks=False, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=color, edgecolor=edge, alpha=0.8)
            for med in bp["medians"]:
                med.set(color=RED, lw=1.0)
            for wk in bp["whiskers"] + bp["caps"]:
                wk.set(color=edge, lw=0.6)
        if stage == "fit":
            cx, cy, kind, _ = fd["specs"][mkey]
            if kind == "step":
                ax.step(cx, cy, where="mid", color=RED, lw=1.0)
            else:
                ax.plot(cx, cy, color=RED, lw=1.0)
            ax.plot([0, 1], [0, 1], color="gray", lw=0.6, ls=":", alpha=0.6)
        else:
            ax.plot([0, 1], [0, 1], color=RED, lw=1.2, ls="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"k={k}  n={len(yy)}", fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(K, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Per-position {mlabel} — {draft_label}  (box = accept rate "
                 f"mean ± SE per prob-bin, {'30% test' if stage == 'test' else 'train'}){note}",
                 fontsize=11)
    fig.supxlabel("calibrated probability" if stage == "test"
                  else "draft probability p", fontsize=9)
    fig.supylabel("empirical accept rate", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    ap.add_argument("--model-color", default="#1f77b4")
    ap.add_argument("--fit-n-tasks", type=int, default=30)
    ap.add_argument("--test-n-tasks", type=int, default=10)
    ap.add_argument("--max-positions", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    sc, bx = out / "scatter", out / "boxplot"
    ppsc, ppbx = out / "perposition" / "scatter", out / "perposition" / "boxplot"
    for d in (sc, bx, ppsc, ppbx):
        d.mkdir(parents=True, exist_ok=True)

    drafts = [("model", args.model_label, args.model_color, args.model_color),
              ("suffix", "Suffix", SUFFIX_COLOR, SUFFIX_EDGE)]
    n = 0

    # ---- aggregate (30/10 task split, mirrors calib_verify.py) ----
    data, nfit, ntest = load_split(args.pairs, args.fit_n_tasks, args.test_n_tasks)
    print(f"aggregate: fit tasks={nfit} test tasks={ntest}", file=sys.stderr)
    for key, label, color, edge in drafts:
        p_fit, y_fit = data[key]["fit"]
        p_test, y_test = data[key]["test"]
        if p_fit.size == 0 or p_test.size == 0:
            print(f"  WARNING: no {key} pairs; skip aggregate", file=sys.stderr)
            continue
        dk = label.lower()
        centers, rates, _ = binned(p_fit, y_fit)
        specs = _fit_specs(p_fit, y_fit, centers, rates)
        for mkey, mlabel in METHODS:
            cx, cy, kind, predict = specs[mkey]
            ref = (cx, cy, kind)
            scatter_fig(sc / f"calib_fit_{dk}_{mkey}.png", label, color, edge,
                        mlabel, "fit", p_fit, y_fit, ref)
            box_fig(bx / f"calib_fit_{dk}_{mkey}.png", label, color, edge,
                    mlabel, "fit", p_fit, y_fit, ref)
            cal = np.clip(predict(p_test), 0.0, 1.0)
            scatter_fig(sc / f"calib_test_{dk}_{mkey}.png", label, color, edge,
                        mlabel, "test", cal, y_test, None)
            box_fig(bx / f"calib_test_{dk}_{mkey}.png", label, color, edge,
                    mlabel, "test", cal, y_test, None)
            n += 4

    # ---- perposition (70/30 sample split, seed-matched to calib_perposition.py) ----
    pdata = load_per_depth(args.pairs, args.max_positions)
    rng = np.random.default_rng(args.seed)
    for key, label, color, edge in drafts:
        per_depth = pdata[key]
        depths = sorted(per_depth)[:args.max_positions]
        if not depths:
            print(f"  WARNING: no {key} depths; skip perposition", file=sys.stderr)
            continue
        fitted = {k: fit_depth(*per_depth[k], args.train_frac, rng) for k in depths}
        dk = label.lower()
        for mkey, mlabel in METHODS:
            pp_scatter_fig(ppsc / f"pp_{dk}_{mkey}.png", label, mlabel, "fit",
                           mkey, fitted, depths)
            pp_scatter_fig(ppsc / f"pp_test_{dk}_{mkey}.png", label, mlabel,
                           "test", mkey, fitted, depths)
            pp_box_subplots(ppbx / f"pp_{dk}_{mkey}.png", label, color, edge,
                            mlabel, "fit", mkey, fitted, depths)
            pp_box_subplots(ppbx / f"pp_test_{dk}_{mkey}.png", label, color,
                            edge, mlabel, "test", mkey, fitted, depths)
            n += 4

    print(f"wrote {n} figures under {out}/{{scatter,boxplot,perposition/scatter,"
          f"perposition/boxplot}}")


if __name__ == "__main__":
    main()
