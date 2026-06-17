#!/usr/bin/env python3
"""Probability -> accept-rate calibration fits, one image per (draft, method).

For each draft (EAGLE-3, suffix decoding) we draw the empirical reliability
histogram (accept rate per fixed-width prob bin) and overlay, in red, the curve
fit by one of four calibration methods:

  1. histogram binning      -- per-bin empirical accept rate (step)
  2. isotonic regression     -- monotone PAV fit (sklearn IsotonicRegression)
  3. logistic regression     -- Platt scaling, sigmoid(a*p + b)
  4. beta calibration        -- logistic regression on [ln p, ln(1-p)]
                                (Kull et al. 2017)

8 images total: 2 drafts x 4 methods, saved to <dir>/figures/.

Samples come from the RAW select-1 decision log via
fit_chain_hybrid_calib.load_samples:
  EAGLE-3 draft : p = eagle_p  (draft softmax prob)
  suffix  draft : p = suffix_p = raw c/n  (Jeffreys shrink NOT applied)
  label y = 1 if the draft token was accepted (accept_len >= depth+1) else 0.

Usage:
    python3 simulation/scripts/plot_calib_methods.py \
        --dir simulation/results/chain_hybrid_bw_27b \
        [--decisions decisions_select1.jsonl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_chain_hybrid_calib import load_samples  # noqa: E402
from plot_calib_reliability import binned, BIN_W  # noqa: E402

from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

EPS = 1e-6
RED = "crimson"
# (key, pretty label, bar face color, bar edge color)
DRAFTS = [
    ("eagle3", "EAGLE-3", "#1f77b4", "#6baed6"),       # blue
    ("suffix", "suffix decoding", "#ff7f0e", "#fdae6b"),  # orange
]
METHODS = [
    ("histogram", "Histogram binning"),
    ("isotonic", "Isotonic regression"),
    ("logistic", "Logistic regression (Platt scaling)"),
    ("beta", "Beta calibration"),
]
PROBES = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9, 0.95])


def _grid(p: np.ndarray) -> np.ndarray:
    lo = max(EPS, float(np.min(p)))
    hi = min(1.0 - EPS, float(np.max(p)))
    return np.linspace(lo, hi, 400)


def _const_predictor(value: float):
    return lambda x: np.full(np.asarray(x, float).shape, value)


def fit_histogram(p, y, centers, rates):
    """Calibrator = per-bin empirical accept rate (the bar tops themselves)."""
    edges = np.arange(0.0, 1.0 + BIN_W, BIN_W)

    def predict(x):
        x = np.asarray(x, dtype=float)
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1,
                      0, len(rates) - 1)
        return rates[idx]

    valid = ~np.isnan(rates)
    return centers[valid], rates[valid], "step", predict


def fit_isotonic(p, y):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    predict = lambda x: iso.predict(np.asarray(x, dtype=float))
    g = _grid(p)
    return g, predict(g), "line", predict


def fit_logistic(p, y):
    if np.unique(y).size < 2:
        predict = _const_predictor(float(y.mean()))
    else:
        lr = LogisticRegression()
        lr.fit(p.reshape(-1, 1), y)
        predict = (lambda x: lr.predict_proba(
            np.asarray(x, dtype=float).reshape(-1, 1))[:, 1])
    g = _grid(p)
    return g, predict(g), "line", predict


def fit_beta(p, y):
    if np.unique(y).size < 2:
        predict = _const_predictor(float(y.mean()))
    else:
        pc = np.clip(p, EPS, 1.0 - EPS)
        X = np.column_stack([np.log(pc), np.log(1.0 - pc)])
        lr = LogisticRegression()
        lr.fit(X, y)

        def predict(x):
            xc = np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)
            Xg = np.column_stack([np.log(xc), np.log(1.0 - xc)])
            return lr.predict_proba(Xg)[:, 1]
    g = _grid(p)
    return g, predict(g), "line", predict


def make_figure(out_path, draft_label, bar_color, edge_color, method_label,
                centers, rates, n, base, spec):
    cx, cy, kind, _ = spec
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(centers, rates, width=BIN_W * 0.9, color=bar_color,
           edgecolor=edge_color, label="empirical accept (bin)")
    if kind == "step":
        ax.step(cx, cy, where="mid", color=RED, lw=2, label="fitted P(accept)")
    else:
        ax.plot(cx, cy, color=RED, lw=2, label="fitted P(accept)")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", alpha=0.7,
            label="perfect calib (y=x)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("draft probability p")
    ax.set_ylabel("P(accept)")
    ax.set_title(f"{method_label} — {draft_label}\n"
                 f"(n={n}, base accept={base:.3f})", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="simulation/results/chain_hybrid_bw_27b")
    ap.add_argument("--decisions", default="decisions_select1.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    s = load_samples(str(out_dir / args.decisions))

    arrays = {
        "eagle3": (np.asarray([r[0] for r in s["eagle"]], dtype=np.float64),
                   np.asarray([r[2] for r in s["eagle"]], dtype=np.float64)),
        # suffix uses raw c/n (r[0]) — Jeffreys (c+.5)/(n+1) deliberately NOT used.
        "suffix": (np.asarray([r[0] for r in s["suffix"]], dtype=np.float64),
                   np.asarray([r[2] for r in s["suffix"]], dtype=np.float64)),
    }

    written = []
    for key, label, bar_color, edge_color in DRAFTS:
        p, y = arrays[key]
        if p.size == 0:
            print(f"  WARNING: no samples for {key}; skipping", file=sys.stderr)
            continue
        n, base = int(p.size), float(y.mean())
        centers, rates, _counts = binned(p, y)
        specs = {
            "histogram": fit_histogram(p, y, centers, rates),
            "isotonic": fit_isotonic(p, y),
            "logistic": fit_logistic(p, y),
            "beta": fit_beta(p, y),
        }
        print(f"== {label}: n={n}, base accept={base:.3f} ==", file=sys.stderr)
        for mkey, mlabel in METHODS:
            spec = specs[mkey]
            predict = spec[3]
            cal = predict(PROBES)
            print("   " + f"{mkey:9s} p->P(accept): "
                  + ", ".join(f"{q:g}->{c:.3f}"
                              for q, c in zip(PROBES, cal)), file=sys.stderr)
            out = out_dir / "figures" / f"calib_fit_{key}_{mkey}.png"
            make_figure(out, label, bar_color, edge_color, mlabel,
                        centers, rates, n, base, spec)
            written.append(out)

    print(f"wrote {len(written)} figures:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
