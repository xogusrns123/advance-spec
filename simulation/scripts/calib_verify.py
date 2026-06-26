"""Calibration verification: fit on 30 tasks, test (offline) on 10 tasks.

Consumes the (probability, accept) pairs extracted by
experiments/extract_calib_pairs.py for ONE model environment. Two draft kinds:
  model  — EAGLE3 (14B, blue) or MTP (27B, purple): per-edge draft prob.
  suffix — suffix decoding (orange): per-edge count ratio (non-Jeffreys).

Four calibration methods (fit on the FIT tasks only; never updated at test):
  histogram binning / isotonic regression / logistic (Platt) / beta calibration.

Graphs (per draft kind):
  GROUP 1  calib_fit_<kind>_<method>.png — FIT-set probability->accept-rate
           histogram (kind color) + the fitted calibrator as a red line.
  GROUP 2  calib_test_<kind>_<method>.png — apply the frozen calibrator to the
           TEST set, then histogram the CALIBRATED probability vs empirical
           accept rate, with y=x as a red dashed line (perfect-calibration
           target). Closeness to y=x = how well calibration generalized.

Task split: rows are grouped by rid in first-seen order; the first --fit-n-tasks
rids are the fit set, the next --test-n-tasks the test set (disjoint).

Usage:
  python3 simulation/scripts/calib_verify.py \
      --pairs simulation/results/calib_verify/pairs_14b.jsonl.gz \
      --out-dir simulation/results/calib_verify/qwen3_14b \
      --model-label EAGLE3 --model-color "#1f77b4" \
      --fit-n-tasks 30 --test-n-tasks 10
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_calib_methods import (  # noqa: E402
    BIN_W, EPS, RED, binned, fit_beta, fit_histogram, fit_isotonic, fit_logistic,
)

METHODS = [
    ("histogram", "Histogram binning"),
    ("isotonic", "Isotonic regression"),
    ("logistic", "Logistic regression (Platt scaling)"),
    ("beta", "Beta calibration"),
]
SUFFIX_COLOR = "#ff7f0e"
SUFFIX_EDGE = "#e8810b"


def load_split(pairs_path: str, fit_n: int, test_n: int):
    """Group rows by rid (first-seen order); return fit/test (p, y) per kind."""
    per_rid = OrderedDict()  # rid -> {"m":([p],[y]), "s":([p],[y])}
    with gzip.open(pairs_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r["rid"]
            d = per_rid.setdefault(rid, {"m": ([], []), "s": ([], [])})
            for p, y in r.get("m", []):
                d["m"][0].append(p); d["m"][1].append(y)
            for p, y in r.get("s", []):
                d["s"][0].append(p); d["s"][1].append(y)
    rids = list(per_rid)
    fit_rids = rids[:fit_n]
    test_rids = rids[fit_n:fit_n + test_n]
    if len(fit_rids) < fit_n or len(test_rids) < test_n:
        print(f"  WARNING: only {len(rids)} rids; fit={len(fit_rids)} "
              f"test={len(test_rids)} (wanted {fit_n}/{test_n})", file=sys.stderr)

    def gather(rid_list, kind):
        ps, ys = [], []
        for rid in rid_list:
            ps += per_rid[rid][kind][0]
            ys += per_rid[rid][kind][1]
        return np.asarray(ps, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    out = {}
    for key, kind in (("model", "m"), ("suffix", "s")):
        out[key] = {
            "fit": gather(fit_rids, kind),
            "test": gather(test_rids, kind),
        }
    return out, len(fit_rids), len(test_rids)


def _fit_specs(p, y, centers, rates):
    # Continuous objective (target_p): y is q_target in [0,1], not binary {0,1};
    # logistic/beta then need the LinearRegression variants (the binary sklearn
    # classifiers raise on continuous labels). histogram/isotonic are native.
    uniq = np.unique(y)
    is_cont = uniq.size > 2 or not np.all(np.isin(uniq, (0.0, 1.0)))
    if is_cont:
        from calib_perposition import _cont_spec  # noqa: E402
        logistic = _cont_spec("logistic", p, y)
        beta = _cont_spec("beta", p, y)
    else:
        logistic = fit_logistic(p, y)
        beta = fit_beta(p, y)
    return {
        "histogram": fit_histogram(p, y, centers, rates),
        "isotonic": fit_isotonic(p, y),
        "logistic": logistic,
        "beta": beta,
    }


def group1_fig(out, kind_key, label, bar_c, edge_c, mkey, mlabel, spec,
               p_fit, y_fit, centers, rates):
    cx, cy, draw_kind, _ = spec
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(centers, rates, width=BIN_W * 0.9, color=bar_c, edgecolor=edge_c,
           label="empirical accept (fit bin)")
    if draw_kind == "step":
        ax.step(cx, cy, where="mid", color=RED, lw=2, label="fitted calibrator")
    else:
        ax.plot(cx, cy, color=RED, lw=2, label="fitted calibrator")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{label} draft probability p")
    ax.set_ylabel("P(accept)")
    ax.set_title(f"{mlabel} — {label} (fit)\n"
                 f"(fit n={len(p_fit)}, base accept={float(y_fit.mean()):.3f})",
                 fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_p = out / f"calib_fit_{kind_key}_{mkey}.png"
    fig.savefig(out_p, dpi=150); plt.close(fig)
    return out_p


def group2_fig(out, kind_key, label, bar_c, edge_c, mkey, mlabel, spec,
               p_test, y_test):
    predict = spec[3]
    cal = np.clip(predict(p_test), 0.0, 1.0)
    centers, rates, counts = binned(cal, y_test)
    # ECE on test: weighted |mean cal - accept| over non-empty bins.
    m = ~np.isnan(rates)
    w = counts[m] / max(1, counts[m].sum())
    ece = float(np.sum(w * np.abs(centers[m] - rates[m]))) if m.any() else float("nan")
    brier = float(np.mean((cal - y_test) ** 2)) if len(y_test) else float("nan")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(centers, rates, width=BIN_W * 0.9, color=bar_c, edgecolor=edge_c,
           label="empirical accept (test bin)")
    ax.plot([0, 1], [0, 1], color=RED, lw=2, ls="--", label="y = x (target)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"calibrated probability  ({mlabel})")
    ax.set_ylabel("empirical accept rate (test)")
    ax.set_title(f"{mlabel} — {label} (test, offline)\n"
                 f"test n={len(p_test)}, ECE={ece:.3f}, Brier={brier:.3f}",
                 fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_p = out / f"calib_test_{kind_key}_{mkey}.png"
    fig.savefig(out_p, dpi=150); plt.close(fig)
    return out_p, ece, brier


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3",
                    help="label for the model-based draft (EAGLE3 / MTP)")
    ap.add_argument("--model-color", default="#1f77b4",
                    help="bar color for the model-based draft (blue/purple)")
    ap.add_argument("--fit-n-tasks", type=int, default=30)
    ap.add_argument("--test-n-tasks", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out_dir) / "histogram"  # bar reliability plots live here
    out.mkdir(parents=True, exist_ok=True)
    data, nfit, ntest = load_split(args.pairs, args.fit_n_tasks, args.test_n_tasks)
    print(f"fit tasks={nfit} test tasks={ntest}", file=sys.stderr)

    drafts = [
        ("model", args.model_label, args.model_color, args.model_color),
        ("suffix", "Suffix", SUFFIX_COLOR, SUFFIX_EDGE),
    ]
    written = []
    for key, label, bar_c, edge_c in drafts:
        p_fit, y_fit = data[key]["fit"]
        p_test, y_test = data[key]["test"]
        if p_fit.size == 0 or p_test.size == 0:
            print(f"  WARNING: no {key} pairs; skipping", file=sys.stderr)
            continue
        kind_key = label.lower()
        centers, rates, _ = binned(p_fit, y_fit)
        specs = _fit_specs(p_fit, y_fit, centers, rates)
        print(f"== {label}: fit n={p_fit.size} (accept {y_fit.mean():.3f}), "
              f"test n={p_test.size} (accept {y_test.mean():.3f}) ==",
              file=sys.stderr)
        for mkey, mlabel in METHODS:
            spec = specs[mkey]
            written.append(group1_fig(out, kind_key, label, bar_c, edge_c,
                                      mkey, mlabel, spec, p_fit, y_fit,
                                      centers, rates))
            g2, ece, brier = group2_fig(out, kind_key, label, bar_c, edge_c,
                                        mkey, mlabel, spec, p_test, y_test)
            written.append(g2)
            print(f"   {mkey:9s} test ECE={ece:.3f} Brier={brier:.3f}",
                  file=sys.stderr)

    print(f"wrote {len(written)} figures to {out}")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
