"""Numerical calibration quality (fit-to-y=x) on the TEST split, offline.

Fits the 4 calibrators on the first --fit-n-tasks rids and reports, on the next
--test-n-tasks rids, how close the (calibrated) probability is to the empirical
accept rate — i.e. distance from y=x:

  ECE   Expected Calibration Error (PRIMARY): sum_b (n_b/N)|conf_b - acc_b|,
        15 fixed-width bins; conf_b = mean predicted prob in bin. The headline
        "how well does it sit on y=x" number.
  aECE  adaptive ECE: same but 15 EQUAL-MASS (quantile) bins — robust when
        probabilities cluster (EAGLE3 low / suffix discrete).
  MCE   Maximum Calibration Error: worst single-bin |conf - acc|.
  Brier mean (p - y)^2  (proper score: calibration + sharpness).
  NLL   mean negative log-likelihood (proper score).

The "raw" row is the UNCALIBRATED probability — the baseline to beat; a good
calibrator drives ECE down toward 0.

Usage:
  python3 simulation/scripts/calib_metrics.py \
      --pairs simulation/results/calib_verify/pairs_14b.jsonl.gz \
      --model-label EAGLE3 --fit-n-tasks 30 --test-n-tasks 10 \
      --out-json simulation/results/calib_verify/qwen3_14b/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_verify import _fit_specs, load_split  # noqa: E402
from plot_calib_methods import binned  # noqa: E402

METHODS = ["histogram", "isotonic", "logistic", "beta"]


def metrics(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    y = np.asarray(y, float)
    N = len(y)
    # fixed-width ECE / MCE
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        gap = abs(float(p[m].mean()) - float(y[m].mean()))
        ece += (m.sum() / N) * gap
        mce = max(mce, gap)
    # adaptive (equal-mass) ECE
    aece = 0.0
    order = np.argsort(p, kind="stable")
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        gap = abs(float(p[idx].mean()) - float(y[idx].mean()))
        aece += (len(idx) / N) * gap
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-7
    pc = np.clip(p, eps, 1 - eps)
    nll = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return {"ECE": ece, "aECE": aece, "MCE": mce, "Brier": brier, "NLL": nll}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    ap.add_argument("--fit-n-tasks", type=int, default=30)
    ap.add_argument("--test-n-tasks", type=int, default=10)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    data, nfit, ntest = load_split(args.pairs, args.fit_n_tasks, args.test_n_tasks)
    out = {"pairs": args.pairs, "fit_tasks": nfit, "test_tasks": ntest,
           "drafts": {}}
    for key, label in (("model", args.model_label), ("suffix", "Suffix")):
        p_fit, y_fit = data[key]["fit"]
        p_test, y_test = data[key]["test"]
        if p_fit.size == 0 or p_test.size == 0:
            continue
        centers, rates, _ = binned(p_fit, y_fit)
        specs = _fit_specs(p_fit, y_fit, centers, rates)
        rows = {"raw": metrics(p_test, y_test)}
        for mkey in METHODS:
            predict = specs[mkey][3]
            rows[mkey] = metrics(predict(p_test), y_test)
        out["drafts"][label] = {
            "test_base_accept": float(y_test.mean()),
            "test_n": int(p_test.size), "metrics": rows}

        print(f"\n=== {label}  (fit tasks={nfit}, test tasks={ntest}, "
              f"test n={p_test.size}, base accept={y_test.mean():.3f}) ===")
        print(f"  {'method':10s} {'ECE':>8s} {'aECE':>8s} {'MCE':>8s} "
              f"{'Brier':>8s} {'NLL':>8s}")
        for name in ["raw"] + METHODS:
            m = rows[name]
            star = "  <- best ECE" if (name != "raw" and m["ECE"] == min(
                rows[k]["ECE"] for k in METHODS)) else ""
            print(f"  {name:10s} {m['ECE']:8.4f} {m['aECE']:8.4f} {m['MCE']:8.4f} "
                  f"{m['Brier']:8.4f} {m['NLL']:8.4f}{star}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
