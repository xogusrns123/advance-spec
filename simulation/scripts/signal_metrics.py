"""Signal-existence metrics: does a feature predict token acceptance?

Pure functions (no plotting). For each (feature x, binary accept y) we ask how
much SIGNAL x carries about y — NOT how to calibrate x. Metrics:

  AUROC  Area under the ROC curve of the single feature (sklearn). 0.5 = no
         signal; threshold-free and invariant to any monotone transform of x.
         We report the direction-folded value max(a, 1-a) as the headline
         (a feature predicting REJECTION is still signal), plus the signed AUC
         and direction (+1/-1).
  MI     Mutual information I(bin(x); y) (nats), and the normalized fraction
         MI/H(y) in [0,1]. Catches NON-monotone dependence that AUROC misses.
  pbr    Point-biserial correlation (= Pearson r between x and binary y): a
         signed linear/monotone strength; its sign must agree with AUROC.
  within_depth_auc  THE confound control. depth dominates accept rate, so a
         feature can look predictive merely by correlating with depth. We
         compute AUROC within each depth slice and report the sample-weighted
         mean. marginal high + within-depth ~0.5 => the feature is just a depth
         proxy; within-depth staying high => signal BEYOND depth.

A tiny positive MI / |AUC-0.5| is the expected estimator bias floor; compare
against the injected `random` negative control the driver adds.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mutual_info_score, roc_auc_score

NAN = float("nan")


def _bin_ids(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Discrete: rank ids over unique values. Continuous: equal-mass bins."""
    x = np.asarray(x, float)
    uniq = np.unique(x)
    if len(uniq) <= n_bins:
        return np.searchsorted(uniq, x)
    qs = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(qs) <= 2:
        return np.zeros(len(x), dtype=int)
    return np.clip(np.searchsorted(qs[1:-1], x, side="right"), 0, len(qs) - 2)


def auroc(x: np.ndarray, y: np.ndarray):
    """(folded_auc, signed_auc, direction, n) over finite x; None if degenerate."""
    x = np.asarray(x, float)
    y = np.asarray(y)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if len(y) == 0 or y.min() == y.max():
        return None, None, 0, int(len(y))
    a = float(roc_auc_score(y, x))
    return max(a, 1.0 - a), a, (1 if a >= 0.5 else -1), int(len(y))


def mi_binned(x: np.ndarray, y: np.ndarray, n_bins: int):
    """(MI nats, MI/H(y)) over finite x; None if degenerate."""
    x = np.asarray(x, float)
    y = np.asarray(y)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if len(y) == 0 or y.min() == y.max():
        return None, None
    mi = float(mutual_info_score(_bin_ids(x, n_bins), y))
    hy = float(mutual_info_score(y, y))  # = H(y) in nats
    return mi, (mi / hy if hy > 1e-12 else NAN)


def point_biserial(x: np.ndarray, y: np.ndarray):
    """Pearson r between x and binary y (= point-biserial); None if degenerate."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if len(y) < 3 or np.std(x) == 0 or y.min() == y.max():
        return None
    return float(np.corrcoef(x, y)[0, 1])


def within_depth_auc(x, y, depth, min_per_slice: int = 200):
    """Sample-weighted mean of per-depth folded AUROC. (mean, n_slices, per)."""
    x = np.asarray(x, float)
    y = np.asarray(y)
    depth = np.asarray(depth)
    aucs, weights, per = [], [], {}
    for k in np.unique(depth):
        sel = depth == k
        if sel.sum() < min_per_slice:
            continue
        fold, _, _, n = auroc(x[sel], y[sel])
        if fold is None or n < min_per_slice:
            continue
        aucs.append(fold)
        weights.append(n)
        per[int(k)] = {"auc": round(fold, 4), "n": n}
    if not aucs:
        return None, 0, per
    return float(np.average(aucs, weights=weights)), len(aucs), per


def describe(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y)
    fin = np.isfinite(x)
    xf = x[fin]
    q = (np.quantile(xf, [0.1, 0.5, 0.9]) if len(xf) else [NAN, NAN, NAN])
    return {
        "n": int(fin.sum()),
        "base_accept": (float(np.mean(y[fin])) if fin.any() else NAN),
        "x_mean": (float(xf.mean()) if len(xf) else NAN),
        "x_std": (float(xf.std()) if len(xf) else NAN),
        "x_p10": float(q[0]), "x_p50": float(q[1]), "x_p90": float(q[2]),
        "frac_nan": (float(1.0 - fin.mean()) if len(x) else NAN),
    }


def _r(v, nd=4):
    return round(v, nd) if v is not None else None


def feature_signal(x, y, depth, name, kind="per-edge", *, is_depth=False,
                   mi_bins=20, min_per_slice=200) -> dict:
    """All signal metrics for one feature -> one record dict."""
    rec = {"name": name, "kind": kind}
    rec.update(describe(x, y))
    fold, signed, direc, _ = auroc(x, y)
    rec["marginal_auc"] = _r(fold)
    rec["signed_auc"] = _r(signed)
    rec["direction"] = direc
    rec["effect"] = (_r(abs(signed - 0.5) * 2) if signed is not None else None)
    mi, mifrac = mi_binned(x, y, mi_bins)
    rec["mi_nats"] = _r(mi, 5)
    rec["mi_frac"] = _r(mifrac)
    rec["pbr"] = _r(point_biserial(x, y))
    if is_depth:
        rec["within_depth_auc"] = None
        rec["n_slices_used"] = 0
        rec["per_depth_auc"] = {}
    else:
        wauc, nsl, per = within_depth_auc(x, y, depth, min_per_slice)
        rec["within_depth_auc"] = _r(wauc)
        rec["n_slices_used"] = nsl
        rec["per_depth_auc"] = per
    return rec


def signals_for_draft(features, y, depth, *, mi_bins=20, min_per_slice=200):
    """features: list of (name, x_array, kind, is_depth). -> {name: record}."""
    out = {}
    for name, x, kind, is_depth in features:
        out[name] = feature_signal(
            x, y, depth, name, kind=kind, is_depth=is_depth,
            mi_bins=mi_bins, min_per_slice=min_per_slice)
    return out
