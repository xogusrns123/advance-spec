#!/usr/bin/env python3
r"""WHAT does the fitted tail calibration optimize, and why does it miss the MAT
optimum? Extracts the (arctic score, realized tail accept) pairs that scaled/
linear/isotonic are fit on (the SAME pairs _fit_tail_iso uses), on the calibrate
half, then computes:

  rho_scaled  = sum(y)/sum(x)          # what --tail-cal scaled fits (mean-match)
  w_ols0      = sum(xy)/sum(x^2)       # slope that MINIMIZES sum (w*x - y)^2
  a,b (OLS)   = polyfit(x,y,1)         # what --tail-cal linear fits
  MSE(w)      = mean (w*x - y)^2       # the objective the L2 fits minimize
  bias(w)     = mean (w*x - y)

and dumps a JSON per workload so a companion plot can overlay MSE(w) (what fitting
minimizes) against MAT(w) (what we actually want, from the replay sweeps). If the
MSE-minimizing w differs from the MAT-maximizing w, that IS the reason fitting
misses the optimum.

  docker exec sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace \
    python3 scripts/analysis/investigate_tail_objective.py <workload> <record.jsonl>'
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np                       # noqa: E402
import replay_extension as R             # noqa: E402

OUT = Path("results/headweight_tailsmooth/tail_objective")


def main():
    ds, record = sys.argv[1], sys.argv[2]
    gm, max_rounds = "convlabel", 4096
    traces = json.load(open(Path(record).with_suffix(".traces.json")))
    warm = traces["warm_traces"]
    ev = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)
    recs = defaultdict(dict)
    for ln in open(record):
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            recs[r["rid"]][r["pos"]] = r
    _, split_par = R.split_parity(ev, gm)
    calib_rids = {rid for rid in recs if split_par.get(rid, 0) == 0}

    # the exact (score, realized-tail-accept) pairs the L2 tail fits see
    _, xs, ys = R._fit_tail_iso(warm, recs, ev, calib_rids, num_spec, max_rounds,
                                return_pairs=True)
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    n = len(x)
    sx, sy, sxy, sxx = x.sum(), y.sum(), float((x * y).sum()), float((x * x).sum())
    rho_scaled = sy / sx if sx else 0.0                       # scaled: mean-match
    w_ols0 = sxy / sxx if sxx else 0.0                        # min MSE through origin
    a, b = np.polyfit(x, y, 1)                                # linear a*x+b (OLS)

    wgrid = [round(w, 4) for w in np.concatenate([
        np.arange(0.002, 0.02, 0.002), np.arange(0.02, 0.30, 0.01)]).tolist()]
    mse = {w: float(np.mean((w * x - y) ** 2)) for w in wgrid}
    bias = {w: float(np.mean(w * x - y)) for w in wgrid}
    w_mse_min = min(mse, key=mse.get)

    res = {
        "ds": ds, "n_pairs": n,
        "mean_score": float(x.mean()), "mean_realized": float(y.mean()),
        "rho_scaled_meanmatch": float(rho_scaled),   # E[y]/E[x]
        "w_ols_through_origin": float(w_ols0),       # argmin_w E[(wx-y)^2]
        "ols_a": float(a), "ols_b": float(b),
        "w_mse_min_gridded": float(w_mse_min),
        "mse_curve": mse, "bias_curve": bias,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{ds}.json", "w"), indent=1)
    print(f"[{ds}] n={n}  E[score]={x.mean():.2f}  E[realized]={y.mean():.3f}")
    print(f"  rho (scaled, mean-match)   = {rho_scaled:.4f}")
    print(f"  w* (min-MSE thru origin)   = {w_ols0:.4f}")
    print(f"  OLS a,b (linear)           = {a:.4f}, {b:.4f}   (implied slope {a:.4f})")
    print(f"  -> fitted tail targets w ~= {rho_scaled:.3f}-{w_ols0:.3f};  MAT sweep peaks far below")


if __name__ == "__main__":
    main()
