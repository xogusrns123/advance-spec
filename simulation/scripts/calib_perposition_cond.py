"""Per-position calibration verification for the ACCEPT-CONDITIONED (cond-trained)
calibrators — boxplot / histogram / scatter, train (fit overlaid) + test.

This is the calib_perposition.py / plot_calib_scatter_box.py figure set, but the
per-depth samples are drawn with CONDITIONAL SAMPLING (accept-conditioned): at
depth k only rows whose chain prefix stayed on the GT trajectory through k-1 are
kept (the same `_alive_depths` filter the cond-trained serving maps are fit with,
fit_chain_hybrid_calib_perpos.py --accept-conditioned). The 70/30 train/test
split runs on the already-conditioned pool, so BOTH train and test are
accept-conditioned.

Three calibrator MODES (--mode), motivated by the fact that accept-conditioning
starves deep depths of samples so an INDEPENDENT per-depth fit is high-variance:
  per_depth     : one calibrator fit per depth (the original; noisy when sparse).
  global        : ONE calibrator fit on ALL depths' conditioned samples pooled
                  (depth-agnostic); the same red curve is overlaid on every
                  depth's data. Robust, but cannot bend with depth.
  depth_feature : ONE joint model with depth as an input feature. logistic =>
                  P(accept)=sigmoid(w0+w1 p+w2 d+w3 p*d) (interaction lets the
                  curve's slope shift with depth); beta => logit-linear in
                  [ln p, ln(1-p), d, d*ln p, d*ln(1-p)]; continuous target_p uses
                  the LinearRegression analog. histogram/isotonic have no natural
                  depth feature, so they fall back to the pooled global fit.
                  Shares statistical strength across depths while staying
                  depth-aware -> the intended fix for sparse deep depths.

Source = an ORACLE select-1 decision log (has gt_token + oracle_hit). Two
objectives, matching the cond-trained map families:
  accept_rate : label y = (token == gt_token), binary  -> "P(accept)"
  target_p    : label y = q_target(token | GT prefix), continuous in [0,1]
                (needs --target-prob-file from capture_target_probs.py)

Outputs (mirror the calib_verify reference layout):
  <out>/perposition/histogram/pp_{,test_}<draft>_<m>_{subplots,overlay}.png
  <out>/perposition/histogram/pp_test_<draft>_<m>_aggregate.png
  <out>/perposition/boxplot/pp_{,test_}<draft>_<m>.png     <- box = accept rate ±SE + fit
  <out>/perposition/scatter/pp_{,test_}<draft>_<m>.png

Usage (inside sglang-bench container, from /workspace):
  python3 simulation/scripts/calib_perposition_cond.py \
      --decision-log simulation/results/chain_hybrid_perdepth/qwen3_14b_tp_train/decisions_select1_oracle.jsonl \
      --out-dir simulation/results/calib_verify/qwen3_14b_cond_ar_global \
      --objective accept_rate --mode global \
      --model-label EAGLE3 --model-color "#1f77b4" --max-positions 16
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_chain_hybrid_calib_perpos import (  # noqa: E402
    _alive_depths, _fit_depth_feature, load_oracle_samples,
    load_target_prob_samples,
)
from calib_perposition import (  # noqa: E402
    METHODS, ece, fit_depth, draw_subplots, draw_overlay, draw_test_subplots,
    draw_test_overlay, draw_test_aggregate,
)
from calib_verify import _fit_specs  # noqa: E402
from plot_calib_methods import binned  # noqa: E402
from plot_calib_scatter_box import (  # noqa: E402
    pp_box_subplots, pp_scatter_fig, SUFFIX_COLOR, SUFFIX_EDGE,
)

GRID = np.linspace(0.0, 1.0, 400)


def to_per_depth(samples: dict, max_pos: int) -> dict:
    """{'eagle':[(p,depth,y,...)], 'suffix':[...]} -> per-draft {depth:(p[],y[])}.
    Keyed 'model'/'suffix' to match the drawers' draft keys."""
    out = {}
    for key, rows in (("model", samples["eagle"]), ("suffix", samples["suffix"])):
        dp = defaultdict(list)
        dy = defaultdict(list)
        for row in rows:
            depth = int(row[1])
            if 0 <= depth < max_pos:
                dp[depth].append(float(row[0]))
                dy[depth].append(float(row[2]))
        out[key] = {k: (np.asarray(dp[k], np.float64), np.asarray(dy[k], np.float64))
                    for k in dp}
    return out


def _spec_from_predict(predict_1d, kind="line"):
    return (GRID, np.clip(predict_1d(GRID), 0.0, 1.0), kind, predict_1d)


def build_fitted(per_depth: dict, depths: list, mode: str, frac: float,
                 rng, continuous: bool) -> dict:
    """Return the {depth: fit-dict} structure the drawers consume. per_depth uses
    an independent split+fit per depth; global / depth_feature share ONE pooled
    70/30 split and ONE fitted model, sliced back per depth for the panels."""
    if mode == "per_depth":
        return {k: fit_depth(*per_depth[k], frac, rng) for k in depths}

    # one global sample-level split shared across depths (tag each row with depth)
    Ps, Ys, Ds = [], [], []
    for k in depths:
        p, y = per_depth[k]
        Ps.append(p); Ys.append(y); Ds.append(np.full(len(y), k, float))
    P, Y, D = np.concatenate(Ps), np.concatenate(Ys), np.concatenate(Ds)
    idx = rng.permutation(len(Y))
    ntr = max(1, int(round(len(Y) * frac)))
    tr, te = idx[:ntr], idx[ntr:]
    Ptr, Ytr, Dtr = P[tr], Y[tr], D[tr]
    Pte, Yte, Dte = P[te], Y[te], D[te]

    centers_g, rates_g, _ = binned(Ptr, Ytr)
    specs_global = _fit_specs(Ptr, Ytr, centers_g, rates_g)  # {m:(cx,cy,kind,pred)}
    df = {}
    if mode == "depth_feature":
        for m in ("logistic", "beta"):
            df[m] = _fit_depth_feature(m, Ptr, Ytr, Dtr, continuous)

    fitted = {}
    for k in depths:
        p_tr, y_tr = Ptr[Dtr == k], Ytr[Dtr == k]
        p_te, y_te = Pte[Dte == k], Yte[Dte == k]
        if len(y_tr):
            centers, rates, _ = binned(p_tr, y_tr)
        else:
            centers, rates = np.asarray([]), np.asarray([])
        specs = {}
        for m, _ in METHODS:
            if mode == "depth_feature" and m in df:
                pred = (lambda mm, kk: (lambda x: df[mm](x, kk)))(m, k)
                specs[m] = _spec_from_predict(pred, "line")
            else:
                specs[m] = specs_global[m]  # depth-agnostic curve
        eces = {m: (ece(specs[m][3](p_te), y_te) if len(y_te) else float("nan"))
                for m, _ in METHODS}
        fitted[k] = {
            "centers": centers, "rates": rates, "specs": specs, "ece": eces,
            "n": int(len(y_tr)),
            "acc": float(y_tr.mean()) if len(y_tr) else float("nan"),
            "p_tr": p_tr, "y_tr": y_tr, "p_te": p_te, "y_te": y_te,
            "n_te": int(len(y_te))}
    return fitted


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decision-log", required=True,
                    help="ORACLE select-1 decision log (gt_token + oracle_hit)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--objective", choices=["accept_rate", "target_p"],
                    default="accept_rate")
    ap.add_argument("--mode", choices=["per_depth", "global", "depth_feature"],
                    default="per_depth")
    ap.add_argument("--target-prob-file", default=None,
                    help="target_probs.jsonl (required for --objective target_p)")
    ap.add_argument("--model-label", default="EAGLE3")
    ap.add_argument("--model-color", default="#1f77b4")
    ap.add_argument("--max-positions", type=int, default=16)
    ap.add_argument("--min-samples", type=int, default=1,
                    help="drop per-depth panels with fewer alive samples")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.objective == "target_p" and not args.target_prob_file:
        ap.error("--objective target_p requires --target-prob-file")

    alive = _alive_depths(args.decision_log)
    print(f"accept-conditioned: {len(alive)} alive (rid,step,depth) rows",
          file=sys.stderr)
    if args.objective == "target_p":
        samples = load_target_prob_samples(args.decision_log,
                                           args.target_prob_file, alive=alive)
    else:
        samples = load_oracle_samples(args.decision_log, alive=alive)
    per = to_per_depth(samples, args.max_positions)
    continuous = args.objective == "target_p"

    note = f"  [accept-conditioned · {args.objective} · {args.mode}]"
    out = Path(args.out_dir)
    hist = out / "perposition" / "histogram"
    box = out / "perposition" / "boxplot"
    sca = out / "perposition" / "scatter"
    for d in (hist, box, sca):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    drafts = [("model", args.model_label, args.model_color, args.model_color),
              ("suffix", "Suffix", SUFFIX_COLOR, SUFFIX_EDGE)]
    n = 0
    for key, label, color, edge in drafts:
        per_depth = per[key]
        depths = [k for k in sorted(per_depth)[:args.max_positions]
                  if len(per_depth[k][1]) >= args.min_samples]
        if not depths:
            print(f"  WARNING: no {key} depths with >= {args.min_samples} "
                  f"alive samples; skip", file=sys.stderr)
            continue
        fitted = build_fitted(per_depth, depths, args.mode, args.train_frac,
                              rng, continuous)
        dk = label.lower()
        print(f"== {label} [{args.mode}]: {len(depths)} depths "
              f"(k{depths[0]} n={fitted[depths[0]]['n']} acc={fitted[depths[0]]['acc']:.3f}"
              f" -> k{depths[-1]} n={fitted[depths[-1]]['n']} "
              f"acc={fitted[depths[-1]]['acc']:.3f}) ==", file=sys.stderr)
        for mkey, mlabel in METHODS:
            draw_subplots(hist, dk, label, color, edge, mkey, mlabel, depths,
                          fitted, note=note)
            draw_overlay(hist, dk, label, mkey, mlabel, depths, fitted, note=note)
            draw_test_subplots(hist, dk, label, color, edge, mkey, mlabel,
                               depths, fitted, note=note)
            draw_test_overlay(hist, dk, label, mkey, mlabel, depths, fitted,
                              note=note)
            draw_test_aggregate(hist, dk, label, color, edge, mkey, mlabel,
                                depths, fitted, note=note)
            pp_box_subplots(box / f"pp_{dk}_{mkey}.png", label, color, edge,
                            mlabel, "fit", mkey, fitted, depths, note=note)
            pp_box_subplots(box / f"pp_test_{dk}_{mkey}.png", label, color, edge,
                            mlabel, "test", mkey, fitted, depths, note=note)
            pp_scatter_fig(sca / f"pp_{dk}_{mkey}.png", label, mlabel, "fit",
                           mkey, fitted, depths, note=note)
            pp_scatter_fig(sca / f"pp_test_{dk}_{mkey}.png", label, mlabel,
                           "test", mkey, fitted, depths, note=note)
            n += 9

    print(f"wrote {n} figures under {out}/perposition/{{histogram,boxplot,scatter}}")


if __name__ == "__main__":
    main()
