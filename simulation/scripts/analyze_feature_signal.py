"""Per-feature acceptance-SIGNAL analysis + figures (offline).

Reads the feature rows from extract_signal_features.py and, for each draft
(model = EAGLE3/MTP, suffix), quantifies whether each candidate feature carries
signal about token acceptance (AUROC / MI / point-biserial / within-depth AUC;
see signal_metrics.py). This is detection, NOT calibration — no fit/test split.

Per draft it writes:
  stratified_accept_<draft>_<feature>.png  one image PER feature: empirical
                                 accept rate vs feature bin (bars) + dashed
                                 base-accept line. Flat at the base line = no
                                 signal.
  auc_bars_<draft>.png           grouped bars: marginal AUC vs within-depth mean
                                 AUC, sorted; 0.5 reference line. Features whose
                                 marginal is high but within-depth ~0.5 are just
                                 depth proxies.
and a combined signal_summary.json (both drafts) + a stderr table.

A `random` negative-control feature (AUC ~0.5) is injected per draft.

Usage (inside sglang-bench container, from /workspace):
  python3 simulation/scripts/analyze_feature_signal.py \
      --features simulation/results/calib_verify/features_14b.jsonl.gz \
      --out-dir  simulation/results/calib_verify/qwen3_14b/signal \
      --model-label EAGLE3 --model-color "#1f77b4"
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_metrics import feature_signal  # noqa: E402

RED = "crimson"
SUFFIX_COLOR, SUFFIX_EDGE = "#ff7f0e", "#e8810b"

PP_SHARED = ["abs_pos", "gt_remaining", "out_entropy_cum", "out_entropy_win64"]
PP_SUFFIX = ["match_len", "score", "n_nodes"]
M_EDGE = ["edge_prob", "path_prob"]
S_EDGE = ["count_ratio_prob", "log_count", "branch_factor", "subtree_size"]
# probability features -> stratified panels use fixed-width bins on [0,1]
PROB_FEATS = {"edge_prob", "path_prob", "count_ratio_prob"}

# (name, kind, is_depth) in display order
MODEL_FEATS = (
    [("depth", "per-edge", True)]
    + [(f, "per-edge", False) for f in M_EDGE]
    + [(f, "per-position", False) for f in PP_SHARED]
    + [("random", "control", False)])
SUFFIX_FEATS = (
    [("depth", "per-edge", True)]
    + [(f, "per-edge", False) for f in S_EDGE]
    + [(f, "per-position", False) for f in PP_SUFFIX]
    + [(f, "per-position", False) for f in PP_SHARED]
    + [("random", "control", False)])


def _iter_rows(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_features(path):
    """Build per-draft numpy arrays; per-position scalars broadcast to edges."""
    nm = ns = 0
    for r in _iter_rows(path):
        if "m" in r:
            nm += len(r["m"]["y"])
        if "s" in r:
            ns += len(r["s"]["y"])

    def alloc(n, edge_keys, pp_keys):
        d = {"y": np.empty(n, np.int8), "depth": np.empty(n, np.int16)}
        for k in edge_keys + pp_keys:
            d[k] = np.empty(n, np.float32)
        return d

    M = alloc(nm, M_EDGE, PP_SHARED)
    S = alloc(ns, S_EDGE, PP_SUFFIX + PP_SHARED)

    def g(pp, k):
        v = pp.get(k)
        return np.nan if v is None else float(v)

    mi = si = 0
    for r in _iter_rows(path):
        pp = r.get("pp", {})
        m = r.get("m")
        if m:
            L = len(m["y"]); sl = slice(mi, mi + L)
            M["y"][sl] = m["y"]; M["depth"][sl] = m["depth"]
            for k in M_EDGE:
                M[k][sl] = m[k]
            for k in PP_SHARED:
                M[k][sl] = g(pp, k)
            mi += L
        s = r.get("s")
        if s:
            L = len(s["y"]); sl = slice(si, si + L)
            S["y"][sl] = s["y"]; S["depth"][sl] = s["depth"]
            for k in S_EDGE:
                S[k][sl] = s[k]
            for k in PP_SUFFIX + PP_SHARED:
                S[k][sl] = g(pp, k)
            si += L
    return M, S


def _fmt(c):
    if not np.isfinite(c):
        return ""
    return str(int(round(c))) if abs(c - round(c)) < 1e-6 else f"{c:.2g}"


def accept_rate_by_bin(x, y, feat_name, n_bins=25, int_max_unique=32):
    """Accept rate per feature bin, on a REAL numeric axis.

    prob features      -> fixed-width bins on [0,1] (fine, undistorted);
    small-card integer -> one bar per value;
    other continuous   -> fixed-width bins over the robust [p1,p99] range
                          (the <2% out-of-range tail is excluded from the PANEL
                          only; AUROC/MI always use the full data).
    Returns (centers, rates, counts, (xlo, xhi), bar_width).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x); x, y = x[m], y[m]
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), (0.0, 1.0), 0.04
    uniq = np.unique(x)
    is_int = bool(np.all(np.abs(x - np.round(x)) < 1e-9))
    if is_int and len(uniq) <= int_max_unique:
        rates = np.array([y[x == v].mean() for v in uniq])
        counts = np.array([int((x == v).sum()) for v in uniq])
        w = 0.8 * (float(np.min(np.diff(uniq))) if len(uniq) > 1 else 1.0)
        return uniq, rates, counts, (uniq.min() - 1, uniq.max() + 1), w
    if feat_name in PROB_FEATS:
        lo, hi = 0.0, 1.0
    else:
        pr = np.percentile(x, [1, 99])
        lo, hi = float(pr[0]), float(pr[1])
        if hi <= lo:
            lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    sel = (x >= lo) & (x <= hi)
    xs, ys = x[sel], y[sel]
    ids = np.clip(np.searchsorted(edges[1:-1], xs, side="right"), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rates = np.array([ys[ids == b].mean() if (ids == b).any() else np.nan
                      for b in range(n_bins)])
    counts = np.array([int((ids == b).sum()) for b in range(n_bins)])
    return centers, rates, counts, (lo, hi), (hi - lo) / n_bins


def draw_stratified(out, draft_key, label, color, edge, arrays, recs, n_bins):
    """One image PER feature: accept rate vs feature bin (bars) + dashed base
    line. No per-bin count line (the grey twinx was dropped on request)."""
    base = float(np.mean(arrays["y"]))
    written = []
    for r in recs:
        name = r["name"]
        centers, rates, counts, xlim, w = accept_rate_by_bin(
            arrays[name], arrays["y"], name, n_bins)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.bar(centers, rates, width=w * 0.9, color=color, edgecolor=edge,
               lw=0.3, align="center", label="empirical accept (bin)")
        ax.axhline(base, color="gray", ls="--", lw=1.0,
                   label=f"base accept ({base:.2f})")
        if len(centers):
            ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.02)
        auc = r["marginal_auc"]
        title = f"{name} — {label}"
        if auc is not None:
            title += f"  (AUC {auc:.2f})"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("feature value")
        ax.set_ylabel("P(accept)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        p = out / f"stratified_accept_{draft_key}_{name}.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        written.append(p)
    return written


def draw_auc_bars(out, draft_key, label, color, recs):
    names = [r["name"] for r in recs]
    marg = [r["marginal_auc"] if r["marginal_auc"] is not None else 0.5
            for r in recs]
    wd = [r["within_depth_auc"] if r["within_depth_auc"] is not None else np.nan
          for r in recs]
    nsl = [r["n_slices_used"] for r in recs]
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(7.0, len(names) * 0.9), 5.0))
    ax.bar(x - w / 2, marg, w, color=color, edgecolor="black", lw=0.4,
           label="marginal AUC")
    ax.bar(x + w / 2, [v if not np.isnan(v) else 0 for v in wd], w,
           color=color, alpha=0.4, edgecolor="black", lw=0.4,
           label="within-depth AUC")
    ax.axhline(0.5, color=RED, ls="--", lw=1.0, label="no signal (0.5)")
    for xi, v, ns in zip(x, wd, nsl):
        if not np.isnan(v):
            ax.text(xi + w / 2, v + 0.005, str(ns), ha="center", va="bottom",
                    fontsize=6, color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0.45, 1.0); ax.set_ylabel("AUROC")
    ax.set_title(f"Feature acceptance-signal — {label}\n"
                 f"(marginal vs within-depth; number = #depth slices)",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = out / f"auc_bars_{draft_key}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def analyze_draft(arrays, feats, label, draft_key, color, edge, out, rng,
                  mi_bins, min_per_slice, n_bins):
    if len(arrays["y"]) == 0:
        print(f"  WARNING: no {label} edges; skip", file=sys.stderr)
        return None
    arrays["random"] = rng.random(len(arrays["y"])).astype(np.float32)
    recs = {}
    for name, kind, is_depth in feats:
        recs[name] = feature_signal(
            arrays[name], arrays["y"], arrays["depth"], name, kind=kind,
            is_depth=is_depth, mi_bins=mi_bins, min_per_slice=min_per_slice)
    ordered = sorted(recs.values(),
                     key=lambda r: (r["marginal_auc"] or 0.0), reverse=True)
    strat = draw_stratified(out, draft_key, label, color, edge, arrays, ordered,
                            n_bins)
    draw_auc_bars(out, draft_key, label, color, ordered)
    print(f"  wrote {len(strat)} stratified panels + 1 auc_bars for {label}",
          file=sys.stderr)

    base = float(np.mean(arrays["y"]))
    print(f"\n=== {label}  (edges={len(arrays['y'])}, base accept={base:.3f}) ===",
          file=sys.stderr)
    print(f"  {'feature':18s} {'AUC':>6s} {'wdAUC':>6s} {'MIfrac':>7s} "
          f"{'pbr':>7s} {'dir':>4s} {'slices':>6s}", file=sys.stderr)
    for r in ordered:
        wd = r["within_depth_auc"]
        print(f"  {r['name']:18s} {(r['marginal_auc'] or float('nan')):6.3f} "
              f"{(wd if wd is not None else float('nan')):6.3f} "
              f"{(r['mi_frac'] or float('nan')):7.4f} "
              f"{(r['pbr'] if r['pbr'] is not None else float('nan')):7.3f} "
              f"{str(r['direction']):>4s} {r['n_slices_used']:6d}",
              file=sys.stderr)
    return {
        "n_edges": int(len(arrays["y"])), "base_accept": base,
        "features": recs,
        "ranking_marginal": [r["name"] for r in ordered],
        "ranking_within_depth": [
            r["name"] for r in sorted(
                recs.values(),
                key=lambda r: (r["within_depth_auc"] or 0.0), reverse=True)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    ap.add_argument("--model-color", default="#1f77b4")
    ap.add_argument("--mi-bins", type=int, default=20)
    ap.add_argument("--min-per-slice", type=int, default=200)
    ap.add_argument("--n-bins", type=int, default=25,
                    help="fixed-width bins for the stratified accept-rate panels")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"loading {args.features} ...", file=sys.stderr)
    M, S = load_features(args.features)
    print(f"model edges={len(M['y'])}  suffix edges={len(S['y'])}",
          file=sys.stderr)

    summary = {"features_path": args.features, "mi_bins": args.mi_bins,
               "min_per_slice": args.min_per_slice, "drafts": {}}
    model_key = args.model_label.lower()
    res_m = analyze_draft(M, MODEL_FEATS, args.model_label, model_key,
                          args.model_color, args.model_color, out, rng,
                          args.mi_bins, args.min_per_slice, args.n_bins)
    res_s = analyze_draft(S, SUFFIX_FEATS, "Suffix", "suffix",
                          SUFFIX_COLOR, SUFFIX_EDGE, out, rng,
                          args.mi_bins, args.min_per_slice, args.n_bins)
    if res_m:
        summary["drafts"][args.model_label] = res_m
    if res_s:
        summary["drafts"]["Suffix"] = res_s

    with open(out / "signal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}/signal_summary.json + per-feature stratified figures "
          f"+ auc_bars", file=sys.stderr)


if __name__ == "__main__":
    main()
