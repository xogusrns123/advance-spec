"""Per-feature acceptance-SIGNAL analysis for the SUFFIX draft, bucketed by
which suffix tree won the speculation (local / global / tie).

Suffix decoding keeps two trees: a per-request LOCAL tree (this request's prompt
+ its own generated tokens) and a cross-request GLOBAL tree (all prior cached
responses + this request's own tokens). At every position arctic_inference
speculates on both and deploys the higher-scoring draft. extract_signal_features
--tag-winner records, per position, which tree won (row['win'] in
{local,global,tie}); this script splits the deployed suffix edges by that label
and runs the SAME signal analysis as analyze_feature_signal.py (AUROC / MI /
within-depth) inside each bucket, so you can see how each feature's acceptance
signal changes depending on the winning tree.

Output (under --out-dir):
  local/   global/   tie/   all/      one folder per bucket, each with the
      stratified_accept_suffix_<feature>.png panels + auc_bars_suffix.png +
      signal_summary.json  (identical filenames to the combined run, so they
      diff cleanly against simulation/results/.../signal/).
  auc_compare_suffix.png              grouped marginal-AUC bars, one group per
      feature, one bar per bucket — the at-a-glance "how does the signal differ
      by tree" figure.
  bytree_summary.json                 win distribution + per-bucket rankings.

Usage (inside sglang-bench container, from /workspace):
  python3 simulation/scripts/analyze_feature_signal_bytree.py \
      --features simulation/results/calib_verify/features_27b_localglobal.jsonl.gz \
      --out-dir  simulation/results/calib_verify/qwen35_27b_mtp/signal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_feature_signal import (  # noqa: E402
    PP_SHARED, PP_SUFFIX, S_EDGE, SUFFIX_COLOR, SUFFIX_EDGE, SUFFIX_FEATS,
    _iter_rows, analyze_draft,
)

RED = "crimson"
BUCKETS = ("local", "global", "tie", "all")
# distinct color per bucket for the comparison figure
BUCKET_COLOR = {"local": "#1f77b4", "global": "#d62728",
                "tie": "#7f7f7f", "all": SUFFIX_COLOR}


def _alloc(n):
    d = {"y": np.empty(n, np.int8), "depth": np.empty(n, np.int16)}
    for k in S_EDGE + PP_SUFFIX + PP_SHARED:
        d[k] = np.empty(n, np.float32)
    return d


def load_suffix_by_bucket(path):
    """Build per-bucket suffix arrays. 'all' = union of local+global+tie.

    Rows without a 'win' tag (i.e. produced without --tag-winner) are counted
    under 'all' only and reported, so a mis-specified feature file fails loud.
    """
    counts = {b: 0 for b in BUCKETS}
    n_untagged = 0
    for r in _iter_rows(path):
        s = r.get("s")
        if not s:
            continue
        L = len(s["y"])
        counts["all"] += L
        w = r.get("win")
        if w in ("local", "global", "tie"):
            counts[w] += L
        else:
            n_untagged += L
    if n_untagged:
        print(f"WARNING: {n_untagged} suffix edges have no win tag "
              f"(features file lacks --tag-winner?)", file=sys.stderr)

    arr = {b: _alloc(counts[b]) for b in BUCKETS}
    idx = {b: 0 for b in BUCKETS}

    def g(pp, k):
        v = pp.get(k)
        return np.nan if v is None else float(v)

    def fill(d, i, s, pp):
        L = len(s["y"]); sl = slice(i, i + L)
        d["y"][sl] = s["y"]; d["depth"][sl] = s["depth"]
        for k in S_EDGE:
            d[k][sl] = s[k]
        for k in PP_SUFFIX + PP_SHARED:
            d[k][sl] = g(pp, k)
        return L

    for r in _iter_rows(path):
        s = r.get("s")
        if not s:
            continue
        pp = r.get("pp", {})
        idx["all"] += fill(arr["all"], idx["all"], s, pp)
        w = r.get("win")
        if w in ("local", "global", "tie"):
            idx[w] += fill(arr[w], idx[w], s, pp)
    return arr, counts


def draw_auc_compare(out, per_bucket_recs):
    """Grouped marginal-AUC bars: one feature group, one bar per bucket."""
    names = [n for n, _, _ in SUFFIX_FEATS]
    buckets = [b for b in BUCKETS if b in per_bucket_recs]
    x = np.arange(len(names))
    w = 0.8 / max(len(buckets), 1)
    fig, ax = plt.subplots(figsize=(max(8.0, len(names) * 1.1), 5.2))
    for j, b in enumerate(buckets):
        recs = per_bucket_recs[b]
        vals = [(recs.get(n, {}).get("marginal_auc") or np.nan) for n in names]
        ax.bar(x + (j - (len(buckets) - 1) / 2) * w, vals, w,
               color=BUCKET_COLOR.get(b, None), edgecolor="black", lw=0.4,
               label=b)
    ax.axhline(0.5, color=RED, ls="--", lw=1.0, label="no signal (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0.45, 1.0); ax.set_ylabel("marginal AUROC")
    ax.set_title("Suffix feature acceptance-signal by winning tree\n"
                 "(marginal AUC; bar = bucket)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = out / "auc_compare_suffix.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mi-bins", type=int, default=20)
    ap.add_argument("--min-per-slice", type=int, default=200)
    ap.add_argument("--n-bins", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"loading {args.features} ...", file=sys.stderr)
    arr, counts = load_suffix_by_bucket(args.features)
    print(f"suffix edges by bucket: "
          + ", ".join(f"{b}={counts[b]}" for b in BUCKETS), file=sys.stderr)

    summary = {"features_path": args.features, "mi_bins": args.mi_bins,
               "min_per_slice": args.min_per_slice,
               "suffix_edges_by_bucket": counts, "buckets": {}}
    per_bucket_recs = {}
    for b in BUCKETS:
        if counts[b] == 0:
            print(f"  skip {b}: no edges", file=sys.stderr)
            continue
        out_b = out / b
        out_b.mkdir(parents=True, exist_ok=True)
        res = analyze_draft(arr[b], SUFFIX_FEATS, f"Suffix [{b}]", "suffix",
                            SUFFIX_COLOR, SUFFIX_EDGE, out_b, rng,
                            args.mi_bins, args.min_per_slice, args.n_bins)
        if res:
            per_bucket_recs[b] = res["features"]
            summary["buckets"][b] = {
                "n_edges": res["n_edges"], "base_accept": res["base_accept"],
                "ranking_marginal": res["ranking_marginal"],
                "ranking_within_depth": res["ranking_within_depth"],
            }
            with open(out_b / "signal_summary.json", "w") as f:
                json.dump({"bucket": b, **res}, f, indent=2, default=float)

    if per_bucket_recs:
        draw_auc_compare(out, per_bucket_recs)
    with open(out / "bytree_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote per-bucket folders {list(per_bucket_recs)} + "
          f"auc_compare_suffix.png + bytree_summary.json under {out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
