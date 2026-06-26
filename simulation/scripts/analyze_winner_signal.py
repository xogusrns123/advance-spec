"""Is the WINNING SUFFIX TREE (local/global/tie) itself a useful feature?

At speculation time the server knows which tree produced the deployed draft
(local-only match, global-only match, or a tie where both agree). This asks
whether that label carries signal about per-edge acceptance, and crucially
whether it adds anything BEYOND (a) depth and (b) the draft's own quality
feature count_ratio_prob — i.e. is it a real feature or just a proxy.

Metrics on the combined suffix edge set:
  base accept per bucket            descriptive separation.
  MI(win; y) / H(y)                 encoding-free 3-way categorical signal.
  is_tie  AUROC                     tie-vs-rest, the dominant split (no leakage).
  is_tie  within-depth AUROC        survives the depth confound?
  is_tie  within-count_ratio_prob   redundant with draft quality? ~0.5 => yes.
  local-vs-global AUROC (non-tie)   does the local/global distinction matter?

Usage (inside sglang-bench container, from /workspace):
  python3 simulation/scripts/analyze_winner_signal.py \
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
from sklearn.metrics import mutual_info_score, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_feature_signal import _iter_rows  # noqa: E402
from signal_metrics import within_depth_auc  # noqa: E402

WMAP = {"global": 0, "local": 1, "tie": 2}


def load(path):
    y, depth, wcode, crp = [], [], [], []
    for r in _iter_rows(path):
        s = r.get("s")
        w = r.get("win")
        if not s or w not in WMAP:
            continue
        L = len(s["y"])
        y.extend(s["y"]); depth.extend(s["depth"])
        wcode.extend([WMAP[w]] * L)
        crp.extend(s["count_ratio_prob"])
    return (np.asarray(y, np.int8), np.asarray(depth, np.int32),
            np.asarray(wcode, np.int8), np.asarray(crp, np.float32))


def _decile_ids(x, n=10):
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)))
    if len(qs) <= 2:
        return np.zeros(len(x), int)
    return np.clip(np.searchsorted(qs[1:-1], x, side="right"), 0, len(qs) - 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    y, depth, wcode, crp = load(args.features)
    n = len(y)
    base = float(y.mean())
    is_tie = (wcode == 2).astype(np.float32)

    # per-bucket base accept + counts
    buckets = {}
    for name, code in WMAP.items():
        sel = wcode == code
        buckets[name] = {"n_edges": int(sel.sum()),
                         "base_accept": float(y[sel].mean()) if sel.any()
                         else float("nan")}

    # 3-way categorical MI (permutation-invariant on labels)
    hy = float(mutual_info_score(y, y))
    mi = float(mutual_info_score(wcode, y))
    mi_frac = mi / hy if hy > 1e-12 else float("nan")

    # is_tie: marginal / within-depth / within-count_ratio_prob
    tie_auc = float(roc_auc_score(y, is_tie))
    tie_wd, tie_wd_ns, _ = within_depth_auc(is_tie, y, depth)
    tie_wcrp, tie_wcrp_ns, _ = within_depth_auc(is_tie, y, _decile_ids(crp))

    # local vs global, restricted to non-tie edges
    nt = wcode != 2
    is_local = (wcode[nt] == 1).astype(np.float32)
    lg_auc = float(roc_auc_score(y[nt], is_local)) if y[nt].min() != y[nt].max() \
        else float("nan")
    lg_wd, lg_wd_ns, _ = within_depth_auc(is_local, y[nt], depth[nt])

    res = {
        "n_edges": n, "base_accept": base,
        "buckets": buckets,
        "win_categorical": {"mi_nats": round(mi, 5),
                            "mi_frac": round(mi_frac, 4)},
        "is_tie": {"marginal_auc": round(tie_auc, 4),
                   "within_depth_auc": round(tie_wd, 4) if tie_wd else None,
                   "within_depth_slices": tie_wd_ns,
                   "within_count_ratio_prob_auc":
                       round(tie_wcrp, 4) if tie_wcrp else None,
                   "within_count_ratio_prob_bins": tie_wcrp_ns},
        "local_vs_global_nontie": {
            "marginal_auc": round(lg_auc, 4) if lg_auc == lg_auc else None,
            "within_depth_auc": round(lg_wd, 4) if lg_wd else None,
            "within_depth_slices": lg_wd_ns,
            "n_edges": int(nt.sum())},
    }
    with open(out / "winner_feature_signal.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure: accept rate by winning tree (the feature itself) ----
    order = ["global", "local", "tie"]
    rates = [buckets[b]["base_accept"] for b in order]
    cnts = [buckets[b]["n_edges"] for b in order]
    colors = ["#d62728", "#1f77b4", "#7f7f7f"]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    bars = ax.bar(order, rates, color=colors, edgecolor="black", lw=0.5)
    ax.axhline(base, color="gray", ls="--", lw=1.0,
               label=f"base accept ({base:.2f})")
    for b, c in zip(bars, cnts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"n={c:,}", ha="center", va="bottom", fontsize=8, color="0.3")
    ax.set_ylim(0, max(rates) * 1.18)
    ax.set_ylabel("P(accept)")
    ax.set_xlabel("winning suffix tree")
    ax.set_title(f"win_tree — Suffix  (MI/H={mi_frac:.3f}, "
                 f"is_tie AUC {tie_auc:.2f})", fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "stratified_accept_suffix_win_tree.png", dpi=150)
    plt.close(fig)

    # ---- stderr table ----
    print(f"\n=== winning-tree as a feature (edges={n:,}, base={base:.3f}) ===",
          file=sys.stderr)
    for b in order:
        print(f"  {b:7s} n={buckets[b]['n_edges']:>9,}  "
              f"accept={buckets[b]['base_accept']:.3f}", file=sys.stderr)
    print(f"  MI(win;y)/H(y)            {mi_frac:.4f}", file=sys.stderr)
    print(f"  is_tie marginal AUC      {tie_auc:.4f}", file=sys.stderr)
    print(f"  is_tie within-depth AUC  "
          f"{tie_wd:.4f} ({tie_wd_ns} slices)", file=sys.stderr)
    print(f"  is_tie within-crp AUC    "
          f"{tie_wcrp:.4f} ({tie_wcrp_ns} bins)  <- redundancy w/ count_ratio_prob",
          file=sys.stderr)
    print(f"  local-vs-global AUC      "
          f"{lg_auc:.4f} (non-tie n={int(nt.sum()):,})", file=sys.stderr)
    print(f"  local-vs-global wd AUC   "
          f"{lg_wd:.4f} ({lg_wd_ns} slices)" if lg_wd else "  (degenerate)",
          file=sys.stderr)
    print(f"\nwrote winner_feature_signal.json + "
          f"stratified_accept_suffix_win_tree.png under {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
