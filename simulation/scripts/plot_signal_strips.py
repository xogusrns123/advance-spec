"""Signal-STRIP heatmap: do several acceptance signals fire at the same time?

Lays ONE trajectory out along x (each column = a decode position where a suffix
draft was made) and stacks each signal as a horizontal color strip. Color = the
signal's strength at that position, measured as its percentile within the WHOLE
dataset (so red really means "high vs everywhere", not just within this
trajectory) and oriented toward acceptance (red = the signal currently argues
for accept, green = against / weak). The top strip is the realized OUTCOME
(number of suffix tokens actually accepted), so you can eyeball whether red
signal regions line up with actually-accepted regions, and whether the signals
light up together.

Per-position value of each signal:
  per-position scalars  score, match_len, n_nodes, gt_remaining, abs_pos,
                        out_entropy_cum, out_entropy_win64   -> used directly.
  per-edge suffix feats count_ratio_prob, count, subtree_size,
                        branch_factor   -> the ROOT edge (depth 1, the first
                        token to be verified). count = raw suffix-tree node
                        count (recovered from log_count for older files).
  outcome               accept_len = #accepted suffix edges (= sum y).

Usage (inside sglang-bench container, from /workspace):
  python3 simulation/scripts/plot_signal_strips.py \
      --features simulation/results/calib_verify/features_27b_localglobal.jsonl.gz \
      --out-dir  simulation/results/calib_verify/qwen35_27b_mtp/signal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_feature_signal import _iter_rows  # noqa: E402

# per-position scalars taken straight from pp
PP_FEATS = ["score", "match_len", "n_nodes", "gt_remaining", "abs_pos",
            "out_entropy_cum", "out_entropy_win64"]
# per-edge suffix feats -> read the ROOT edge (index 0, depth 1). We show the
# RAW suffix-tree node count (not log_count); older feature files only stored
# log_count, so we recover count = expm1(log_count) as a fallback.
ROOT_FEATS = ["count_ratio_prob", "count", "subtree_size", "branch_factor"]
ALL_FEATS = ROOT_FEATS + PP_FEATS


def collect(path):
    """-> list of per-position dicts (only positions with a suffix draft)."""
    rows = []
    for r in _iter_rows(path):
        s = r.get("s")
        if not s or not s.get("y"):
            continue
        pp = r.get("pp", {})
        rec = {"rid": r["rid"], "ci": r.get("ci", 0), "pos": r.get("pos", 0),
               "accept_len": float(sum(s["y"])), "win": r.get("win")}
        for k in ("count_ratio_prob", "subtree_size", "branch_factor"):
            rec[k] = float(s[k][0]) if s.get(k) else np.nan
        # raw count: prefer the stored raw count, else recover from log_count
        if s.get("count"):
            rec["count"] = float(s["count"][0])
        elif s.get("log_count"):
            rec["count"] = float(np.expm1(s["log_count"][0]))
        else:
            rec["count"] = np.nan
        for k in PP_FEATS:
            v = pp.get(k)
            rec[k] = float(v) if v is not None else np.nan
        rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rid", default=None,
                    help="request id to plot (default: a representative long "
                         "trajectory near --length-pct)")
    ap.add_argument("--select", choices=["pct", "longest"], default="pct",
                    help="pct: trajectory nearest --length-pct length "
                         "(avoids pathological outliers); longest: the maximum")
    ap.add_argument("--length-pct", type=float, default=90.0,
                    help="percentile of trajectory length to target (--select pct)")
    ap.add_argument("--max-positions", type=int, default=1500,
                    help="cap columns (subsample by stride if exceeded)")
    ap.add_argument("--name", default=None,
                    help="output basename (default depends on --simplified)")
    ap.add_argument("--simplified", action="store_true",
                    help="only a fixed 6-row set: accept_len, match_len, "
                         "out_entropy_win64, count_ratio_prob, count, "
                         "tree_source (which suffix tree won). tree_source "
                         "needs --tag-winner features; dropped with a warning "
                         "if absent.")
    ap.add_argument("--color", choices=["value", "percentile"], default="value",
                    help="value: color = the raw feature value on a per-row "
                         "linear scale (low=green, high=red). percentile: rank "
                         "vs whole dataset, accept-oriented (old behavior).")
    args = ap.parse_args()
    if args.name is None:
        args.name = ("signal_strips_simplified" if args.simplified
                     else "signal_strips_traj")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = collect(args.features)
    if not rows:
        print("no suffix positions found", file=sys.stderr); return

    # global percentile reference + accept-direction per feature
    allvals = {f: np.array([r[f] for r in rows], float) for f in ALL_FEATS}
    acc_all = np.array([r["accept_len"] for r in rows], float)
    sorted_ref, direction = {}, {}
    for f in ALL_FEATS:
        v = allvals[f]
        fin = np.isfinite(v)
        sorted_ref[f] = np.sort(v[fin]) if fin.any() else np.array([0.0])
        if fin.sum() > 2 and np.std(v[fin]) > 0:
            c = np.corrcoef(v[fin], acc_all[fin])[0, 1]
            direction[f] = 1.0 if c >= 0 else -1.0
        else:
            direction[f] = 1.0
    acc_sorted = np.sort(acc_all)

    # linear value ranges (robust p1-p99) for --color value
    def _range(v):
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return (0.0, 1.0)
        lo, hi = (float(x) for x in np.percentile(v, [1, 99]))
        if hi <= lo:
            lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi
    vrange = {f: _range(allvals[f]) for f in ALL_FEATS}
    acc_lo, acc_hi = _range(acc_all)

    # tree_source available only when features carry the --tag-winner label
    has_win = any(r.get("win") in ("local", "global", "tie") for r in rows)

    # choose trajectory
    from collections import defaultdict
    by_traj = defaultdict(list)
    for r in rows:
        by_traj[(r["rid"], r["ci"])].append(r)
    if args.rid is not None:
        keys = [k for k in by_traj if str(k[0]) == args.rid]
        if not keys:
            print(f"rid {args.rid} not found; have e.g. "
                  f"{list(by_traj)[:3]}", file=sys.stderr); return
        key = max(keys, key=lambda k: len(by_traj[k]))
    elif args.select == "longest":
        key = max(by_traj, key=lambda k: len(by_traj[k]))
    else:  # representative: nearest to the requested length percentile
        lens = np.array([len(v) for v in by_traj.values()])
        target = np.percentile(lens, args.length_pct)
        key = min(by_traj, key=lambda k: abs(len(by_traj[k]) - target))
    traj = sorted(by_traj[key], key=lambda r: r["pos"])
    full = len(traj)
    stride = 1
    if full > args.max_positions:
        # subsample evenly across the WHOLE trajectory (keep the full arc, full
        # per-column resolution at the sampled points) rather than truncating.
        stride = int(np.ceil(full / args.max_positions))
        traj = traj[::stride]
    P = len(traj)
    print(f"trajectory rid={key[0]} ci={key[1]}: {P} of {full} positions "
          f"(stride {stride})", file=sys.stderr)

    def cnorm(f, vals):
        """color value for feature f: raw value on a linear scale, or rank."""
        if args.color == "value":
            lo, hi = vrange[f]
            return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
        ref = sorted_ref[f]; n = len(ref)
        p = np.searchsorted(ref, vals, side="right") / max(n, 1)
        return (1.0 - p) if direction[f] < 0 else p

    def cnorm_acc(vals):
        """color for accept_len / tree_source, on the accept_len scale."""
        if args.color == "value":
            return np.clip((vals - acc_lo) / (acc_hi - acc_lo), 0.0, 1.0)
        return np.searchsorted(acc_sorted, vals, side="right") / max(
            len(acc_sorted), 1)

    # build strength matrix: row 0 = outcome, then the feature rows
    tree_source_row = False
    if args.simplified:
        # fixed, user-requested row set (count derived from log_count)
        feat_order = ["match_len", "out_entropy_win64", "count_ratio_prob",
                      "count"]
        tree_source_row = has_win
        if not has_win:
            print("WARNING: features have no 'win' tag (need --tag-winner); "
                  "dropping tree_source row", file=sys.stderr)
    else:
        feat_order = sorted(ALL_FEATS, key=lambda f: abs(np.corrcoef(
            allvals[f][np.isfinite(allvals[f])],
            acc_all[np.isfinite(allvals[f])])[0, 1])
            if np.isfinite(allvals[f]).sum() > 2 else 0.0, reverse=True)

    # continuous rows (value/percentile colormap)
    cont_labels = ["accept_len (OUTCOME)"] + feat_order
    M = np.full((len(cont_labels), P), np.nan)
    acc_traj = np.array([r["accept_len"] for r in traj], float)
    M[0] = cnorm_acc(acc_traj)
    for i, f in enumerate(feat_order, start=1):
        vals = np.array([r[f] for r in traj], float)
        s = cnorm(f, vals)
        s[~np.isfinite(vals)] = np.nan
        M[i] = s

    # tree_source = a CATEGORICAL strip (distinct colors + legend); its numeric
    # value is meaningless, so it gets its own discrete colormap, not the value
    # scale.
    TS_CATS = ["local", "global", "tie"]
    TS_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd"]   # blue / orange / purple
    ts_code = None
    if tree_source_row:
        code = {"local": 0, "global": 1, "tie": 2}
        ts_code = np.array([code.get(r.get("win"), np.nan) for r in traj], float)

    Mm = np.ma.masked_invalid(M)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="0.85")
    n_cont = len(cont_labels)
    width = max(12.0, min(48.0, P * 0.03 + 6))
    xlab = (f"trajectory position  (ordered decode steps; {P} of {full}"
            + (f", every {stride}th" if stride > 1 else "") + ")")

    if tree_source_row:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        fig, (ax, axts) = plt.subplots(
            2, 1, sharex=True, figsize=(width, 0.46 * n_cont + 2.4),
            gridspec_kw={"height_ratios": [n_cont, 1.0], "hspace": 0.12})
    else:
        fig, ax = plt.subplots(figsize=(width, 0.46 * n_cont + 1.6))
        axts = None

    im = ax.imshow(Mm, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_yticks(range(n_cont))
    ax.set_yticklabels(cont_labels, fontsize=9)
    ax.axhline(0.5, color="black", lw=2.0)  # divider under the outcome strip

    if args.color == "value":
        sub = "color = raw value per row (linear, low=green → high=red, p1–p99)"
        cblab = "value (per-row linear, p1–p99)"
    else:
        sub = ("red = strong / argues-accept, green = weak; percentile vs "
               "whole dataset, accept-oriented")
        cblab = "signal strength (percentile, →accept)"
    ax.set_title(f"Signal strips along one trajectory  (rid={key[0]}, "
                 f"ci={key[1]}){'  [simplified]' if args.simplified else ''}\n"
                 f"{sub}", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
    cb.set_label(cblab, fontsize=9)

    if tree_source_row:
        ts_cmap = ListedColormap(TS_COLORS)
        ts_cmap.set_bad(color="0.85")
        axts.imshow(np.ma.masked_invalid(ts_code)[None, :], aspect="auto",
                    cmap=ts_cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
        axts.set_yticks([0]); axts.set_yticklabels(["tree_source"], fontsize=9)
        axts.legend(handles=[Patch(facecolor=c, edgecolor="black", lw=0.4,
                                   label=n) for n, c in zip(TS_CATS, TS_COLORS)],
                    ncol=3, fontsize=8, loc="upper center",
                    bbox_to_anchor=(0.5, -0.6), frameon=False)
        axts.set_xlabel(xlab, fontsize=10)
    else:
        ax.set_xlabel(xlab, fontsize=10)

    fig.tight_layout()
    p = out / f"{args.name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
