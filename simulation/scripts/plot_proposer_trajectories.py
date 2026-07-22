"""Per-proposer accept-length trajectory for a SINGLE request, ALIGNED on a shared
output-token-position x-axis so you can see whether different proposers get their
long accepts at the SAME points in the trajectory.

Reads a chain-hybrid select-1 ORACLE decisions log (decisions_select1_oracle.jsonl).
That log records, per (rid, decode_step, depth), each arm's own drafted token
(<arm>_token) vs gt_token, plus a per-step accept_len (the COMBINED oracle advance).

For each decode_step we reconstruct every arm's INDEPENDENT accept length =
#consecutive depths from 0 where <arm>_token == gt_token (faithful: while an arm
keeps matching gt the committed prefix IS gt, so its chain is gt-conditioned). All
arms are scored at the SAME committed positions (the oracle trajectory), so the
x-axis (cumulative output token position = sum of accept_len+1) is identical across
arms -> a spike at position P in two subplots means both would fire at the same tok.

NOTE this differs from a per-proposer solo rollout (each advancing by its OWN
accept): here the step grid is the oracle's, which is the only way to align arms.
The accepts are still each arm's own; the shared grid just fixes the x-positions.

Run inside sglang-bench (matplotlib):
  docker exec sglang-bench python3 /workspace/simulation/scripts/plot_proposer_trajectories.py \
    --decisions .../qwen35_27b_3way_real_full/decisions_select1_oracle.jsonl \
    --arms eagle=MTP dflash=DFlash suffix=suffix \
    --label "Qwen3.5-27B  bfcl_v4 web_search" --out .../qwen35_27b_proposers.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# unified proposer color convention (shared with plot_suffix_trajectory.py)
PROPOSER_COLORS = {
    "suffix": "#e8820e",   # orange
    "eagle3": "#2b6cb0",   # blue
    "eagle": "#2b6cb0",
    "mtp": "#7c3aed",      # purple
    "dflash": "#e11d48",   # red
}
REF_GRAY = "#4a5568"       # reference lines (mean), not a proposer


def color_for(label):
    return PROPOSER_COLORS.get(label.strip().lower(), REF_GRAY)


def scan_lengths(path):
    """Pass 1: rid appearance order + per-rid {decode_step: accept_len}."""
    order, steps = [], defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        t = r.get("type")
        if t == "req":
            order.append(r["rid"])
        elif t == "step":
            steps[r["rid"]][r["decode_step"]] = r["accept_len"]
    return order, steps


def load_rid_decisions(path, rid):
    """Pass 2: {decode_step: {depth: row}} for one rid."""
    dec = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r.get("type") == "decision" and r.get("rid") == rid:
            dec[r["decode_step"]].setdefault(r["depth"], r)
    return dec


def arm_accept(depths, arm):
    """#consecutive depths from 0 where <arm>_token == gt_token."""
    a, d = 0, 0
    while d in depths:
        row = depths[d]
        tok, gt = row.get(arm + "_token"), row.get("gt_token")
        if gt is None or tok is None or tok != gt:
            break
        a += 1
        d += 1
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="arm_key=Label pairs, e.g. eagle=MTP dflash=DFlash suffix=suffix")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rid", default="auto")
    ap.add_argument("--target-len", type=int, default=600,
                    help="auto rid pick: normal request nearest this output length")
    ap.add_argument("--min-len", type=int, default=350)
    ap.add_argument("--max-len", type=int, default=1200)
    args = ap.parse_args()

    arms = [(p.split("=", 1)[0], p.split("=", 1)[1]) for p in args.arms]

    order, steps = scan_lengths(args.decisions)
    tot = {rid: sum(v + 1 for v in steps[rid].values()) for rid in steps}
    if args.rid == "auto":
        cands = [rid for rid in order
                 if args.min_len <= tot.get(rid, 0) <= args.max_len]
        if not cands:
            raise SystemExit("no request in [min-len,max-len]")
        rid = min(cands, key=lambda r: abs(tot[r] - args.target_len))
    else:
        rid = args.rid
    dec = load_rid_decisions(args.decisions, rid)

    # reconstruct per-arm accept + shared position axis
    ds_sorted = sorted(dec)
    pos, cum = [], 0
    acc = {k: [] for k, _ in arms}
    for ds in ds_sorted:
        depths = dec[ds]
        pos.append(cum)
        for k, _ in arms:
            acc[k].append(arm_accept(depths, k))
        cum += steps[rid].get(ds, 0) + 1
    pos = np.array(pos, float)
    acc = {k: np.array(v, float) for k, v in acc.items()}
    out_len = cum

    # console summary + pairwise co-occurrence of accepts across shared steps
    print(f"[{args.label}] rid={rid[:8]} out_len={out_len} steps={len(ds_sorted)}")
    keys = [k for k, _ in arms]
    for k, lab in arms:
        v = acc[k]
        print(f"  {lab:<8} mean accept={v.mean():.2f}  "
              f"zero%={(v == 0).mean()*100:.0f}  max={int(v.max())}")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = acc[keys[i]], acc[keys[j]]
            r = np.corrcoef(a, b)[0, 1] if a.std() and b.std() else float("nan")
            both = ((a >= 2) & (b >= 2)).sum()
            either = ((a >= 2) | (b >= 2)).sum()
            print(f"  corr({keys[i]},{keys[j]})={r:+.2f}  "
                  f"both>=2 / either>=2 = {both}/{either}")

    # figure: one aligned subplot per proposer (shared x)
    ymax = max(float(acc[k].max()) for k, _ in arms) * 1.20 + 1.5
    n = len(arms)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.35 * n + 0.6), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (k, lab) in zip(axes, arms):
        c = color_for(lab)
        v = acc[k]
        ax.vlines(pos, 0, v, color=c, linewidth=1.0, alpha=0.9)
        m = v.mean()
        ax.axhline(m, color=REF_GRAY, lw=1.2, ls="--")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("accept length")
        ax.text(0.008, 0.94, f"{lab}   (mean {m:.2f}, max {int(v.max())})",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="top", color=c,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
        # label the large spikes with their accept-length value
        thr = max(3.0, 0.5 * v.max())
        for i in np.where(v >= thr)[0]:
            ax.annotate(f"{int(v[i])}", (pos[i], v[i]),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=7, color=c)
        ax.margins(x=0.004)
    axes[-1].set_xlabel("output token position  (shared oracle-committed trajectory)")
    fig.suptitle(
        f"Per-proposer accept-length, aligned  —  {args.label}   "
        f"(rid {rid[:8]}, out {out_len} tok, {len(ds_sorted)} steps)",
        fontsize=12.5, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
