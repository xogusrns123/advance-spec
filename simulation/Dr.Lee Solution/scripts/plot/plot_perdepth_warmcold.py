#!/usr/bin/env python3
"""REAL versions of required_figures_synthesized_ver figures 2-3:

  conditional_accept_rate_per_draft_position_<wl>.png
      a_i = P(accept depth i | accepted all < i), three series:
      DFlash (chain hazard from dflash_match leading run, Wilson 95% band) vs
      Suffix conditioned on WARM regions vs Suffix conditioned on COLD regions.
  survival_rate_per_draft_positition_<wl>.png
      S_i = P(first i tokens ALL correct) — same three series.

warm/cold = the plot_traj_warmcold TRAJECTORY-REGION decomposition (positions
inside realized copy runs >= --run-len, gaps closed / islands dropped — same
params, same code), applied to ALL eval units of the workload and pooled.
Chain identity: accept-length L makes both stats exact: #(L>=i)/#(L>=i-1) and
#(L>=i)/#class. Curves stop when the denominator < --min-n.

  python3 scripts/plot_perdepth_warmcold.py --records results/perpos_bfcl_full/bfcl_v4_full.jsonl
  python3 scripts/plot_perdepth_warmcold.py --records results/perpos_swebench/swebench_quick.jsonl \
      --pool swebench
  python3 scripts/plot_perdepth_warmcold.py --records results/perpos_bfcl_full/bfcl_v4_full.jsonl \
      results/perpos_swebench/swebench_quick.jsonl --pool all   # single-panel deck version
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_traj_warmcold import run_cover, segments_from_hits  # same criterion

BASE = Path(__file__).resolve().parent.parent
BLUE, ORANGE, GRAY = "#4C78A8", "#F58518", "#8A8A8A"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - hw, c + hw


def lead_run(match):
    i = 0
    while i < len(match) and match[i] == 1:
        i += 1
    return i


def curves(lengths, max_depth, min_n):
    """accept-length list -> (a_i list, wilson bands, S_i list, depths kept)."""
    a, band, S, depths = [], [], [], []
    n_tot = len(lengths)
    for i in range(1, max_depth + 1):
        n_prev = sum(1 for v in lengths if v >= i - 1) if i > 1 else n_tot
        n_i = sum(1 for v in lengths if v >= i)
        if n_prev < min_n:
            break
        a.append(n_i / n_prev)
        band.append(wilson(n_i, n_prev))
        S.append(n_i / n_tot if n_tot else 0.0)
        depths.append(i)
    return a, band, S, depths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--pool", default=None, help="pool everything into one workload name")
    ap.add_argument("--run-len", type=int, default=4)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--min-seg", type=int, default=16)
    ap.add_argument("--max-depth", type=int, default=15)
    ap.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args()

    # pooled per workload: suffix accept lengths by region class + dflash lead runs
    suf = defaultdict(lambda: {"warm": [], "cold": []})
    dfl = defaultdict(list)
    meta = defaultdict(lambda: {"pos": 0, "warm": 0, "units": 0})

    for rec_path in args.records:
        rp = Path(rec_path)
        traces = json.load(open(rp.with_suffix(".traces.json")))
        ev = traces["eval_traces"]
        has_conv = any("conv" in t for t in ev)
        srows = defaultdict(dict)                   # rid -> pos -> (suffix_match, ad)
        for l in open(rp):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            srows[r["rid"]][r["pos"]] = (int(r.get("suffix_match_warm", 0)),
                                         lead_run(r.get("dflash_match", [])))
        units = defaultdict(lambda: {"label": None, "sm": [], "ad": []})
        for t in ev:
            rid = t["rid"]
            if rid not in srows:
                continue
            uid = (str(rp), t.get("conv", rid) if has_conv else rid)
            u = units[uid]
            u["label"] = t.get("task", "all")
            by_pos = srows[rid]
            for p in range(1, max(by_pos) + 1):
                sm, ad = by_pos.get(p, (0, 0))
                u["sm"].append(sm); u["ad"].append(ad)
        for u in units.values():
            wl = args.pool or u["label"]
            segs = segments_from_hits(run_cover(u["sm"], args.run_len),
                                      args.gap, args.min_seg)
            warm_idx = set()
            for s, e in segs:
                warm_idx.update(range(s, e))
            for i, (sm, ad) in enumerate(zip(u["sm"], u["ad"])):
                suf[wl]["warm" if i in warm_idx else "cold"].append(sm)
                dfl[wl].append(ad)
            meta[wl]["pos"] += len(u["sm"]); meta[wl]["warm"] += len(warm_idx)
            meta[wl]["units"] += 1

    fig_dir = BASE / "readable_outputs" / "figures" / "perdepth"
    fig_dir.mkdir(parents=True, exist_ok=True)
    series = [("DFlash (model head)", BLUE, "o", "dfl"),
              ("Suffix — warm region", ORANGE, "s", "warm"),
              ("Suffix — cold region", GRAY, "^", "cold")]

    for wl in sorted(suf):
        data = {"dfl": dfl[wl], "warm": suf[wl]["warm"], "cold": suf[wl]["cold"]}
        cur = {k: curves(v, args.max_depth, args.min_n) for k, v in data.items()}
        m = meta[wl]
        sub = (f"{m['units']} convs, {m['pos']} positions, warm {m['warm']/max(m['pos'],1):.0%}   "
               f"[copy-run>={args.run_len}, gap={args.gap}, min={args.min_seg}]")

        # crossing = hand-off depth k*: first depth where suffix-warm a_i >= dflash a_i
        aw, ad_ = cur["warm"][0], cur["dfl"][0]
        kstar = next((d for d, (w, v) in enumerate(zip(aw, ad_), start=1) if w >= v), None)

        # -- conditional acceptance ------------------------------------------
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        for name, col, mk, key in series:
            a, band, _, depths = cur[key]
            if not depths:
                continue
            n0 = len(data[key])
            ax.plot(depths, a, marker=mk, ms=5, lw=1.8, color=col,
                    label=f"{name}  (n={n0})")
            if key == "dfl":
                ax.fill_between(depths, [b[0] for b in band], [b[1] for b in band],
                                color=col, alpha=0.15, lw=0)
        if kstar is not None and kstar > 1:
            ax.axvline(kstar - 0.5, color="#333333", ls="--", lw=1.2)
            ax.text(kstar - 0.42, 0.04, f"hand-off depth k* = {kstar}",
                    fontsize=9, rotation=90, va="bottom", color="#333333")
        elif kstar == 1:
            ax.text(0.98, 0.05, "suffix-warm ≥ model from depth 1\n(warm regions: hand off immediately)",
                    transform=ax.transAxes, ha="right", fontsize=8.5, color="#7A4A10")
        if not cur["cold"][0] or all(w < v for w, v in zip(cur["cold"][0], cur["dfl"][0])):
            ax.text(0.98, 0.22, "suffix-cold: no crossing → model alone",
                    transform=ax.transAxes, ha="right", fontsize=8.5, color=GRAY)
        ax.set_xlabel("draft position i  (depth from the same verified prefix)", fontsize=10)
        ax.set_ylabel("conditional acceptance  a_i = P(accept i | accept < i)", fontsize=10)
        ax.set_title(f"Per-depth conditional acceptance — {wl}  [measured]\n{sub}", fontsize=9.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(cur["dfl"][3] or list(range(1, args.max_depth + 1)))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fp = fig_dir / f"conditional_accept_rate_per_draft_position_{wl}.png"
        fig.savefig(fp, dpi=150); plt.close(fig)

        # -- survival ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for name, col, mk, key in series:
            _, _, S, depths = cur[key]
            if depths:
                ax.plot(depths, S, marker=mk, ms=5, lw=1.8, color=col, label=name)
        ax.set_xlabel("draft position i", fontsize=10)
        ax.set_ylabel("survival  S_i = P(first i tokens ALL correct)", fontsize=10)
        ax.set_title(f"Survival per draft position — {wl}  [measured]\n{sub}", fontsize=9.5)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(cur["dfl"][3] or list(range(1, args.max_depth + 1)))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fp2 = fig_dir / f"survival_rate_per_draft_positition_{wl}.png"
        fig.savefig(fp2, dpi=150); plt.close(fig)

        a1 = {k: (cur[k][0][0] if cur[k][0] else 0) for k in ("dfl", "warm", "cold")}
        print(f"[{wl}] pos={m['pos']} warm={m['warm']/max(m['pos'],1):.0%} "
              f"a1: dflash={a1['dfl']:.2f} warm={a1['warm']:.2f} cold={a1['cold']:.2f} "
              f"k*={kstar} -> {fp.name}, {fp2.name}")


if __name__ == "__main__":
    main()
