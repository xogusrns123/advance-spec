#!/usr/bin/env python3
"""Chain hand-off oracle analysis (O0-O3) over the dense table.

Consumes the per-position table produced by
simulation/scripts/experiments/run_chain_handoff_oracle.py and derives,
per workload x k (= num_draft_tokens, the total draft-token cap of the
composed chain) x suffix variant:

  variant "grd" (HEADLINE) — suffix drafts a single TOP-1 chain: the trie
      is only an intermediate; the deployable chain follows the
      max-count child at every node (validated ~96% prefix-identical to
      speculate(use_tree_spec=False)).
  variant "orc" (reference) — best path inside the suffix tree w.r.t.
      ground truth, i.e. path selection is also oracle. Upper reference
      only; NOT a chain method.

Oracles:
  O0_eagle  — eagle-only chain, fixed source:        mean min(A_e, k)
  O0_suffix — suffix-only chain, fixed source:       mean val(t, 0, k)
  O1        — per-step root selection oracle:        mean max(eagle, suffix)
  O2        — best FIXED hand-off depth j:           max_j mean_t val(t, j, k)
  O3        — per-step ADAPTIVE hand-off depth j(t): mean_t max_j val(t, j, k)

with val(t, j, k) = min(j, A_e, k) + [j <= min(A_e, k)] * min(A_s(t, j), k - j).

Each metric is reported two ways:
  * pos  — uniform mean over all token positions (dense view), and
  * traj — renewal trajectory walk (pos -> pos + accept + 1, missing
           positions advance 1 with accept 0), i.e. mean accepted tokens
           per DECODE STEP, which is what a real decode sees. For O2 the
           fixed j is chosen by the same metric it is reported under.

Also emits the j*(t) distribution for O3 (ties -> min j), both over all
positions and over trajectory-visited steps (plus best>0-only variants),
and figures at a single headline k.

Usage:
  python3 -m simulation.scripts.analyze_chain_handoff \
      --inputs bfcl_v4=...jsonl.gz specbench=...jsonl.gz \
      --out-json simulation/results/chain_handoff_oracle/summary.json \
      --fig-dir simulation/notebooks/figures/chain_handoff \
      [--ks 4,8,16,32,64,inf] [--variants grd,orc] \
      [--fig-k 8] [--fig-variant grd]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fixed hand-off depths considered (0..chain depth). Default matches the
# steps=8 captures; override via --j-max (or make_figures(j_max=...)) for
# deeper chains. Mutated IN PLACE so all consumers see the same grid.
J_GRID = list(range(0, 9))

# Colors: oracles in the red/warm family per project convention (oracle=red);
# O0 sources in neutral grays/blues to stay clear of EAGLE3/suffix/hybrid hues.
COLORS = {
    "O0_eagle": "#7f7f7f",
    "O0_suffix": "#4c72b0",
    "O1": "#f1a340",
    "O2": "#e3120b",
    "O3": "#8b0000",
}
WL_COLORS = {  # per-workload line colors (fixed-j figure)
    "bfcl_v4": "#4c72b0",
    "specbench": "#55a868",
    "swebench_verified": "#8172b2",
}


def load_rows(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
    return rows


def _as_map(row, variant):
    """j -> A_s(t, j) for the chosen suffix variant."""
    col = 1 if variant == "orc" else 2
    return {s[0]: s[col] for s in row.get("sfx", [])}


def val(ae, asv, j, k):
    je = min(j, ae, k)
    if j <= min(ae, k) and j in asv:
        return je + min(asv[j], k - j)
    return je


def best_j(ae, asv, k):
    """(max val, min argmax j) over j in 0..min(ae, k)."""
    bv, bj = -1, 0
    for j in range(0, min(ae, k) + 1):
        v = val(ae, asv, j, k)
        if v > bv:
            bv, bj = v, j
    return bv, bj


def analyze_workload(rows, ks, variant):
    out = {}
    # Pre-extract once per variant.
    table = [(r["rid"], r["ci"], r["pos"], r["ae"], _as_map(r, variant))
             for r in rows]
    # Sequence index for trajectory walks.
    seqs = defaultdict(dict)
    for rid, ci, pos, ae, asv in table:
        seqs[(rid, ci)][pos] = (ae, asv)

    for K in ks:
        kv = 10 ** 9 if K == "inf" else int(K)
        n = len(table)
        if n == 0:
            continue

        # ---------- per-position means ----------
        sum_e = sum_s = sum_o1 = sum_o3 = 0
        per_j = [0.0] * len(J_GRID)
        jstar_hist = [0] * len(J_GRID)
        # _nz: only steps where the best achievable accept is > 0 —
        # separates "suffix-at-root wins" from "nothing accepts anyway"
        # (all-zero steps tie at val=0 and min-j tie-break dumps them in j*=0).
        jstar_hist_nz = [0] * len(J_GRID)
        # Tie-aware breakdown over ALL positions: unique argmax / tie binned
        # at min j (original spec) / accepts nothing at any j.
        jbd_pos_strict = [0] * len(J_GRID)
        jbd_pos_tie = [0] * len(J_GRID)
        jbd_pos_zero = 0
        for _, _, _, ae, asv in table:
            e = min(ae, kv)
            s = val(ae, asv, 0, kv)
            sum_e += e
            sum_s += s
            sum_o1 += max(e, s)
            vals = [val(ae, asv, j, kv) for j in range(0, min(ae, kv) + 1)]
            v3 = max(vals)
            j3 = vals.index(v3)  # first index = min-j tie-break
            sum_o3 += v3
            jstar_hist[min(j3, len(J_GRID) - 1)] += 1
            if v3 > 0:
                jstar_hist_nz[min(j3, len(J_GRID) - 1)] += 1
            if v3 == 0:
                jbd_pos_zero += 1
            elif vals.count(v3) == 1:
                jbd_pos_strict[min(j3, len(J_GRID) - 1)] += 1
            else:
                jbd_pos_tie[min(j3, len(J_GRID) - 1)] += 1
            for ji, j in enumerate(J_GRID):
                per_j[ji] += val(ae, asv, j, kv)
        per_j = [x / n for x in per_j]
        o2_pos = max(per_j)
        o2_j_pos = J_GRID[per_j.index(o2_pos)]  # index() -> first = min j

        # ---------- trajectory walks ----------
        def walk(accept_fn):
            """Renewal walk over every sequence; returns (mat, n_steps,
            visited (ae, asv) entries for jstar)."""
            tot_acc = 0
            tot_steps = 0
            visited = []
            for key, posmap in seqs.items():
                if not posmap:
                    continue
                pos = min(posmap)
                last = max(posmap)
                while pos <= last:
                    entry = posmap.get(pos)
                    if entry is None:
                        tot_steps += 1
                        pos += 1
                        continue
                    a = accept_fn(entry[0], entry[1])
                    tot_acc += a
                    tot_steps += 1
                    visited.append((entry[0], entry[1]))
                    pos += a + 1
            mat = tot_acc / tot_steps if tot_steps else 0.0
            return mat, tot_steps, visited

        traj_e, _, _ = walk(lambda ae, asv: min(ae, kv))
        traj_s, _, _ = walk(lambda ae, asv: val(ae, asv, 0, kv))
        traj_o1, _, _ = walk(
            lambda ae, asv: max(min(ae, kv), val(ae, asv, 0, kv)))
        traj_per_j = []
        for j in J_GRID:
            m, _, _ = walk(lambda ae, asv, _j=j: val(ae, asv, _j, kv))
            traj_per_j.append(m)
        traj_o2 = max(traj_per_j)
        traj_o2_j = J_GRID[traj_per_j.index(traj_o2)]
        traj_o3, traj_steps, visited = walk(
            lambda ae, asv: best_j(ae, asv, kv)[0])
        jstar_hist_traj = [0] * len(J_GRID)
        jstar_hist_traj_nz = [0] * len(J_GRID)
        # Tie-aware breakdown of the traj j* distribution: each step has
        # either a unique argmax j, a tie across several j (binned at min j
        # per the original spec), or accepts nothing at any j (degenerate
        # all-zero tie -> lands in the j*=0 bin).
        jbd_strict = [0] * len(J_GRID)
        jbd_tie = [0] * len(J_GRID)
        jbd_zero = 0
        for ae, asv in visited:
            vals = [val(ae, asv, j, kv) for j in range(0, min(ae, kv) + 1)]
            v3 = max(vals)
            j3 = vals.index(v3)  # first index = min-j tie-break
            jstar_hist_traj[min(j3, len(J_GRID) - 1)] += 1
            if v3 > 0:
                jstar_hist_traj_nz[min(j3, len(J_GRID) - 1)] += 1
            if v3 == 0:
                jbd_zero += 1
            elif vals.count(v3) == 1:
                jbd_strict[min(j3, len(J_GRID) - 1)] += 1
            else:
                jbd_tie[min(j3, len(J_GRID) - 1)] += 1

        o3_pos = sum_o3 / n
        out[str(K)] = {
            "n_positions": n,
            "pos": {
                "O0_eagle": sum_e / n,
                "O0_suffix": sum_s / n,
                "O1": sum_o1 / n,
                "O2": o2_pos, "O2_j": o2_j_pos,
                "O2_per_j": {str(j): per_j[ji]
                             for ji, j in enumerate(J_GRID)},
                "O3": o3_pos,
                "gap_O3_O2_abs": o3_pos - o2_pos,
                "gap_O3_O2_rel": (o3_pos / o2_pos - 1.0) if o2_pos else 0.0,
            },
            "traj": {
                "O0_eagle": traj_e,
                "O0_suffix": traj_s,
                "O1": traj_o1,
                "O2": traj_o2, "O2_j": traj_o2_j,
                "O2_per_j": {str(j): traj_per_j[ji]
                             for ji, j in enumerate(J_GRID)},
                "O3": traj_o3,
                "n_steps_O3": traj_steps,
                "gap_O3_O2_abs": traj_o3 - traj_o2,
                "gap_O3_O2_rel": (traj_o3 / traj_o2 - 1.0) if traj_o2 else 0.0,
            },
            "jstar_hist": jstar_hist,
            "jstar_hist_nz": jstar_hist_nz,
            "jstar_hist_traj": jstar_hist_traj,
            "jstar_hist_traj_nz": jstar_hist_traj_nz,
            "jstar_pos_breakdown": {
                "strict": jbd_pos_strict,
                "tie_min_j": jbd_pos_tie,
                "all_zero": jbd_pos_zero,
            },
            "jstar_traj_breakdown": {
                "strict": jbd_strict,
                "tie_min_j": jbd_tie,
                "all_zero": jbd_zero,
            },
        }
    return out


def check_invariants(res, workload, variant):
    """Sanity checks; prints WARN lines, returns count of violations."""
    bad = 0
    eps = 1e-9
    for K, r in res.items():
        for view in ("pos", "traj"):
            v = r[view]
            checks = [
                ("O0_suffix == O2_per_j[0]",
                 abs(v["O0_suffix"] - v["O2_per_j"]["0"]) < 1e-6),
                ("O0_eagle <= O3", v["O0_eagle"] <= v["O3"] + eps),
                ("O0_suffix <= O1", v["O0_suffix"] <= v["O1"] + eps),
                ("O1 <= O3", v["O1"] <= v["O3"] + eps),
                ("O2 <= O3", v["O2"] <= v["O3"] + eps),
            ]
            # O0_eagle <= max_j O2(j) only strictly holds per-position
            # (k >= 8 makes val(t,8,k) >= min(ae,k)); on trajectories the
            # walks visit different positions, so check pos view only.
            if view == "pos":
                checks.append(
                    ("O0_eagle <= max_j O2_per_j",
                     v["O0_eagle"] <= max(v["O2_per_j"].values()) + 1e-6))
            for name, ok in checks:
                if not ok:
                    bad += 1
                    print(f"WARN invariant [{workload}/{variant}/k={K}/"
                          f"{view}] {name} FAILED: {v}", file=sys.stderr)
    return bad


def make_figures(summary, fig_dir, fig_k="8", variant="grd",
                 fig_workloads=None, view="pos", base_label="eagle",
                 j_max=None):
    """Consolidated figures at a single headline k (= num_draft_tokens).

    view="pos"  — dense per-position means (every step, no skipping);
                  files named *_k{K}.png.
    view="traj" — renewal-walk means (what a real decode sees);
                  files named *_k{K}_traj.png, titles marked "traj".
    j_max       — hand-off depth grid bound; must match the --j-max the
                  summary was computed with (default: current J_GRID).
    """
    if j_max is not None:
        J_GRID[:] = list(range(0, j_max + 1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = ["O0_eagle", "O0_suffix", "O1", "O2", "O3"]
    workloads = [w for w in summary
                 if variant in summary[w] and fig_k in summary[w][variant]
                 and (fig_workloads is None or w in fig_workloads)]
    if not workloads:
        print(f"WARN: no workload has k={fig_k}/{variant}; no figures",
              file=sys.stderr)
        return
    klabel = f"k = num_draft_tokens = {fig_k}"
    if view == "traj":
        klabel += ", traj"
    fsuf = "" if view == "pos" else f"_{view}"

    def disp(m):
        """Display name for a method key (base proposer may be MTP, not
        EAGLE3 — see project_eagle_label_means_mtp)."""
        return m.replace("eagle", base_label.lower())

    # 1) methods bar. Single workload: methods on the x-axis, one bar each.
    #    Multiple workloads: grouped bars per workload with a method legend.
    if len(workloads) == 1:
        wl = workloads[0]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ys = [summary[wl][variant][fig_k][view][m] for m in methods]
        bars = ax.bar(np.arange(len(methods)), ys, 0.6,
                      color=[COLORS[m] for m in methods])
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
        o2j = summary[wl][variant][fig_k][view]["O2_j"]
        labels = [disp(m) if m != "O2" else f"O2 (j={o2j})" for m in methods]
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(labels)
        ax.set_title(f"{wl} chain hand-off oracles ({klabel})",
                     fontsize=10.5)
    else:
        fig, ax = plt.subplots(
            figsize=(max(6.5, 2.0 + 1.8 * len(workloads)), 4.2))
        x = np.arange(len(workloads))
        w = 0.16
        for mi, m in enumerate(methods):
            ys = [summary[wl][variant][fig_k][view][m] for wl in workloads]
            bars = ax.bar(x + (mi - 2) * w, ys, w, label=disp(m),
                          color=COLORS[m])
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
        ax.set_xticks(x)
        ax.set_xticklabels(workloads)
        ax.set_title(f"Chain hand-off oracles ({klabel})", fontsize=10.5)
        ax.legend(fontsize=8)
    ax.set_ylabel("mean accepted tokens / step")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"methods_k{fig_k}{fsuf}.png", dpi=140)
    plt.close(fig)

    # 2) fixed-j curve: one line per workload
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for wl in workloads:
        r = summary[wl][variant][fig_k][view]
        pj = r["O2_per_j"]
        color = WL_COLORS.get(wl)
        ax.plot(J_GRID, [pj[str(j)] for j in J_GRID], marker="o",
                label=wl, color=color)
        bj = r["O2_j"]
        ax.scatter([bj], [pj[str(bj)]], s=110, facecolors="none",
                   edgecolors="black", zorder=5)
    ax.set_xlabel(f"fixed hand-off depth j ({base_label.upper()} "
                  "for depth <= j)")
    ax.set_ylabel("mean accepted tokens / step")
    if len(workloads) == 1:
        ax.set_title(f"{workloads[0]} O2 fixed-j curve ({klabel})")
    else:
        ax.set_title(f"O2 fixed-j curve ({klabel})")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"fixedj_k{fig_k}{fsuf}.png", dpi=140)
    plt.close(fig)

    # 3) j* histogram: one panel per workload (all positions — same dense
    #    per-position view as the other figures; no renewal skipping)
    fig, axes = plt.subplots(
        1, len(workloads),
        figsize=(max(7.0, 3.4 * len(workloads)), 3.4), sharey=True)
    if len(workloads) == 1:
        axes = [axes]
    for axi, wl in zip(axes, workloads):
        bd = summary[wl][variant][fig_k][f"jstar_{view}_breakdown"]
        strict, tie, zero = bd["strict"], bd["tie_min_j"], bd["all_zero"]
        axi.bar(J_GRID, strict, 0.6, color=COLORS["O3"],
                label="unique best j")
        axi.bar(J_GRID, tie, 0.6, bottom=strict, color="#e8a09a",
                label="tie (min j)")
        axi.bar([0], [zero], 0.6, bottom=[strict[0] + tie[0]],
                color="#cccccc", label="no accept at any j")
        axi.set_title(wl, fontsize=10)
        axi.set_xlabel("j*(t)")
        axi.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Optimal hand-off depth j*(t) ({klabel})", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / f"jstar_k{fig_k}{fsuf}.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="workload=path.jsonl.gz pairs")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--ks", default="4,8,16,32,64,inf",
                    help="k (= num_draft_tokens) caps to evaluate")
    ap.add_argument("--variants", default="grd,orc",
                    help="grd = suffix top-1 chain (headline); "
                         "orc = oracle path in suffix tree (reference)")
    ap.add_argument("--fig-k", default="8",
                    help="single k used for figures")
    ap.add_argument("--fig-variant", default="grd")
    ap.add_argument("--fig-workloads", default=None,
                    help="comma list; restrict figures to these workloads "
                         "(default: all)")
    ap.add_argument("--j-max", type=int, default=8,
                    help="max hand-off depth j (= chain depth of the "
                         "capture; 8 for steps8 captures, 16 for steps16)")
    args = ap.parse_args()

    J_GRID[:] = list(range(0, args.j_max + 1))
    ks = [k.strip() for k in args.ks.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]

    summary = {}
    n_bad = 0
    for spec in args.inputs:
        workload, path = spec.split("=", 1)
        print(f"loading {workload}: {path}", file=sys.stderr)
        rows = load_rows(path)
        print(f"  {len(rows)} positions", file=sys.stderr)
        summary[workload] = {}
        for variant in variants:
            res = analyze_workload(rows, ks, variant)
            n_bad += check_invariants(res, workload, variant)
            summary[workload][variant] = res
            for K in ks:
                if str(K) not in res:
                    continue
                t = res[str(K)]["traj"]
                print(f"  [{workload}/{variant}/k={K}] traj "
                      f"O0_e={t['O0_eagle']:.3f} O0_s={t['O0_suffix']:.3f} "
                      f"O1={t['O1']:.3f} O2={t['O2']:.3f}(j={t['O2_j']}) "
                      f"O3={t['O3']:.3f} "
                      f"gap={t['gap_O3_O2_rel'] * 100:+.1f}%")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"j_grid": J_GRID, "ks": ks,
                   "k_meaning": "num_draft_tokens (total draft cap of the "
                                "composed chain)",
                   "headline": {"k": args.fig_k, "variant": args.fig_variant},
                   "workloads": summary}, f, indent=2)
    print(f"summary -> {out_path}", file=sys.stderr)

    if args.fig_dir:
        fig_wls = (args.fig_workloads.split(",")
                   if args.fig_workloads else None)
        make_figures(summary, args.fig_dir, fig_k=args.fig_k,
                     variant=args.fig_variant, fig_workloads=fig_wls)
        print(f"figures -> {args.fig_dir}", file=sys.stderr)

    if n_bad:
        print(f"WARNING: {n_bad} invariant violations", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
