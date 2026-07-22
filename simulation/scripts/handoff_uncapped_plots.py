#!/usr/bin/env python3
"""Uncapped-tail hand-off curves (MAT + latency-aware speedup) vs a*.

Fixes two earlier mistakes:
  (1) the suffix TAIL was capped by the val() budget k=16 -> use k=inf so the
      tail runs to its NATURAL suffix-decoding length (bounded only by the
      suffix's own max_spec_factor, which is legitimate). DFlash head still 16.
  (2) verify latency was taken flat at the ~1-token base (39.5ms). A longer tail
      means a longer verify block, so use the MEASURED DFlash verify-vs-block
      curve V(n) from budget_tradeoff/qwen35_27b/DFlash.json.

Per a* (= T/(1+T) threshold on the DFlash hazard dflash_p):
  head length  k* = policy_threshold(hazards, a*)              (deployable)
  accept       = val(A_d, A_s(k*), k*, inf)   (tail uncapped)
  verify block = k* + proposed_suffix_run(k*) = k* + gcl[k*]
  step time    = V(block) + dflash_draft + suffix_query
Aggregate throughput speedup vs vanilla (1 tok / V(1) forward):
  speedup = (sum committed) * V(1) / (sum step_time)

Endpoints integrated onto the a* axis: a*=0 -> pure DFlash-only (16-block, no
tail), a*=1 -> pure suffix-only (natural run). O3 (tree/oracle ceiling) is drawn
as a separate marker at the far right.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.scripts.analyze_handoff_policy import (  # noqa: E402
    load_haz, val, policy_threshold,
)

INF = 10**9

# ---- latency components (Qwen3.5-27B), verify held CONSTANT ----
# The 27B target forward is memory-bound at bs=1 (weight-read dominated), so the
# verify cost is ~flat vs block length up to n~160 (methodology). Per user, hold
# verify FIXED at the measured base and do NOT let it grow with the tail.
V_VERIFY = 39.475         # verify per step (CONSTANT, legacy summary target fwd)
V_BASE_1TOK = 39.475      # vanilla: 1 token / target forward
DFLASH_OH = 2.769         # DFlash draft 1.952 + others 0.817 (step overhead)
SUFFIX_QUERY = 0.152      # suffix trie query (adds a tail)
SUFFIX_OH = 0.202         # suffix-only step overhead (draft 0.152 + others 0.05)


def V(n):
    """verify cost — held constant regardless of block length n."""
    return V_VERIFY


def load_table(path, variant="grd"):
    """(rid,ci) -> {pos: dict(ae, asv, gclj, gcl0)}.
    asv[j]  = realized suffix accept A_s(t,j) (grd col 2)
    gclj[j] = proposed suffix run length at hand-off depth j (col 4)
    gcl0    = gclj[0]"""
    col = 1 if variant == "orc" else 2
    seqs = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            sfx = r["sfx"]
            asv = {s[0]: s[col] for s in sfx}
            gclj = {s[0]: s[4] for s in sfx}
            gcl0 = sfx[0][4] if sfx else 0
            score0 = sfx[0][6] if sfx else 0.0   # suffix RUN score at j=0 = T(t)
            seqs[(r["rid"], r["ci"])][r["pos"]] = dict(
                ae=r["ae"], asv=asv, gclj=gclj, gcl0=gcl0, score0=score0)
    return seqs


def attach_haz(seqs, haz):
    for (rid, ci), pm in seqs.items():
        for pos in pm:
            pm[pos]["haz"] = haz.get((rid, pos), [])


def walk_agg(seqs, accept_fn, block_fn, draft_ms):
    """Renewal walk. Returns dict(mat, mean_block, speedup)."""
    committed = 0
    steps = 0
    total_time = 0.0
    for pm in seqs.values():
        if not pm:
            continue
        pos, last = min(pm), max(pm)
        while pos <= last:
            e = pm.get(pos)
            if e is None:
                # gap: vanilla-like single step (no draft) — count as 1 token
                committed += 1
                steps += 1
                total_time += V(1)
                pos += 1
                continue
            a = accept_fn(e)
            b = block_fn(e)
            committed += a + 1
            steps += 1
            total_time += V(b) + draft_ms
            pos += a + 1
    mat = (committed - steps) / max(steps, 1)
    vanilla_time = committed * V_BASE_1TOK
    speedup = vanilla_time / max(total_time, 1e-9)
    return dict(mat=mat, speedup=speedup, steps=steps)


# ---- accept / block functions per mode (tail uncapped: k=inf) ----
def acc_dflash(e):
    return min(e["ae"], INF)


def blk_dflash(e):
    return 16


def acc_suffix(e):
    return val(e["ae"], e["asv"], 0, INF)


def blk_suffix(e):
    return max(e["gcl0"], 1)


def acc_handoff(e, astar):
    k = policy_threshold(e["haz"], astar)
    return val(e["ae"], e["asv"], k, INF)


def blk_handoff(e, astar):
    k = policy_threshold(e["haz"], astar)
    return k + e["gclj"].get(k, 0)


def _astar_of(e):
    """per-position variable threshold a*(t) = T/(1+T), T = suffix run score."""
    T = e["score0"]
    return T / (1.0 + T) if T > 0 else 0.0


def acc_score(e):
    k = policy_threshold(e["haz"], _astar_of(e))
    return val(e["ae"], e["asv"], k, INF)


def blk_score(e):
    k = policy_threshold(e["haz"], _astar_of(e))
    return k + e["gclj"].get(k, 0)


def acc_o3(e):
    ae, asv = e["ae"], e["asv"]
    best = 0
    for j in range(0, min(ae, max(asv) if asv else 0) + 1):
        best = max(best, val(ae, asv, j, INF))
    return best


def blk_o3(e):
    ae, asv, gclj = e["ae"], e["asv"], e["gclj"]
    best_v, best_j = -1, 0
    for j in range(0, min(ae, max(asv) if asv else 0) + 1):
        v = val(ae, asv, j, INF)
        if v > best_v:
            best_v, best_j = v, j
    return best_j + gclj.get(best_j, 0)


def compute(seqs):
    astars = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
              0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
              0.93, 0.95, 0.97]
    curve = []
    for a in astars:
        r = walk_agg(seqs, lambda e, a=a: acc_handoff(e, a),
                     lambda e, a=a: blk_handoff(e, a),
                     DFLASH_OH + SUFFIX_QUERY)
        curve.append((a, r["mat"], r["speedup"]))
    dflash = walk_agg(seqs, acc_dflash, blk_dflash, DFLASH_OH)
    suffix = walk_agg(seqs, acc_suffix, blk_suffix, SUFFIX_OH)
    o3 = walk_agg(seqs, acc_o3, blk_o3, DFLASH_OH + SUFFIX_QUERY)
    # deployable score-based variable-threshold policy (per-position a*=T/(1+T))
    score = walk_agg(seqs, acc_score, blk_score, DFLASH_OH + SUFFIX_QUERY)
    avals = sorted(_astar_of(e) for pm in seqs.values() for e in pm.values())
    score["astar_med"] = avals[len(avals) // 2] if avals else 0.0
    return dict(curve=curve, dflash=dflash, suffix=suffix, o3=o3, score=score)


def plot(res, regime, model, outdir, tag="uncapped",
         struct_label="Chain hand-off",
         curve_label="DFlash head + suffix tail (uncapped)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = res["curve"]
    xs = [0.0] + [c[0] for c in curve] + [1.0]
    mats = [res["dflash"]["mat"]] + [c[1] for c in curve] + [res["suffix"]["mat"]]
    sps = [res["dflash"]["speedup"]] + [c[2] for c in curve] + [res["suffix"]["speedup"]]
    o3_x = 1.11

    for kind, ys, o3y, ylab, title in [
        ("MAT", mats, res["o3"]["mat"], "MAT (mean accepted tokens / step)",
         f"{struct_label} ({regime}): MAT vs a*  —  {model}"),
        ("speedup", sps, res["o3"]["speedup"],
         "latency-aware speedup vs vanilla (verify held constant)",
         f"{struct_label} ({regime}): speedup vs a*  —  {model}"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        # interior handoff curve
        ax.plot(xs[1:-1], ys[1:-1], "-o", color="#1f77b4", ms=4.5, lw=1.8,
                label=curve_label, zorder=3)
        # integrated endpoints
        ax.plot([0.0], [ys[0]], "s", color="#d62728", ms=11,
                label=f"DFlash-only: {ys[0]:.2f}", zorder=5)
        ax.plot([1.0], [ys[-1]], "^", color="#2ca02c", ms=12,
                label=f"suffix-only: {ys[-1]:.2f}", zorder=5)
        # connect endpoints to the interior curve
        ax.plot([0.0, xs[1]], [ys[0], ys[1]], "-", color="#1f77b4", lw=1.8,
                alpha=0.6, zorder=2)
        ax.plot([xs[-2], 1.0], [ys[-2], ys[-1]], "-", color="#1f77b4", lw=1.8,
                alpha=0.6, zorder=2)
        # O3 separate at far right
        ax.axhline(o3y, color="#7f7f7f", ls=":", lw=1.0, alpha=0.6, zorder=1)
        ax.plot([o3_x], [o3y], "*", color="#9467bd", ms=20,
                label=f"O3 oracle (tree ceiling): {o3y:.2f}", zorder=6)
        ax.annotate("O3", (o3_x, o3y), textcoords="offset points",
                    xytext=(6, 6), fontsize=10, color="#9467bd", weight="bold")

        # score-based variable-threshold policy (offline sim), at its median a*
        sc = res["score"]
        scy = sc["mat"] if kind == "MAT" else sc["speedup"]
        ax.plot([sc["astar_med"]], [scy], "D", color="#ff7f0e", ms=11,
                mec="black", mew=1.2, zorder=8,
                label=f"score-based T/(1+T) sim (a*≈{sc['astar_med']:.2f}): {scy:.2f}")

        ax.set_xlabel("a*  =  T / (1+T)   (hand-off threshold on DFlash hazard)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.04, 1.18)
        ax.grid(True, alpha=0.3)
        loc = "lower left" if kind == "speedup" else "best"
        ax.legend(loc=loc, fontsize=9, framealpha=0.92)
        fig.tight_layout()
        out = Path(outdir) / f"handoff_astar_{regime}_{kind}_{tag}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--dflash-proposals", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--model", default="Qwen3.5-27B (SpecBench)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    seqs = load_table(args.table)
    haz = load_haz(args.dflash_proposals)
    attach_haz(seqs, haz)
    res = compute(seqs)

    print(f"\n=== {args.regime} ===")
    print(f"  DFlash-only : MAT={res['dflash']['mat']:.3f}  "
          f"speedup={res['dflash']['speedup']:.2f}x")
    print(f"  suffix-only : MAT={res['suffix']['mat']:.3f}  "
          f"speedup={res['suffix']['speedup']:.2f}x")
    print(f"  O3 oracle   : MAT={res['o3']['mat']:.3f}  "
          f"speedup={res['o3']['speedup']:.2f}x")
    best = max(res["curve"], key=lambda c: c[1])
    print(f"  best handoff: a*={best[0]:.2f} MAT={best[1]:.3f} "
          f"speedup={best[2]:.2f}x")
    for a, m, s in res["curve"]:
        print(f"    a*={a:.2f}  MAT={m:.3f}  speedup={s:.2f}x")

    print(f"  score-based sim: a*_med={res['score']['astar_med']:.3f} "
          f"MAT={res['score']['mat']:.3f} speedup={res['score']['speedup']:.2f}x")
    plot(res, args.regime, args.model, args.outdir)
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
