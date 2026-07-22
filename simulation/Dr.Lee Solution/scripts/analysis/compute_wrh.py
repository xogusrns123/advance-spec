#!/usr/bin/env python3
r"""WRH — Warm Re-entry Headroom: the warm/cold-NATIVE form of the grafting
headroom. The objects are the warm/cold segmentation's (theta=4): cold gaps,
boundaries, warm runs. The VALUE LAW on those objects is the exact G identity
(no summary-stat approximation — a first-order form using (g, d, a(b)) only
recovers 31% of the mass and is kept here as `v1` for the record):

  boundary value  V_j = max_{p in gap_j} G(p)
                  (the best packing composition can achieve when crossing THIS
                   boundary: G looks forward into the warm run, so deep onramps
                   beyond the theta-crossing are priced correctly)
  WRH/100tok      = sum_j V_j / N * 100
  WRH_round       = WRH/100 * (K_switch + 2)          (tok/round units)
  remainder       = G mass at warm positions (reported as a share; the part a
                    boundary decomposition cannot carry)

Decomposition, one factor per party:
  boundary density (Kim)  x  mean boundary value E[V_j]  (weight both missed);
  V_j > 0 requires a DFlash-reachable onramp (Dr.Lee's bridgeability, priced
  not gated).

  PYTHONPATH=/workspace python3 scripts/compute_wrh.py \
      --dir results/interp_validation --names spider swebench bfcl specbench --toy
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import argparse
import gzip
import json
import os
from collections import defaultdict

THETA, W_BLOCK = 4, 15

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_headroom import graft_and_switch  # noqa: E402


def load_curves(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            out[r["rid"]] = r
    return out


def gaps_of(cu):
    """(gap_start, gap_len, reentry_depth, a_at_start, s_at_start) for every
    cold run followed by a warm run (leading gaps included)."""
    s, a = cu["s"], cu["a"]
    L = len(s)
    w = [1 if v >= THETA else 0 for v in s]
    out, i = [], 0
    runs = []
    while i < L:
        j = i
        while j < L and w[j] == w[i]:
            j += 1
        runs.append((i, j, w[i]))
        i = j
    for idx in range(1, len(runs)):
        b, e, iw = runs[idx]
        if not iw:
            continue
        gb, ge, _ = runs[idx - 1]
        out.append((gb, ge - gb, s[b], max(a[gb], 0), s[gb]))
    return out, L


def wrh_of(rids, curves, B):
    n_gap = n_pos = N = 0
    val_sum = v1_sum = warm_g = total_g = 0.0
    for rid in rids:
        cu = curves[rid]
        s, a = cu["s"], cu["a"]
        gaps, L = gaps_of(cu)
        N += L
        # exact per-boundary value: best G within the gap
        for gb, g, d, a0, s0 in gaps:
            n_gap += 1
            V = 0.0
            for p in range(gb, gb + g):
                gg, sw = graft_and_switch(s, a, p, B)
                if gg - sw > V:
                    V = gg - sw
            val_sum += V
            if V > 0:
                n_pos += 1
            if a0 >= g:                       # first-order form, for the record
                v1_sum += max(0, g + min(d, B - g) - max(a0, s0))
        # warm-side remainder (G mass a boundary decomposition cannot carry)
        w = [1 if v >= THETA else 0 for v in s]
        for p in range(L):
            if a[p] < 0:
                continue
            gg, sw = graft_and_switch(s, a, p, B)
            dG = gg - sw
            total_g += dG
            if w[p]:
                warm_g += dG
    return dict(N=N, n_gap=n_gap,
                wrh100=100.0 * val_sum / max(1, N),
                v1_100=100.0 * v1_sum / max(1, N),
                bound100=100.0 * n_gap / max(1, N),
                p_bridge=n_pos / max(1, n_gap),
                value_mean=val_sum / max(1, n_gap),
                warm_g_share=warm_g / max(1e-9, total_g))


def sim_attr(rids, curves, B, max_rounds=4096):
    """Round-weighted G along the graft (compose-oracle) trajectory, split by
    the warm/cold label of the round-start position. This is an EXACT
    attribution: WRH := E_rounds[G] = crossing_rate*E[G|cold] + warm-part."""
    n_r = n_cold = n_cold_pos = n_warm_pos = 0
    g_cold = g_warm = 0.0
    for rid in rids:
        cu = curves[rid]
        s, a = cu["s"], cu["a"]
        w = [1 if v >= THETA else 0 for v in s]
        m, rounds = 0, 0
        gt_len = len(s) + 1
        while rounds < max_rounds and m < gt_len:
            i = m
            if i >= len(s) or a[i] < 0:
                break
            gg, sw = graft_and_switch(s, a, i, B)
            G = gg - sw
            n_r += 1
            if w[i]:
                g_warm += G
                n_warm_pos += 1 if G > 0 else 0
            else:
                n_cold += 1
                g_cold += G
                n_cold_pos += 1 if G > 0 else 0
            m += 1 + gg + 1
            rounds += 1
    n_r = max(1, n_r)
    return dict(rounds=n_r,
                wrh=(g_cold + g_warm) / n_r,
                cold_part=g_cold / n_r, warm_part=g_warm / n_r,
                cross_rate=n_cold_pos / n_r,
                cross_value=g_cold / max(1, n_cold_pos),
                cold_round_share=n_cold / n_r)


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            for k in range(i, j):
                r[order[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--toy", action="store_true")
    args = ap.parse_args()

    hd = {(r["wl"], r["task"]): r
          for r in json.load(open(os.path.join(args.dir, "headroom.json")))}

    rows = []
    print(f"{'unit':<28}{'WRH':>8}{'x-rate':>7}{'x-val':>6}{'warmP':>7}"
          f"{'C_graft':>8}{'meas':>7} |{'bnd100':>7}{'stat100':>8}{'statrnd':>8}")
    print("-" * 100)
    for name in args.names:
        rep = json.load(open(os.path.join(args.dir, f"report_{name}.json")))
        B = rep["num_spec"]
        curves = load_curves(os.path.join(args.dir, f"curves_{name}.jsonl.gz"))
        by_task = defaultdict(list)
        for rid, cu in curves.items():
            by_task[cu.get("task") or "all"].append(rid)
        for scope in ["__all__"] + sorted(by_task):
            rids = ([r for v in by_task.values() for r in v]
                    if scope == "__all__" else by_task[scope])
            m = wrh_of(rids, curves, B)
            at = sim_attr(rids, curves, B)
            h = hd.get((name, scope))
            if h is None:
                continue
            ksw = h["K"]["switch"]
            wrh_round = m["wrh100"] / 100.0 * (ksw + 2)
            rows.append(dict(wl=name, task=scope, **m, wrh_round=wrh_round,
                             **{f"at_{k}": v for k, v in at.items()},
                             c_graft=h["c_graft"], meas=h["meas_struct"],
                             g_kim=h["g_kim"], g_drlee=h["g_drlee"]))
            r = rows[-1]
            print(f"{name + ':' + scope:<28}{at['wrh']:>8.2f}"
                  f"{at['cross_rate']:>7.1%}{at['cross_value']:>6.1f}"
                  f"{at['warm_part']:>7.2f}{h['c_graft']:>8.2f}"
                  f"{h['meas_struct']:>+7.2f} |{m['bound100']:>7.2f}"
                  f"{m['wrh100']:>8.2f}{wrh_round:>8.2f}")

    tu = [r for r in rows if r["task"] != "__all__" and r["meas"] == r["meas"]]
    if len(tu) >= 4:
        print(f"\n== validation across {len(tu)} task units ==")
        for xcol, xl in (("at_wrh", "WRH (round-weighted, attributed)"),
                         ("wrh_round", "static per-gap maxG (no traj)")):
            for tgt, lab in (("c_graft", "C_graft"), ("meas", "measured gap")):
                ys = [r[tgt] for r in tu]
                xs = [r[xcol] for r in tu]
                ratio = sum(xs) / max(1e-9, sum(ys))
                print(f"  {xl:<36} vs {lab:<13} Spearman "
                      f"{spearman(xs, ys):+.2f}, mean ratio {ratio:.2f}")
        print(f"  cold-part only (x-rate*x-val)          vs measured gap  Spearman "
              f"{spearman([r['at_cold_part'] for r in tu], [r['meas'] for r in tu]):+.2f}")

    if args.toy:
        print("\n== multislot toy ==")
        print(f"{'k':<3}{'WRHrnd':>8}{'C_graft':>8}{'meas':>7}{'bnd100':>8}"
              f"{'P_br':>7}{'val':>6}")
        for k in (0, 1, 2, 4, 8):
            cp = os.path.join(args.dir, f"curves_ms_k{k}.jsonl.gz")
            rp = os.path.join(args.dir, f"report_ms_k{k}.json")
            if not (os.path.exists(cp) and os.path.exists(rp)):
                continue
            rep = json.load(open(rp))
            curves = load_curves(cp)
            m = wrh_of(list(curves), curves, rep["num_spec"])
            A = rep["arms"]
            ksw = A["switch_oracle"]["K"]
            wr = m["wrh100"] / 100.0 * (ksw + 2)
            print(f"{k:<3}{wr:>8.2f}"
                  f"{A['handoff_oracle']['K'] - A['switch_oracle']['K']:>8.2f}"
                  f"{A['handoff_oracle']['K'] - A['switch_oracle']['K']:>+7.2f}"
                  f"{m['bound100']:>8.2f}{m['p_bridge']:>7.1%}{m['value_mean']:>6.1f}")

    json.dump(rows, open(os.path.join(args.dir, "wrh.json"), "w"), indent=1)
    print(f"\nsaved -> {os.path.join(args.dir, 'wrh.json')}")


if __name__ == "__main__":
    main()
