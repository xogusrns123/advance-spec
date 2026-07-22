#!/usr/bin/env python3
"""The FUNDAMENTAL quantity behind 'where does composition work': grafting
headroom, derived from the mechanism instead of fitted to it.

Mechanical identity: if the DFlash head survives k tokens they equal gt, so the
grafted suffix tail sees exactly the context of position p+k. Hence per round
at position p (teacher-forced, budget B=num_spec, block W):

  compose_oracle(p) = max_{0<=k<=min(a(p),W)}  k + min(s(p+k), B-k)
  switch_oracle(p)  = max(a(p), s(p))
  G(p)              = compose_oracle(p) - switch_oracle(p)      >= 0

G is a PARAMETER-FREE functional of the workload's dual profile (a(p), s(p)) —
no warm threshold, no gap gate, no segmentation. Its round-mean C_graft is in
the same units as MAT gains, so it predicts magnitude, not just rank.

Two capacities decompose the whole ladder (all from the same curves):
  C_sel   = K(switch_sim) - max(K(dflash_sim), K(suffix_sim))   selection value
            (complementarity: the proposers win at different places)
  C_graft = K(graft_sim) - K(switch_sim)                        composition value
            (the head converts foresight into reaching a better suffix onramp)

Validation performed here:
  1. K(graft_sim) vs the LIVE handoff-oracle K from the replay (identity check)
  2. C_graft vs measured (handoff_OR - switch_OR) per task unit (magnitude+rank)
  3. C_sel + C_graft vs compose-best_single ladder positions
  4. Where G lives: share of G mass at positions with s(p) < 4 (the slot/boundary
     support both parties describe — emergent, not assumed)

  PYTHONPATH=/workspace python3 scripts/compute_headroom.py \
      --dir results/interp_validation --names spider swebench bfcl specbench
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict

W_BLOCK = 15


def load_curves(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            out[r["rid"]] = r
    return out


def graft_and_switch(s, a, i, B):
    L = len(s)
    a_i = a[i] if a[i] > 0 else 0
    s_i = min(s[i], B)
    sw = max(a_i, s_i)
    g = sw
    for k in range(0, min(a_i, W_BLOCK) + 1):
        sp = s[i + k] if i + k < L else 0
        v = k + min(sp, B - k)
        if v > g:
            g = v
    return g, sw


def sim(curve, B, pick, max_rounds=4096):
    """Round trajectory from curves (same advance rule as the replay arms)."""
    s, a = curve["s"], curve["a"]
    Ks, m, rounds = [], 0, 0
    gt_len = len(s) + 1
    while rounds < max_rounds and m < gt_len:
        i = m
        if i >= len(s) or a[i] < 0:
            break
        Ks.append(pick(s, a, i, B))
        m += 1 + Ks[-1] + 1
        rounds += 1
    return Ks


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


PICKS = dict(
    dflash=lambda s, a, i, B: a[i] if a[i] > 0 else 0,
    suffix=lambda s, a, i, B: min(s[i], B),
    switch=lambda s, a, i, B: graft_and_switch(s, a, i, B)[1],
    graft=lambda s, a, i, B: graft_and_switch(s, a, i, B)[0],
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--toy", action="store_true", help="include multislot ks")
    args = ap.parse_args()

    units = {(u["wl"], u["task"]): u
             for u in json.load(open(os.path.join(args.dir, "units.json")))}

    rows = []
    print(f"{'unit':<28}{'K_graft':>8}{'K_hOR':>7}{'K_sw':>6}{'K_swOR':>7} |"
          f"{'C_sel':>7}{'C_graft':>8}{'meas':>7} |{'G>0%':>6}{'Gmass@cold':>11}")
    print("-" * 104)
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
            K = {}
            for arm, pick in PICKS.items():
                ks = [k for rid in rids for k in sim(curves[rid], B, pick)]
                K[arm] = sum(ks) / len(ks) if ks else 0.0
            # per-position G diagnostics
            gpos = gmass = gmass_cold = npos = 0
            for rid in rids:
                cu = curves[rid]
                for i in range(len(cu["s"])):
                    if cu["a"][i] < 0:
                        continue
                    g, sw = graft_and_switch(cu["s"], cu["a"], i, B)
                    d = g - sw
                    npos += 1
                    if d > 0:
                        gpos += 1
                        gmass += d
                        if cu["s"][i] < 4:
                            gmass_cold += d
            u = units.get((name, scope))
            meas = (u["K"]["handoff_oracle"] - u["K"]["switch_oracle"]) if u else float("nan")
            live = u["K"]["handoff_oracle"] if u else float("nan")
            c_sel = K["switch"] - max(K["dflash"], K["suffix"])
            c_graft = K["graft"] - K["switch"]
            rows.append(dict(wl=name, task=scope, K=K, live_hOR=live,
                             c_sel=c_sel, c_graft=c_graft, meas_struct=meas,
                             g_pos_share=gpos / max(1, npos),
                             g_mass_cold_share=gmass_cold / max(1e-9, gmass),
                             g_drlee=u["g_drlee"] if u else float("nan"),
                             g_kim=u["g_kim"] if u else float("nan"),
                             g_struct=u["g_struct"] if u else float("nan")))
            r = rows[-1]
            print(f"{name + ':' + scope:<28}{K['graft']:>8.2f}{live:>7.2f}"
                  f"{K['switch']:>6.2f}"
                  f"{(u['K']['switch_oracle'] if u else float('nan')):>7.2f} |"
                  f"{c_sel:>+7.2f}{c_graft:>+8.2f}{meas:>+7.2f} |"
                  f"{r['g_pos_share']:>6.1%}{r['g_mass_cold_share']:>11.1%}")

    tu = [r for r in rows if r["task"] != "__all__" and r["task"] in
          {k[1] for k in units if k[0] == r["wl"]}]
    tu = [r for r in tu if r["meas_struct"] == r["meas_struct"]]
    if len(tu) >= 4:
        print(f"\n== validation across {len(tu)} task units ==")
        print(f"  C_graft vs measured structural gap (hOR-swOR): "
              f"Spearman {spearman([r['c_graft'] for r in tu], [r['meas_struct'] for r in tu]):+.2f}, "
              f"mean ratio {sum(r['c_graft'] for r in tu) / max(1e-9, sum(r['meas_struct'] for r in tu)):.2f}")
        for g, lab in (("g_struct", "g_struct"), ("g_kim", "g_kim (realized)"),
                       ("g_drlee", "g_drlee (realized)")):
            ys = [r[g] for r in tu]
            print(f"  {lab:<18} vs C_graft {spearman([r['c_graft'] for r in tu], ys):+.2f}"
                  f"   vs C_sel {spearman([r['c_sel'] for r in tu], ys):+.2f}"
                  f"   vs C_sel+C_graft {spearman([r['c_sel'] + r['c_graft'] for r in tu], ys):+.2f}")

    if args.toy:
        print("\n== multislot toy ==")
        print(f"{'k':<3}{'K_graft':>8}{'K_hOR':>7}{'C_sel':>7}{'C_graft':>8}"
              f"{'meas_struct':>12}{'gDr':>7}{'gKim':>7}")
        for k in (0, 1, 2, 4, 8):
            rp = os.path.join(args.dir, f"report_ms_k{k}.json")
            cp = os.path.join(args.dir, f"curves_ms_k{k}.jsonl.gz")
            if not (os.path.exists(rp) and os.path.exists(cp)):
                continue
            rep = json.load(open(rp))
            B = rep["num_spec"]
            curves = load_curves(cp)
            K = {}
            for arm, pick in PICKS.items():
                ks = [x for rid in curves for x in sim(curves[rid], B, pick)]
                K[arm] = sum(ks) / len(ks) if ks else 0.0
            A = rep["arms"]
            print(f"{k:<3}{K['graft']:>8.2f}{A['handoff_oracle']['K']:>7.2f}"
                  f"{K['switch'] - max(K['dflash'], K['suffix']):>+7.2f}"
                  f"{K['graft'] - K['switch']:>+8.2f}"
                  f"{A['handoff_oracle']['K'] - A['switch_oracle']['K']:>+12.2f}"
                  f"{A['compose']['K'] - max(A['dflash']['K'], A['suffix']['K']):>+7.2f}"
                  f"{A['compose']['K'] - A['switch_real']['K']:>+7.2f}")

    json.dump(rows, open(os.path.join(args.dir, "headroom.json"), "w"), indent=1)
    print(f"\nsaved -> {os.path.join(args.dir, 'headroom.json')}")


if __name__ == "__main__":
    main()
