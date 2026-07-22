#!/usr/bin/env python3
"""Candidate COMMON metrics for the Dr.Lee(multislot) / Kim(boundary) debate,
evaluated against both parties' gains on the interp_validation data.

Every candidate is computable from the arm-independent trajectory profile
(s(p) suffix copy depth, a(p) DFlash leading match) — no compose run needed —
so it can be used to CLASSIFY a new workload up front, which is what both
parties want the metric for.

Events: cold->warm entries. A warm run W_j with preceding cold gap g_j is ONE
object read two ways — Dr.Lee: "slot g_j interrupting the template", Kim:
"boundary packed into one verify step". Candidates differ in how they WEIGH it:

  slot100   #interior gaps <= W per 100 tok                (Dr.Lee's count)
  bound100  #warm<->cold transitions per 100 tok           (Kim's count)
  warm      warm share (theta=4 pointwise)                 (mass, no events)
  BM        sum over entries of unlocked depth s(entry)    (depth-weighted count)
  BCM       sum over entries with gap<=W of min(s(entry), num_spec - gap)
            = tokens the composition packs into the ONE verify step that a
            switch would split (gap<=W: the head can cross in one block)
  BCMx      BCM x min(1, a(gap_start)/gap)  (soft DFlash-crossability credit)
  EWS       reachable warm mass: sum of warm-run lengths whose entry gap <= W
  WxS       warm x short-gap share (two-factor product)

Outputs Spearman vs g_drlee (compose-best_single), g_kim (compose-switch_real),
g_struct (handoffOR-switchOR) across task units, within-workload checks, and
the same candidates on the multislot toy next to the toy gains.

  PYTHONPATH=/workspace python3 scripts/metric_candidates.py \
      --dir results/interp_validation --names spider swebench bfcl specbench
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

W_BLOCK, THETA, NSPEC = 15, 4, 32


def load_curves(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            out[r["rid"]] = r
    return out


def runs_of(s, theta):
    w = [1 if v >= theta else 0 for v in s]
    runs, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and w[j] == w[i]:
            j += 1
        runs.append((i, j, w[i]))
        i = j
    return runs


def call_metrics(cu):
    """Accumulate candidate numerators + structure counts for one call."""
    s, a = cu["s"], cu["a"]
    L = len(s)
    if L == 0:
        return None
    runs = runs_of(s, THETA)
    m = dict(L=L, warm=0, bm=0.0, bcm=0.0, bcmx=0.0, ews=0.0,
             n_slot=0, n_bound=max(0, len(runs) - 1), n_gap=0, n_short=0)
    for idx, (b, e, iw) in enumerate(runs):
        if not iw:
            continue
        m["warm"] += e - b
        # preceding cold gap (0 when the call starts warm: free entry)
        if idx == 0:
            glen, g0 = 0, b
        else:
            pb, pe, _ = runs[idx - 1]
            glen, g0 = pe - pb, pb
            m["n_gap"] += 1
            if glen <= W_BLOCK:
                m["n_short"] += 1
                if 0 < idx < len(runs) - 1:
                    m["n_slot"] += 1
        depth = min(s[b], NSPEC)
        m["bm"] += depth
        if glen <= W_BLOCK:
            v = min(depth, NSPEC - glen)
            m["bcm"] += v
            cross = 1.0 if glen == 0 else min(1.0, max(a[g0], 0) / glen)
            m["bcmx"] += cross * v
            m["ews"] += e - b
    return m


def unit_metrics(curves, rids):
    tot = defaultdict(float)
    for rid in rids:
        m = call_metrics(curves[rid])
        if m:
            for k, v in m.items():
                tot[k] += v
    L = tot["L"] or 1.0
    gaps = tot["n_gap"] or 1.0
    return dict(
        warm=tot["warm"] / L,
        slot100=100.0 * tot["n_slot"] / L,
        bound100=100.0 * tot["n_bound"] / L,
        BM=100.0 * tot["bm"] / L,
        BCM=100.0 * tot["bcm"] / L,
        BCMx=100.0 * tot["bcmx"] / L,
        EWS=100.0 * tot["ews"] / L,
        WxS=(tot["warm"] / L) * (tot["n_short"] / gaps),
    )


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


METS = ["slot100", "bound100", "warm", "BM", "BCM", "BCMx", "EWS", "WxS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    units = json.load(open(os.path.join(args.dir, "units.json")))
    tu = [u for u in units if u["task"] != "__all__"]

    rows = []
    for name in args.names:
        curves = load_curves(os.path.join(args.dir, f"curves_{name}.jsonl.gz"))
        by_task = defaultdict(list)
        for rid, cu in curves.items():
            by_task[cu.get("task") or "all"].append(rid)
        for u in tu:
            if u["wl"] != name:
                continue
            rids = by_task.get(u["task"])
            if not rids:
                continue
            met = unit_metrics(curves, rids)
            rows.append(dict(wl=name, task=u["task"], g_drlee=u["g_drlee"],
                             g_kim=u["g_kim"], g_struct=u["g_struct"], **met))

    hdr = (f"{'unit':<28}{'gDr':>7}{'gKim':>7}{'gStr':>7} |"
           + "".join(f"{m:>8}" for m in METS))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['wl'] + ':' + r['task']:<28}{r['g_drlee']:>+7.2f}"
              f"{r['g_kim']:>+7.2f}{r['g_struct']:>+7.2f} |"
              + "".join(f"{r[m]:>8.2f}" for m in METS))

    print(f"\n== Spearman across {len(rows)} task units ==")
    print(f"{'gain':<26}" + "".join(f"{m:>8}" for m in METS))
    for g, lab in (("g_drlee", "compose-best_single"), ("g_kim", "compose-switch"),
                   ("g_struct", "handoffOR-switchOR")):
        ys = [r[g] for r in rows]
        print(f"{lab:<26}" + "".join(
            f"{spearman([r[m] for r in rows], ys):>8.2f}" for m in METS))

    for wl in args.names:
        sub = [r for r in rows if r["wl"] == wl]
        if len(sub) < 4:
            continue
        print(f"\n-- within {wl} ({len(sub)} tasks) --")
        for g, lab in (("g_drlee", "vs best_single"), ("g_kim", "vs switch")):
            ys = [r[g] for r in sub]
            print(f"{lab:<26}" + "".join(
                f"{spearman([r[m] for r in sub], ys):>8.2f}" for m in METS))

    # ---- toy sweep: candidates vs k next to toy gains
    print("\n== multislot toy (same candidates; gains from report_ms_k*.json) ==")
    print(f"{'k':<4}{'gDr':>7}{'gKim':>7}{'gStr':>7} |"
          + "".join(f"{m:>8}" for m in METS))
    for k in (0, 1, 2, 4, 8):
        rp = os.path.join(args.dir, f"report_ms_k{k}.json")
        cp = os.path.join(args.dir, f"curves_ms_k{k}.jsonl.gz")
        if not (os.path.exists(rp) and os.path.exists(cp)):
            continue
        rep = json.load(open(rp))
        A = rep["arms"]
        gdr = A["compose"]["K"] - max(A["dflash"]["K"], A["suffix"]["K"])
        gk = A["compose"]["K"] - A["switch_real"]["K"]
        gs = A["handoff_oracle"]["K"] - A["switch_oracle"]["K"]
        curves = load_curves(cp)
        met = unit_metrics(curves, list(curves))
        print(f"{k:<4}{gdr:>+7.2f}{gk:>+7.2f}{gs:>+7.2f} |"
              + "".join(f"{met[m]:>8.2f}" for m in METS))

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
