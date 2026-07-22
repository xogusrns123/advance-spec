#!/usr/bin/env python3
"""Synthesize validate_interpretations.py outputs across workloads/tasks:

  unit = (workload, task) with enough rounds. For each unit:
    g_drlee   = K(compose) - max(K(dflash), K(suffix))     [vs best single]
    g_kim     = K(compose) - K(switch_real)                [vs per-step binary pick]
    g_kim_or  = K(compose) - K(switch_oracle)
    g_struct  = K(handoff_oracle) - K(switch_oracle)       [structure-only ceiling gap]
  and structure stats (slot_per100, bound_per100, warm_share, ...) plus a
  re-entry-weighted candidate metric computed from the raw curves:
    BM  = sum over cold->warm entries of min(s(entry), num_spec) / positions * 100
    BMb = same, but only entries whose preceding cold gap the DFlash block could
          bridge (gap <= W and a(gap_start) >= gap_len)

  Then Spearman correlations of each gain against each structure metric, and a
  head-vs-gap check on oracle 'g'-labelled rounds (is the chosen head length
  the slot length?).

  PYTHONPATH=/workspace python3 scripts/synthesize_interp.py \
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


def load_curves(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            out[r["rid"]] = r
    return out


def entries_of(curve, W, theta, num_spec):
    """Cold->warm entry events: (entry_pos, gap_len, bridgeable, unlocked_depth).
    gap = the cold run immediately before the warm run (leading run included)."""
    s, a = curve["s"], curve["a"]
    L = len(s)
    w = [1 if v >= theta else 0 for v in s]
    ev = []
    i = 0
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
        glen = ge - gb
        br = (glen <= W and a[gb] >= glen and a[gb] >= 0)
        ev.append((b, glen, br, min(s[b], num_spec)))
    return ev, L


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
            avg = (i + j - 1) / 2.0
            for k2 in range(i, j):
                r[order[k2]] = avg
            i = j
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--min-rounds", type=int, default=200)
    ap.add_argument("--theta", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    units = []          # dict per unit
    for name in args.names:
        rep = json.load(open(os.path.join(args.dir, f"report_{name}.json")))
        num_spec, W = rep["num_spec"], rep["W"]
        curves = load_curves(os.path.join(args.dir, f"curves_{name}.jsonl.gz"))

        # per-task curve-derived metrics (incl. BM)
        by_task = defaultdict(lambda: dict(L=0, bm=0.0, bmb=0.0, n_entry=0))
        for rid, cu in curves.items():
            t = cu.get("task") or "all"
            ev, L = entries_of(cu, W, args.theta, num_spec)
            d = by_task[t]
            d["L"] += L
            d["n_entry"] += len(ev)
            d["bm"] += sum(x[3] for x in ev)
            d["bmb"] += sum(x[3] for x in ev if x[2])

        arms = rep["arms"]
        tasks = sorted(rep["structure_by_task"].keys())
        rows = []
        for scope in ["__all__"] + tasks:
            if scope == "__all__":
                st = rep["structure"]
                K = {a: arms[a]["K"] for a in arms}
                nr = arms["compose"]["rounds"] if "compose" in arms else 0
                cby = arms.get("compose", {}).get("cf_by_label", {})
                bm = sum(d["bm"] for d in by_task.values())
                bmb = sum(d["bmb"] for d in by_task.values())
                L = sum(d["L"] for d in by_task.values()) or 1
                ne = sum(d["n_entry"] for d in by_task.values())
            else:
                st = rep["structure_by_task"][scope]
                K = {}
                for a in ("dflash", "suffix", "switch_real", "switch_oracle"):
                    K[a] = arms[a]["by_task"].get(scope, 0.0)
                for a in ("compose", "handoff_oracle"):
                    if a in arms and scope in arms[a]["by_task"]:
                        K[a] = arms[a]["by_task"][scope]["K"]
                nr = (arms["compose"]["by_task"][scope]["rounds"]
                      if "compose" in arms and scope in arms["compose"]["by_task"] else 0)
                cby = (arms["compose"]["by_task"][scope].get("cf_by_label", {})
                       if "compose" in arms and scope in arms["compose"]["by_task"] else {})
                d = by_task.get(scope, dict(L=1, bm=0, bmb=0, n_entry=0))
                bm, bmb, L, ne = d["bm"], d["bmb"], d["L"], d["n_entry"]
            if "compose" not in K or nr < args.min_rounds:
                continue
            best_single = max(K["dflash"], K["suffix"])
            rows.append(dict(
                wl=name, task=scope, rounds=nr,
                K=K,
                g_drlee=K["compose"] - best_single,
                g_drlee_rel=(K["compose"] / best_single - 1) if best_single else 0,
                g_kim=K["compose"] - K["switch_real"],
                g_kim_or=K["compose"] - K["switch_oracle"],
                g_struct=(K.get("handoff_oracle", 0) - K["switch_oracle"])
                          if "handoff_oracle" in K else float("nan"),
                warm=st["warm_share"], bound100=st["bound_per100"],
                slot100=st["slot_per100"], gap_mean=st["gap_mean"],
                bridge=st["bridgeable_share"], reentry=st["reentry_mean"],
                warm_run=st["warm_run_mean"],
                bm100=100.0 * bm / L, bmb100=100.0 * bmb / L,
                entry100=100.0 * ne / L,
                cf_by_label={k: round(v[0], 1) for k, v in cby.items()},
            ))
        units.extend(rows)

    hdr = (f"{'unit':<28}{'rnds':>6} {'df':>6}{'sf':>6}{'swR':>6}{'swO':>6}"
           f"{'comp':>6}{'hOR':>6} | {'gDr':>6}{'gKim':>6}{'gKimO':>6}{'gStr':>6}"
           f" | {'warm':>5}{'bnd':>6}{'slot':>6}{'BM':>7}{'BMb':>7}{'reent':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in units:
        K = r["K"]
        print(f"{r['wl'] + ':' + r['task']:<28}{r['rounds']:>6} "
              f"{K['dflash']:>6.2f}{K['suffix']:>6.2f}{K['switch_real']:>6.2f}"
              f"{K['switch_oracle']:>6.2f}{K['compose']:>6.2f}"
              f"{K.get('handoff_oracle', float('nan')):>6.2f} | "
              f"{r['g_drlee']:>+6.2f}{r['g_kim']:>+6.2f}{r['g_kim_or']:>+6.2f}"
              f"{r['g_struct']:>+6.2f} | {r['warm']:>5.0%}{r['bound100']:>6.2f}"
              f"{r['slot100']:>6.2f}{r['bm100']:>7.1f}{r['bmb100']:>7.1f}"
              f"{r['reentry']:>6.1f}")

    # correlations on TASK-level units only (exclude __all__ to avoid double count)
    tu = [r for r in units if r["task"] != "__all__"]
    if len(tu) >= 4:
        print(f"\n== Spearman across {len(tu)} task units ==")
        gains = [("g_drlee", "compose-best_single"), ("g_kim", "compose-switch_real"),
                 ("g_kim_or", "compose-switch_oracle"),
                 ("g_struct", "handoffOR-switchOR")]
        mets = ["slot100", "bound100", "bm100", "bmb100", "warm", "reentry",
                "entry100", "gap_mean"]
        print(f"{'gain':<24}" + "".join(f"{m:>9}" for m in mets))
        for g, label in gains:
            xs = [r[g] for r in tu]
            row = []
            for m in mets:
                ys = [r[m] for r in tu]
                row.append(spearman(ys, xs))
            print(f"{label:<24}" + "".join(f"{v:>9.2f}" for v in row))

    if args.out:
        json.dump(units, open(args.out, "w"), indent=1)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
