#!/usr/bin/env python3
"""Is the s(p) FIELD's first moment — mean copy depth, MCD = E[min(s,B)] —
the single fundamental scalar behind 'where composition works'?

Hypothesis (from the whole study): the two parties' quantities (slots,
boundaries) are derived features of one underlying field s(p); the predictor
side a(p) is approximately workload-invariant; hence one mass number of the
s-field should carry the cross-workload signal, with mosaic granularity only a
second-order correction inside the stitchable regime (gap scale <~ head reach).

  PYTHONPATH=/workspace python3 scripts/test_field_mass.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import gzip
import json
from collections import defaultdict

D = "results/interp_validation"


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    def rank(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[o[j]] == v[o[i]]:
                j += 1
            for k in range(i, j):
                r[o[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


units = json.load(open(f"{D}/units.json"))
tu = [u for u in units if u["task"] != "__all__"]

vals = {}
for name in ["spider", "swebench", "bfcl", "specbench"]:
    per_task = defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0, 0, 0, 0])
    with gzip.open(f"{D}/curves_{name}.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l)
            t = r.get("task") or "all"
            d = per_task[t]
            for s, a in zip(r["s"], r["a"]):
                d[0] += s
                d[1] += max(a, 0)
                d[2] += 1
                d[3] += min(s, 32)
                d[4] += 1 if a >= 4 else 0
                d[5] += 1 if s >= 2 else 0
                d[6] += 1 if s >= 4 else 0
                d[7] += 1 if s >= 8 else 0
    for t, d in per_task.items():
        vals[(name, t)] = dict(mcd=d[3] / d[2], es=d[0] / d[2],
                               ea=d[1] / d[2], pred4=d[4] / d[2],
                               w2=d[5] / d[2], w4=d[6] / d[2], w8=d[7] / d[2])

rows = []
for u in tu:
    v = vals.get((u["wl"], u["task"]))
    if v:
        rows.append(dict(u, **v))

print(f"{'unit':<28}{'MCD':>6}{'E[a]':>6}{'a>=4':>6} |  gDr   gKim  gStr")
for r in rows:
    print(f"{r['wl'] + ':' + r['task']:<28}{r['mcd']:>6.2f}{r['ea']:>6.2f}"
          f"{r['pred4']:>6.1%} | {r['g_drlee']:+.2f}  {r['g_kim']:+.2f}"
          f"  {r['g_struct']:+.2f}")

print("\n== Spearman across task units ==")
for g, lab in [("g_drlee", "vs best_single (Dr.Lee frame)"),
               ("g_kim", "vs switch (Kim frame)"),
               ("g_struct", "structural (oracle-oracle)")]:
    ys = [r[g] for r in rows]
    print(f"  {lab:<30} MCD {spearman([r['mcd'] for r in rows], ys):+.2f}"
          f"   E[a] {spearman([r['ea'] for r in rows], ys):+.2f}"
          f"   P(s>=2) {spearman([r['w2'] for r in rows], ys):+.2f}"
          f"   P(s>=4) {spearman([r['w4'] for r in rows], ys):+.2f}"
          f"   P(s>=8) {spearman([r['w8'] for r in rows], ys):+.2f}")

for wl in ["swebench", "bfcl", "specbench"]:
    sub = [r for r in rows if r["wl"] == wl]
    if len(sub) < 4:
        continue
    line = f"  within {wl:<10}"
    for g in ["g_drlee", "g_kim"]:
        ys = [r[g] for r in sub]
        line += f"  {g}: MCD {spearman([r['mcd'] for r in sub], ys):+.2f}"
    print(line)

ms = [r["mcd"] for r in rows]
ma = [r["ea"] for r in rows]
print(f"\nfield variability across units: MCD {min(ms):.2f}-{max(ms):.2f} "
      f"(x{max(ms) / min(ms):.1f})   E[a] {min(ma):.2f}-{max(ma):.2f} "
      f"(x{max(ma) / min(ma):.1f})")

print("\ntoy:  k   MCD  E[a]    gDr   gKim  gStr")
for k in [0, 1, 2, 4, 8]:
    ssum = asum = n = 0
    with gzip.open(f"{D}/curves_ms_k{k}.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l)
            for s, a in zip(r["s"], r["a"]):
                ssum += min(s, 32)
                asum += max(a, 0)
                n += 1
    A = json.load(open(f"{D}/report_ms_k{k}.json"))["arms"]
    print(f"  {k}  {ssum / n:5.2f} {asum / n:5.2f}  "
          f"{A['compose']['K'] - max(A['dflash']['K'], A['suffix']['K']):+.2f}"
          f"  {A['compose']['K'] - A['switch_real']['K']:+.2f}"
          f"  {A['handoff_oracle']['K'] - A['switch_oracle']['K']:+.2f}")
