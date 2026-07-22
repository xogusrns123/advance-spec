#!/usr/bin/env python3
"""Micro-check of the multislot bridge mechanism on real workloads: at oracle
rounds that START inside a short cold gap (label 'g'), does the chosen head
length k track the remaining gap length (head = slot bridge, tail = resync)?

Reads curves_{name}.jsonl.gz + rounds_{name}.jsonl.gz from validate_interpretations.

  PYTHONPATH=/workspace python3 scripts/check_bridge_micro.py \
      --dir results/interp_validation --names bfcl swebench spider specbench --theta 4
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


def spans(curve, theta):
    s = curve["s"]
    w = [1 if v >= theta else 0 for v in s]
    seg = {}                      # pos-index -> (run_start, run_end_excl, is_warm)
    i, L = 0, len(s)
    while i < L:
        j = i
        while j < L and w[j] == w[i]:
            j += 1
        for p in range(i, j):
            seg[p] = (i, j, w[i])
        i = j
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--theta", type=int, default=4)
    args = ap.parse_args()

    for name in args.names:
        curves = {}
        with gzip.open(os.path.join(args.dir, f"curves_{name}.jsonl.gz"), "rt") as f:
            for l in f:
                r = json.loads(l)
                curves[r["rid"]] = r
        segs = {rid: spans(cu, args.theta) for rid, cu in curves.items()}

        rows = []
        with gzip.open(os.path.join(args.dir, f"rounds_{name}.jsonl.gz"), "rt") as f:
            for l in f:
                r = json.loads(l)
                if r["mode"] == "oracle" and r["lab"] == "g":
                    rows.append(r)

        n = pear_n = 0
        ks, gaps, tails = [], [], []
        exact = within2 = resync = 0
        for r in rows:
            seg = segs[r["rid"]].get(r["p"] - 1)
            if seg is None:
                continue
            b, e, iw = seg
            if iw:
                continue
            rem = e - (r["p"] - 1)               # remaining gap from round start
            ks.append(r["k"]); gaps.append(rem); tails.append(r["tail"])
            if r["k"] == rem:
                exact += 1
            if abs(r["k"] - rem) <= 2:
                within2 += 1
            if r["k"] >= rem and r["tail"] > 0:
                resync += 1                       # head crossed gap AND tail resynced
            n += 1
        if not n:
            print(f"{name}: no oracle 'g' rounds")
            continue
        mk = sum(ks) / n
        mg = sum(gaps) / n
        mx = sum(x * y for x, y in zip(ks, gaps)) / n
        vx = sum(x * x for x in ks) / n - mk * mk
        vy = sum(y * y for y in gaps) / n - mg * mg
        pear = (mx - mk * mg) / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else float("nan")
        print(f"{name}: {n} oracle slot('g') rounds | mean k={mk:.1f} vs remaining "
              f"gap={mg:.1f} | corr(k, gap)={pear:+.2f} | k==gap {exact/n:.0%}, "
              f"|k-gap|<=2 {within2/n:.0%} | bridged+resynced (k>=gap & tail>0) "
              f"{resync/n:.0%} | mean tail after bridge "
              f"{(sum(t for x, g_, t in zip(ks, gaps, tails) if x >= g_) / max(1, sum(1 for x, g_ in zip(ks, gaps) if x >= g_))):.1f}")


if __name__ == "__main__":
    main()
