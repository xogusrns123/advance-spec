#!/usr/bin/env python3
"""Subset an existing capture (record jsonl + .traces.json) to the first N eval
conversations per task label (and optionally the first M calls per conv) — the
zero-GPU equivalent of re-running capture_traj.py with --per-task-eval-convs /
--max-calls-per-conv on an already-collected record. Warm traces are unchanged.

  python3 scripts/subset_record.py --record results/perpos_bfcl_full/bfcl_v4_full.jsonl \
      --per-task-convs 3 --out results/perpos_bfcl_full/bfcl_v4_sub3.jsonl
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task-convs", type=int, default=0, help="0 = all")
    ap.add_argument("--max-calls-per-conv", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rp = Path(args.record)
    traces = json.load(open(rp.with_suffix(".traces.json")))
    ev = traces["eval_traces"]                      # capture temporal order

    order, lab_of = [], {}
    for t in ev:
        c = t.get("conv", t["rid"])
        if c not in lab_of:
            order.append(c); lab_of[c] = t.get("task", "")
    keep = set(order)
    if args.per_task_convs:
        cnt, keep = {}, set()
        for c in order:
            l = lab_of[c]
            if cnt.get(l, 0) < args.per_task_convs:
                keep.add(c); cnt[l] = cnt.get(l, 0) + 1

    ncall, keep_rids = {}, set()
    ev_out = []
    for t in ev:
        c = t.get("conv", t["rid"])
        if c not in keep:
            continue
        ncall[c] = ncall.get(c, 0) + 1
        if args.max_calls_per_conv and ncall[c] > args.max_calls_per_conv:
            continue
        keep_rids.add(t["rid"]); ev_out.append(t)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(outp, "w") as f:
        for l in open(rp):
            l = l.strip()
            if not l:
                continue
            if json.loads(l)["rid"] in keep_rids:
                f.write(l + "\n"); n += 1
    json.dump({**traces, "eval_traces": ev_out}, open(outp.with_suffix(".traces.json"), "w"))
    from collections import Counter
    per = Counter(lab_of[t.get("conv", t["rid"])] for t in ev_out)
    print(f"subset: {len(keep)} convs / {len(ev_out)} calls / {n} records "
          f"(per task: {dict(per)}) -> {outp}")


if __name__ == "__main__":
    main()
