#!/usr/bin/env python3
"""Trim warm_traces to the eval-set size (warm:eval = 1:1) on the canonical
records. Backs up the original .traces.json to *.traces.json.orig (idempotent:
always trims from the backup). Eval set is untouched (can't grow it without a
fresh dflash capture). Deterministic subsample (seed 0)."""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json, os, random

RECS = {"specbench": "results/perpos_specbench_alleval/specbench_4way",
        "bfcl": "results/perpos_bfcl_alleval/bfcl_4way",
        "swebench": "results/perpos_swebench_alleval/swebench_4way",
        "spider": "results/perpos_spider_alleval/spider_4way",
        "tau2": "results/perpos_tau2_alleval/tau2_4way",
        # interp-family records (feed units.json + boundary figures)
        "specbench_interp": "results/perpos_specbench_full/specbench",
        "bfcl_interp": "results/perpos_bfcl_full/bfcl_v4_full"}

for wl, stem in RECS.items():
    tp = stem + ".traces.json"
    orig = tp + ".orig"
    src = orig if os.path.exists(orig) else tp
    t = json.load(open(src))
    if not os.path.exists(orig):
        os.rename(tp, orig)
    w, e = t["warm_traces"], t["eval_traces"]
    ne = len(e)
    if len(w) > ne:
        random.seed(0)
        t["warm_traces"] = random.sample(w, ne)
    nw = len(t["warm_traces"])
    json.dump(t, open(tp, "w"))
    print("%-10s warm %d->%d  eval %d  ratio %.2f:1" % (wl, len(w), nw, ne, nw / ne))
print("DONE (originals -> *.traces.json.orig)")
