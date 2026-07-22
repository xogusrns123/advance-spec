#!/usr/bin/env python3
"""How many TASKS (= conversations; each = one full agent trajectory) went to the
warm corpus vs the eval set, per workload, on the ORIGINAL (restored) split.
Split is conversation-level parity (even->warm, odd->eval; label-rank for subtask
datasets). eval tasks are read exactly from the record; warm tasks from the
conv_map's warm-parity conversations (the mode is auto-detected by matching the
record's eval convs)."""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import glob
import json

REC = {"specbench": "results/perpos_specbench_alleval/specbench_4way",
       "bfcl": "results/perpos_bfcl_alleval/bfcl_4way",
       "swebench": "results/perpos_swebench_alleval/swebench_4way",
       "spider": "results/perpos_spider_alleval/spider_4way",
       "tau2": "results/perpos_tau2_alleval/tau2_4way"}
CMAP_DIR = {"specbench": "results/perpos_specbench_full",
            "bfcl": "results/perpos_bfcl_full",
            "swebench": "results/perpos_swebench_step250",
            "spider": "results/perpos_spider",
            "tau2": "results/perpos_tau2_full"}


def parity_conv(convs):
    return {c["conv"]: c["conv"] % 2 for c in convs}


def parity_labelrank(convs):
    rank, par = {}, {}
    for c in convs:                       # collection order
        l = c.get("label")
        par[c["conv"]] = rank.get(l, 0) % 2
        rank[l] = rank.get(l, 0) + 1
    return par


print(f"{'workload':<11}{'warm tasks':>11}{'eval tasks':>11}{'(warm calls)':>14}{'(eval calls)':>13}  split-mode")
for wl, stem in REC.items():
    t = json.load(open(stem + ".traces.json"))
    ev = t["eval_traces"]
    E = {e["conv"] for e in ev}
    warm_calls, eval_calls = len(t["warm_traces"]), len(ev)
    # pick a conv_map that covers the eval convs
    best = None
    for cm in sorted(glob.glob(f"{CMAP_DIR[wl]}/conv_map*.json")):
        try:
            d = json.load(open(cm))
        except Exception:
            continue
        convs = d.get("convs") or []
        ids = {c["conv"] for c in convs}
        if E <= ids:
            best = (cm, convs)
            break
    if not best:
        print(f"{wl:<11}{'?':>11}{len(E):>11}{warm_calls:>14}{eval_calls:>13}  (no conv_map superset)")
        continue
    cm, convs = best
    for mode, parf in (("conv", parity_conv), ("label-rank", parity_labelrank)):
        par = parf(convs)
        eval_par = {c for c, p in par.items() if p == 1}
        if E <= eval_par:                 # this mode's eval side contains all evaluated convs
            warm_tasks = sum(1 for p in par.values() if p == 0)
            print(f"{wl:<11}{warm_tasks:>11}{len(E):>11}{warm_calls:>14}{eval_calls:>13}"
                  f"  {mode}  (cmap={cm.split('/')[-1]})")
            break
    else:
        # fallback: warm = total convs - evaluated convs
        tot = len({c['conv'] for c in convs})
        print(f"{wl:<11}{tot - len(E):>11}{len(E):>11}{warm_calls:>14}{eval_calls:>13}"
              f"  total-minus-eval  (cmap={cm.split('/')[-1]})")
