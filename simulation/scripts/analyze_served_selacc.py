"""REAL-SERVING decisive selection accuracy, read straight from each arm's own
served decision log (now carries oracle_hit + chosen under pin). Alive-conditioned
(walk chain, stop at first non-ALIVE). No estimation."""
from __future__ import annotations
import json, os
from collections import defaultdict
DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_ar"
ALIVE = {"eagle", "suffix", "both"}

def selacc(fname, label):
    p = f"{DIR}/{fname}"
    if not os.path.exists(p):
        print(f"{label:26s} (missing)"); return
    chains = defaultdict(list)
    for line in open(p):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains: chains[k].sort(key=lambda r: r["depth"])
    n = correct = 0
    for rs in chains.values():
        alive = True
        for r in rs:
            if not alive: break
            h = r.get("oracle_hit")
            if h in ("eagle", "suffix"):
                n += 1
                ps = (r.get("chosen") == "suffix")
                correct += (ps and h == "suffix") or ((not ps) and h == "eagle")
            if h not in ALIVE:
                alive = False
    print(f"{label:26s} decisive n={n:6d}  SERVED sel.acc = {correct/max(n,1):.4f}")

print(f"=== REAL-SERVING decisive selection accuracy ({DIR}) ===")
for f, lab in [
    ("decisions_select1.jsonl", "raw (sp>ep)"),
    ("decisions_select1_calib_histogram.jsonl", "calib histogram"),
    ("decisions_select1_calib_isotonic.jsonl", "calib isotonic"),
    ("decisions_select1_calib_logistic.jsonl", "calib logistic"),
    ("decisions_select1_calib_beta.jsonl", "calib beta"),
    ("decisions_select1_disc_beta.jsonl", "disc beta (Bayes-ceiling proxy)"),
    ("decisions_select1_oracle.jsonl", "oracle (GT)"),
]:
    selacc(f, lab)
