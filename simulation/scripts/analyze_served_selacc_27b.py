"""REAL-SERVING decisive selection accuracy + MAT, 27B MTP cell. Reads each arm's
own served decision log (oracle_hit + chosen, logged under pin) — no estimation.
calib arms use the perdepth cond-trained filenames (*_cond-trained)."""
from __future__ import annotations
import json, os
from collections import defaultdict
DIR = "simulation/results/chain_hybrid_perdepth/qwen35_27b_ar"
ALIVE = {"eagle", "suffix", "both"}

def stats(fname, label):
    p = f"{DIR}/{fname}"
    if not os.path.exists(p):
        print(f"{label:30s} (missing {fname})"); return
    chains = defaultdict(list); acc = {}
    for line in open(p):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
        elif o.get("type") == "step":
            acc[(o["rid"], o["decode_step"])] = o.get("accept_len")
    for k in chains: chains[k].sort(key=lambda r: r["depth"])
    n = correct = 0; Ls = []
    for k, rs in chains.items():
        alive = True
        for r in rs:
            if not alive: break
            h = r.get("oracle_hit")
            if h in ("eagle", "suffix"):
                n += 1
                ps = (r.get("chosen") == "suffix")
                correct += (ps and h == "suffix") or ((not ps) and h == "eagle")
            if h not in ALIVE: alive = False
        if k in acc and acc[k] is not None:
            Ls.append(acc[k])
    sa = correct / max(n, 1)
    mat = sum(Ls) / max(len(Ls), 1)
    print(f"{label:30s} dec.n={n:6d}  sel.acc={sa:.4f}  MAT={mat:.4f}")

print(f"=== 27B MTP REAL-SERVING (held-out) — {DIR} ===")
for f, lab in [
    ("decisions_select1.jsonl", "raw (sp>ep)"),
    ("decisions_select1_calib_histogram_cond-trained.jsonl", "calib histogram"),
    ("decisions_select1_calib_isotonic_cond-trained.jsonl", "calib isotonic"),
    ("decisions_select1_calib_logistic_cond-trained.jsonl", "calib logistic"),
    ("decisions_select1_calib_beta_cond-trained.jsonl", "calib beta"),
    ("decisions_select1_disc_beta.jsonl", "disc beta (Bayes-ceiling)"),
    ("decisions_select1_oracle.jsonl", "oracle (GT)"),
]:
    stats(f, lab)
