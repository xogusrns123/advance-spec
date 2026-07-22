"""Why does calibration recover the MAT gap on some cells but not others?
Per cell, on DECISIVE-ALIVE positions, measure per-proposer:
  hit-rate   = P(proposer token == gt)            -> is anyone reliable when it matters
  AUC        = roc_auc(prob -> hit)               -> does the prob RANK its own correctness (info)
  meanp      = mean raw prob                       -> confidence level
  overconf   = meanp - hit-rate                    -> miscalibration magnitude (fixable scale)
  dp(c-w)    = mean prob|correct - mean prob|wrong -> separation in prob units
The hypothesis: calibration recovers the gap when SOME proposer has BOTH high
reliability AND high AUC (info present + mis-scaled). Low AUC everywhere = info-capped.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = {
    "14B 2-way (eagle3+sfx)": {"dir": "qwen3_14b_ar",
        "props": [("eagle3", "eagle_token", "eagle_p"), ("suffix", "suffix_token", "suffix_p")]},
    "27B 2-way (mtp+sfx)": {"dir": "qwen35_27b_ar",
        "props": [("mtp", "eagle_token", "eagle_p"), ("suffix", "suffix_token", "suffix_p")]},
    "8B 3-way (e3+df+sfx)": {"dir": "qwen3_8b_dflash_e3_ceiling20",
        "props": [("dflash", "eagle_token", "eagle_p"), ("eagle3", "e3_token", "e3_p"),
                  ("suffix", "suffix_token", "suffix_p")]},
    "27B 3-way (mtp+df+sfx)": {"dir": "qwen35_27b_3way_real_full",
        "props": [("mtp", "eagle_token", "eagle_p"), ("dflash", "dflash_token", "dflash_p"),
                  ("suffix", "suffix_token", "suffix_p")]},
}
RECOVERY = {"14B 2-way (eagle3+sfx)": 8.6, "27B 2-way (mtp+sfx)": 42.3,
            "8B 3-way (e3+df+sfx)": 3.5, "27B 3-way (mtp+df+sfx)": 30.4}


def load_chains(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains:
        chains[k].sort(key=lambda r: r["depth"])
    return chains


def loopy_rids(d, thresh=0.5, n=4):
    dd = Path(d)
    gt = {}
    gtf = dd / "gt_tokens.jsonl"
    if not gtf.exists():
        return set()
    reqs = {}
    for line in open(dd / "decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "req":
            reqs[o["rid"]] = tuple(o["input_ids"])
    for line in open(gtf):
        r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
    bad = set()
    for rid, ids in reqs.items():
        out = gt.get(ids)
        if not out or len(out) < n + 1:
            continue
        g = [tuple(out[i:i + n]) for i in range(len(out) - n + 1)]
        if len(set(g)) / max(len(g), 1) < thresh:
            bad.add(rid)
    return bad


def analyze(name, cfg):
    d = f"{ROOT}/{cfg['dir']}"
    chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
    bad = loopy_rids(d)
    chains = {k: v for k, v in chains.items() if k[0] not in bad}
    props = cfg["props"]
    coll = {p[0]: {"p": [], "h": []} for p in props}
    n_dec = 0; n_pos = 0
    for rs in chains.values():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            toks = {p[0]: r.get(p[1]) for p in props}
            probs = {p[0]: (r.get(p[2]) if r.get(p[2]) is not None else None) for p in props}
            avail = [p[0] for p in props if toks[p[0]] is not None]
            hits = [nm for nm in avail if gt is not None and toks[nm] == gt]
            n_pos += 1
            if 0 < len(hits) < len(avail):
                n_dec += 1
                for nm in avail:
                    if probs[nm] is None:
                        continue
                    coll[nm]["p"].append(float(probs[nm]))
                    coll[nm]["h"].append(1 if nm in hits else 0)
            if gt is not None and len(hits) == 0:
                alive = False
    print(f"\n=== {name}   [MAT-gap recovered (bayes): {RECOVERY[name]:.1f}%]   "
          f"loopy-excl={len(bad)}  decisive={n_dec}/{n_pos} ({100*n_dec/max(n_pos,1):.0f}%) ===")
    print(f"  {'proposer':10} {'n':>6} {'hit-rate':>9} {'AUC':>7} {'meanP':>7} "
          f"{'overconf':>9} {'dp(c-w)':>8}")
    best_auc = 0.0; best_rel = 0.0
    for nm, _, _ in props:
        h = np.array(coll[nm]["h"]); p = np.array(coll[nm]["p"])
        if len(h) < 20 or h.min() == h.max():
            print(f"  {nm:10} {len(h):>6}  (degenerate)"); continue
        hr = h.mean(); meanp = p.mean()
        auc = roc_auc_score(h, p)
        dp = p[h == 1].mean() - p[h == 0].mean()
        print(f"  {nm:10} {len(h):>6} {hr:>9.3f} {auc:>7.3f} {meanp:>7.3f} "
              f"{meanp-hr:>+9.3f} {dp:>+8.3f}")
        best_auc = max(best_auc, auc); best_rel = max(best_rel, hr)
    print(f"  -> best proposer: reliability={best_rel:.3f}  AUC={best_auc:.3f}")
    return best_rel, best_auc


print("DECISIVE-position proposer reliability & confidence-informativeness per cell")
rows = []
for name, cfg in CELLS.items():
    try:
        br, ba = analyze(name, cfg)
        rows.append((name, RECOVERY[name], br, ba))
    except Exception as e:
        print(f"\n=== {name}: ERROR {e} ===")

print("\n\n=== SUMMARY: does best-proposer reliability/AUC track MAT recovery? ===")
print(f"  {'cell':24} {'recover%':>9} {'best-rel':>9} {'best-AUC':>9}")
for name, rec, br, ba in sorted(rows, key=lambda x: x[1]):
    print(f"  {name:24} {rec:>8.1f}% {br:>9.3f} {ba:>9.3f}")
