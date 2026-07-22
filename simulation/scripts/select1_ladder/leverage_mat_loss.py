"""Per-depth MAT-loss + leverage analysis for chain select-1, and the leverage
weight curve used by the NEXT step (leverage-weighted selection training).

WHY: MAT is a run-length; a selection miss at depth d forfeits oracle's expected
remaining chain from d (= leverage(d)). leverage(d) falls ~7.6->1 over depth (27B),
while the raw miss rate is ~flat (~15-20%), so MAT loss concentrates at SHALLOW
depths: d<=2 = 52% (27B 3-way) / 61% (8B 3-way) of total raw->oracle MAT loss.
=> total selacc is info-capped, but MAT = sum_d leverage(d)*survival(d), so
REALLOCATING the fixed accuracy budget to shallow (high-leverage) positions raises
MAT without breaking the cap.

leverage(d) = E[oracle remaining chain length | oracle alive at depth d], measured
from the oracle decision log. Use it as sample_weight when fitting the selector
(candidate A in the plan). w(d)=max_depth-d or 1/(d+1) are cheap proxies.

Run: python3 simulation/scripts/select1_ladder/leverage_mat_loss.py
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = {
    "27B 3-way": ("qwen35_27b_3way_real_full",
                  [("mtp","eagle_token","eagle_p"),("dflash","dflash_token","dflash_p"),("suffix","suffix_token","suffix_p")]),
    "8B 3-way":  ("qwen3_8b_dflash_e3_ceiling20",
                  [("dflash","eagle_token","eagle_p"),("e3","e3_token","e3_p"),("suffix","suffix_token","suffix_p")]),
    "14B 2-way": ("qwen3_14b_ar", [("mtp","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
    "27B 2-way": ("qwen35_27b_ar", [("mtp","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
}

def load(path, props):
    raw = defaultdict(list)
    for line in open(path):
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            raw[(o["rid"], o["decode_step"])].append(o)
    B = {}
    for k, rs in raw.items():
        rs.sort(key=lambda r: r["depth"]); pos = []
        for r in rs:
            e = {"depth": r["depth"], "gt": r.get("gt_token"), "P": {}}
            for nm, tk, pk in props:
                t = r.get(tk); p = r.get(pk)
                if t is not None and p is not None:
                    e["P"][nm] = (t, float(p))
            pos.append(e)
        B[k] = pos
    return B

def loopy(d):
    dd = Path(d); gtf = dd / "gt_tokens.jsonl"; reqs = {}
    for line in open(dd / "decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "req": reqs[o["rid"]] = tuple(o["input_ids"])
    bad = set()
    if gtf.exists():
        gt = {}
        for line in open(gtf):
            r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
        for rid, ids in reqs.items():
            out = gt.get(ids)
            if not out or len(out) < 5: continue
            g = [tuple(out[i:i+4]) for i in range(len(out)-3)]
            if len(set(g)) / max(len(g), 1) < 0.5: bad.add(rid)
    return bad

def leverage_curve(blocks, maxd=16):
    """E[oracle remaining | oracle alive at d]."""
    lev_sum = np.zeros(maxd); reach = np.zeros(maxd)
    for pos in blocks.values():
        D = len(pos)
        for i, e in enumerate(pos):
            gt = e["gt"]
            if gt is None or not e["P"]: D = i; break
            if not any(t == gt for t, _ in e["P"].values()): D = i; break
        for i in range(min(D, maxd)):
            reach[i] += 1; lev_sum[i] += (D - i)
    return np.where(reach > 0, lev_sum / np.maximum(reach, 1), 0.0), reach

def main():
    for name, (dd, props) in CELLS.items():
        d = f"{ROOT}/{dd}"
        B = load(f"{d}/decisions_select1_oracle.jsonl", props)
        bad = loopy(d); B = {k: v for k, v in B.items() if k[0] not in bad}
        lev, reach = leverage_curve(B)
        nb = len(B)
        print(f"\n### {name}  blocks={nb}")
        print("  leverage(d):", " ".join(f"{x:.2f}" for x in lev[:16]))

if __name__ == "__main__":
    main()
