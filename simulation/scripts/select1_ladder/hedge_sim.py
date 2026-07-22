"""OFFLINE (block-anchored) simulation of a HedgeSpec-style per-position FULL-INFORMATION
online hedge over our co-resident proposers, vs the static per-position selectors.

Motivation: HedgeSpec (Not-a-Bandit, arXiv 2510.20064) does online full-info drafter selection
because verify reveals all drafters' outcomes. We already observe every proposer's (token,gt-match)
at each position for free, so we can run a per-position hedge that adapts weights to which proposer
has been winning RECENTLY. This can only beat the static selector if "which proposer is the
decisively-correct one" is AUTOCORRELATED along the sequence. So we (1) measure that lag-1
autocorrelation, and (2) simulate the hedge's block-anchored MAT vs raw (eta=0) and oracle.

Hedge: per-position pick = argmax_i [ log prob_i - eta * R_i ], R_i = discounted cumulative
full-info loss (1 - [token_i==gt]) with decay delta, RESET per request (rid) like MetaSD's
per-query reinit. eta=0 -> argmax prob = raw. Sweep (eta, delta).

NOTE: block-anchored proxy (oracle trajectory). Realized validation pending GPU0 free.
Run: python3 simulation/scripts/select1_ladder/hedge_sim.py
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = [
    ("27B 2-way", "qwen35_27b_ar", [("eagle","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
    ("14B 2-way", "qwen3_14b_ar",  [("eagle","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
]

def load(path, props):
    raw = defaultdict(list)
    for line in open(path):
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            raw[(o["rid"], o["decode_step"])].append(o)
    by_rid = defaultdict(list)   # rid -> list of (decode_step, [pos dicts depth-sorted])
    for (rid, ds), rs in raw.items():
        rs.sort(key=lambda r: r["depth"]); pos = []
        for r in rs:
            e = {"depth": r["depth"], "gt": r.get("gt_token"), "P": {}}
            for nm, tk, pk in props:
                t = r.get(tk); p = r.get(pk)
                if t is not None and p is not None:
                    e["P"][nm] = (t, float(p))
            pos.append(e)
        by_rid[rid].append((ds, pos))
    for rid in by_rid:
        by_rid[rid].sort(key=lambda x: x[0])
    return by_rid

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
            if len(set(g))/max(len(g),1) < 0.5: bad.add(rid)
    return bad

def decisive_winner_seq(by_rid):
    """Per rid, sequence over DECISIVE positions of indicator: is suffix the unique gt-match?
    Returns lag-1 autocorrelation and base rate."""
    base_hits = 0; base_n = 0
    pairs_same = 0; pairs_n = 0
    for rid, blocks in by_rid.items():
        seq = []
        for ds, pos in blocks:
            for e in pos:
                gt = e["gt"]
                if gt is None: continue
                P = e["P"]; avail = list(P)
                hits = [nm for nm in avail if P[nm][0] == gt]
                if len(avail) >= 2 and 0 < len(hits) < len(avail):   # decisive
                    seq.append(1 if ("suffix" in hits) else 0)   # suffix is the right pick?
        base_hits += sum(seq); base_n += len(seq)
        for a, b in zip(seq, seq[1:]):
            pairs_same += (a == b); pairs_n += 1
    base = base_hits / max(base_n, 1)
    # P(same as prev) vs chance P(same)=base^2+(1-base)^2 ; and P(suffix|prev suffix)
    return base, (pairs_same / max(pairs_n, 1)), base_n, pairs_n

def cond_prob(by_rid):
    """P(suffix-correct at t | suffix-correct at t-1) and P(.. | eagle-correct at t-1) over decisive seq."""
    a_after_s = 0; n_after_s = 0; a_after_e = 0; n_after_e = 0
    for rid, blocks in by_rid.items():
        seq = []
        for ds, pos in blocks:
            for e in pos:
                gt = e["gt"]
                if gt is None: continue
                P = e["P"]; avail = list(P); hits = [nm for nm in avail if P[nm][0] == gt]
                if len(avail) >= 2 and 0 < len(hits) < len(avail):
                    seq.append(1 if ("suffix" in hits) else 0)
        for prev, cur in zip(seq, seq[1:]):
            if prev == 1: a_after_s += cur; n_after_s += 1
            else:         a_after_e += cur; n_after_e += 1
    return (a_after_s / max(n_after_s, 1)), (a_after_e / max(n_after_e, 1))

def hedge_mat(by_rid, eta, delta, hist_only=False, block_causal=True):
    """block-anchored MAT under the online hedge. eta=0 -> raw (argmax prob).
    block_causal=True (serving-honest): R is FROZEN within a block (only the previous
    verified blocks' outcomes are known when picking); R is updated from the block's
    positions only AFTER the block. block_causal=False leaks within-block future matches."""
    tot = 0; n = 0
    for rid, blocks in by_rid.items():
        R = defaultdict(float)            # discounted cumulative loss per proposer (reset per rid)
        for ds, pos in blocks:
            Rsnap = dict(R) if block_causal else None
            run = 0; alive = True
            for e in pos:
                gt = e["gt"]; P = e["P"]
                if gt is None or not P:
                    if alive: alive = False
                    continue
                present = list(P)
                Ruse = Rsnap if block_causal else R
                if hist_only:
                    score = {i: -eta * Ruse.get(i, 0.0) for i in present}
                else:
                    score = {i: math.log(max(P[i][1], 1e-9)) - eta * Ruse.get(i, 0.0) for i in present}
                pick = max(present, key=lambda i: score[i])
                if alive:
                    if P[pick][0] == gt: run += 1
                    else: alive = False
                if not block_causal:                    # leaky: update immediately
                    for i in present:
                        R[i] = delta * R[i] + (0.0 if P[i][0] == gt else 1.0)
            if block_causal:                            # honest: update only at block end
                for e in pos:
                    P = e["P"]; gt = e["gt"]
                    if gt is None or not P: continue
                    for i in P:
                        R[i] = delta * R[i] + (0.0 if P[i][0] == gt else 1.0)
            tot += run; n += 1
    return tot / max(n, 1)

def oracle_mat(by_rid):
    tot = 0; n = 0
    for rid, blocks in by_rid.items():
        for ds, pos in blocks:
            run = 0
            for e in pos:
                gt = e["gt"]; P = e["P"]
                if gt is None or not P: break
                if any(t == gt for t, _ in P.values()): run += 1
                else: break
            tot += run; n += 1
    return tot / max(n, 1)

def main():
    for name, d, props in CELLS:
        dd = f"{ROOT}/{d}"
        by_rid = load(f"{dd}/decisions_select1_oracle.jsonl", props)
        bad = loopy(dd); by_rid = {k: v for k, v in by_rid.items() if k not in bad}
        nblk = sum(len(b) for b in by_rid.values())
        base, psame, ndec, npair = decisive_winner_seq(by_rid)
        p_s_s, p_s_e = cond_prob(by_rid)
        print(f"\n{'='*78}\n### {name}  rids={len(by_rid)} blocks={nblk} decisive={ndec}")
        print(f"  base P(suffix is right pick | decisive) = {base:.3f}")
        print(f"  AUTOCORR: P(suffix-right | prev suffix-right)={p_s_s:.3f}  vs  P(suffix-right | prev eagle-right)={p_s_e:.3f}")
        print(f"           lift = {p_s_s - p_s_e:+.3f}   (>0 => decisive winner is autocorrelated => hedge can help)")
        orc = oracle_mat(by_rid)
        raw = hedge_mat(by_rid, eta=0.0, delta=1.0)
        print(f"  raw(eta=0) MAT={raw:.3f}   oracle MAT={orc:.3f}")
        print(f"  -- prob-primed hedge: argmax[log p_i - eta*R_i]; HONEST=block-causal, LEAK=within-block (upper bnd) --")
        print(f"     {'eta':>5} {'delta':>6} {'MAT honest':>11} {'dMAT':>7} | {'MAT leak':>9} {'dMAT':>7}")
        best_h = (raw, 0, 1); best_l = (raw, 0, 1)
        for eta in (0.3, 1.0, 3.0, 10.0):
            for delta in (0.5, 0.8, 0.95, 1.0):
                mh = hedge_mat(by_rid, eta, delta, block_causal=True)
                ml = hedge_mat(by_rid, eta, delta, block_causal=False)
                if mh > best_h[0]: best_h = (mh, eta, delta)
                if ml > best_l[0]: best_l = (ml, eta, delta)
                print(f"     {eta:>5.1f} {delta:>6.2f} {mh:>11.3f} {mh-raw:>+7.3f} | {ml:>9.3f} {ml-raw:>+7.3f}")
        print(f"  BEST honest hedge: MAT={best_h[0]:.3f} (eta={best_h[1]},delta={best_h[2]}) dMAT={best_h[0]-raw:+.3f}"
              f"  ||  best leaky(upper bnd): {best_l[0]:.3f} dMAT={best_l[0]-raw:+.3f}")

if __name__ == "__main__":
    main()
