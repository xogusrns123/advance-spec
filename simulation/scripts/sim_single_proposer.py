"""Offline single-proposer (EAGLE3-only / suffix-only) accept-length simulation
over the GT trajectory recorded by an ORACLE chain-hybrid run.

A single proposer does NO per-depth cross-proposer re-selection, so its MAT needs
no real serving: along the fixed GT trajectory each proposer's accepted length is
just where its proposed tokens stop matching GT. We:
  1. reconstruct each absolute GT position's eagle/suffix/gt token from the oracle
     decision log. position = L_k + depth, where L_k is the step's start
     (L_0=0, L_{k+1}=L_k + accept_len_k + 1 from the step records). At (k,d) the
     committed prefix is GT[0..L_k+d-1] (oracle commits GT), so the proposer token
     there is its proposal given GT[0..p-1] — well-defined per absolute position.
  2. RE-CHUNK greedily by EACH proposer's OWN acceptance (advance pos by
     accept_len+1) — its own step structure, NOT the oracle arm's chunking
     (which over-accepts and biases a naive consec count).

Validation: the simulated EAGLE3-only MAT should match the separately-measured
real `baseline` arm. Writes timing_{baseline,suffix}.jsonl (accept_lengths schema
plot_o4 reads) into --out-dir and records the MATs in run.json (marked simulated).

Usage:
  python3 simulation/scripts/sim_single_proposer.py \
    --oracle-log <dir>/decisions_select1_oracle.jsonl --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    accept_len = {}
    dec = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "step":
                accept_len[(r["rid"], r["decode_step"])] = int(r["accept_len"])
            elif r.get("type") == "decision" and not r.get("tail"):
                dec[r["rid"]].append(r)
    return accept_len, dec


def rechunk(pos_tok, pos_gt, maxpos):
    """greedy single-proposer accept lengths over a contiguous GT position map."""
    accs = []
    pos = 0
    while pos <= maxpos and pos in pos_gt:
        a = 0
        while (pos + a) in pos_gt and pos_tok.get(pos + a) is not None \
                and pos_tok[pos + a] == pos_gt[pos + a]:
            a += 1
        accs.append(a)
        pos += a + 1
    return accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-log", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    accept_len, dec = load(args.oracle_log)

    eag_all, suf_all = [], []
    n_gap = 0
    for rid, decs in dec.items():
        bystep = defaultdict(dict)
        for r in decs:
            bystep[r["decode_step"]][int(r["depth"])] = r
        L = 0
        pe, ps, pg = {}, {}, {}
        maxpos = -1
        for k in sorted(bystep):
            for d, r in bystep[k].items():
                p = L + d
                gt = r.get("gt_token")
                if gt is None:
                    continue
                pg[p] = gt
                pe[p] = r.get("eagle_token")
                ps[p] = r.get("suffix_token")
                if p > maxpos:
                    maxpos = p
            al = accept_len.get((rid, k))
            if al is None:
                break
            L = L + al + 1
        # contiguity check (positions 0..maxpos should mostly be present)
        if maxpos >= 0:
            miss = sum(1 for p in range(maxpos + 1) if p not in pg)
            n_gap += miss
        eag_all += rechunk(pe, pg, maxpos)
        suf_all += rechunk(ps, pg, maxpos)

    eag = np.asarray(eag_all, float)
    suf = np.asarray(suf_all, float)
    print(f"requests={len(dec)}  position-map gaps={n_gap}")
    print(f"EAGLE3-only (sim, re-chunked): steps={len(eag)} MAT={eag.mean():.3f}")
    print(f"suffix-only (sim, re-chunked): steps={len(suf)} MAT={suf.mean():.3f}")
    for d in (1, 2, 3, 5, 8):
        print(f"  survival>=%d: eagle=%.3f suffix=%.3f"
              % (d, (eag >= d).mean(), (suf >= d).mean()))

    out = Path(args.out_dir)
    for arm, arr in (("baseline", eag), ("suffix", suf)):
        with open(out / f"timing_{arm}.jsonl", "w") as f:
            for v in arr:
                f.write(json.dumps({"phase": "decode",
                                    "accept_lengths": [int(v)]}) + "\n")
    rj = out / "run.json"
    s = json.load(open(rj))
    s["arms"]["baseline"] = {"accept_length_mean": float(eag.mean()),
                             "n_samples": int(len(eag)), "simulated": True}
    s["arms"]["suffix"] = {"accept_length_mean": float(suf.mean()),
                           "n_samples": int(len(suf)), "simulated": True}
    json.dump(s, open(rj, "w"), indent=2)
    print(f"wrote timing_baseline/suffix.jsonl + updated {rj} (marked simulated)")


if __name__ == "__main__":
    main()
