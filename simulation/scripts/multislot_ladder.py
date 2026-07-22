"""Selection ladder on Dr.Lee's multislot per-position data (DFlash vs Suffix,
warm vs cold, across novelty k). Tests whether a WARM suffix corpus widens the
DFlash->oracle gap and whether a realistic selector captures it.

Each perpos row = one committed position:
  a_d  = DFlash block accept length (leading 1s in dflash_match)
  a_sw = suffix_match_warm  (warm-corpus suffix greedy-walk accept length)
  a_sc = suffix_match_cold
  c_d  = dflash_conf[0]      (model confidence in its first token)  [select signal]
  T_w  = suffix_T_warm       (suffix warmth score)                  [select signal]

Ladder per k: dflash-only, suffix-only(cold/warm), realistic selector
(calibrated E[accept|signal], LOO by prompt), oracle = per-position max.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from collections import defaultdict
from sklearn.isotonic import IsotonicRegression


def _ad(match):
    i = 0
    while i < len(match) and match[i] == 1:
        i += 1
    return i


def _fit(sig, y):
    if len(set(sig)) < 3:
        m = float(np.mean(y)) if len(y) else 0.0
        return lambda v: m
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(np.asarray(sig, float), np.asarray(y, float))
    return lambda v: float(ir.predict([v])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="simulation/Dr.Lee Solution/results/perpos/multislot_k*.jsonl")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    print(f"files={len(files)}\n")
    hdr = (f"{'k':>4}{'npos':>6}{'dflash':>8}{'sfx_cold':>9}{'sfx_warm':>9}"
           f"{'sel_warm':>9}{'orac_cold':>10}{'orac_warm':>10}"
           f"{'gap_c%':>8}{'gap_w%':>8}{'sel_w%':>8}")
    print(hdr)
    for fn in files:
        k = os.path.basename(fn).replace("multislot_k", "").replace(".jsonl", "")
        rows = [json.loads(l) for l in open(fn) if l.strip()]
        by_rid = defaultdict(list)
        recs = []
        for r in rows:
            a_d = _ad(r["dflash_match"])
            a_sw = int(r.get("suffix_match_warm", 0))
            a_sc = int(r.get("suffix_match_cold", 0))
            c_d = float(r["dflash_conf"][0]) if r.get("dflash_conf") else 0.0
            T_w = float(r.get("suffix_T_warm", 0.0))
            rec = dict(rid=r["rid"], a_d=a_d, a_sw=a_sw, a_sc=a_sc, c_d=c_d, T_w=T_w)
            recs.append(rec); by_rid[r["rid"]].append(rec)

        ad = np.array([r["a_d"] for r in recs]); asw = np.array([r["a_sw"] for r in recs])
        asc = np.array([r["a_sc"] for r in recs])
        oc = np.maximum(ad, asc); ow = np.maximum(ad, asw)

        # realistic warm selector: calibrate E[a_d|c_d], E[a_sw|T_w], LOO by prompt(rid)
        rids = sorted(by_rid)
        sel = []
        for held in rids:
            tr = [r for r in recs if r["rid"] != held]
            fd = _fit([r["c_d"] for r in tr], [r["a_d"] for r in tr])
            fs = _fit([r["T_w"] for r in tr], [r["a_sw"] for r in tr])
            for r in by_rid[held]:
                pick_s = fs(r["T_w"]) > fd(r["c_d"])
                sel.append(r["a_sw"] if pick_s else r["a_d"])
        sel = np.array(sel)

        dm, scm, swm = ad.mean(), asc.mean(), asw.mean()
        ocm, owm, selm = oc.mean(), ow.mean(), sel.mean()
        best_c = max(dm, scm); best_w = max(dm, swm)
        gapc = 100 * (0) if ocm <= best_c else 100 * (ocm - best_c) / best_c
        gapw = 100 * (owm - best_w) / best_w if best_w > 0 else 0.0
        # selector gap-closed over best-single-warm toward oracle-warm
        selw = 100 * (selm - best_w) / (owm - best_w) if owm > best_w else 0.0
        print(f"{k:>4}{len(recs):>6}{dm:>8.3f}{scm:>9.3f}{swm:>9.3f}{selm:>9.3f}"
              f"{ocm:>10.3f}{owm:>10.3f}{gapc:>7.1f}%{gapw:>7.1f}%{selw:>7.1f}%")


if __name__ == "__main__":
    main()
