#!/usr/bin/env python3
"""WHY do target_p and token_gt calibration give near-identical MAT (esp. hist/iso)?

Three empirical probes:
 (A) q_target peakedness — if the target softmax is peaked (q_gt~1, others~0), the
     CONTINUOUS target collapses onto the BINARY token==gt target, so any fitter that
     tracks the target (histogram/isotonic) learns near-identical maps.
 (B) per-method curve distance between the two objectives' maps (def=token_gt vs
     tp=target_p), over the shared x-grid, per (group, depth). Hypothesis: hist/iso
     tiny; logistic/beta larger (different FIT PROCEDURE: LogisticRegression on
     binary vs LinearRegression on continuous).
 (C) selection-flip rate: replay a decision log through BOTH map sets and count how
     often the pick (suffix vs eagle, i.e. sign of calib_suffix - calib_eagle)
     changes. MAT can only differ where the pick flips → this bounds the MAT gap.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

BASE = Path("simulation/results/o4_perdepth")
METHODS = ["histogram", "isotonic", "logistic", "beta"]


def load_map(d, method):
    return json.load(open(BASE / d / f"calib_pp_{method}.json"))["groups"]


def interp(curve, p):
    x, y = curve["x"], curve["y"]
    # x is a uniform 0..1 grid of 501 pts; nearest-bin lookup (matches serving)
    n = len(x)
    i = min(n - 1, max(0, int(round(p * (n - 1)))))
    return y[i]


def probeA():
    rows = [json.loads(l) for l in open(BASE / "qwen3_14b_tp_train" / "target_probs.jsonl")]
    print(f"(A) q_target peakedness  (n={len(rows)})")
    for key in ("q_gt", "q_eagle", "q_suffix"):
        v = sorted(r[key] for r in rows if r.get(key) is not None)
        n = len(v)
        lo = sum(1 for x in v if x < 0.1) / n
        hi = sum(1 for x in v if x > 0.9) / n
        mid = 1 - lo - hi
        mean = sum(v) / n
        med = v[n // 2]
        print(f"  {key:9s} mean={mean:.3f} median={med:.3f} | "
              f"frac<0.1={lo:.3f}  0.1-0.9={mid:.3f}  >0.9={hi:.3f}")
    # binary-collapse check: when the drafted token IS gt (q==q_gt), what is q?
    eag_is_gt = [r for r in rows if r["q_eagle"] is not None and abs(r["q_eagle"] - r["q_gt"]) < 1e-6]
    eag_not = [r for r in rows if r["q_eagle"] is not None and abs(r["q_eagle"] - r["q_gt"]) >= 1e-6]
    if eag_is_gt:
        print(f"  eagle==gt rows: {len(eag_is_gt)/sum(1 for r in rows if r['q_eagle'] is not None):.3f} "
              f"of eagle; mean q_eagle there={sum(r['q_eagle'] for r in eag_is_gt)/len(eag_is_gt):.3f} (→1 means peaked)")
    if eag_not:
        print(f"  eagle!=gt rows: mean q_eagle there={sum(r['q_eagle'] for r in eag_not)/len(eag_not):.3f} (→0 means peaked)")


def probeB():
    print("\n(B) per-method curve distance  token_gt(def) vs target_p(tp)")
    print(f"  {'method':10s} {'group':7s}  mean|Δy|  max|Δy|   (avg over 16 depths)")
    for m in METHODS:
        dg = load_map("qwen3_14b_def", m)
        tg = load_map("qwen3_14b_tp", m)
        for grp in ("eagle", "suffix"):
            md = mx = 0.0; cnt = 0
            for dep in dg[grp]:
                if dep not in tg[grp]:
                    continue
                yd, yt = dg[grp][dep]["y"], tg[grp][dep]["y"]
                diffs = [abs(a - b) for a, b in zip(yd, yt)]
                md += sum(diffs) / len(diffs); mx = max(mx, max(diffs)); cnt += 1
            if cnt:
                print(f"  {m:10s} {grp:7s}  {md/cnt:.4f}   {mx:.4f}")


def probeC(decision_log):
    if not Path(decision_log).exists():
        print(f"\n(C) skipped — decision log not found: {decision_log}")
        return
    print(f"\n(C) selection-flip rate on {decision_log}")
    # keep only compact tuples for decisions that CAN flip (both candidates present)
    rows = []  # (depth_str, eagle_p, suffix_p)
    n_total = n_eagle_only = 0
    for l in open(decision_log):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("type") != "decision" or r.get("tail"):
            continue
        n_total += 1
        if r.get("eagle_p") is None or r.get("suffix_p") is None:
            n_eagle_only += 1
            continue
        rows.append((str(int(r["depth"])), r["eagle_p"], r["suffix_p"]))
    print(f"  decisions total={n_total}  eagle-only(no suffix cand)={n_eagle_only/max(n_total,1):.3f}  "
          f"contested(both)={len(rows)/max(n_total,1):.3f}")
    for m in METHODS:
        dg = load_map("qwen3_14b_def", m)
        tg = load_map("qwen3_14b_tp", m)
        flips = same_pick = both = tg_suf = tp_suf = 0
        for dep, ep, sp in rows:
            if dep not in dg["eagle"] or dep not in dg["suffix"]:
                continue
            both += 1
            # pick suffix iff calib(suffix) > calib(eagle), under each objective's maps
            p_tg = interp(dg["suffix"][dep], sp) > interp(dg["eagle"][dep], ep)
            p_tp = interp(tg["suffix"][dep], sp) > interp(tg["eagle"][dep], ep)
            tg_suf += p_tg; tp_suf += p_tp
            if p_tg == p_tp:
                same_pick += 1
            else:
                flips += 1
        if both:
            print(f"  {m:10s} contested={both}  same-pick={same_pick/both:.4f}  "
                  f"FLIP={flips/both:.4f}  | picks-suffix: token_gt={tg_suf/both:.3f} "
                  f"target_p={tp_suf/both:.3f}")


if __name__ == "__main__":
    probeA()
    probeB()
    # use the token_gt eval decision log (shared trajectory T_def) if present
    dl = sys.argv[1] if len(sys.argv) > 1 else str(BASE / "qwen3_14b_def" / "decisions_select1.jsonl")
    probeC(dl)
