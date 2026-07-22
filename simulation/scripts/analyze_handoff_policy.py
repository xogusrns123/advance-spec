#!/usr/bin/env python3
"""Deployable hazard hand-off policy vs the O3 oracle (chain head=DFlash, tail=suffix).

The O3 chain-handoff oracle (analyze_chain_handoff.py) picks the head length j
per position WITH oracle knowledge of the realized accepts: O3 = max_j val(t,j,k).
This script evaluates the DEPLOYABLE closed-form rule from the Extension write-up,
which picks the head length using only signals available at serving time:

  hazard a_k        = the DFlash draft model's own confidence dflash_p at depth k-1
  survival   S_m    = a_1 a_2 ... a_m
  head value G_m    = S_1 + ... + S_m
  tail value T      = a scalar estimate of the suffix's expected accepted tokens
  Value(m)          = G_m + S_m * T          (head value + surviving tail value)

  threshold policy  : extend head while a_k >= a* with a* = T/(T+1); stop at the
                      first depth that fails (= argmax Value if hazards are
                      monotone non-increasing).
  argmax  policy    : m* = argmax_{m=0..K_head} Value(m)  (robust to non-monotone).

The chosen head length m is then SCORED on the ground truth with the exact same
value function the oracle uses, so the only difference vs O3 is the selection
signal (deployable confidence + scalar T) vs oracle hindsight:

  val(ae, asv, j, k) = min(j,ae,k) + [j<=min(ae,k)] * min(asv[j], k-j)

i.e. if the policy over-commits the head past the true DFlash accept length ae,
it loses the suffix tail entirely (val falls back to min(ae,k)). MATs are reported
two ways: pos (uniform per-position mean) and traj (renewal walk = real decode).

Hazards (dflash_p) are identical across the three suffix-corpus regimes (the DFlash
main chain is unchanged); only the suffix table A_s and the tail value T differ.

Usage:
  python3 simulation/scripts/analyze_handoff_policy.py \
      --table   .../qwen35_27b_dflash/specbench_dense.jsonl.gz \
      --dflash-proposals .../dflash_proposals_dense.jsonl \
      --label online --ks 8,16,inf
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict


def load_table(path, variant):
    """(rid,ci) -> {pos: (ae, asv, el, gcl0)} with asv[j]=A_s(t,j) for `variant`,
    gcl0 = the suffix draft's proposed greedy-chain length at j=0 (sfx col 4) —
    a DEPLOYABLE per-position tail-length estimate (known before verification)."""
    col = 1 if variant == "orc" else 2  # grd = col 2
    seqs = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            sfx = r["sfx"]
            asv = {s[0]: s[col] for s in sfx}
            gcl0 = sfx[0][4] if sfx else 0   # greedy_chain_len at j=0
            score0 = sfx[0][6] if sfx else 0.0  # suffix RUN score at j=0 = T(t)
            seqs[(r["rid"], r["ci"])][r["pos"]] = (r["ae"], asv, r["el"],
                                                   gcl0, score0)
    return seqs


def load_haz(path):
    """(rid, decode_step) -> [dflash_p ordered by depth]."""
    tmp = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tmp[(r["rid"], r["decode_step"])][r["depth"]] = r["dflash_p"]
    return {k: [dm[d] for d in sorted(dm)] for k, dm in tmp.items()}


def val(ae, asv, j, k):
    je = min(j, ae, k)
    if j <= min(ae, k) and j in asv:
        return je + min(asv[j], k - j)
    return je


def o3_accept(ae, asv, k):
    return max(val(ae, asv, j, k) for j in range(0, min(ae, k) + 1))


def policy_threshold(haz, astar):
    m = 0
    for p in haz:
        if p >= astar:
            m += 1
        else:
            break
    return m


def policy_argmax(haz, T):
    """m* = argmax_{m=0..len(haz)} Value(m); Value(0)=T (head 0 -> tail only)."""
    S, G, best, bestm = 1.0, 0.0, T, 0
    for m, p in enumerate(haz, start=1):
        S *= p
        G += S
        v = G + S * T
        if v > best:
            best, bestm = v, m
    return bestm


def walk(posmap, accept_fn):
    """Renewal walk: pos -> pos+accept+1; missing pos advances 1 with accept 0."""
    if not posmap:
        return 0, 0
    pos, last, tot, steps = min(posmap), max(posmap), 0, 0
    while pos <= last:
        e = posmap.get(pos)
        if e is None:
            steps += 1
            pos += 1
            continue
        a = accept_fn(pos, e)
        tot += a
        steps += 1
        pos += a + 1
    return tot, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="dense jsonl.gz from run_chain_handoff_dflash")
    ap.add_argument("--dflash-proposals", required=True, help="dense dflash_proposals jsonl (hazards)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--variant", default="grd", choices=["grd", "orc"])
    ap.add_argument("--ks", default="8,16,inf")
    ap.add_argument("--t-source", default="suffix_inf",
                    help="scalar T: 'suffix_inf' = mean A_s(t,0) (uncapped). "
                         "Used for a* and Value.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    ks = [10**9 if x == "inf" else int(x) for x in args.ks.split(",")]
    klabels = args.ks.split(",")
    seqs = load_table(args.table, args.variant)
    haz = load_haz(args.dflash_proposals)
    # attach hazards to each position
    npos = 0
    nposnohaz = 0
    for (rid, ci), pm in seqs.items():
        for pos in pm:
            ae, asv, el, gcl0, score0 = pm[pos]
            h = haz.get((rid, pos), [])
            if not h:
                nposnohaz += 1
            # e[3]=hazards, e[4]=gcl0, e[5]=score0 (suffix run score = T(t))
            pm[pos] = (ae, asv, el, h, gcl0, score0)
            npos += 1
    print(f"[{args.label}] positions={npos} no_haz={nposnohaz}", file=sys.stderr)

    def mat(fn):
        tot = steps = 0
        for pm in seqs.values():
            t, s = walk(pm, fn)
            tot += t
            steps += s
        return tot / max(steps, 1)

    # T grid for the single-scalar-T sweep: dense at the low end (cold suffix),
    # sparser at the high end (warm/self-match suffix).
    Tgrid = sorted(set(
        [round(i * 0.1, 3) for i in range(0, 30)]      # 0.0 .. 2.9
        + [round(3 + i * 0.5, 3) for i in range(0, 20)]  # 3 .. 12.5
        + [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]))

    rows = {}
    for kv, kl in zip(ks, klabels):
        m_d = mat(lambda pos, e, kv=kv: min(e[0], kv))
        m_s = mat(lambda pos, e, kv=kv: val(e[0], e[1], 0, kv))
        m_o3 = mat(lambda pos, e, kv=kv: o3_accept(e[0], e[1], kv))
        base = max(m_d, m_s)
        head = m_o3 - base

        def closed(p):
            return (p - base) / head if head > 1e-9 else float("nan")

        # sweep scalar T for the argmax policy; record the best
        best_arg = (-1.0, None)
        sweep = {}
        for T in Tgrid:
            m = mat(lambda pos, e, T=T, kv=kv:
                    val(e[0], e[1], policy_argmax(e[3], T), kv))
            sweep[T] = m
            if m > best_arg[0]:
                best_arg = (m, T)
        m_arg_best, T_best = best_arg
        # threshold policy at the same best T (for reference)
        astar_best = T_best / (T_best + 1.0)
        m_thr_best = mat(lambda pos, e, a=astar_best, kv=kv:
                         val(e[0], e[1], policy_threshold(e[3], a), kv))
        # PER-POSITION T(t): deployable = the suffix draft's proposed length at
        # this position (gcl0, known before verify); oracle = the realized suffix
        # accept A_s(t,0) (hindsight ceiling for a per-position tail estimate).
        m_pp_deploy = mat(lambda pos, e, kv=kv:
                          val(e[0], e[1], policy_argmax(e[3], e[4]), kv))
        m_pp_oracle = mat(lambda pos, e, kv=kv:
                          val(e[0], e[1], policy_argmax(e[3], e[1].get(0, 0)), kv))
        # *** CORRECTED HAND-OFF (the deployable served rule) ***
        # T(t) = the SUFFIX RUN SCORE at this position (sfx[0].score = expected
        # tail length); a*(t)=T/(1+T); extend the DFlash head while the hazard
        # dflash_p >= a*(t), then hand off to the suffix tail. This is exactly
        # what the served select1_handoff arm does.
        def _acc_scorethr(pos, e, kv=kv):
            T = e[5]
            a = T / (1.0 + T) if T > 0 else 0.0
            return val(e[0], e[1], policy_threshold(e[3], a), kv)
        def _acc_scorearg(pos, e, kv=kv):
            return val(e[0], e[1], policy_argmax(e[3], e[5]), kv)
        m_score_thr = mat(_acc_scorethr)
        m_score_arg = mat(_acc_scorearg)

        rows[kl] = {
            "O0_dflash": m_d, "O0_suffix": m_s, "O3": m_o3,
            "best_single": base, "o3_headroom": head,
            "policy_argmax_bestT": m_arg_best, "T_best": T_best,
            "astar_best": astar_best,
            "policy_threshold_bestT": m_thr_best,
            "policy_perpos_deploy": m_pp_deploy,
            "policy_perpos_oracleT": m_pp_oracle,
            "handoff_score_threshold": m_score_thr,
            "handoff_score_argmax": m_score_arg,
            "arg_gap_closed": closed(m_arg_best),
            "thr_gap_closed": closed(m_thr_best),
            "ppdeploy_gap_closed": closed(m_pp_deploy),
            "pporacle_gap_closed": closed(m_pp_oracle),
            "handoff_score_thr_gap_closed": closed(m_score_thr),
            "handoff_score_arg_gap_closed": closed(m_score_arg),
            "sweep": sweep,
        }
        print(f"  k={kl:>4} | dflash={m_d:6.3f} suffix={m_s:7.3f} "
              f"|| HANDOFF(T=suffix-score): thr={m_score_thr:7.3f} "
              f"arg={m_score_arg:7.3f} || O3={m_o3:7.3f} "
              f"|| gap-closed: score-thr={closed(m_score_thr)*100:5.1f}% "
              f"score-arg={closed(m_score_arg)*100:5.1f}% "
              f"(bestFixedT-arg={m_arg_best:.2f}/{closed(m_arg_best)*100:.0f}%)")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"label": args.label, "variant": args.variant,
                       "ks": klabels, "rows": rows}, f, indent=2)
    return rows


if __name__ == "__main__":
    main()
