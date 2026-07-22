#!/usr/bin/env python3
"""Diagnostic: decompose the calib->oracle handoff gap per round (test split).

On the SAME calib-policy trajectory, per round evaluate every candidate handoff
k in 0..W (realized accept + arctic tail score sc_k), then compare choices:
  calib   : argmax 1+G_k+S_k*T0      (current — scalar T from k=0 probe)
  perk    : argmax 1+G_k+S_k*sc_k    (per-k RAW arctic tail score, budget-aware)
  perkcal : argmax 1+G_k+S_k*g(sc_k) (per-k score through calibrated map g)
  oracle  : argmax realized accept
Also fits g on the calibrate split (chain-policy replay) and prints the
score->realized reliability table.

Run in docker:  cd "/workspace/simulation/Dr.Lee Solution" && \
  python3 scripts/_diag_gap.py --record results/perpos_bfcl_conv/bfcl_v4.jsonl --group-mode lenreset
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
from fusion_tree import build_extension_chain  # noqa: E402
from measure_k_fusion import ArcticSuffix  # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402
from replay_extension import _ad, _fit_logistic  # noqa: E402


def load(record):
    traces = json.load(open(Path(record).with_suffix(".traces.json")))
    recs = defaultdict(dict)
    for l in open(record):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    return traces, recs, eval_traces


def conv_map(eval_traces, mode):
    conv = {}
    if mode == "lenreset":
        cid, prev = -1, float("inf")
        for rid in sorted(eval_traces):
            L = len(eval_traces[rid]["prompt_ids"])
            if L < prev:
                cid += 1
            conv[rid] = cid
            prev = L
    else:
        conv = {rid: rid for rid in eval_traces}
    return conv


def spec_full(suffix, ctx, budget):
    """Direct arctic speculate exposing (tokens, score, match_len, probs)."""
    d = suffix.cache.speculate(
        suffix._eval_rid, np.asarray([int(t) for t in ctx], dtype=np.int32),
        max_spec_tokens=int(budget), max_spec_factor=suffix.msf,
        min_token_prob=suffix.mtp, use_tree_spec=False)
    toks = [int(t) for t in d.token_ids] if getattr(d, "token_ids", None) is not None else []
    probs = list(getattr(d, "probs", None) or [])
    return toks, float(getattr(d, "score", 0.0) or 0.0), int(getattr(d, "match_len", 0) or 0), probs


def tail_feats(sc, ml, toks, probs):
    """Feature vector for the tail-expectation model g."""
    return [sc, float(ml), float(len(toks)), float(probs[0]) if probs else 0.0]


def per_k_eval(suffix, ctx_list, block_full, gt, m, W, num_spec):
    """For each k in 0..W: (realized accept, arctic score, feature vector)."""
    accs, scs, fts = [], [], []
    for kk in range(W + 1):
        budget = max(0, num_spec - kk)
        if budget > 0:
            tk, sc, ml, pr = spec_full(suffix, ctx_list + block_full[:kk], budget)
        else:
            tk, sc, ml, pr = [], 0.0, 0, []
        tr = build_extension_chain(block_full[:kk], tk[:budget])
        pth = greedy_tree_walk_path(list(tr.tokens), list(tr.parents), gt[m + 1:])
        accs.append(len(pth))
        scs.append(float(sc))
        fts.append(tail_feats(sc, ml, tk, pr))
    return accs, scs, fts


def argmax_k(conf, W_, haz, Tof):
    """argmax_k 1+G_k+S_k*T(k); Tof(k) supplies the tail expectation."""
    S_k, G_k = 1.0, 0.0
    best_val, k = 1.0 + 1.0 * Tof(0), 0
    for j in range(W_):
        S_k *= haz(conf[j]); G_k += S_k
        val = 1.0 + G_k + S_k * Tof(j + 1)
        if val > best_val:
            best_val, k = val, j + 1
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--group-mode", default="lenreset", choices=["rid", "lenreset"])
    ap.add_argument("--max-rounds", type=int, default=int(os.environ.get("ROUNDS", "8")))
    args = ap.parse_args()

    traces, recs, eval_traces = load(args.record)
    warm_traces = traces["warm_traces"]
    num_spec = traces.get("num_spec", 32)
    conv = conv_map(eval_traces, args.group_mode)
    test_rids = {rid for rid in recs if conv.get(rid, 0) % 2 == 1}
    calib_rids = {rid for rid in recs if conv.get(rid, 0) % 2 == 0}

    # hazard calibrator (same as replay_extension) + depth-featured variant
    tr_pairs = []
    for rid in calib_rids:
        for r in recs[rid].values():
            c, mt = r["dflash_conf"], r["dflash_match"]
            ad = _ad(mt)
            for d in range(min(ad + 1, len(c))):
                tr_pairs.append((float(c[d]), d, int(mt[d])))
    cal_hazard = _fit_logistic([x for x, _, _ in tr_pairs], [y for _, _, y in tr_pairs])
    from sklearn.linear_model import LogisticRegression
    _X2 = np.array([[x, d] for x, d, _ in tr_pairs], float)
    _y2 = np.array([y for _, _, y in tr_pairs], int)
    if len(set(_y2.tolist())) > 1:
        _lr2 = LogisticRegression(C=1.0, solver="lbfgs").fit(_X2, _y2)
        cal_hazard2 = lambda c, d: float(_lr2.predict_proba([[float(c), float(d)]])[0, 1])  # noqa: E731
    else:
        cal_hazard2 = lambda c, d: cal_hazard(c)  # noqa: E731

    # ---- pass 1: collect (sc_k, realized tail accept) on the CALIBRATE split ----
    # chain-policy trajectory; only k <= ad (head survives) yields a valid tail label.
    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    sc_pairs = []
    for rid in sorted(calib_rids):
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids = tr["output_ids"], tr["prompt_ids"]
        rby = recs[rid]
        W = next(iter(rby.values()))["W"]
        suffix.new_eval(pids)
        m = 0
        for _ in range(args.max_rounds):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            block_full, cf, mt = rec["dflash_tok"], rec["dflash_conf"], rec["dflash_match"]
            root = gt[m]
            ctx_list = pids + gt[:m] + [root]
            ad = _ad(mt)
            for kk in range(min(ad, W) + 1):
                budget = max(0, num_spec - kk)
                if budget <= 0:
                    break
                tk, sc, ml, pr = spec_full(suffix, ctx_list + block_full[:kk], budget)
                fut = gt[m + 1 + kk:]
                r_acc = 0
                for t, g in zip(tk[:budget], fut):
                    if t == g:
                        r_acc += 1
                    else:
                        break
                sc_pairs.append((tail_feats(sc, ml, tk, pr), r_acc))
            # advance with deployed chain policy for trajectory realism
            T0 = suffix.probe(ctx_list, num_spec)[1]
            k = argmax_k(cf, min(num_spec, len(cf)),
                         lambda c: min(1.0, max(0.0, 0.69 * float(c) + 0.29)), lambda _k: T0)
            tk = suffix.speculate(ctx_list + block_full[:k], num_spec)
            trd = build_extension_chain(block_full[:k], tk[:max(0, num_spec - k)])
            pth = greedy_tree_walk_path(list(trd.tokens), list(trd.parents), gt[m + 1:])
            acc = len(pth)
            accepted = [trd.tokens[i] for i in pth]
            nxt = [root] + accepted
            if m + 1 + acc < len(gt):
                nxt.append(gt[m + 1 + acc])
            suffix.add_response(nxt)
            m += 1 + acc + 1
            if m >= len(gt):
                break

    F = np.array([f for f, _ in sc_pairs], float)
    ys = np.array([y for _, y in sc_pairs], float)
    xs = F[:, 0]
    print(f"score->realized pairs (calibrate split): n={len(sc_pairs)}  "
          f"corr={np.corrcoef(xs, ys)[0,1]:.3f}" if len(sc_pairs) > 2 else "few pairs")
    print(f"{'sc bin':>12}{'n':>7}{'mean sc':>9}{'mean acc':>9}")
    bins = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 64]
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (xs >= lo) & (xs < hi)
        if sel.sum():
            print(f"[{lo:>4},{hi:>4})  {sel.sum():>6}{xs[sel].mean():>9.2f}{ys[sel].mean():>9.2f}")

    # calibrated map g: isotonic on (sc, realized)
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(xs, ys)
    g = lambda v: float(ir.predict([max(0.0, v)])[0])  # noqa: E731

    # GBM tail model on [score, match_len, draft_len, probs0]
    from sklearn.ensemble import GradientBoostingRegressor
    gbm = GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.1,
                                    subsample=0.9, random_state=0).fit(F, ys)
    g2 = lambda ft: max(0.0, float(gbm.predict([ft])[0]))  # noqa: E731
    print("GBM feat importance [sc,ml,len,p0]:", np.round(gbm.feature_importances_, 3).tolist())

    # ---- pass 2: per-round counterfactual on the TEST split (calib trajectory) ----
    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    stats = defaultdict(list)
    kdist = defaultdict(list)
    n_rounds = 0
    for rid in sorted(test_rids):
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids = tr["output_ids"], tr["prompt_ids"]
        rby = recs[rid]
        W = next(iter(rby.values()))["W"]
        suffix.new_eval(pids)
        m = 0
        for _ in range(args.max_rounds):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            block_full, cf = rec["dflash_tok"], rec["dflash_conf"]
            root = gt[m]
            ctx_list = pids + gt[:m] + [root]
            W_ = min(num_spec, len(cf))
            accs, scs, fts = per_k_eval(suffix, ctx_list, block_full, gt, m, min(W, W_), num_spec)
            T0 = suffix.probe(ctx_list, num_spec)[1]
            k_cal = argmax_k(cf, W_, cal_hazard, lambda _k: T0)
            k_perk = argmax_k(cf, W_, cal_hazard, lambda kk: scs[min(kk, len(scs) - 1)])
            k_pcal = argmax_k(cf, W_, cal_hazard, lambda kk: g(scs[min(kk, len(scs) - 1)]))
            k_pgbm = argmax_k(cf, W_, cal_hazard, lambda kk: g2(fts[min(kk, len(fts) - 1)]))
            # depth-featured hazard + isotonic g
            S_k, G_k = 1.0, 0.0
            bv, k_pd = 1.0 + g(scs[0]), 0
            for j in range(W_):
                S_k *= cal_hazard2(cf[j], j); G_k += S_k
                val = 1.0 + G_k + S_k * g(scs[min(j + 1, len(scs) - 1)])
                if val > bv:
                    bv, k_pd = val, j + 1
            k_orc = int(np.argmax(accs))
            for name, kk in [("calib", k_cal), ("perk", k_perk), ("perkcal", k_pcal),
                             ("perkgbm", k_pgbm), ("pcal+dhaz", k_pd), ("oracle", k_orc)]:
                stats[name].append(accs[min(kk, len(accs) - 1)])
                kdist[name].append(kk)
            n_rounds += 1
            # advance along the CURRENT calib policy (the reported arm)
            acc = accs[min(k_cal, len(accs) - 1)]
            # rebuild the chosen tree to get accepted tokens
            budget = max(0, num_spec - k_cal)
            tk = suffix.speculate(ctx_list + block_full[:k_cal], budget) if budget > 0 else []
            trd = build_extension_chain(block_full[:k_cal], tk[:budget])
            pth = greedy_tree_walk_path(list(trd.tokens), list(trd.parents), gt[m + 1:])
            accepted = [trd.tokens[i] for i in pth]
            nxt = [root] + accepted
            if m + 1 + len(pth) < len(gt):
                nxt.append(gt[m + 1 + len(pth)])
            suffix.add_response(nxt)
            m += 1 + len(pth) + 1
            if m >= len(gt):
                break

    print(f"\n== per-round counterfactual on SAME calib trajectory  (test rounds={n_rounds}) ==")
    for name in ["calib", "perk", "perkcal", "perkgbm", "pcal+dhaz", "oracle"]:
        a = np.array(stats[name], float)
        kk = np.array(kdist[name], float)
        agree = float(np.mean(np.array(kdist[name]) == np.array(kdist["oracle"])))
        print(f"  {name:>8}: mean acc={a.mean():.3f}  mean k={kk.mean():.2f}  k==oracle_k {agree*100:.0f}%")
    loss = np.array(stats["oracle"], float) - np.array(stats["calib"], float)
    print(f"  calib loss vs oracle: mean={loss.mean():.3f}  P(loss>0)={np.mean(loss>0)*100:.0f}%  "
          f"p90={np.percentile(loss,90):.1f}  max={loss.max():.0f}")
    dk = np.array(kdist["calib"]) - np.array(kdist["oracle"])
    lossy = loss > 0
    print(f"  when lossy: calib k - oracle k  mean={dk[lossy].mean():.2f}  "
          f"(neg=handed off too early)  median={np.median(dk[lossy]):.0f}")


if __name__ == "__main__":
    main()
