"""GOAL-2 selection ladder in the CHAIN all-competition regime: raw vs calib vs
oracle. (Policy A/B gating discarded.)

Every depth is a competition depth. At each depth d (while a proposer is still on
the gt-path: a>=d and has a token) we commit the WINNER's token by one of:
  - raw   : argmax(eagle_p, suffix_p)                    [scale-mismatched compare]
  - calib : argmax(cal_e(eagle_p), cal_s(suffix_p))      [per-proposer MARGINAL
            isotonic calibration of P(token==gt | prob); NOT joint/Bayes]
  - oracle: pick whoever actually has gt                 [ceiling]
A wrong commit stops the run. all_eagle (never compete) is the baseline.

Calibrators are fit with leave-one-task-out (fit on the other tasks' (prob,hit)
pairs, evaluate the held-out task) to avoid in-sample leakage. eagle_p = draft
conditional prob (reslice cumulative ratio); suffix_p = SuffixDecodingCache node prob.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from arctic_inference.suffix_decoding import SuffixDecodingCache
from sklearn.isotonic import IsotonicRegression
from simulation.scripts.goal2_realistic_policy import _eagle_chain, _suffix_chain, _mlen


def _walk(a_e, Le, e_cond, a_s, Ls, s_cond, mode, cal_e=None, cal_s=None):
    """all-competition walk. mode in {raw, calib, oracle}."""
    d = 0
    while True:
        e_av = (a_e >= d) and (d < Le)
        s_av = (a_s >= d) and (d < Ls)
        if e_av and s_av:
            if mode == "oracle":
                pick_e = (a_e >= d + 1) or not (a_s >= d + 1)
            elif mode == "raw":
                pick_e = e_cond[d] >= s_cond[d]
            else:  # calib
                pick_e = cal_e(e_cond[d]) >= cal_s(s_cond[d])
        elif e_av:
            pick_e = True
        elif s_av:
            pick_e = False
        else:
            break
        hit = (a_e >= d + 1) if pick_e else (a_s >= d + 1)
        if hit:
            d += 1
        else:
            break
    return d


def _fit_iso(pairs):
    if len(pairs) < 5:
        return lambda x: x
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(x, y)
    return lambda v: float(ir.predict([v])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--budgets", default="8,16,32,64")
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=8)
    ap.add_argument("--max-spec-factor", type=float, default=100.0)
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    S, K = args.capture_steps, args.capture_topk
    d = json.load(open(args.record))

    # group turns by task (question) for leave-one-task-out
    tasks = []
    for q in d["questions"]:
        turns = []
        for turn in q["agent_metrics"]["steps"]:
            ents = (turn.get("spec_decode") or {}).get("oracle_vanilla_entries") or []
            gt, pools = [], []
            for e in ents:
                if not e.get("tokens"):
                    continue
                gt.append(e["tokens"][0][0]); pools.append(e.get("eagle3_pool_full") or {})
            if len(gt) >= 2:
                turns.append((gt, pools))
        if turns:
            tasks.append(turns)
    npos = sum(len(g) - 1 for t in tasks for g, _ in t)
    print(f"tasks={len(tasks)}  positions={npos}  regime=CHAIN all-competition\n")

    print(f"{'budget':>7}{'all_eagle':>11}{'raw':>9}{'calib':>9}{'oracle':>9}"
          f"{'raw_gap%':>10}{'calib_gap%':>12}")
    for B in budgets:
        # Pass 1: per-task per-position (a_e, e_cond, Le, a_s, s_cond, Ls)
        cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
        rid = 0
        per_task = []
        for turns in tasks:
            recs = []
            for gt, pools in turns:
                cache.start_request(rid, np.asarray(gt[:1], dtype=np.int32))
                ctx = [gt[0]]
                for p in range(1, len(gt)):
                    fut = gt[p:]
                    e_tok, e_cond = _eagle_chain(pools[p], S, K, B) if p < len(pools) else ([], [])
                    a_e = _mlen(e_tok, fut)
                    try:
                        dr = cache.speculate(rid, np.asarray(ctx, dtype=np.int32),
                                             max_spec_tokens=B, max_spec_factor=args.max_spec_factor,
                                             min_token_prob=0.0, use_tree_spec=True)
                        s_tok, s_cond = _suffix_chain(dr, B) if dr.token_ids else ([], [])
                    except Exception:
                        s_tok, s_cond = [], []
                    a_s = _mlen(s_tok, fut)
                    recs.append((a_e, len(e_tok), e_cond, a_s, len(s_tok), s_cond))
                    cache.add_active_response(rid, [int(gt[p])]); ctx.append(gt[p])
                cache.stop_request(rid); rid += 1
            per_task.append(recs)

        # calibration training pairs per task
        e_pairs_t, s_pairs_t = [], []
        for recs in per_task:
            ep, sp = [], []
            for a_e, Le, e_cond, a_s, Ls, s_cond in recs:
                for dd in range(Le):
                    if a_e >= dd:  # available
                        ep.append((e_cond[dd], 1.0 if a_e >= dd + 1 else 0.0))
                for dd in range(Ls):
                    if a_s >= dd:
                        sp.append((s_cond[dd], 1.0 if a_s >= dd + 1 else 0.0))
            e_pairs_t.append(ep); s_pairs_t.append(sp)

        acc = {k: [] for k in ("eagle", "raw", "calib", "oracle")}
        for ti in range(len(tasks)):
            # fit on other tasks
            tr_e = [x for tj in range(len(tasks)) if tj != ti for x in e_pairs_t[tj]]
            tr_s = [x for tj in range(len(tasks)) if tj != ti for x in s_pairs_t[tj]]
            cal_e, cal_s = _fit_iso(tr_e), _fit_iso(tr_s)
            for a_e, Le, e_cond, a_s, Ls, s_cond in per_task[ti]:
                acc["eagle"].append(a_e)
                acc["raw"].append(_walk(a_e, Le, e_cond, a_s, Ls, s_cond, "raw"))
                acc["calib"].append(_walk(a_e, Le, e_cond, a_s, Ls, s_cond, "calib", cal_e, cal_s))
                acc["oracle"].append(_walk(a_e, Le, e_cond, a_s, Ls, s_cond, "oracle"))
        e, r, c, o = (np.mean(acc[k]) for k in ("eagle", "raw", "calib", "oracle"))
        gr = 100 * (r - e) / (o - e) if o > e else 0.0
        gc = 100 * (c - e) / (o - e) if o > e else 0.0
        print(f"{B:>7}{e:>11.3f}{r:>9.3f}{c:>9.3f}{o:>9.3f}{gr:>10.1f}{gc:>12.1f}")


if __name__ == "__main__":
    main()
