"""GOAL-2 depth-gated policies with a REALISTIC per-depth selector (CHAIN regime).

At each competition depth we compare the ACTUAL confidences eagle_p (draft-model
conditional prob of eagle's chain token) vs suffix_p (SuffixDecodingCache node prob)
and commit the WINNER's token — a wrong bet stops the run (this is what makes the
realistic number < oracle). eagle-only depths always commit eagle's token.

A proposer is "available" at depth d only while still on the gt-path (matched gt
through d-1, i.e. a>=d) AND it has a token at depth d. If the committed token != gt,
the run stops. Reports, per budget, each policy under `real` (prob-pick) and `oracle`
(pick whoever actually has gt) so the selector's loss is visible. all_eagle is
selector-free (identical either way; sanity vs prior 1.391).

Chain: eagle = reslice(S,1) top-1 chain (cond prob = cumulative[d]/cumulative[d-1]);
suffix = top-1 path of the speculate tree (cond prob = dr.probs on that path).
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from arctic_inference.suffix_decoding import SuffixDecodingCache
from simulation.pipeline.pool_reslicer import reslice_eagle3_pool


def _eagle_chain(pool, S, K, B):
    """Return (tokens, cond_probs) for eagle's top-1 chain, capped at B."""
    try:
        ids, par, pp = reslice_eagle3_pool(
            list(pool["draft_tokens"]), list(pool["parent_list"]),
            list(pool["path_probs"]), int(pool["pool_size"]), S, K, S, 1)
    except Exception:
        return [], []
    ids, pp = ids[:B], pp[:B]
    cond, prev = [], 1.0
    for d in range(len(ids)):
        cond.append(pp[d] / prev if prev > 0 else 0.0)
        prev = pp[d]
    return ids, cond


def _suffix_chain(dr, B):
    """Top-1 (first-child) path of the suffix tree: (tokens, cond_probs)."""
    tids, pids, probs = list(dr.token_ids), list(dr.parents), list(dr.probs)
    toks, cond, node = [], [], -1
    while len(toks) < B:
        child = None
        for i in range(len(pids)):
            if pids[i] == node:
                child = i
                break
        if child is None:
            break
        toks.append(tids[child]); cond.append(probs[child]); node = child
    return toks, cond


def _mlen(toks, fut):
    a = 0
    for d in range(len(toks)):
        if d < len(fut) and toks[d] == fut[d]:
            a += 1
        else:
            break
    return a


def _walk(e_tok, e_cond, a_e, s_tok, s_cond, a_s, gate, mode):
    """gate(d) -> True if depth d is competition. mode: 'real' | 'oracle'."""
    d = 0
    Le, Ls = len(e_tok), len(s_tok)
    while True:
        e_av = (a_e >= d) and (d < Le)   # eagle still on gt-path & has a token
        s_av = (a_s >= d) and (d < Ls)
        if not gate(d):                  # eagle-only depth
            if not e_av:
                break
            if a_e >= d + 1:             # eagle token == gt
                d += 1
            else:
                break
            continue
        # competition depth
        if e_av and s_av:
            if mode == "real":
                pick_e = e_cond[d] >= s_cond[d]
            else:  # oracle: pick a hitter if one exists (eagle on tie)
                pick_e = (a_e >= d + 1) or not (a_s >= d + 1)
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


GATES = {
    "all_eagle": lambda d: False,
    "policyA":   lambda d: d >= 1,
    "policyB":   lambda d: d % 2 == 1,
    "all_comp":  lambda d: True,
}


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

    turns = []
    for q in d["questions"]:
        for turn in q["agent_metrics"]["steps"]:
            ents = (turn.get("spec_decode") or {}).get("oracle_vanilla_entries") or []
            gt, pools = [], []
            for e in ents:
                if not e.get("tokens"):
                    continue
                gt.append(e["tokens"][0][0])
                pools.append(e.get("eagle3_pool_full") or {})
            if len(gt) >= 2:
                turns.append((gt, pools))
    npos = sum(len(g) - 1 for g, _ in turns)
    print(f"turns={len(turns)}  positions={npos}  regime=CHAIN  selector=raw(prob-pick)\n")

    print(f"{'budget':>7}  {'all_eagle':>9} | "
          f"{'A_real':>7}{'A_orac':>7} | {'B_real':>7}{'B_orac':>7} | "
          f"{'comp_real':>10}{'comp_orac':>10} | {'A_gapReal%':>11}")
    for B in budgets:
        cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
        acc = {f"{k}_{m}": [] for k in GATES for m in ("real", "oracle")}
        rid = 0
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
                for k, g in GATES.items():
                    for m in ("real", "oracle"):
                        acc[f"{k}_{m}"].append(_walk(e_tok, e_cond, a_e, s_tok, s_cond, a_s, g, m))
                cache.add_active_response(rid, [int(gt[p])]); ctx.append(gt[p])
            cache.stop_request(rid); rid += 1
        M = {k: np.mean(v) for k, v in acc.items()}
        e = M["all_eagle_real"]
        o = M["all_comp_oracle"]
        gapAr = 100 * (M["policyA_real"] - e) / (o - e) if o > e else 0.0
        print(f"{B:>7}  {e:>9.3f} | "
              f"{M['policyA_real']:>7.3f}{M['policyA_oracle']:>7.3f} | "
              f"{M['policyB_real']:>7.3f}{M['policyB_oracle']:>7.3f} | "
              f"{M['all_comp_real']:>10.3f}{M['all_comp_oracle']:>10.3f} | {gapAr:>11.1f}")


if __name__ == "__main__":
    main()
