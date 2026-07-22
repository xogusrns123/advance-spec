#!/usr/bin/env python3
"""WHY does calibrated compose beat the best single on swe/spider but not on
specbench(/bfcl web)? Round-level decomposition of the SAME calib arm the
figures use (two-way in-sample, beta hazard + isotonic tail):

  per round:  acc = a_head - head_loss + tail_part
    a_head    = min(leading match run, W)   what DFlash alone would accept at
                                            this position (counterfactual)
    head_loss = max(0, a_head - k)          tokens thrown away by handing off
                                            BEFORE the block died (k < a_head)
    tail_part = acc - min(a_head, k)        tokens the suffix tail added after
                                            a surviving head

  =>  MAT(compose) - MAT(dflash@same positions) = mean(tail_part) - mean(head_loss)

Also reports suffix-tracking coverage (copy-run>=4, the standard criterion) as
the upstream cause, tail burstiness, and beyond-block rounds (acc > W — only
reachable through the tail, DFlash is capped at W=block-1).

  python3 scripts/analyze_compose_gap.py --records \
      specbench=results/perpos_specbench_full/specbench.jsonl ...
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

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
from replay_extension import _ad, _fit_beta, _fit_tail_iso  # noqa: E402
from measure_k_fusion import ArcticSuffix  # noqa: E402
from fusion_tree import build_extension_chain  # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402
from plot_mat_sets import coverage_of_record  # noqa: E402


def replay_calib_logged(record, max_rounds=4096):
    traces = json.load(open(Path(record).with_suffix(".traces.json")))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)

    recs = defaultdict(dict)
    for l in open(record):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r

    # in-sample calibrators, exactly like replay_extension --calib-insample --hazard-fit beta
    pairs = []
    for rby in recs.values():
        for r in rby.values():
            conf, match = r["dflash_conf"], r["dflash_match"]
            ad = _ad(match)
            for d in range(min(ad + 1, len(conf))):
                pairs.append((float(conf[d]), int(match[d])))
    cal_h = _fit_beta([x for x, _ in pairs], [y for _, y in pairs])
    cal_t = _fit_tail_iso(warm_traces, recs, eval_traces, set(recs), num_spec, max_rounds)

    rows = []
    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    for rid, rby in recs.items():
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids = tr["output_ids"], tr["prompt_ids"]
        task = tr.get("task", "all")
        suffix.new_eval(pids)
        m = 0
        for _ in range(max_rounds):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            block_full, conf, match = rec["dflash_tok"], rec["dflash_conf"], rec["dflash_match"]
            root = gt[m]
            ctx_list = pids + gt[:m] + [root]
            W_ = min(num_spec, len(conf))
            tails = []
            for kk in range(W_ + 1):
                budget = num_spec - kk
                tails.append(suffix._spec(ctx_list + block_full[:kk], budget)
                             if budget > 0 else ([], 0.0))
            S_k, G_k = 1.0, 0.0
            best_val, k = 1.0 + cal_t(tails[0][1]), 0
            for j in range(W_):
                S_k *= cal_h(conf[j]); G_k += S_k
                val = 1.0 + G_k + S_k * cal_t(tails[j + 1][1])
                if val > best_val:
                    best_val, k = val, j + 1
            tree = build_extension_chain(block_full[:k], tails[k][0][:max(0, num_spec - k)])
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            acc = len(path)
            a_head = min(_ad(match), W_)
            head_part = min(acc, k)
            rows.append(dict(task=task, W=W_, k=k, a_head=a_head, acc=acc,
                             head_loss=max(0, a_head - k),
                             tail=acc - head_part,
                             t_pred=cal_t(tails[k][1])))
            accepted_toks = [tree.tokens[i] for i in path]
            nxt = [root] + accepted_toks
            if m + 1 + acc < len(gt):
                nxt.append(gt[m + 1 + acc])
            suffix.add_response(nxt)
            m += 1 + acc + 1
            if m >= len(gt):
                break
    return rows


def agg(rows):
    n = len(rows) or 1
    mean = lambda f: sum(f(r) for r in rows) / n           # noqa: E731
    tails = sorted(r["tail"] for r in rows)
    p95 = tails[int(0.95 * (len(tails) - 1))] if tails else 0
    W = rows[0]["W"] if rows else 0
    return dict(
        rounds=len(rows),
        mat=mean(lambda r: r["acc"]),
        dflash_cf=mean(lambda r: r["a_head"]),
        head_loss=mean(lambda r: r["head_loss"]),
        tail_mean=mean(lambda r: r["tail"]),
        tail_pos=sum(1 for r in rows if r["tail"] > 0) / n,
        tail_p95=p95,
        tail_share=(sum(r["tail"] for r in rows) / max(1, sum(r["acc"] for r in rows))),
        premature=sum(1 for r in rows if r["k"] < r["a_head"]) / n,
        k_mean=mean(lambda r: r["k"]),
        k0=sum(1 for r in rows if r["k"] == 0) / n,
        kW=sum(1 for r in rows if r["k"] == r["W"]) / n,
        beyond=sum(1 for r in rows if r["acc"] > W) / n,
        beyond_mat=(lambda b: sum(r["acc"] for r in b) / max(1, len(b)))(
            [r for r in rows if r["acc"] > W]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True,
                    help="name=path/to/record.jsonl ...")
    ap.add_argument("--max-rounds", type=int, default=4096)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = {}
    for spec in args.records:
        name, rec = spec.split("=", 1)
        rows = replay_calib_logged(rec, args.max_rounds)
        cov = coverage_of_record(rec)
        w = sum(v[0] for v in cov.values()); t = sum(v[1] for v in cov.values())
        out[name] = dict(agg(rows), coverage=(w / t if t else 0.0))
        a = out[name]
        print(f"\n== {name} (rounds={a['rounds']}, coverage={a['coverage']:.0%})")
        print(f"  MAT compose            {a['mat']:.2f}")
        print(f"  DFlash counterfactual  {a['dflash_cf']:.2f}   "
              f"(same positions; delta = {a['mat'] - a['dflash_cf']:+.2f} "
              f"= tail {a['tail_mean']:.2f} - head_loss {a['head_loss']:.2f})")
        print(f"  tail: >0 in {a['tail_pos']:.0%} rounds, mean {a['tail_mean']:.2f}, "
              f"p95 {a['tail_p95']:.0f}, share of accepts {a['tail_share']:.0%}")
        print(f"  handoff: mean k {a['k_mean']:.1f}, k=0 {a['k0']:.0%}, k=W {a['kW']:.0%}, "
              f"premature (k<a_head) {a['premature']:.0%}")
        print(f"  beyond-block rounds (acc>W): {a['beyond']:.1%} at mean {a['beyond_mat']:.1f} tok")
        sys.stdout.flush()
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
