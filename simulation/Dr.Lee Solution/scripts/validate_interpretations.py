#!/usr/bin/env python3
"""Validate the two readings of WHY DFlash+Suffix composition wins on agentic
workloads, on the same capture_perpos records the 4-way study uses.

  Dr.Lee (multislot):  outputs are a shared template + novel SLOTS. Suffix copies
      the template, DFlash bridges the slots; more slots => composition wins more.
      Baseline: best SINGLE proposer.
  Kim (boundary):      outputs alternate WARM (suffix-copyable) and COLD regions.
      The Suffix-Decoding-paper hybrid picks ONE proposer per verify step, so
      every warm<->cold boundary costs an extra verify step; composition packs
      the boundary into one step. Baseline: per-step binary SWITCH.

What this script adds over replay_extension.py:
  1. ARM-INDEPENDENT per-position ground curves along gt (teacher-forced):
       s(p) = realized suffix copy depth at position p (tree state = committed
              prefix, identical for every arm at the same position)
       a(p) = DFlash block leading-match run at p (from the dense capture)
       sc(p) = arctic probe score at p
     From these: warm/cold segmentation -> slot & boundary densities.
  2. Two SWITCH arms (Kim's baseline, absent from replay_extension):
       switch_oracle: per round max(dflash-only, suffix-only) accept
       switch_real:   per round pick by the SAME calibrated signals compose uses
     Both derived exactly from the curves (binary choice never grafts a tail).
  3. Per-round decomposition of the calib (compose) and handoff-oracle arms:
     head/tail split, bridge rounds (head>0 AND tail>0), same-position
     counterfactual gain vs switch_oracle, position labels (warm / slot / cold).

Run inside sglang-bench (root, CPU only):
  cd "/workspace/simulation/Dr.Lee Solution"
  PYTHONPATH=/workspace python3 scripts/validate_interpretations.py \
      --record results/perpos_spider_alleval/spider_4way.jsonl --name spider \
      --out results/interp_validation
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
from replay_extension import _ad, _fit_beta, _fit_tail_iso  # noqa: E402
from measure_k_fusion import ArcticSuffix  # noqa: E402
from fusion_tree import build_extension_chain  # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def load_record(record, limit_rids=0):
    traces = json.load(open(Path(record).with_suffix(".traces.json")))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)
    recs, task_of = {}, {}
    for l in open(record):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        rid = r["rid"]
        if rid not in recs:
            if limit_rids and len(recs) >= limit_rids:
                continue
            recs[rid] = {}
        # keep only what the replay needs (halves memory on the 100MB records)
        recs[rid][r["pos"]] = dict(W=r["W"], dflash_tok=r["dflash_tok"],
                                   dflash_conf=r["dflash_conf"],
                                   dflash_match=r["dflash_match"])
        task_of[rid] = r["task"]
    return warm_traces, eval_traces, recs, task_of, num_spec


# ---------------------------------------------------------------------------
# pass 1: arm-independent ground curves
# ---------------------------------------------------------------------------
def ground_curves(warm_traces, eval_traces, recs, num_spec, max_rounds):
    """s(p), a(p), sc(p) for p = 1..len(gt)-1 per rid, with the suffix tree in
    exactly the state any replay arm has at a round starting at m = p-1
    (committed = gt[:p-1]; ctx = prompt + gt[:p])."""
    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    curves = {}
    for rid in recs:
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids = tr["output_ids"], tr["prompt_ids"]
        suffix.new_eval(pids)
        S, A, SC = [], [], []
        rby = recs[rid]
        for p in range(1, len(gt)):
            toks, sc = suffix._spec(pids + gt[:p], num_spec)
            S.append(lcp(toks[:num_spec], gt[p:]))
            rec = rby.get(p)
            A.append(min(_ad(rec["dflash_match"]), rec["W"]) if rec else -1)
            SC.append(float(sc))
            suffix.add_response([gt[p - 1]])
        if len(gt) >= 1:
            suffix.add_response([gt[-1]])
        curves[rid] = dict(s=S, a=A, sc=SC)
    return curves


# ---------------------------------------------------------------------------
# structure from curves: warm/cold segments, slots, boundaries
# ---------------------------------------------------------------------------
def structure_of(curve, W, theta):
    """Segment the position axis by warm(p) = s(p) >= theta. Returns per-call
    structure stats + a per-position label array:
      'w' warm, 'g' interior cold gap <= W (slot), 'G' interior gap > W,
      'l' leading cold (before first warm), 't' trailing cold."""
    s, a = curve["s"], curve["a"]
    L = len(s)
    if L == 0:
        return None
    w = [1 if v >= theta else 0 for v in s]
    lab = ["t"] * L
    # runs
    runs = []                                    # (start, end_excl, is_warm)
    i = 0
    while i < L:
        j = i
        while j < L and w[j] == w[i]:
            j += 1
        runs.append((i, j, w[i]))
        i = j
    warm_share = sum(w) / L
    nb = sum(1 for i in range(1, L) if w[i] != w[i - 1])
    gaps, slots, bridge_ok, reentry = [], 0, 0, []
    warm_runs = [(b, e) for b, e, iw in runs if iw]
    for idx, (b, e, iw) in enumerate(runs):
        if iw:
            for p in range(b, e):
                lab[p] = "w"
            continue
        interior = 0 < idx < len(runs) - 1
        glen = e - b
        c = "l" if idx == 0 else ("t" if idx == len(runs) - 1 else
                                  ("g" if glen <= W else "G"))
        for p in range(b, e):
            lab[p] = c
        if interior:
            gaps.append(glen)
            if glen <= W:
                slots += 1
                if a[b] >= glen >= 0:
                    bridge_ok += 1
                reentry.append(s[e] if e < L else 0)
    return dict(
        L=L, warm_share=warm_share, n_bound=nb,
        bound_per100=100.0 * nb / L,
        gaps=gaps, n_slot=slots,
        slot_per100=100.0 * slots / L,
        bridgeable=bridge_ok,
        reentry=reentry,
        warm_run_lens=[e - b for b, e in warm_runs],
        lab=lab,
    )


# ---------------------------------------------------------------------------
# curve-simulated arms (identical mechanics to replay_extension trajectories)
# ---------------------------------------------------------------------------
def sim_from_curves(curve, gt_len, max_rounds, pick):
    """Walk the round trajectory using only curve lookups. pick(p) -> acc.
    Mirrors replay advance: m += 1 + acc + 1, break on missing record/curve."""
    Ks, m, rounds = [], 0, 0
    while rounds < max_rounds and m < gt_len:
        p = m + 1
        if p > len(curve["s"]) or curve["a"][p - 1] < 0:
            break
        acc = pick(p - 1)
        Ks.append(acc)
        m += 1 + acc + 1
        rounds += 1
    return Ks


# ---------------------------------------------------------------------------
# live loop arms with per-round logging (calib compose + handoff oracle)
# ---------------------------------------------------------------------------
def replay_logged(name, warm_traces, eval_traces, recs, task_of, num_spec,
                  max_rounds, mode, cal=None, curves=None, structs=None):
    """mode='calib' (deployed compose: in-sample beta hazard + isotonic tail,
    argmax objective) or 'oracle' (handoff oracle). Logs one row per round."""
    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    rows = []
    for rid in recs:
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids = tr["output_ids"], tr["prompt_ids"]
        cu, st = curves.get(rid), structs.get(rid)
        rby = recs[rid]
        suffix.new_eval(pids)
        m, rounds = 0, 0
        while rounds < max_rounds and m < len(gt):
            rec = rby.get(m + 1)
            if rec is None:
                break
            block_full, conf = rec["dflash_tok"], rec["dflash_conf"]
            root = gt[m]
            ctx_list = pids + gt[:m] + [root]
            W_ = min(num_spec, len(conf))
            if mode == "oracle":
                best_acc, k_best, path, tree = 0, 0, [], build_extension_chain([], [])
                for kk in range(W_ + 1):
                    tk = suffix.speculate(ctx_list + block_full[:kk], num_spec)
                    trd = build_extension_chain(block_full[:kk],
                                                tk[:max(0, num_spec - kk)])
                    pth = greedy_tree_walk_path(list(trd.tokens), list(trd.parents),
                                                gt[m + 1:])
                    if len(pth) > best_acc:
                        best_acc, k_best, path, tree = len(pth), kk, pth, trd
                k, acc = k_best, best_acc
            else:                                   # calib compose
                cal_h, cal_t = cal
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
                tree = build_extension_chain(block_full[:k],
                                             tails[k][0][:max(0, num_spec - k)])
                path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents),
                                             gt[m + 1:])
                acc = len(path)
            p_i = m                                  # curve index for position m+1
            s0 = cu["s"][p_i] if p_i < len(cu["s"]) else 0
            a0 = cu["a"][p_i] if p_i < len(cu["a"]) else 0
            lab = st["lab"][p_i] if st and p_i < len(st["lab"]) else "?"
            head_acc = min(acc, k)
            rows.append(dict(rid=rid, task=task_of.get(rid, "all"), p=m + 1,
                             k=k, acc=acc, head=head_acc,
                             tail=acc - head_acc, s0=s0, a0=max(a0, 0),
                             lab=lab, W=W_))
            accepted_toks = [tree.tokens[i] for i in path]
            nxt = [root] + accepted_toks
            if m + 1 + acc < len(gt):
                nxt.append(gt[m + 1 + acc])
            suffix.add_response(nxt)
            m += 1 + acc + 1
            rounds += 1
    return rows


# ---------------------------------------------------------------------------
def K_of(Ks_all):
    n = sum(len(k) for k in Ks_all)
    return (sum(sum(k) for k in Ks_all) / n if n else 0.0), n


def agg_rows(rows):
    n = len(rows) or 1
    tot = sum(r["acc"] for r in rows)
    cf = [r["acc"] - max(r["a0"], r["s0"]) for r in rows]     # vs switch oracle @ same pos
    bridge = [r for r in rows if r["head"] > 0 and r["tail"] > 0]
    beyond = [r for r in rows if r["acc"] > r["W"]]
    cf_pos = sum(c for c in cf if c > 0)
    lab_gain = defaultdict(float)
    lab_cnt = defaultdict(int)
    for r, c in zip(rows, cf):
        lab_gain[r["lab"]] += c
        lab_cnt[r["lab"]] += 1
    bridge_cf = sum(c for r, c in zip(rows, cf) if r["head"] > 0 and r["tail"] > 0)
    bridge_cf_pos = sum(c for r, c in zip(rows, cf)
                        if c > 0 and r["head"] > 0 and r["tail"] > 0)
    return dict(
        rounds=len(rows), K=tot / n,
        cf_gain_mean=sum(cf) / n,
        cf_gain_pos_share=(sum(1 for c in cf if c > 0) / n),
        bridge_share=len(bridge) / n,
        bridge_tok_share=(sum(r["acc"] for r in bridge) / tot if tot else 0.0),
        bridge_cf_share=(bridge_cf / sum(cf) if sum(cf) > 0 else 0.0),
        bridge_cfpos_share=(bridge_cf_pos / cf_pos if cf_pos > 0 else 0.0),
        beyond_share=len(beyond) / n,
        cf_by_label={k: (lab_gain[k], lab_cnt[k]) for k in lab_gain},
        head_mean=sum(r["head"] for r in rows) / n,
        tail_mean=sum(r["tail"] for r in rows) / n,
        cf_total=sum(cf), cf_pos_total=cf_pos,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/interp_validation")
    ap.add_argument("--max-rounds", type=int, default=4096)
    ap.add_argument("--theta", type=int, default=4)
    ap.add_argument("--limit-rids", type=int, default=0)
    ap.add_argument("--skip-live", action="store_true",
                    help="curves + curve-simulated arms only (no calib/oracle)")
    ap.add_argument("--no-calib", action="store_true",
                    help="RAW prob: cal_h=cal_t=identity (no beta hazard / isotonic tail). "
                         "compose->compose_raw, switch_real->switch_raw.")
    ap.add_argument("--skip-oracle", action="store_true",
                    help="only run the compose(calib) replay, skip handoff-oracle replay")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    warm_traces, eval_traces, recs, task_of, num_spec = load_record(
        args.record, args.limit_rids)
    W_glob = next(iter(next(iter(recs.values())).values()))["W"]
    print(f"[{args.name}] {len(recs)} evals, {len(warm_traces)} warm, "
          f"num_spec={num_spec}, W={W_glob}", flush=True)

    # ---- pass 1: curves + structure
    curves = ground_curves(warm_traces, eval_traces, recs, num_spec, args.max_rounds)
    structs, structs2 = {}, {}
    for rid, cu in curves.items():
        st = structure_of(cu, W_glob, args.theta)
        if st:
            structs[rid] = st
        st2 = structure_of(cu, W_glob, 2)
        if st2:
            structs2[rid] = st2
    print(f"[{args.name}] curves done ({time.time()-t0:.0f}s)", flush=True)

    with gzip.open(os.path.join(args.out, f"curves_{args.name}.jsonl.gz"), "wt") as f:
        for rid, cu in curves.items():
            f.write(json.dumps(dict(rid=rid, task=task_of.get(rid), s=cu["s"],
                                    a=cu["a"],
                                    sc=[round(x, 3) for x in cu["sc"]])) + "\n")

    # ---- calibrators (in-sample, mirrors --calib-insample --hazard-fit beta)
    if args.no_calib:
        cal_h = cal_t = (lambda x: x)          # RAW prob, no calibration
        print(f"[{args.name}] NO-CALIB: raw prob (identity cals)", flush=True)
    else:
        pairs = []
        for rby in recs.values():
            for r in rby.values():
                conf, match = r["dflash_conf"], r["dflash_match"]
                ad = _ad(match)
                for d in range(min(ad + 1, len(conf))):
                    pairs.append((float(conf[d]), int(match[d])))
        cal_h = _fit_beta([x for x, _ in pairs], [y for _, y in pairs])
        cal_t = _fit_tail_iso(warm_traces, recs, eval_traces, set(recs), num_spec,
                              args.max_rounds)
        print(f"[{args.name}] calibrators fit ({time.time()-t0:.0f}s)", flush=True)

    # ---- curve-simulated arms
    def run_sim(pick_fn):
        per_task = defaultdict(list)
        allK = []
        for rid, cu in curves.items():
            gt_len = len(eval_traces[rid]["output_ids"])
            Ks = sim_from_curves(cu, gt_len, args.max_rounds,
                                 lambda i, cu=cu: pick_fn(cu, i))
            allK.append(Ks)
            per_task[task_of.get(rid, "all")].append(Ks)
        return allK, per_task

    def ghat(conf):
        S, G = 1.0, 0.0
        for c in conf:
            S *= cal_h(c); G += S
        return G

    # expected block accept needs conf at that position -> from recs
    conf_at = {rid: {p: r["dflash_conf"] for p, r in rby.items()}
               for rid, rby in recs.items()}

    arms = {}
    arms["dflash"] = run_sim(lambda cu, i: max(cu["a"][i], 0))
    arms["suffix"] = run_sim(lambda cu, i: cu["s"][i])
    arms["switch_oracle"] = run_sim(lambda cu, i: max(cu["a"][i], cu["s"][i]))

    def real_pick(rid):
        ca = conf_at[rid]
        def pick(cu, i):
            g = ghat(ca[i + 1][:num_spec]) if (i + 1) in ca else 0.0
            t = cal_t(cu["sc"][i])
            return max(cu["a"][i], 0) if g > t else cu["s"][i]
        return pick

    per_task_sw = defaultdict(list)
    allK_sw = []
    for rid, cu in curves.items():
        gt_len = len(eval_traces[rid]["output_ids"])
        Ks = sim_from_curves(cu, gt_len, args.max_rounds,
                             lambda i, cu=cu, rid=rid: real_pick(rid)(cu, i))
        allK_sw.append(Ks)
        per_task_sw[task_of.get(rid, "all")].append(Ks)
    arms["switch_real"] = (allK_sw, per_task_sw)
    print(f"[{args.name}] curve arms done ({time.time()-t0:.0f}s)", flush=True)

    # ---- live logged arms
    out = dict(name=args.name, record=args.record, num_spec=num_spec, W=W_glob,
               theta=args.theta, n_eval=len(recs))
    rows_by_mode = {}
    if not args.skip_live:
        for mode in (("calib",) if args.skip_oracle else ("calib", "oracle")):
            rows = replay_logged(args.name, warm_traces, eval_traces, recs, task_of,
                                 num_spec, args.max_rounds, mode,
                                 cal=(cal_h, cal_t), curves=curves, structs=structs)
            rows_by_mode[mode] = rows
            print(f"[{args.name}] {mode} replay done ({time.time()-t0:.0f}s)",
                  flush=True)
        with gzip.open(os.path.join(args.out, f"rounds_{args.name}.jsonl.gz"),
                       "wt") as f:
            for mode, rows in rows_by_mode.items():
                for r in rows:
                    f.write(json.dumps(dict(r, mode=mode)) + "\n")

    # ---- aggregate
    def struct_agg(sts):
        L = sum(s["L"] for s in sts) or 1
        gaps = [g for s in sts for g in s["gaps"]]
        slots = sum(s["n_slot"] for s in sts)
        reent = [x for s in sts for x in s["reentry"]]
        return dict(
            positions=L,
            warm_share=sum(s["warm_share"] * s["L"] for s in sts) / L,
            bound_per100=100.0 * sum(s["n_bound"] for s in sts) / L,
            slot_per100=100.0 * slots / L,
            n_gap=len(gaps),
            gap_hist={"1-3": sum(1 for g in gaps if g <= 3),
                      "4-15": sum(1 for g in gaps if 4 <= g <= 15),
                      ">15": sum(1 for g in gaps if g > 15)},
            gap_mean=(sum(gaps) / len(gaps) if gaps else 0.0),
            bridgeable_share=(sum(s["bridgeable"] for s in sts) / slots
                              if slots else 0.0),
            reentry_mean=(sum(reent) / len(reent) if reent else 0.0),
            warm_run_mean=(lambda w: sum(w) / len(w) if w else 0.0)(
                [x for s in sts for x in s["warm_run_lens"]]),
        )

    out["structure"] = struct_agg(list(structs.values()))
    out["structure_theta2"] = struct_agg(list(structs2.values()))
    out["structure_by_task"] = {}
    by_task_st = defaultdict(list)
    for rid, st in structs.items():
        by_task_st[task_of.get(rid, "all")].append(st)
    for t, sts in sorted(by_task_st.items()):
        out["structure_by_task"][t] = struct_agg(sts)

    out["arms"] = {}
    for arm, (allK, per_task) in arms.items():
        K, n = K_of(allK)
        out["arms"][arm] = dict(K=K, rounds=n,
                                by_task={t: K_of(v)[0]
                                         for t, v in sorted(per_task.items())})
    for mode, rows in rows_by_mode.items():
        name = "compose" if mode == "calib" else "handoff_oracle"
        a = agg_rows(rows)
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)
        a["by_task"] = {t: agg_rows(v) for t, v in sorted(by_task.items())}
        out["arms"][name] = a

    json.dump(out, open(os.path.join(args.out, f"report_{args.name}.json"), "w"),
              indent=1)

    # ---- text summary
    st = out["structure"]
    print(f"\n===== {args.name} (theta={args.theta}, {st['positions']} positions) =====")
    ladder = []
    for arm in ("dflash", "suffix", "switch_real", "switch_oracle"):
        ladder.append(f"{arm}={out['arms'][arm]['K']:.2f}")
    for arm in ("compose", "handoff_oracle"):
        if arm in out["arms"]:
            ladder.append(f"{arm}={out['arms'][arm]['K']:.2f}")
    print("  K ladder: " + "  ".join(ladder))
    print(f"  structure: warm {st['warm_share']:.0%} | bound/100 {st['bound_per100']:.2f}"
          f" | slot/100 {st['slot_per100']:.2f} | gaps {st['gap_hist']} "
          f"(mean {st['gap_mean']:.1f}) | bridgeable {st['bridgeable_share']:.0%}"
          f" | reentry {st['reentry_mean']:.1f} | warm-run {st['warm_run_mean']:.1f}")
    if "compose" in out["arms"]:
        best_single = max(out["arms"]["dflash"]["K"], out["arms"]["suffix"]["K"])
        c = out["arms"]["compose"]
        print(f"  Dr.Lee: compose-best_single = {c['K']-best_single:+.2f} "
              f"({(c['K']/best_single-1) if best_single else 0:+.0%}) | bridge rounds "
              f"{c['bridge_share']:.0%} carry {c['bridge_tok_share']:.0%} of accepts, "
              f"{c['bridge_cfpos_share']:.0%} of positive cf-gain")
        sw, swo = out["arms"]["switch_real"]["K"], out["arms"]["switch_oracle"]["K"]
        steps_saved = 100.0 / (sw + 2) - 100.0 / (c["K"] + 2)
        print(f"  Kim: compose-switch_real = {c['K']-sw:+.2f} | -switch_oracle = "
              f"{c['K']-swo:+.2f} | steps saved/100tok vs switch_real {steps_saved:+.2f}"
              f" | cf>0 rounds {c['cf_gain_pos_share']:.0%} | cf by label "
              + str({k: (round(v[0], 1), v[1]) for k, v in sorted(c["cf_by_label"].items())}))
        if "handoff_oracle" in out["arms"]:
            ho = out["arms"]["handoff_oracle"]
            print(f"  structural (oracle-vs-oracle): handoff_oracle-switch_oracle = "
                  f"{ho['K']-swo:+.2f} ({(ho['K']/swo-1) if swo else 0:+.0%}) | "
                  f"oracle bridge rounds {ho['bridge_share']:.0%}, "
                  f"{ho['bridge_cfpos_share']:.0%} of positive cf-gain")
    print(f"[{args.name}] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
