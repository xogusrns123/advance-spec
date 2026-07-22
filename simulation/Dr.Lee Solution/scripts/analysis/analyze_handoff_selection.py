#!/usr/bin/env python3
"""Handoff-position selection quality vs the per-round decision oracle (specbench).

The compose controller picks the DFlash->Suffix handoff length k_sel by gain
argmax_k [1 + G_k + S_k * cal_t(score_k)] (cal_h logistic head, cal_t isotonic
tail; fit in-sample on specbench). At each realized round we also scan every k
to get the realized accept acc(k), so acc_opt = max_k acc(k), K* = argmax set,
k_lo = min K*. Classify the selection:
    optimal  : acc(k_sel) == acc_opt      (as good as the decision oracle)
    early    : acc(k_sel) <  acc_opt AND k_sel < k_lo   (handed off too early)
    late     : acc(k_sel) <  acc_opt AND k_sel >= k_lo  (handed off too late)
Per-round regret = acc_opt - acc_sel (tokens); decomposed into early / late.
Trajectory advances along the SELECTED handoff (on-policy), so the decision
oracle is the counterfactual best at each compose-visited state.

Same TEST half as the deck (three-way convlabel). Emits to
readable_outputs/figures/0716_regenerated/:
  handoff_position_ratio.png     early / optimal / late proportions
  handoff_matloss_decomp.png     early-loss / late-loss (tokens/step) vs oracle

  python3 scripts/analysis/analyze_handoff_selection.py --workload specbench
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from replay_extension import (ArcticSuffix, build_extension_chain,  # noqa: E402
                              greedy_tree_walk_path, split_parity, OnlineCalib)

BASE = Path("/workspace/simulation/Dr.Lee Solution")
BV = BASE / "results" / "bias_variance"
OUT = BASE / "readable_outputs" / "figures" / "0716_regenerated"
REC = {
    "specbench": "results/perpos_specbench_alleval/specbench_4way.jsonl",
    "bfcl": "results/perpos_bfcl_alleval/bfcl_4way.jsonl",
    "swebench": "results/perpos_swebench_alleval/swebench_4way.jsonl",
    "spider": "results/perpos_spider_alleval/spider_4way.jsonl",
    "tau2": "results/perpos_tau2_alleval/tau2_4way.jsonl",
}
C_EARLY, C_OPT, C_LATE = "#d1603d", "#4C9F70", "#3E6E9E"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="specbench", choices=list(REC))
    ap.add_argument("--calib", default="raw", choices=["raw", "online"],
                    help="handoff-selection signals: raw = identity (raw DFlash prob "
                         "+ raw arctic score, the uncalibrated controller); online = "
                         "OnlineCalib (logistic head + isotonic tail warmed on stream)")
    ap.add_argument("--max-rounds", type=int, default=4096)
    args = ap.parse_args()
    WL = args.workload; CALIB = args.calib

    rec_path = Path(REC[WL])
    traces = json.load(open(rec_path.with_suffix(".traces.json")))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)
    recs = defaultdict(dict)
    for l in open(rec_path):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r
    _, split_par = split_parity(eval_traces, "convlabel")
    test_rids = [rid for rid in recs if split_par.get(rid, 0) == 1]

    # Handoff-selection signals. raw = IDENTITY calibrators = the uncalibrated
    # controller (raw DFlash prob as head hazard, raw arctic score as tail value);
    # this exposes the selection misses that RAW signals make and that calibration
    # exists to fix (compose ~= raw-cals arm, e.g. specbench 2.75). online =
    # OnlineCalib (deployed calibrated arm) for comparison.
    if CALIB == "online":
        oc = OnlineCalib(window=8000, refit_every=200, head="logistic", tail="isotonic")
        cal_h, cal_t = oc.cal_h, oc.cal_t
    else:
        oc = None
        cal_h = lambda c: float(c)          # raw DFlash prob
        cal_t = lambda s: float(s)          # raw arctic suffix score

    counts = {"early": 0, "optimal": 0, "late": 0}
    loss_by = {"early": 0.0, "optimal": 0.0, "late": 0.0}
    N = 0; sum_opt = 0.0; sum_sel = 0.0

    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    for rid in test_rids:
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids, rby = tr["output_ids"], tr["prompt_ids"], recs[rid]
        suffix.new_eval(pids)
        m = 0
        for _ in range(args.max_rounds):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            block_full = rec["dflash_tok"]; conf = rec["dflash_conf"]
            root = gt[m]; ctx_list = pids + gt[:m] + [root]
            W_ = min(num_spec, len(conf))
            accs = np.zeros(W_ + 1, int); scores = np.zeros(W_ + 1)
            tk_by = []
            for kk in range(W_ + 1):
                budget = num_spec - kk
                tk, sc = suffix._spec(ctx_list + block_full[:kk], budget) if budget > 0 else ([], 0.0)
                tk_by.append(tk); scores[kk] = sc
                tree = build_extension_chain(block_full[:kk], tk[:max(0, budget)])
                pth = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
                accs[kk] = len(pth)
            acc_opt = int(accs.max())
            k_lo = int(np.argmax(accs))                      # first k achieving the max
            # compose selection
            S_k, G_k = 1.0, 0.0
            best_val = 1.0 + cal_t(scores[0]); k_sel = 0
            for j in range(W_):
                S_k *= cal_h(conf[j]); G_k += S_k
                val = 1.0 + G_k + S_k * cal_t(scores[j + 1])
                if val > best_val:
                    best_val, k_sel = val, j + 1
            acc_sel = int(accs[k_sel])
            if acc_sel == acc_opt:
                cat = "optimal"
            elif k_sel < k_lo:
                cat = "early"
            else:
                cat = "late"
            counts[cat] += 1; loss_by[cat] += (acc_opt - acc_sel)
            N += 1; sum_opt += acc_opt; sum_sel += acc_sel
            # online arm: feed the calibrator the verify-time labels it observes
            # for free (chosen-head leading match + realized tail accept).
            if oc is not None:
                head_acc = 0
                for j in range(k_sel):
                    if m + 1 + j < len(gt) and block_full[j] == gt[m + 1 + j]:
                        head_acc += 1
                    else:
                        break
                tail_acc = (acc_sel - k_sel) if head_acc == k_sel else None
                oc.observe(conf, k_sel, head_acc, scores[k_sel], tail_acc)
            # advance along the SELECTED handoff (on-policy)
            tree = build_extension_chain(block_full[:k_sel],
                                         tk_by[k_sel][:max(0, num_spec - k_sel)])
            pth = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            accepted = [tree.tokens[i] for i in pth]
            nxt = [root] + accepted
            if m + 1 + len(pth) < len(gt):
                nxt.append(gt[m + 1 + len(pth)])
            suffix.add_response(nxt)
            m += 1 + len(pth) + 1

    frac = {k: counts[k] / N for k in counts}
    early_loss = loss_by["early"] / N
    late_loss = loss_by["late"] / N
    # MAT = mean accepted DRAFT tokens / round, EXCLUDING the bonus token (user
    # convention = project "crossover K"). Loss is a per-round difference.
    oracle_mat = sum_opt / N; compose_mat = sum_sel / N
    clab = "raw prob" if CALIB == "raw" else "online calib"
    print(f"{WL} [{clab}]: N={N} rounds")
    print(f"  early={frac['early']:.3f} optimal={frac['optimal']:.3f} late={frac['late']:.3f}")
    print(f"  oracle(decision) MAT={oracle_mat:.3f}  compose MAT={compose_mat:.3f}  "
          f"gap={oracle_mat - compose_mat:.3f}  (accepted tokens/round)")
    print(f"  early-loss={early_loss:.3f}  late-loss={late_loss:.3f}  accepted tokens/round")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- fig A: early / optimal / late proportions ----
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cats = ["early handoff", "optimal", "late handoff"]
    vals = [frac["early"] * 100, frac["optimal"] * 100, frac["late"] * 100]
    cols = [C_EARLY, C_OPT, C_LATE]
    b = ax.bar(cats, vals, color=cols, edgecolor="k", linewidth=0.5, width=0.62)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("share of handoff decisions (%)")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title(f"Handoff-position selection vs decision oracle\n({WL}, {clab} signals)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / f"handoff_position_ratio_{WL}_{CALIB}.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    print("saved ->", OUT / f"handoff_position_ratio_{WL}_{CALIB}.png")

    # ---- fig B: MAT-loss decomposition (early vs late) vs oracle ----
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    names = ["early handoff", "late handoff"]
    vals = [early_loss, late_loss]
    b = ax.bar(names, vals, color=[C_EARLY, C_LATE], edgecolor="k", linewidth=0.5, width=0.55)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12)
    ax.axhline(oracle_mat - compose_mat, ls="--", color="#555555", lw=1.6,
               label=f"total oracle gap = {oracle_mat - compose_mat:.3f}")
    ax.set_ylabel("MAT loss vs decision oracle (accepted tokens/round)")
    ax.set_ylim(0, max(vals + [oracle_mat - compose_mat]) * 1.2)
    ax.set_title(f"Handoff MAT loss decomposition ({WL}, {clab} signals)\n"
                 f"oracle {oracle_mat:.2f}  vs  compose {compose_mat:.2f} "
                 f"(accepted tokens/round)", fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / f"handoff_matloss_decomp_{WL}_{CALIB}.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    print("saved ->", OUT / f"handoff_matloss_decomp_{WL}_{CALIB}.png")

    json.dump({"workload": WL, "calib": CALIB, "N": N, "frac": frac,
               "early_loss": early_loss, "late_loss": late_loss,
               "oracle_mat": oracle_mat, "compose_mat": compose_mat},
              open(BV / f"handoff_{WL}_{CALIB}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
