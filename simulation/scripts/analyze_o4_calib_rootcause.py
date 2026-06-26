"""Root cause of why per-proposer calibration's SELECTION loses to raw.

Calibration is a per-group MONOTONE warp, so its only effect on the eagle-vs-
suffix decision is to MOVE the boundary off the raw diagonal (suffix_p>eagle_p).
We decompose why that move hurts by separating three hypotheses, all measured as
selection accuracy on GT-labeled decisive+contested decisions:

  H1 overfitting/variance : per-position maps (few samples at deep depths) overfit
        -> compare GLOBAL (low-variance) vs PER-POSITION maps, and TRAIN vs TEST.
  H2 selection bias       : the deployed maps are fit on raw-CHOSEN subsets (eagle
        samples only where eagle won, suffix only where suffix won)
        -> compare UNBIASED maps (fit on oracle labels: token==gt for ALL
        decisions) vs the deployed biased calib_pp map.
  H3 mis-specification     : even a perfect marginal calibration compares MARGINALS
        not the joint conditional -> does the BEST calib variant still lose to raw?

Maps are isotonic (non-parametric). Unbiased maps are fit on the TRAIN-oracle log
(eagle: P(eagle_token==gt | eagle_p); suffix: P(suffix_token==gt | suffix_p)) and
evaluated on the disjoint TEST-oracle log; train accuracy is also reported to
expose the train/test gap (overfitting). The deployed biased map is loaded from
the per-position calib_pp_isotonic.json (non-Jeffreys) as shipped.

Usage (container):
  python3 simulation/scripts/analyze_o4_calib_rootcause.py \
    --train-oracle simulation/results/o4_perdepth/qwen3_14b_disc_train/decisions_select1_oracle.jsonl \
    --test-oracle  simulation/results/o4_perdepth/qwen3_14b_disc/decisions_select1_oracle.jsonl \
    --deployed-map simulation/results/o4_perdepth/qwen3_14b_replay/calib_pp_isotonic.json \
    --out simulation/results/o4_perdepth/qwen3_14b_disc/figures/o4_calib_rootcause.png
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "/workspace/simulation/oracle")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
from chain_hybrid_patch import _ServingIsoCalibrator  # noqa: E402

MIN_N = 500


def load(path):
    """contested+decisive rows: eagle_p, suffix_p, depth, e_correct, s_correct.
    Also return the per-group marginal-fit arrays (each proposer's own proposals)."""
    ep, sp, dep, ec, sc = [], [], [], [], []
    # marginal-fit pools (unbiased: every proposal, label token==gt)
    e_p, e_y, e_d = [], [], []
    s_p, s_y, s_d = [], [], []
    # rule-conditioned pools (SAME log, only the subset the RAW rule would select
    # that proposer on). Each carries BOTH labels: token==gt (correctness) and
    # survival (accept_len>=depth+1, what the deployed fitter uses). Comparing the
    # two on identical rows isolates the LABEL from everything else.
    erc_p, erc_y, erc_sy, erc_d = [], [], [], []
    src_p, src_y, src_sy, src_d = [], [], [], []
    # phase 1: step records -> per-(rid,decode_step) accept_len (chain survival)
    accept_len = {}
    decisions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "step":
                accept_len[(r["rid"], r["decode_step"])] = r["accept_len"]
            elif r.get("type") == "decision" and not r.get("tail"):
                decisions.append(r)
    # phase 2: build pools
    for r in decisions:
            gt = r.get("gt_token")
            if gt is None:
                continue
            d = int(r["depth"])
            al = accept_len.get((r["rid"], r["decode_step"]))
            surv = None if al is None else (1 if al >= d + 1 else 0)
            ep_v = r.get("eagle_p"); sp_v = r.get("suffix_p")
            if ep_v is not None and r.get("eagle_token") is not None:
                e_p.append(float(ep_v)); e_d.append(d)
                e_y.append(1 if r["eagle_token"] == gt else 0)
                if (sp_v is None or float(ep_v) >= float(sp_v)) and surv is not None:
                    erc_p.append(float(ep_v)); erc_d.append(d)
                    erc_y.append(1 if r["eagle_token"] == gt else 0)
                    erc_sy.append(surv)
            if sp_v is not None and r.get("suffix_token") is not None:
                s_p.append(float(sp_v)); s_d.append(d)
                s_y.append(1 if r["suffix_token"] == gt else 0)
                if ep_v is not None and float(sp_v) > float(ep_v) and surv is not None:
                    src_p.append(float(sp_v)); src_d.append(d)
                    src_y.append(1 if r["suffix_token"] == gt else 0)
                    src_sy.append(surv)
            # decisive+contested eval rows
            if r.get("oracle_hit") in ("eagle", "suffix") \
                    and r.get("eagle_p") is not None and r.get("suffix_p") is not None:
                ep.append(float(r["eagle_p"])); sp.append(float(r["suffix_p"]))
                dep.append(d)
                ec.append(1 if r["eagle_token"] == gt else 0)
                sc.append(1 if r["suffix_token"] == gt else 0)
    ev = dict(ep=np.array(ep), sp=np.array(sp), dep=np.array(dep, int),
              ec=np.array(ec, int), sc=np.array(sc, int))
    fit = dict(e=(np.array(e_p), np.array(e_y, int), np.array(e_d, int)),
               s=(np.array(s_p), np.array(s_y, int), np.array(s_d, int)),
               e_rc=(np.array(erc_p), np.array(erc_y, int), np.array(erc_d, int)),
               s_rc=(np.array(src_p), np.array(src_y, int), np.array(src_d, int)),
               e_rc_surv=(np.array(erc_p), np.array(erc_sy, int), np.array(erc_d, int)),
               s_rc_surv=(np.array(src_p), np.array(src_sy, int), np.array(src_d, int)))
    return ev, fit


def iso_global(p, y):
    return IsotonicRegression(out_of_bounds="clip").fit(p, y)


def iso_perpos(p, y, d):
    g = iso_global(p, y)
    maps = {}
    for dd in np.unique(d):
        m = d == dd
        if m.sum() >= MIN_N:
            maps[int(dd)] = IsotonicRegression(out_of_bounds="clip").fit(p[m], y[m])
    return g, maps


def pp_predict(g, maps, p, d):
    out = np.empty(len(p))
    for dd in np.unique(d):
        m = d == dd
        mdl = maps.get(int(dd), g)
        out[m] = mdl.predict(p[m])
    return out


def sel_acc(cal_s, cal_e, ev):
    """accuracy + picks_suffix on decisive+contested. correct pick = the proposer
    whose token == gt (decisive: exactly one)."""
    pick_s = (cal_s > cal_e).astype(int)
    correct = np.where(pick_s == 1, ev["sc"], ev["ec"])
    return float(correct.mean()), float(pick_s.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-oracle", required=True)
    ap.add_argument("--test-oracle", required=True)
    ap.add_argument("--deployed-map", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tr_ev, tr_fit = load(args.train_oracle)
    te_ev, _ = load(args.test_oracle)
    opt_s = float(tr_ev["sc"].mean())  # suffix-right rate (train) for reference
    te_opt = float((te_ev["sc"] >= te_ev["ec"]).mean())  # always-pick-suffix-ish
    print(f"train eval n={len(tr_ev['ep'])} suffix-right={tr_ev['sc'].mean():.3f} | "
          f"test eval n={len(te_ev['ep'])} suffix-right={te_ev['sc'].mean():.3f}")

    # fit unbiased maps on TRAIN
    egp, egy, egd = tr_fit["e"]; sgp, sgy, sgd = tr_fit["s"]
    e_glob = iso_global(egp, egy); s_glob = iso_global(sgp, sgy)
    e_g2, e_pp = iso_perpos(egp, egy, egd)
    s_g2, s_pp = iso_perpos(sgp, sgy, sgd)
    print(f"unbiased fit: eagle n={len(egy)} ({len(e_pp)} per-pos depths), "
          f"suffix n={len(sgy)} ({len(s_pp)} per-pos depths)")

    rows = []  # (label, train_acc, test_acc, test_picks_suffix)

    def raw(ev): return ev["sp"], ev["ep"]
    rows.append(("raw (suffix_p>eagle_p)",
                 *sel_acc(*raw(tr_ev), tr_ev)[:1],
                 *sel_acc(*raw(te_ev), te_ev)))

    def glob(ev):
        return s_glob.predict(ev["sp"]), e_glob.predict(ev["ep"])
    rows.append(("calib GLOBAL unbiased",
                 sel_acc(*glob(tr_ev), tr_ev)[0],
                 *sel_acc(*glob(te_ev), te_ev)))

    def perpos(ev):
        return (pp_predict(s_g2, s_pp, ev["sp"], ev["dep"]),
                pp_predict(e_g2, e_pp, ev["ep"], ev["dep"]))
    rows.append(("calib PER-POS unbiased",
                 sel_acc(*perpos(tr_ev), tr_ev)[0],
                 *sel_acc(*perpos(te_ev), te_ev)))

    # rule-conditioned: SAME oracle log + SAME token==gt label, but each map fit
    # ONLY on the subset the raw rule would select that proposer on. Isolates the
    # selection-conditioning bias from trajectory/label-definition differences.
    ercp, ercy, ercd = tr_fit["e_rc"]; srcp, srcy, srcd = tr_fit["s_rc"]
    e_rcg, e_rcpp = iso_perpos(ercp, ercy, ercd)
    s_rcg, s_rcpp = iso_perpos(srcp, srcy, srcd)
    print(f"rule-conditioned fit: eagle n={len(ercy)} ({len(e_rcpp)} depths), "
          f"suffix n={len(srcy)} ({len(s_rcpp)} depths)")

    def rulecond(ev):
        return (pp_predict(s_rcg, s_rcpp, ev["sp"], ev["dep"]),
                pp_predict(e_rcg, e_rcpp, ev["ep"], ev["dep"]))
    rows.append(("calib PER-POS rule-cond [token==gt]",
                 sel_acc(*rulecond(tr_ev), tr_ev)[0],
                 *sel_acc(*rulecond(te_ev), te_ev)))

    # SAME rule-conditioned rows, but the SURVIVAL label (accept_len>=depth+1) the
    # deployed fitter actually uses (no GT in the raw run). Isolates the LABEL.
    esp, esy, esd = tr_fit["e_rc_surv"]; ssp, ssy, ssd = tr_fit["s_rc_surv"]
    e_sg, e_spp = iso_perpos(esp, esy, esd)
    s_sg, s_spp = iso_perpos(ssp, ssy, ssd)

    def survlab(ev):
        return (pp_predict(s_sg, s_spp, ev["sp"], ev["dep"]),
                pp_predict(e_sg, e_spp, ev["ep"], ev["dep"]))
    rows.append(("calib PER-POS rule-cond [SURVIVAL]",
                 sel_acc(*survlab(tr_ev), tr_ev)[0],
                 *sel_acc(*survlab(te_ev), te_ev)))

    # deployed biased per-position map (fit on raw-chosen subsets, non-Jeffreys)
    cal = _ServingIsoCalibrator.load(args.deployed_map)

    def deployed(ev):
        cs = np.array([cal.predict("suffix", ev["sp"][i], ev["sp"][i], int(ev["dep"][i]))
                       for i in range(len(ev["sp"]))])
        ce = np.array([cal.predict("eagle", ev["ep"][i], ev["ep"][i], int(ev["dep"][i]))
                       for i in range(len(ev["ep"]))])
        return cs, ce
    rows.append(("calib DEPLOYED (biased,per-pos)",
                 sel_acc(*deployed(tr_ev), tr_ev)[0],
                 *sel_acc(*deployed(te_ev), te_ev)))

    print(f"\n  {'variant':34s} {'train_acc':>9s} {'test_acc':>8s} "
          f"{'test_pick_s':>11s}")
    for lab, tra, tea, ps in rows:
        flag = ""
        if "raw" not in lab:
            flag = "  <overfit gap %.3f>" % (tra - tea)
        print(f"  {lab:34s} {tra:9.3f} {tea:8.3f} {ps:11.3f}{flag}")
    print(f"  {'ORACLE':34s} {1.0:9.3f} {1.0:8.3f} "
          f"{te_ev['sc'].mean():11.3f}")
    print(f"\n  test suffix-right (optimal pick_s) = {te_ev['sc'].mean():.3f}; "
          f"raw test pick_s above tells over/under-picking")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    labs = [r[0].replace(" ", "\n", 1) for r in rows] + ["ORACLE"]
    tea = [r[2] for r in rows] + [1.0]
    tra = [r[1] for r in rows] + [1.0]
    x = np.arange(len(labs)); w = 0.38
    ax1.bar(x - w/2, tra, w, color="#aec7e8", label="train acc")
    ax1.bar(x + w/2, tea, w, color="#1f77b4", label="test acc")
    ax1.axhline(rows[0][2], color="k", ls="--", lw=0.9,
                label=f"raw test={rows[0][2]:.3f}")
    for i, (a, b) in enumerate(zip(tra, tea)):
        ax1.text(i, max(a, b) + 0.005, f"{b:.3f}", ha="center", fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace("\n", " ") for l in labs], fontsize=6.5,
                        rotation=18, ha="right")
    ax1.set_ylabel("selection accuracy"); ax1.set_ylim(0.5, 1.02)
    ax1.set_title("Selection accuracy (train vs test)\n"
                  "train>>test = overfitting; all<raw = mis-specified", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    # variance panel: suffix marginal map (train) vs test empirical, by prob bin
    edges = np.linspace(0, 1, 16)
    sgp_te, sgy_te = None, None
    te_full, _ = load(args.test_oracle)  # reuse loader's fit pools via second call
    # recompute test suffix pool
    s_p_te, s_y_te = [], []
    with open(args.test_oracle) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail") or r.get("gt_token") is None:
                continue
            if r.get("suffix_p") is not None and r.get("suffix_token") is not None:
                s_p_te.append(float(r["suffix_p"]))
                s_y_te.append(1 if r["suffix_token"] == r["gt_token"] else 0)
    s_p_te = np.array(s_p_te); s_y_te = np.array(s_y_te)
    centers, tr_rate, te_rate, te_se = [], [], [], []
    idx_tr = np.clip(np.digitize(sgp, edges) - 1, 0, len(edges) - 2)
    idx_te = np.clip(np.digitize(s_p_te, edges) - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        mt = idx_tr == b; me = idx_te == b
        if mt.sum() < 20 or me.sum() < 20:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        tr_rate.append(sgy[mt].mean()); te_rate.append(s_y_te[me].mean())
        te_se.append(s_y_te[me].std() / np.sqrt(me.sum()))
    ax2.plot(centers, tr_rate, "-o", color="#d62728", ms=4,
             label="suffix accept-rate (TRAIN fit target)")
    ax2.errorbar(centers, te_rate, yerr=te_se, fmt="s--", color="#2ca02c", ms=4,
                 capsize=2, label="suffix accept-rate (TEST, ±SE)")
    ax2.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5)
    ax2.set_xlabel("suffix_p (raw count ratio c/n)")
    ax2.set_ylabel("P(suffix token == gt)")
    ax2.set_title("Is the prob->accept map stable train vs test?\n"
                  "(divergence = the variance/overfit the calib map chases)",
                  fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    fig.suptitle("Why per-proposer calibration's selection loses to raw — "
                 "overfit / bias / mis-spec decomposition (EAGLE3, 14B)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
