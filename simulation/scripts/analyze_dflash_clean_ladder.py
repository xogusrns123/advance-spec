"""Qwen3-8B DFlash-vs-suffix OFFLINE select-1 ladder: decisive selection
accuracy + simulated MAT for raw / calib(histogram,isotonic,logistic,beta) /
oracle. DFlash has no live select-1 (block drafter), so this is a counterfactual
select-1 simulation along ground truth from the full-block capture
(capture_dflash_vs_suffix.py). 8B web_search has no runaway, so nothing to clean.

Each arm's calibrator maps a proposer prob (+depth) -> P(token==gt); pick suffix
iff cal_suffix > cal_dflash. Calibrators are fit GroupKFold-by-rid (held-out OOF).
MAT(policy) = mean over decode rounds of #consecutive-correct picks from depth 0.

Same bar style as the EAGLE3 mat_compare.png (via _ladder_style)."""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_style import ladder_bar, SELECT1_COLORS  # noqa: E402

CELL = "qwen3_8b"
PATH = f"simulation/results/chain_hybrid_perdepth/{CELL}_dflash/decisions_dflash_suffix.jsonl"
FIGDIR = "simulation/results/calib_why_analysis/figures"
XLABELS = ["raw\n(sp>dp)", "calib\nhistogram", "calib\nisotonic",
           "calib\nlogistic", "calib\nbeta", "oracle\n(GT)"]
EPS = 1e-6


def fit_predict(method, Xtr, ytr, Xte):
    """Calibrate prob(+depth)->P(correct). X cols = [prob, depth]."""
    p_tr, d_tr = Xtr[:, 0], Xtr[:, 1]; p_te = Xte[:, 0]
    if method == "histogram":                      # 2D prob x depth bin means
        pb = np.clip((p_tr * 8).astype(int), 0, 7)
        dbk = np.minimum(d_tr.astype(int), 3)
        tbl = {}; g = ytr.mean()
        for b in range(8):
            for dd in range(4):
                m = (pb == b) & (dbk == dd)
                if m.sum() >= 5: tbl[(b, dd)] = ytr[m].mean()
        pbe = np.clip((p_te * 8).astype(int), 0, 7)
        dbe = np.minimum(Xte[:, 1].astype(int), 3)
        return np.array([tbl.get((b, dd), g) for b, dd in zip(pbe, dbe)])
    if method == "isotonic":                        # 1D isotonic on prob
        ir = IsotonicRegression(out_of_bounds="clip").fit(p_tr, ytr)
        return ir.predict(p_te)
    if method == "logistic":                        # logistic on [prob, depth]
        lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        return lr.predict_proba(Xte)[:, 1]
    if method == "beta":                            # Kull beta calibration on prob
        pc = np.clip(p_tr, EPS, 1 - EPS)
        F = np.c_[np.log(pc), -np.log(1 - pc)]
        lr = LogisticRegression(max_iter=1000).fit(F, ytr)
        pce = np.clip(p_te, EPS, 1 - EPS)
        Fe = np.c_[np.log(pce), -np.log(1 - pce)]
        return lr.predict_proba(Fe)[:, 1]
    raise ValueError(method)


def oof_calibrate(rows, method):
    """Return per-row (cal_dflash, cal_suffix) via GroupKFold-by-rid OOF."""
    grp = np.array([r["rid"] for r in rows])
    dP = np.array([[r["dflash_p"], r["depth"]] for r in rows], float)
    sP = np.array([[(r["suffix_p"] if r["suffix_p"] is not None else 0.0), r["depth"]] for r in rows], float)
    yd = np.array([1 if r["dflash_tok"] == r["gt_tok"] else 0 for r in rows])
    ys = np.array([1 if (r["suffix_tok"] is not None and r["suffix_tok"] == r["gt_tok"]) else 0 for r in rows])
    cd = np.zeros(len(rows)); cs = np.zeros(len(rows))
    ng = len(set(grp.tolist()))
    for tr, te in GroupKFold(n_splits=min(5, ng)).split(dP, yd, grp):
        cd[te] = fit_predict(method, dP[tr], yd[tr], dP[te])
        cs[te] = fit_predict(method, sP[tr], ys[tr], sP[te])
    return cd, cs


def pick_suffix_arrays(rows, method):
    """Return per-row boolean: does the policy pick suffix?"""
    if method == "raw":
        return np.array([(r["suffix_p"] is not None and r["suffix_p"] > r["dflash_p"]) for r in rows])
    if method == "oracle":
        # pick the correct arm when exactly/at-least one is right; suffix iff suffix right
        out = []
        for r in rows:
            s_hit = r["suffix_tok"] is not None and r["suffix_tok"] == r["gt_tok"]
            e_hit = r["dflash_tok"] == r["gt_tok"]
            out.append(s_hit and not e_hit)   # prefer dflash on ties (both); only need suffix when only suffix right
        return np.array(out)
    cd, cs = oof_calibrate(rows, method)
    # suffix unavailable -> never pick it
    avail = np.array([r["suffix_tok"] is not None for r in rows])
    return (cs > cd) & avail


def selacc(rows, pick_suffix):
    """Decisive selection accuracy (alive-conditioned per round)."""
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["rid"], r["round"])].append((r["depth"], i, r))
    n = correct = 0
    for k in by:
        seq = sorted(by[k]); alive = True
        for depth, i, r in seq:
            if not alive: break
            e_hit = r["dflash_tok"] == r["gt_tok"]
            s_hit = r["suffix_tok"] is not None and r["suffix_tok"] == r["gt_tok"]
            oh = "both" if (e_hit and s_hit) else "eagle" if e_hit else "suffix" if s_hit else "none"
            if oh in ("eagle", "suffix"):
                n += 1
                ps = bool(pick_suffix[i])
                correct += (ps and oh == "suffix") or ((not ps) and oh == "eagle")
            if oh not in ("eagle", "suffix", "both"): alive = False
    return correct / max(n, 1)


def mat(rows, pick_suffix):
    """Simulated select-1 MAT = mean #consecutive-correct picks from depth 0."""
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["rid"], r["round"])].append((r["depth"], i, r))
    Ls = []
    for k in by:
        seq = sorted(by[k]); acc = 0
        for depth, i, r in seq:
            ps = bool(pick_suffix[i])
            tok = r["suffix_tok"] if ps else r["dflash_tok"]
            if tok is not None and tok == r["gt_tok"]:
                acc += 1
            else:
                break
        Ls.append(acc)
    return sum(Ls) / max(len(Ls), 1)


def main():
    rows = [json.loads(l) for l in open(PATH) if l.strip()]
    print(f"=== {CELL} DFlash-vs-suffix offline select-1 ladder ({len(rows)} rows) ===")
    methods = ["raw", "histogram", "isotonic", "logistic", "beta", "oracle"]
    sels, mats = [], []
    for m in methods:
        ps = pick_suffix_arrays(rows, m)
        sa = selacc(rows, ps); mt = mat(rows, ps)
        sels.append(sa); mats.append(mt)
        print(f"  {m:10s} sel.acc={sa:.4f}  MAT={mt:.4f}")

    ladder_bar(mats, XLABELS, "MAT (offline select-1 simulation)",
               "MAT: raw vs calibration (4 methods) vs oracle\n"
               "(Qwen3-8B DFlash vs suffix, offline select-1 sim on GT)",
               f"{FIGDIR}/dflash_mat_compare_clean.png", fmt="{:.3f}",
               colors=SELECT1_COLORS, star_idx=5)
    ladder_bar(sels, XLABELS, "decisive selection accuracy (alive-conditioned)",
               "Selection accuracy: raw vs calibration (4 methods) vs oracle\n"
               "(Qwen3-8B DFlash vs suffix, held-out OOF)",
               f"{FIGDIR}/dflash_selacc_compare_clean.png", fmt="{:.3f}",
               colors=SELECT1_COLORS, star_idx=5)


if __name__ == "__main__":
    main()
