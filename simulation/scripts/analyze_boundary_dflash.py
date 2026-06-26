"""OFFLINE Panel-B boundary ladder for DFlash (Qwen3-8B) — same format as the
served MTP/EAGLE3 boundary figures, but OFFLINE: DFlash is a block drafter and
cannot run live chain-hybrid select-1, so every bar is a counterfactual select-1
simulation on the captured DFlash-vs-suffix decisions (consistent with all other
DFlash analysis being offline).

selacc (5 bars): raw / best calib(method) / best-monotone ceiling / 0.5-Bayes / oracle
MAT   (7 bars): DFlash-only, suffix-only (single proposers) + the above 5.
Classifiers (mono/Bayes) fit on (dflash_p, suffix_p, depth), decisive rows,
GroupKFold-by-rid OOF. Colors match: mono=darkorange, Bayes=lime."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_style import ladder_bar  # noqa: E402
from analyze_dflash_clean_ladder import oof_calibrate, selacc, mat, pick_suffix_arrays  # noqa: E402

CELL = "qwen3_8b"
PATH = f"simulation/results/chain_hybrid_perdepth/{CELL}_dflash/decisions_dflash_suffix.jsonl"
FIGDIR = "simulation/results/calib_why_analysis/figures"
METHODS = ("histogram", "isotonic", "logistic", "beta")
C_DRAFT = "#17becf"; C_SUFFIX = "#8c564b"; C_RAW = "#7f7f7f"; C_CALIB = "#e377c2"
C_MONO = "darkorange"; C_BAYES = "lime"; C_ORACLE = "#d62728"


def oof_gbm(rows, monotone):
    """Per-row OOF P(suffix-wins) from a (dflash_p, suffix_p, depth) GBM fit on
    decisive rows. monotone -> calibration-framework ceiling."""
    X = np.array([[r["dflash_p"], (r["suffix_p"] if r["suffix_p"] is not None else 0.0),
                   r["depth"]] for r in rows], float)
    grp = np.array([r["rid"] for r in rows])
    isdec = np.array([r["oracle_hit"] in ("eagle", "suffix") for r in rows])
    y = np.array([1 if r["oracle_hit"] == "suffix" else 0 for r in rows])
    pred = np.zeros(len(rows))
    cst = [-1, 1, 0] if monotone else None
    ng = len(set(grp.tolist()))
    for tr, te in GroupKFold(n_splits=min(5, ng)).split(X, y, grp):
        trd = tr[isdec[tr]]
        if len(set(y[trd].tolist())) < 2:
            continue
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                           monotonic_cst=cst).fit(X[trd], y[trd])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def main():
    rows = [json.loads(l) for l in open(PATH) if l.strip()]
    avail = np.array([r["suffix_tok"] is not None for r in rows])
    # best calibration method by selacc
    cal_sel = {m: selacc(rows, pick_suffix_arrays(rows, m)) for m in METHODS}
    best_m = max(cal_sel, key=lambda m: cal_sel[m])
    pick_mono = (oof_gbm(rows, True) > 0.5) & avail
    pick_bayes = (oof_gbm(rows, False) > 0.5) & avail

    policies = [
        ("raw\n(sp>dp)", pick_suffix_arrays(rows, "raw"), C_RAW),
        (f"best calib\n({best_m})", pick_suffix_arrays(rows, best_m), C_CALIB),
        ("calib ceiling\n(best-mono)", pick_mono, C_MONO),
        ("0.5-Bayes", pick_bayes, C_BAYES),
        ("oracle\n(GT)", pick_suffix_arrays(rows, "oracle"), C_ORACLE),
    ]
    labels = [p[0] for p in policies]; cols = [p[2] for p in policies]
    sels = [selacc(rows, p[1]) for p in policies]
    mats = [mat(rows, p[1]) for p in policies]
    print(f"=== Qwen3-8B DFlash boundary (best calib={best_m}) OFFLINE ===")
    for lab, sa, mt in zip(labels, sels, mats):
        print(f"  {lab.replace(chr(10),' '):22s} selacc={sa:.4f}  MAT={mt:.4f}")

    ladder_bar(sels, labels, "decisive selection accuracy (alive-conditioned)",
               "Selection accuracy by Panel-B boundary policy (OFFLINE)\n"
               "(Qwen3-8B DFlash vs suffix, simulated select-1 on GT)",
               f"{FIGDIR}/boundary_selacc_dflash.png", fmt="{:.3f}", colors=cols)

    # MAT: prepend single-proposer floors (always-pick-X select-1 sim)
    draft_only = np.zeros(len(rows), bool)
    suffix_only = avail.copy()
    d_mat = mat(rows, draft_only); s_mat = mat(rows, suffix_only)
    print(f"  DFlash-only={d_mat:.4f}  suffix-only={s_mat:.4f}")
    m_labels = ["DFlash\nonly", "suffix\nonly"] + labels
    m_vals = [d_mat, s_mat] + mats
    m_cols = [C_DRAFT, C_SUFFIX] + cols
    ladder_bar(m_vals, m_labels, "MAT (offline select-1 simulation)",
               "MAT by Panel-B boundary policy (OFFLINE)\n"
               "(Qwen3-8B DFlash vs suffix, simulated select-1 on GT; single proposers = always-pick-X)",
               f"{FIGDIR}/boundary_mat_dflash.png", fmt="{:.3f}", colors=m_cols)


if __name__ == "__main__":
    main()
