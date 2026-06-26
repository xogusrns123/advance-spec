"""Selection accuracy + MAT bars for the Panel-B boundary-based selection
policies, one set per drafter (EAGLE3 / MTP). All simulated from the cell's
oracle decision log (ep, sp, oracle_hit) so every policy shares the same chains.

Bars (6):
  draft-only          single proposer (always pick the draft arm)
  suffix-only         single proposer (always pick suffix)
  best calib (METHOD) best of the 4 cond-trained calibration maps
  calib ceiling       best-monotone boundary  (Panel B ORANGE line)
  0.5-Bayes           unconstrained Bayes 0.5 boundary (Panel B LIME line)
  oracle              perfect selection

Colors match Panel B: calib-ceiling = darkorange, 0.5-Bayes = lime.
Classifiers (mono / bayes) fit on (ep, sp) over decisive-alive rows, GroupKFold-
by-rid OOF. best-calib applies the train-fit cond-trained maps (held-out)."""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_style import ladder_bar  # noqa: E402

ROOT = "simulation/results/chain_hybrid_perdepth"
FIGDIR = "simulation/results/calib_why_analysis/figures"
ALIVE = {"eagle", "suffix", "both"}
METHODS = ("histogram", "isotonic", "logistic", "beta")
CELLS = {
    "eagle3": {"dir": f"{ROOT}/qwen3_14b_ar", "draft_color": "#1f77b4",
               "draft_label": "EAGLE3", "title": "Qwen3-14B EAGLE3", "tag": "eagle3"},
    "mtp": {"dir": f"{ROOT}/qwen35_27b_ar", "draft_color": "#9467bd",
            "draft_label": "MTP", "title": "Qwen3.5-27B MTP", "tag": "mtp"},
}
# Panel-B-matched + non-overlapping
C_SUFFIX = "#8c564b"; C_RAW = "#7f7f7f"; C_CALIB = "#e377c2"
C_CEIL = "darkorange"; C_BAYES = "lime"; C_ORACLE = "#d62728"


def load_chains(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains: chains[k].sort(key=lambda r: r["depth"])
    return chains


def load_calmaps(d):
    maps = {}
    for m in METHODS:
        p = f"{d}/calib_cond-trained/calib_pp_{m}.json"
        try:
            g = json.load(open(p))["groups"]
            maps[m] = {grp: {int(dep): (np.asarray(v["x"], float), np.asarray(v["y"], float))
                             for dep, v in dd.items()} for grp, dd in g.items()}
        except Exception:
            pass
    return maps


def cal(mp, grp, p, d):
    m = mp[grp]
    if d in m: xs, ys = m[d]
    else:
        le = [k for k in m if k <= d]; xs, ys = m[max(le)] if le else m[max(m)]
    return max(float(np.interp(p, xs, ys)), 1e-6)


def correct_under(pick_suffix, h):
    """Is the policy's pick correct at this row? (h = oracle_hit)"""
    if pick_suffix:
        return h in ("suffix", "both")
    return h in ("eagle", "both")


def selacc_mat(chains, pick_of):
    """pick_of(row) -> bool pick_suffix. Returns (decisive selacc, MAT)."""
    n = corr = 0; Ls = []
    for rs in chains.values():
        alive = True
        Lp = 0; pol_alive = True
        for r in rs:
            h = r.get("oracle_hit")
            ps = pick_of(r)
            if alive and h in ("eagle", "suffix"):      # decisive selacc, alive prefix
                n += 1
                corr += int(correct_under(ps, h))
            if pol_alive:
                if h in ALIVE and correct_under(ps, h): Lp += 1
                else: pol_alive = False
            if h not in ALIVE: alive = False
        Ls.append(Lp)
    return corr / max(n, 1), sum(Ls) / max(len(Ls), 1)


def oof_pred(chains, monotone):
    """GroupKFold-by-rid OOF suffix-prob for every (ep,sp)-present row. Fit on
    decisive-alive rows; predict held-out groups in BATCH. Features = (ep, sp,
    depth) so the classifier is depth-aware like the cond-trained calib maps
    (fair comparison). monotone=True -> calibration-framework ceiling
    (monotone in ep DOWN / sp UP, depth unconstrained); False -> unconstrained."""
    all_rows = []; all_X = []; all_rid = []
    fX = []; fy = []; frid = []
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            h = r.get("oracle_hit"); ep = r.get("eagle_p"); sp = r.get("suffix_p"); dep = r["depth"]
            if ep is not None and sp is not None:
                all_rows.append(r); all_X.append([ep, sp, dep]); all_rid.append(rid)
                if alive and h in ("eagle", "suffix"):
                    fX.append([ep, sp, dep]); fy.append(1 if h == "suffix" else 0); frid.append(rid)
            if h not in ALIVE: alive = False
    all_X = np.array(all_X); all_rid = np.array(all_rid)
    fX = np.array(fX); fy = np.array(fy); frid = np.array(frid)
    pred = np.zeros(len(all_rows))
    cst = [-1, 1, 0] if monotone else None    # ep DOWN, sp UP, depth unconstrained
    ng = len(set(frid.tolist()))
    for tr, te in GroupKFold(n_splits=min(5, ng)).split(fX, fy, frid):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                             monotonic_cst=cst).fit(fX[tr], fy[tr])
        mask = np.isin(all_rid, list(set(frid[te].tolist())))
        if mask.any():
            pred[mask] = clf.predict_proba(all_X[mask])[:, 1]
    return {id(all_rows[i]): float(pred[i]) for i in range(len(all_rows))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["eagle3", "mtp", "both"], default="both")
    args = ap.parse_args()
    items = CELLS.items() if args.cell == "both" else [(args.cell, CELLS[args.cell])]
    for key, c in items:
        chains = load_chains(f"{c['dir']}/decisions_select1_oracle.jsonl")
        maps = load_calmaps(c["dir"])

        # best calibration method (by selacc)
        cal_sel = {}
        for m in maps:
            def pk(r, _m=m):
                sp = r.get("suffix_p"); ep = r.get("eagle_p")
                if sp is None or ep is None: return False
                return cal(maps[_m], "suffix", sp, r["depth"]) > cal(maps[_m], "eagle", ep, r["depth"])
            cal_sel[m] = selacc_mat(chains, pk)
        best_m = max(cal_sel, key=lambda m: cal_sel[m][0]) if cal_sel else None

        pred_mono = oof_pred(chains, monotone=True)
        pred_bayes = oof_pred(chains, monotone=False)

        def pk_calib(r):
            sp = r.get("suffix_p"); ep = r.get("eagle_p")
            if sp is None or ep is None: return False
            return cal(maps[best_m], "suffix", sp, r["depth"]) > cal(maps[best_m], "eagle", ep, r["depth"])

        def pk_raw(r):
            sp = r.get("suffix_p"); ep = r.get("eagle_p")
            return sp is not None and ep is not None and sp > ep

        policies = [
            (f"{c['draft_label']}\nonly", lambda r: False, c["draft_color"]),
            ("suffix\nonly", lambda r: True, C_SUFFIX),
            ("raw\n(sp>ep)", pk_raw, C_RAW),
            (f"best calib\n({best_m})", pk_calib, C_CALIB),
            ("calib ceiling\n(best-mono)", lambda r: pred_mono.get(id(r), 0.0) > 0.5, C_CEIL),
            ("0.5-Bayes", lambda r: pred_bayes.get(id(r), 0.0) > 0.5, C_BAYES),
            ("oracle\n(GT)", lambda r: r.get("oracle_hit") == "suffix", C_ORACLE),
        ]
        labels = [p[0] for p in policies]; colors = [p[2] for p in policies]
        sels = []; mats = []
        for _, pk, _c in policies:
            sa, mt = selacc_mat(chains, pk)
            sels.append(sa); mats.append(mt)
        print(f"=== {c['title']} (best calib={best_m}) ===")
        for l, sa, mt in zip(labels, sels, mats):
            print(f"  {l.replace(chr(10),' '):22s} selacc={sa:.4f}  MAT={mt:.4f}")

        # selection accuracy: single proposers make NO selection (they always
        # pick one arm) -> exclude them; only the selection policies belong here.
        si = [i for i, l in enumerate(labels) if "only" not in l]
        ladder_bar([sels[i] for i in si], [labels[i] for i in si],
                   "decisive selection accuracy (alive-conditioned)",
                   "Selection accuracy by Panel-B boundary policy\n"
                   f"({c['title']}, all tasks; simulated on oracle trajectory)",
                   f"{FIGDIR}/boundary_selacc_{c['tag']}.png", fmt="{:.3f}",
                   colors=[colors[i] for i in si])
        # MAT: single-proposer MAT IS meaningful -> keep all bars
        ladder_bar(mats, labels, "MAT (per-step accept length)",
                   "MAT by Panel-B boundary policy\n"
                   f"({c['title']}, all tasks; simulated on oracle trajectory)",
                   f"{FIGDIR}/boundary_mat_{c['tag']}.png", fmt="{:.3f}", colors=colors)


if __name__ == "__main__":
    main()
