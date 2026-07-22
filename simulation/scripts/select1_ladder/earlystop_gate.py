"""Early-stop gate enabled by calibrated per-depth accept-probability.

The first analysis noted: in the calibrated [0,1] domain, `max_i f_i(depth) < tau`
is a principled early-stop -- stop drafting when even the best proposer's
accept-conditioned accept-prob is low, since deeper draft tokens will be
rejected anyway. Raw softmax/count-ratio scores can't do this. This quantifies
the wall-clock payoff OFFLINE on the oracle-log chains (GT-pinned trajectory).

Method:
  * accept label per (proposer, depth) = token==gt (counterfactual, both proposers).
  * ACCEPT-CONDITIONED fit: keep only alive-prefix rows (all shallower depths hit)
    -> OOF GBM P(accept | prob, depth[, match_len, lcnt]) per proposer.
  * gate g_d = max(cal_eagle_d, cal_suffix_d).
  * true accept boundary of a step = # leading hits under the ORACLE policy
    (survive at d iff oracle_hit in {eagle,suffix,both}); drafted-without-gate = S.
  * early-stop at d* = first alive depth with g_d < tau; drafted = d*+1 (we still
    draft d* to read its gate), realized accept = min(boundary, d*).
  * sweep tau: report MAT retained vs mean drafted depth (cost). The knee = free
    tail we can stop drafting with ~no MAT loss.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import irreducible_cases as ic  # noqa: E402

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = {"27b": "qwen35_27b_ar", "14b": "qwen3_14b_ar"}
HIT = ("eagle", "suffix", "both")


def oof_cal(X, y, g):
    """OOF P(accept). Returns array aligned to X (mean-fallback if degenerate)."""
    pred = np.full(len(y), float(y.mean()) if len(y) else 0.0)
    if len(y) < 40 or len(set(y)) < 2:
        return pred
    ng = len(set(g))
    for tr, te in GroupKFold(min(5, ng)).split(X, y, g):
        if len(set(y[tr])) < 2:
            pred[te] = y[tr].mean(); continue
        m = HGB(max_depth=3, max_iter=200, learning_rate=0.06,
                l2_regularization=1.0).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def run(tag, d):
    path = f"{ROOT}/{d}/decisions_select1_oracle.jsonl"
    chains = ic.load_chains(path)
    bad = ic.loopy_rids(f"{ROOT}/{d}")
    chains = {k: v for k, v in chains.items() if k[0] not in bad}

    # accept-conditioned rows for calibration fit: alive iff all shallower hit.
    rows = []  # (rid, ds, depth, ep, et, sp, st, ml, lcnt, gt, oh)
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            oh = r.get("oracle_hit"); gt = r.get("gt_token")
            rows.append((rid, ds, int(r["depth"]),
                         r.get("eagle_p"), r.get("eagle_token"),
                         r.get("suffix_p"), r.get("suffix_token"),
                         r.get("match_len"), r.get("suffix_count"), gt, oh))
            if gt is None or oh not in HIT:
                alive = False

    # fit per-proposer OOF calibrated accept-prob on these alive rows
    def fit(kind):
        X, y, g, idx = [], [], [], []
        for i, t in enumerate(rows):
            rid, ds, dep, ep, et, sp, st, ml, cnt, gt, oh = t
            if gt is None:
                continue
            if kind == "eagle" and ep is not None and et is not None:
                X.append([float(ep), float(dep)]); y.append(1 if et == gt else 0)
                g.append(rid); idx.append(i)
            if kind == "suffix" and sp is not None and st is not None:
                X.append([float(sp), float(dep),
                          float(ml) if ml is not None else np.nan,
                          float(np.log1p(cnt)) if cnt is not None else np.nan])
                y.append(1 if st == gt else 0); g.append(rid); idx.append(i)
        if not y:
            return {}
        p = oof_cal(np.asarray(X, float), np.asarray(y), np.asarray(g))
        return {idx[j]: float(p[j]) for j in range(len(idx))}

    cal_e = fit("eagle"); cal_s = fit("suffix")
    # gate per row = max calibrated accept-prob (raw fallback where uncalibrated)
    gate = {}
    for i, t in enumerate(rows):
        ce = cal_e.get(i, t[3] if t[3] is not None else 0.0)
        cs = cal_s.get(i, t[5] if t[5] is not None else 0.0)
        gate[(t[0], t[1], t[2])] = max(ce, cs)

    # per-step: oracle accept boundary + S + gated stop depth
    steps = defaultdict(list)
    for t in rows:
        steps[(t[0], t[1])].append(t)
    for k in steps:
        steps[k].sort(key=lambda r: r[2])

    def sim(tau):
        mats, drafted = [], []
        for k, rs in steps.items():
            S = len(rs)
            # oracle accept boundary = leading hits
            boundary = 0
            for r in rs:
                if r[9] is not None and r[10] in HIT:
                    boundary += 1
                else:
                    break
            # gate stop: first depth with gate < tau
            dstar = S
            for r in rs:
                if gate[(r[0], r[1], r[2])] < tau:
                    dstar = r[2]  # stop BEFORE drafting deeper; we drafted up to here
                    break
            mats.append(min(boundary, dstar))
            drafted.append(min(S, dstar + 1))
        return np.mean(mats), np.mean(drafted), np.mean([len(v) for v in steps.values()])

    full_mat, _, S_mean = sim(-1.0)  # tau<0 never stops
    print(f"\n===== {tag} early-stop gate (oracle policy, accept-conditioned calib) =====")
    print(f"  no-gate: MAT(oracle boundary)={full_mat:.3f}  mean chain len S={S_mean:.2f}  "
          f"(mean wasted draft depth = S - MAT = {S_mean-full_mat:.2f})")
    print(f"  {'tau':>5} {'MAT':>7} {'MAT%kept':>9} {'drafted':>8} {'draft%':>7} {'saved/step':>11}")
    for tau in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
        mat, dr, _ = sim(tau)
        print(f"  {tau:>5.2f} {mat:>7.3f} {100*mat/full_mat:>8.1f}% {dr:>8.2f} "
              f"{100*dr/S_mean:>6.1f}% {S_mean-dr:>10.2f}")


def main():
    for tag, d in CELLS.items():
        run(tag, d)


if __name__ == "__main__":
    main()
