"""How many decisive-contested positions are UNFIXABLE by any feature-based selector?

The user's microscopic question: eagle_p=0.6, suffix_p=0.2, but SUFFIX is right.
No monotone calibration (and, if the features carry no signal, no joint selector
either) can fix this. We separate three nested populations on the DECISIVE-CONTESTED
set (accept-conditioned: walk each chain, stop at the first depth where NO available
proposer's token==gt; a position is decisive iff 0 < |hits| < |available|):

  1. RAW-INVERSION   : raw argmax(prob) picks a wrong proposer.  (headroom; a warp MIGHT fix)
     - confident vs near-tie by margin = p_rawwinner - max_{k in hits} p_k.
  2. JOINT-BAYES floor: fit OOF per-proposer P(k correct | FULL feature vector) with a
     flexible GBM (HistGradientBoosting, NaN-aware). pick = argmax_k p_k over available.
     1 - selacc(joint) ~= IRREDUCIBLE decisive-error: the floor NO selector beats.
  3. UNFIXABLE-CONFIDENT: confident raw-inversions where the JOINT model ALSO picks wrong
     -> the features genuinely point at the wrong proposer (the user's exact case).

Ladder of decisive selacc: raw  <  marginal-GBM (prob-only, == calibration ceiling)
  <  joint-GBM (full features, == Bayes)  <  oracle(=1). Plus per-proposer AUC(prob->own
correctness) on the decisive set (is the score even informative), and, within the
confident-inversion region, AUC of the losing-but-correct proposer's OWN extra features
(match_len, log count, depth) -> tests "even the other features are low-signal there".

Run (host, sklearn ok -- pure stdout, no figures/pickle):
  python3 simulation/scripts/select1_ladder/irreducible_cases.py
"""
import json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = "simulation/results/chain_hybrid_perdepth"
# name : (token_field, prob_field, has_suffix_extras)
CELLS = {
    "14B 2-way (eagle3+sfx)": ("qwen3_14b_ar", [
        ("eagle3", "eagle_token", "eagle_p", False),
        ("suffix", "suffix_token", "suffix_p", True)]),
    "27B 2-way (mtp+sfx)": ("qwen35_27b_ar", [
        ("mtp", "eagle_token", "eagle_p", False),
        ("suffix", "suffix_token", "suffix_p", True)]),
    "8B 3-way (df+e3+sfx)": ("qwen3_8b_dflash_e3_ceiling20", [
        ("dflash", "eagle_token", "eagle_p", False),
        ("eagle3", "e3_token", "e3_p", False),
        ("suffix", "suffix_token", "suffix_p", True)]),
    "27B 3-way (mtp+df+sfx)": ("qwen35_27b_3way_real_full", [
        ("mtp", "eagle_token", "eagle_p", False),
        ("dflash", "dflash_token", "dflash_p", False),
        ("suffix", "suffix_token", "suffix_p", True)]),
}
CONF_MARGIN = 0.20   # "confident" inversion threshold (p_winner - p_best_correct)
TIE_MARGIN = 0.05    # near-tie threshold


def load_chains(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains:
        chains[k].sort(key=lambda r: r["depth"])
    return chains


def loopy_rids(d, thresh=0.5, n=4):
    dd = Path(d); gtf = dd / "gt_tokens.jsonl"
    if not gtf.exists():
        return set()
    reqs = {}
    for line in open(dd / "decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "req":
            reqs[o["rid"]] = tuple(o["input_ids"])
    gt = {}
    for line in open(gtf):
        r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
    bad = set()
    for rid, ids in reqs.items():
        out = gt.get(ids)
        if not out or len(out) < n + 1:
            continue
        g = [tuple(out[i:i + n]) for i in range(len(out) - n + 1)]
        if len(set(g)) / max(len(g), 1) < thresh:
            bad.add(rid)
    return bad


def build_decisive(chains, props):
    """Return list of position dicts on the accept-conditioned decisive-contested set.
    Each: {rid, depth, avail:[names], hits:[names], prob:{name:p},
           feat:{name:[..own feats..]}, fullkeys for the joint vector}."""
    rows = []
    names = [p[0] for p in props]
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            tok = {p[0]: r.get(p[1]) for p in props}
            prob = {p[0]: r.get(p[2]) for p in props}
            avail = [nm for nm in names if tok[nm] is not None and prob[nm] is not None]
            if not avail:
                continue
            hits = [nm for nm in avail if gt is not None and tok[nm] == gt]
            if 0 < len(hits) < len(avail):
                rows.append({
                    "rid": rid, "depth": int(r["depth"]),
                    "avail": avail, "hits": set(hits),
                    "prob": {nm: float(prob[nm]) for nm in avail},
                    "match_len": (float(r["match_len"]) if r.get("match_len") is not None else np.nan),
                    "lcnt": (math.log1p(float(r["suffix_count"])) if r.get("suffix_count") is not None else np.nan),
                })
            if gt is not None and len(hits) == 0:
                alive = False
    return rows, names


def full_vector(rows, names):
    """Joint feature matrix: [prob_name1..probN, depth, match_len, lcnt]. Missing
    proposer prob -> NaN (HGB handles natively)."""
    X = []
    for e in rows:
        v = [e["prob"].get(nm, np.nan) for nm in names]
        v += [float(e["depth"]), e["match_len"], e["lcnt"]]
        X.append(v)
    return np.asarray(X, float)


def own_vector(rows, nm, has_extra):
    """Per-proposer marginal feature matrix on rows where nm is available:
    [prob_nm, depth] (+[match_len, lcnt] for suffix). Returns X, y(1=correct), g(rid), idx."""
    X, y, g, idx = [], [], [], []
    for i, e in enumerate(rows):
        if nm not in e["avail"]:
            continue
        v = [e["prob"][nm], float(e["depth"])]
        if has_extra:
            v += [e["match_len"], e["lcnt"]]
        X.append(v); y.append(1 if nm in e["hits"] else 0); g.append(e["rid"]); idx.append(i)
    return np.asarray(X, float), np.asarray(y), np.asarray(g), np.asarray(idx)


def oof_gbm(X, y, g):
    pred = np.full(len(y), float(y.mean()) if len(y) else 0.0)
    if len(y) < 20 or len(set(y)) < 2:
        return pred
    ng = len(set(g))
    for tr, te in GroupKFold(min(5, ng)).split(X, y, g):
        if len(set(y[tr])) < 2:
            pred[te] = y[tr].mean(); continue
        m = HGB(max_depth=3, max_iter=200, learning_rate=0.06, l2_regularization=1.0).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def selacc(rows, pick_per_row):
    ok = [1.0 if pick_per_row[i] in rows[i]["hits"] else 0.0 for i in range(len(rows))]
    return float(np.mean(ok))


def analyze(name, dirname, props):
    d = f"{ROOT}/{dirname}"
    chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
    bad = loopy_rids(d)
    chains = {k: v for k, v in chains.items() if k[0] not in bad}
    rows, names = build_decisive(chains, props)
    N = len(rows)
    n_chains = len(chains)
    print(f"\n{'='*78}\n{name}   dir={dirname}  loopy-excl={len(bad)}  chains={n_chains}  decisive={N}")
    if N < 30:
        print("  too few decisive positions; skip"); return None

    # --- per-proposer OOF probs: marginal (own feats) and joint (full vector) ---
    Xfull = full_vector(rows, names)
    pj = {nm: np.full(N, np.nan) for nm in names}   # joint P(k correct|full x)
    pm = {nm: np.full(N, np.nan) for nm in names}   # marginal P(k correct|own feats)
    base = {}
    auc_prob = {}
    for nm, _, _, extra in props:
        Xo, y, g, idx = own_vector(rows, nm, extra)
        if not len(y):
            continue
        base[nm] = float(y.mean())
        # AUC of raw prob -> own correctness (is the score informative at all)
        try:
            auc_prob[nm] = roc_auc_score(y, Xo[:, 0]) if len(set(y)) == 2 else float("nan")
        except Exception:
            auc_prob[nm] = float("nan")
        pm_k = oof_gbm(Xo, y, g)
        for j, i in enumerate(idx):
            pm[nm][i] = pm_k[j]
        # joint: same label/rows, but features = full joint vector
        Xj = Xfull[idx]
        pj_k = oof_gbm(Xj, y, g)
        for j, i in enumerate(idx):
            pj[nm][i] = pj_k[j]

    def argmax_pick(score):
        out = []
        for i, e in enumerate(rows):
            av = e["avail"]
            out.append(max(av, key=lambda nm: (score[nm][i] if not np.isnan(score[nm][i]) else -1)))
        return out

    raw_pick = [max(e["avail"], key=lambda nm: e["prob"][nm]) for e in rows]
    mrg_pick = argmax_pick(pm)
    jnt_pick = argmax_pick(pj)

    sa_raw = selacc(rows, raw_pick)
    sa_mrg = selacc(rows, mrg_pick)
    sa_jnt = selacc(rows, jnt_pick)

    # --- inversion decomposition ---
    inv = confident = tie = unfix_conf = 0
    margins = []
    for i, e in enumerate(rows):
        w = raw_pick[i]
        if w in e["hits"]:
            continue
        inv += 1
        best_correct_p = max(e["prob"][nm] for nm in e["hits"])
        m = e["prob"][w] - best_correct_p
        margins.append(m)
        if m >= CONF_MARGIN:
            confident += 1
            if jnt_pick[i] not in e["hits"]:
                unfix_conf += 1
        elif m < TIE_MARGIN:
            tie += 1
    margins = np.asarray(margins) if margins else np.asarray([0.0])

    print(f"  base correctness (proposer right | available): "
          + ", ".join(f"{nm}={base.get(nm, float('nan')):.3f}" for nm in names))
    print(f"  AUC(raw prob -> own correctness, decisive): "
          + ", ".join(f"{nm}={auc_prob.get(nm, float('nan')):.3f}" for nm in names))
    print(f"\n  decisive selacc ladder:")
    print(f"     raw (argmax prob)         = {sa_raw:.4f}")
    print(f"     marginal-GBM (calib ceil) = {sa_mrg:.4f}   (+{sa_mrg-sa_raw:+.4f} vs raw)")
    print(f"     joint-GBM (Bayes)         = {sa_jnt:.4f}   (+{sa_jnt-sa_raw:+.4f} vs raw)")
    print(f"     oracle                    = 1.0000")
    print(f"  IRREDUCIBLE decisive-error (1 - joint Bayes) = {1-sa_jnt:.4f}  "
          f"({1-sa_jnt:.1%} of decisive picks unfixable by ANY feature-selector)")
    print(f"\n  RAW-INVERSION = {inv}/{N} ({inv/N:.1%} of decisive) "
          f"[headroom for any selector]")
    print(f"     near-tie  (margin<{TIE_MARGIN}) = {tie} ({tie/max(inv,1):.1%} of inversions)  "
          f"<- recalibration-fixable region")
    print(f"     CONFIDENT (margin>={CONF_MARGIN}) = {confident} ({confident/max(inv,1):.1%} of inversions)  "
          f"<- the eagle0.6/sfx0.2 case")
    print(f"        of which JOINT-Bayes ALSO wrong = {unfix_conf} "
          f"({unfix_conf/max(confident,1):.1%} of confident) -> UNFIXABLE-CONFIDENT "
          f"= {unfix_conf}/{N} ({unfix_conf/N:.2%} of decisive)")
    print(f"     inversion margin: mean={margins.mean():.3f} median={np.median(margins):.3f} "
          f"p90={np.percentile(margins,90):.3f}")

    # --- region-deadness: within confident inversions, do the losing-correct proposer's
    #     OWN extra features separate "it is actually right"? (the user's worry) ---
    #     measured as AUC over the confident-inversion-prone region per proposer
    print(f"  region-deadness (confident-inversion rows, AUC of own feats -> correct):")
    for nm, _, _, extra in props:
        # rows where nm is the LOSING-but-correct proposer in a confident inversion
        sub_p, sub_d, sub_ml, sub_lc, sub_y = [], [], [], [], []
        for i, e in enumerate(rows):
            w = raw_pick[i]
            if w in e["hits"] or nm not in e["avail"]:
                continue
            best_correct_p = max(e["prob"][k] for k in e["hits"])
            if e["prob"][w] - best_correct_p < CONF_MARGIN:
                continue
            # this is a confident inversion; is nm right here?
            sub_p.append(e["prob"][nm]); sub_d.append(e["depth"])
            sub_ml.append(e["match_len"]); sub_lc.append(e["lcnt"])
            sub_y.append(1 if nm in e["hits"] else 0)
        sy = np.asarray(sub_y)
        if len(sy) < 30 or len(set(sy)) < 2:
            print(f"     {nm:8} (n={len(sy)}, degenerate)"); continue
        feats = {"prob": np.asarray(sub_p), "depth": np.asarray(sub_d, float)}
        if extra:
            feats["match_len"] = np.asarray(sub_ml); feats["lcnt"] = np.asarray(sub_lc)
        aucs = []
        for fn, fv in feats.items():
            ok = ~np.isnan(fv)
            if ok.sum() < 30 or len(set(sy[ok])) < 2:
                aucs.append(f"{fn}=na"); continue
            try:
                a = roc_auc_score(sy[ok], fv[ok])
            except Exception:
                a = float("nan")
            aucs.append(f"{fn}={a:.3f}")
        print(f"     {nm:8} n={len(sy)} P(right)={sy.mean():.3f}  AUC: " + ", ".join(aucs))

    return dict(name=name, N=N, sa_raw=sa_raw, sa_mrg=sa_mrg, sa_jnt=sa_jnt,
                irr=1-sa_jnt, inv=inv/N, conf=confident/N, unfix_conf=unfix_conf/N)


def main():
    summ = []
    for name, (dirname, props) in CELLS.items():
        try:
            r = analyze(name, dirname, props)
            if r:
                summ.append(r)
        except Exception as e:
            import traceback
            print(f"\n{name}: ERROR {e}"); traceback.print_exc()
    print(f"\n\n{'='*78}\nSUMMARY  (decisive-contested, accept-conditioned)")
    print(f"{'cell':24} {'N':>7} {'raw':>6} {'calib':>6} {'bayes':>6} {'IRRED':>6} "
          f"{'inv%':>6} {'conf%':>6} {'unfix%':>7}")
    for r in summ:
        print(f"{r['name']:24} {r['N']:>7} {r['sa_raw']:>6.3f} {r['sa_mrg']:>6.3f} "
              f"{r['sa_jnt']:>6.3f} {r['irr']:>6.3f} {r['inv']*100:>5.1f} "
              f"{r['conf']*100:>5.1f} {r['unfix_conf']*100:>6.2f}")
    print("\n  raw/calib/bayes = decisive selacc; IRRED = 1-bayes (irreducible floor);")
    print("  inv% = raw-inversion rate; conf% = confident-inversion rate (margin>=.2);")
    print("  unfix% = confident inversions the JOINT Bayes ALSO gets wrong (truly unfixable).")


if __name__ == "__main__":
    main()
