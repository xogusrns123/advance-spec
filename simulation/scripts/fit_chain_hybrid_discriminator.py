"""Fit the JOINT discriminator arms (logistic + beta) from a TRAIN-slice oracle
decision log.

The comparative label (which proposer == GT, oracle_hit) is logged ONLY in
oracle mode, so the discriminator must be fit from an ORACLE run on the TRAIN
tasks — disjoint from the test slice the arms are later evaluated on.

Feature set (our-Bayes spec): suffix_p, eagle_p, match_len, suffix_count,
suffix_total, + depth (--with-depth, default ON). The logistic arm uses the two
probs raw; the beta arm encodes each prob as [ln p, ln(1-p)] (beta calibration
generalised to several inputs). The feature VECTOR is built by
chain_hybrid_patch._ServingDiscriminator.features so fitting and serving stay
byte-for-byte identical.

Training population: ALIVE-conditioned by default (--accept-conditioned) -- only
decisive rows whose chain prefix was still alive (every shallower depth of the
(rid, decode_step) had oracle_hit in {both,eagle,suffix}). This matches the
offline "our-Bayes" selector (alive-decisive AUC ~0.87 vs pooled ~0.78) and the
positions that actually drive MAT; --no-accept-conditioned reproduces the legacy
pooled fit. The blob stamps {accept_conditioned, with_depth} so serving asserts
the same feature layout.

Writes disc_logistic.json + disc_beta.json into --out-dir, each carrying
{kind, feature_names, mean, std, coef, intercept} — everything serving needs to
predict P(pick suffix) with no sklearn (standardize, dot, sigmoid). Prints train
+ held-out (GroupKFold by rid) AUC and verifies the serving predictor reproduces
sklearn.

Usage:
  python3 simulation/scripts/fit_chain_hybrid_discriminator.py \
      --oracle-log <train_dir>/decisions_select1_oracle.jsonl \
      --out-dir   <eval_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import base64
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/workspace/simulation/oracle")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
from chain_hybrid_patch import _ServingDiscriminator  # noqa: E402

# Human-readable names for the columns _ServingDiscriminator.features() emits.
FEATURE_NAMES = {
    "logistic": ["suffix_p", "eagle_p", "match_len", "suffix_count",
                 "suffix_total"],
    "beta": ["ln_suffix_p", "ln_1m_suffix_p", "ln_eagle_p", "ln_1m_eagle_p",
             "match_len", "suffix_count", "suffix_total"],
}


_ALIVE = {"eagle", "suffix", "both"}


def load(path, accept_conditioned=True):
    """Decisive rows (oracle_hit in eagle/suffix) with their depth.

    accept_conditioned (our-Bayes spec, default ON): only keep a decisive row if
    the chain prefix was still ALIVE when it was reached -- i.e. every shallower
    depth of the same (rid, decode_step) had oracle_hit in {both,eagle,suffix}.
    This matches the population the selector actually decides for MAT and drops
    the dead-branch drifted rows that the pooled fit over-weights. With the flag
    OFF the loader reproduces the original pooled behaviour exactly.
    """
    from collections import defaultdict
    chains = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail"):
                continue
            chains[(r.get("rid", "?"), r.get("decode_step"))].append(r)
    raw, rids, y, depths = [], [], [], []
    for rs in chains.values():
        rs.sort(key=lambda r: r.get("depth", 0))
        alive = True
        for r in rs:
            if accept_conditioned and not alive:
                break
            h = r.get("oracle_hit")
            if (h in ("eagle", "suffix") and r.get("eagle_p") is not None
                    and r.get("suffix_p") is not None):
                raw.append((float(r["suffix_p"]), float(r["eagle_p"]),
                            r.get("match_len"), r.get("suffix_count"),
                            r.get("suffix_total")))
                rids.append(r.get("rid", "?"))
                y.append(1 if h == "suffix" else 0)
                depths.append(int(r.get("depth", 0)))
            if h not in _ALIVE:
                alive = False
    return raw, np.asarray(rids), np.asarray(y, int), np.asarray(depths, int)


def design(raw, kind, depths=None):
    if depths is not None:
        return np.array([_ServingDiscriminator.features(kind, sp, ep, ml, c, n, depth=d)
                         for (sp, ep, ml, c, n), d in zip(raw, depths)], dtype=float)
    return np.array([_ServingDiscriminator.features(kind, sp, ep, ml, c, n)
                     for (sp, ep, ml, c, n) in raw], dtype=float)


def fit_gbm_arms(raw, rids, y, depths, with_depth, accept_conditioned, out):
    """Fit the Panel-B BOUNDARY arms as servable GBMs on (suffix_p, eagle_p,
    [depth]) ONLY (the calibration info set): disc_mono.json = best-monotone
    (calibration-framework ceiling, monotonic sp UP / ep DOWN), disc_bayes.json =
    unconstrained Bayes. Same TRAIN-fit / held-out / pinned-serving footing as
    the calib + logistic/beta disc arms. The fitted sklearn model is pickled into
    the blob so the serving _ServingDiscriminator can apply predict_proba."""
    sp = np.array([r[0] for r in raw]); ep = np.array([r[1] for r in raw])
    cols = [sp, ep] + ([depths.astype(float)] if with_depth else [])
    X = np.column_stack(cols)
    names = ["suffix_p", "eagle_p"] + (["depth"] if with_depth else [])
    ng = len(set(rids.tolist()))
    for kind, monotone in (("gbm_mono", True), ("gbm_bayes", False)):
        cst = ([1, -1] + ([0] if with_depth else [])) if monotone else None
        aucs = []
        for tr, te in GroupKFold(n_splits=min(5, ng)).split(X, y, rids):
            if len(set(y[tr].tolist())) < 2:
                continue
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                               learning_rate=0.06, monotonic_cst=cst).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        heldout = float(np.mean(aucs)) if aucs else None
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                             learning_rate=0.06, monotonic_cst=cst).fit(X, y)
        train_auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
        blob = dict(kind=kind, feature_names=names,
                    model_b64=base64.b64encode(pickle.dumps(clf)).decode(),
                    monotone=monotone, with_depth=bool(with_depth),
                    accept_conditioned=bool(accept_conditioned),
                    n_train=int(len(y)), suffix_right=float(y.mean()),
                    train_auc=train_auc, heldout_auc=heldout)
        fp = out / f"disc_{kind}.json"
        with open(fp, "w") as f:
            json.dump(blob, f)
        ho = f"{heldout:.4f}" if heldout is not None else "n/a"
        print(f"  {kind:9s} train_auc={train_auc:.4f} heldout_auc={ho} -> {fp}")
        disc = _ServingDiscriminator.load(str(fp))
        chk = np.array([disc.predict(*raw[i], depth=int(depths[i])) for i in range(min(200, len(y)))])
        skl = clf.predict_proba(X[:min(200, len(y))])[:, 1]
        print(f"    serving-vs-sklearn max|Δ|: {float(np.max(np.abs(chk - skl))):.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-log", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gbm", action="store_true",
                    help="fit servable Panel-B boundary GBMs (disc_mono/disc_bayes) "
                         "on (suffix_p,eagle_p,depth) instead of the logistic/beta disc")
    # our-Bayes spec: both ON by default. --no-... to recover the legacy fit.
    ap.add_argument("--accept-conditioned", action=argparse.BooleanOptionalAction,
                    default=True, help="alive-prefix conditioning (our-Bayes; default on)")
    ap.add_argument("--with-depth", action=argparse.BooleanOptionalAction,
                    default=True, help="append depth as a feature (our-Bayes; default on)")
    args = ap.parse_args()
    raw, rids, y, depths = load(args.oracle_log,
                                accept_conditioned=args.accept_conditioned)
    n = len(y)
    if n < 200:
        sys.exit(f"too few labeled decisions ({n}); need a real train-oracle log")
    ng = len(set(rids.tolist()))
    print(f"train decisive+contested n={n}  suffix-right={y.mean():.3f}  "
          f"unique rids={ng}  accept_conditioned={args.accept_conditioned}  "
          f"with_depth={args.with_depth}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.gbm:
        fit_gbm_arms(raw, rids, y, depths, args.with_depth,
                     args.accept_conditioned, out)
        return
    dep_arg = depths if args.with_depth else None

    for kind, base_names in FEATURE_NAMES.items():
        names = base_names + (["depth"] if args.with_depth else [])
        X = design(raw, kind, dep_arg)
        # held-out AUC (GroupKFold by rid) for an honest read on the fit
        aucs = []
        gkf = GroupKFold(n_splits=min(5, ng))
        for tr, te in gkf.split(X, y, rids):
            if len(set(y[tr].tolist())) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(
                y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
        heldout = float(np.mean(aucs)) if aucs else None
        # final model on ALL train data
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(sc.transform(X), y)
        train_auc = float(roc_auc_score(y, clf.predict_proba(sc.transform(X))[:, 1]))
        blob = dict(
            kind=kind, feature_names=names,
            mean=sc.mean_.tolist(), std=sc.scale_.tolist(),
            coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
            n_train=int(n), suffix_right=float(y.mean()),
            train_auc=train_auc, heldout_auc=heldout,
            accept_conditioned=bool(args.accept_conditioned),
            with_depth=bool(args.with_depth),
        )
        fp = out / f"disc_{kind}.json"
        with open(fp, "w") as f:
            json.dump(blob, f, indent=2)
        ho = f"{heldout:.4f}" if heldout is not None else "n/a"
        print(f"  {kind:9s} train_auc={train_auc:.4f} heldout_auc={ho} -> {fp}")
        # sanity: the serving-side predictor must reproduce sklearn exactly
        disc = _ServingDiscriminator.load(str(fp))
        m = min(200, n)
        chk = np.array([disc.predict(*raw[i], depth=int(depths[i])) for i in range(m)])
        skl = clf.predict_proba(sc.transform(X[:m]))[:, 1]
        print(f"    serving-vs-sklearn max|Δ| ({m} rows): "
              f"{float(np.max(np.abs(chk - skl))):.2e}")
        # coefficient readout (standardized -> comparable magnitudes)
        order = np.argsort(-np.abs(clf.coef_[0]))
        print("    standardized coef (|desc|): " + ", ".join(
            f"{names[i]}={clf.coef_[0][i]:+.2f}" for i in order))


if __name__ == "__main__":
    main()
