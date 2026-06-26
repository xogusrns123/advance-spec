#!/usr/bin/env python3
"""Offline, serving-independent comparison of calibration OBJECTIVES for the
chain-hybrid per-depth selector: raw vs token_gt-calib vs target_p-calib.

Motivation: on sglang 0.5.12 the chain-hybrid SERVING instrumentation is
unreliable (the select hook's eagle_token does not track the real prediction;
oracle/injection broken). This script answers the target_p hypothesis WITHOUT
serving, using a TRUSTWORTHY oracle decision log produced on a working sglang
(e.g. the legacy qwen3_14b_full run, eagle==gt depth0 ~0.43) plus the offline
q_target capture (capture_target_probs.py).

For each proposer group (eagle, suffix) we fit a per-depth monotone calibrator
on the TRAIN rids under two objectives:
  token_gt : y = 1[token == gt_token]      (the deployed binary-accept label)
  target_p : y = q_target(token | GT prefix)  (the proposed continuous label)
and a per-depth isotonic map raw_prob -> calibrated score (sklearn Isotonic,
>= --min-samples per depth else that group's global fit).

We then evaluate, on the held-out TEST rids' DECISIVE decisions (exactly one of
eagle/suffix == gt), the SELECTION ACCURACY of three rules (pick suffix iff
score_suffix > score_eagle):
  raw      : score = raw prob (suffix_p vs eagle_p)
  token_gt : score = token_gt-calibrated
  target_p : score = target_p-calibrated
Accuracy = fraction of decisive decisions where the picked proposer's token==gt.
This is the project's primary selection-quality signal (analyze_o4_calib_*).

Usage:
  python3 simulation/scripts/compare_target_p_offline.py \
    --decision-log .../qwen3_14b_full/decisions_select1_oracle.jsonl \
    --target-probs .../qwen3_14b_full/target_probs.jsonl \
    --train-frac 0.7 --min-samples 200 --out-json .../target_p_offline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict, defaultdict

import numpy as np
from sklearn.isotonic import IsotonicRegression


def load(decision_log, target_probs):
    """Return per-decision rows with both labels + raw probs, grouped by rid."""
    q = {}
    with open(target_probs) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q[(r["rid"], r["decode_step"], int(r["depth"]))] = r
    rows_by_rid = OrderedDict()
    n_no_q = 0
    with open(decision_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail"):
                continue
            gt = r.get("gt_token")
            if gt is None:
                continue
            ep, sp = r.get("eagle_p"), r.get("suffix_p")
            et, st_ = r.get("eagle_token"), r.get("suffix_token")
            depth = int(r["depth"])
            tq = q.get((r["rid"], r["decode_step"], depth))
            if tq is None:
                n_no_q += 1
                continue
            row = {
                "rid": r["rid"], "depth": depth,
                "ep": float(ep) if ep is not None else None,
                "sp": float(sp) if sp is not None else None,
                "e_ok": (1.0 if et == gt else 0.0) if et is not None else None,
                "s_ok": (1.0 if st_ == gt else 0.0) if st_ is not None else None,
                "qe": tq.get("q_eagle"), "qs": tq.get("q_suffix"),
            }
            rows_by_rid.setdefault(r["rid"], []).append(row)
    print(f"  rids={len(rows_by_rid)} rows w/o joined q dropped={n_no_q}",
          file=sys.stderr)
    return rows_by_rid


class PerDepthIso:
    """Per-depth isotonic raw_prob -> score; global fallback for sparse depths."""

    def __init__(self, min_samples):
        self.min_samples = min_samples
        self.glob = None
        self.maps = {}

    def fit(self, p, y, d):
        p, y, d = np.asarray(p, float), np.asarray(y, float), np.asarray(d, int)
        ok = ~(np.isnan(p) | np.isnan(y))
        p, y, d = p[ok], y[ok], d[ok]
        if len(y) == 0:
            self.glob = lambda x: np.full(np.shape(x), 0.0)
            return self
        g = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        self.glob = g.predict
        for dd in np.unique(d):
            m = d == dd
            if m.sum() >= self.min_samples and len(np.unique(y[m])) > 1:
                self.maps[int(dd)] = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p[m], y[m])
        return self

    def predict(self, p, d):
        p = np.asarray(p, float)
        out = np.empty(len(p))
        for dd in np.unique(np.asarray(d, int)):
            m = np.asarray(d, int) == dd
            mdl = self.maps.get(int(dd))
            out[m] = (mdl.predict(p[m]) if mdl is not None
                      else self.glob(p[m]))
        return np.clip(out, 0.0, 1.0)


def fit_group(rows, group, label):
    """label in {'token_gt','target_p'}; group in {'e','s'}."""
    pk = "ep" if group == "e" else "sp"
    ok = "e_ok" if group == "e" else "s_ok"
    qk = "qe" if group == "e" else "qs"
    p, y, d = [], [], []
    for r in rows:
        if r[pk] is None:
            continue
        yv = r[ok] if label == "token_gt" else r[qk]
        if yv is None:
            continue
        p.append(r[pk]); y.append(float(yv)); d.append(r["depth"])
    return PerDepthIso(MIN).fit(p, y, d)


def sel_acc(rows, score_e, score_s):
    """Selection accuracy + picks_suffix on DECISIVE decisions (exactly one
    of eagle/suffix == gt). score_* are arrays aligned to `dec` rows."""
    pick_s = (score_s > score_e).astype(int)
    e_ok = np.array([r["e_ok"] for r in rows])
    s_ok = np.array([r["s_ok"] for r in rows])
    correct = np.where(pick_s == 1, s_ok, e_ok)
    return float(np.mean(correct)), float(np.mean(pick_s))


MIN = 200


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decision-log", required=True)
    ap.add_argument("--target-probs", required=True)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--min-samples", type=int, default=200)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    global MIN
    MIN = args.min_samples

    by_rid = load(args.decision_log, args.target_probs)
    rids = list(by_rid)
    ntr = max(1, int(round(len(rids) * args.train_frac)))
    train_rids, test_rids = rids[:ntr], rids[ntr:]
    train = [r for rid in train_rids for r in by_rid[rid]]
    test = [r for rid in test_rids for r in by_rid[rid]]
    print(f"train rids={len(train_rids)} rows={len(train)} | "
          f"test rids={len(test_rids)} rows={len(test)}", file=sys.stderr)

    # fit maps on TRAIN, both objectives, both groups
    maps = {
        "token_gt": {"e": fit_group(train, "e", "token_gt"),
                     "s": fit_group(train, "s", "token_gt")},
        "target_p": {"e": fit_group(train, "e", "target_p"),
                     "s": fit_group(train, "s", "target_p")},
    }

    # TEST decisive decisions: exactly one of eagle/suffix == gt, both present
    dec = [r for r in test if r["e_ok"] is not None and r["s_ok"] is not None
           and r["ep"] is not None and r["sp"] is not None
           and (r["e_ok"] != r["s_ok"])]
    n = len(dec)
    sr = float(np.mean([r["s_ok"] for r in dec])) if n else float("nan")
    ep = np.array([r["ep"] for r in dec])
    sp = np.array([r["sp"] for r in dec])
    dd = np.array([r["depth"] for r in dec])
    print(f"\nTEST decisive decisions n={n}  suffix-is-gt rate={sr:.3f} "
          f"(optimal pick_s)\n", file=sys.stderr)

    results = OrderedDict()
    # raw
    results["raw"] = sel_acc(dec, ep, sp)
    # calibrated
    for lab in ("token_gt", "target_p"):
        ce = maps[lab]["e"].predict(ep, dd)
        cs = maps[lab]["s"].predict(sp, dd)
        results[lab] = sel_acc(dec, ce, cs)

    print(f"  {'rule':10s} {'sel_acc':>8s} {'picks_suffix':>13s} "
          f"{'vs_raw':>8s}")
    raw_acc = results["raw"][0]
    out = {"n_decisive": n, "suffix_gt_rate": sr, "rules": {}}
    for lab, (acc, ps) in results.items():
        delta = acc - raw_acc
        flag = "" if lab == "raw" else (f"{delta:+.4f}")
        print(f"  {lab:10s} {acc:8.4f} {ps:13.3f} {flag:>8s}")
        out["rules"][lab] = {"sel_acc": acc, "picks_suffix": ps,
                             "delta_vs_raw": delta}
    print(f"\n  optimal (always pick gt proposer) = 1.000; "
          f"suffix-gt rate {sr:.3f}")
    winner = max(results, key=lambda k: results[k][0])
    print(f"  BEST: {winner} (acc {results[winner][0]:.4f})")
    out["winner"] = winner

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    main()
