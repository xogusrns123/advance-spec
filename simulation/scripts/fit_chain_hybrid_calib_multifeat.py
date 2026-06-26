"""Direction 2: MULTI-FEATURE per-proposer calibration.

Calibrate each proposer with its OWN features (NOT fusion across proposers):
  eagle : [eagle_p, depth]
  suffix: [suffix_p, log1p(count), log1p(total), match_len, depth]
onto accept_rate (binary token==gt -> logistic) or target_p (continuous q -> linear).
depth is a POOLED feature (one model, depth-aware). alive-prefix conditional via
--accept-conditioned. Frozen fit; serving evaluates the linear/logistic form from
the saved standardized coefficients (no sklearn at serving).

Output JSON:
  {"meta": {"label", "accept_conditioned", "kind"},
   "groups": {"eagle": {"features":[...], "mean":[...], "std":[...], "coef":[...],
                        "intercept": b0, "kind": "logistic"|"linear"}, "suffix": {...}}}

Usage:
  python3 fit_chain_hybrid_calib_multifeat.py --decision-log <oracle.jsonl> \
     [--target-prob-labels <target_probs.jsonl>] [--accept-conditioned] --out <map.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_chain_hybrid_calib_perpos import _alive_depths  # noqa: E402

EAGLE_FEATS = ["eagle_p", "depth"]
SUFFIX_FEATS = ["suffix_p", "log1p_count", "log1p_total", "match_len", "depth"]


def _q_join(target_prob_file):
    q = {}
    if not target_prob_file:
        return q
    with open(target_prob_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q[(r["rid"], r["decode_step"], int(r["depth"]))] = r
    return q


def load_feats(decision_log, target_prob_file=None, alive=None):
    q = _q_join(target_prob_file)
    cont = bool(target_prob_file)
    ex, ey, sx, sy = [], [], [], []
    with open(decision_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail"):
                continue
            depth = r.get("depth")
            if depth is None:
                continue
            if alive is not None and (r.get("rid"), r.get("decode_step"),
                                      int(depth)) not in alive:
                continue
            tq = q.get((r.get("rid"), r.get("decode_step"), int(depth))) if cont else None
            gt = r.get("gt_token")
            ep, et = r.get("eagle_p"), r.get("eagle_token")
            if ep is not None and et is not None:
                y = (tq.get("q_eagle") if tq else None) if cont else (
                    None if gt is None else (1.0 if et == gt else 0.0))
                if y is not None:
                    ex.append([float(ep), float(depth)]); ey.append(float(y))
            sp, stk = r.get("suffix_p"), r.get("suffix_token")
            c, n, ml = r.get("suffix_count"), r.get("suffix_total"), r.get("match_len")
            if sp is not None and stk is not None and None not in (c, n, ml):
                y = (tq.get("q_suffix") if tq else None) if cont else (
                    None if gt is None else (1.0 if stk == gt else 0.0))
                if y is not None:
                    sx.append([float(sp), float(np.log1p(c)), float(np.log1p(n)),
                               float(ml), float(depth)]); sy.append(float(y))
    return {"eagle": (np.asarray(ex, float), np.asarray(ey, float)),
            "suffix": (np.asarray(sx, float), np.asarray(sy, float))}


def fit_group(X, y, kind, feats):
    mean = X.mean(0); std = X.std(0); std[std == 0] = 1.0
    Xs = (X - mean) / std
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000).fit(Xs, (y >= 0.5).astype(int))
        coef, intc = m.coef_[0].tolist(), float(m.intercept_[0])
    else:
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(Xs, y)
        coef, intc = m.coef_.tolist(), float(m.intercept_)
    return {"features": feats, "mean": mean.tolist(), "std": std.tolist(),
            "coef": coef, "intercept": intc, "kind": kind}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decision-log", required=True)
    ap.add_argument("--target-prob-labels", default=None)
    ap.add_argument("--accept-conditioned", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cont = bool(args.target_prob_labels)
    kind = "linear" if cont else "logistic"
    alive = _alive_depths(args.decision_log) if args.accept_conditioned else None
    if alive is not None:
        print(f"  accept-conditioned: {len(alive)} alive rows", file=sys.stderr)
    s = load_feats(args.decision_log, args.target_prob_labels, alive)
    groups = {}
    for g, feats in (("eagle", EAGLE_FEATS), ("suffix", SUFFIX_FEATS)):
        X, y = s[g]
        if X.shape[0] < 50:
            print(f"  WARN {g}: only {X.shape[0]} samples", file=sys.stderr)
            continue
        groups[g] = fit_group(X, y, kind, feats)
        print(f"  {g}: n={X.shape[0]} kind={kind} "
              f"coef={[round(c,3) for c in groups[g]['coef']]} "
              f"(feats={feats})", file=sys.stderr)
    blob = {"meta": {"label": "target_p" if cont else "accept_rate",
                     "accept_conditioned": bool(args.accept_conditioned),
                     "kind": kind, "source": args.decision_log},
            "groups": groups}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
