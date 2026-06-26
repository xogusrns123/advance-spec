"""Export PER-POSITION per-method calibration maps for real-serving select1_calib.

From a raw select-1 decision log (decisions_select1_train.jsonl) build per-depth
samples — depth = the COMPOSED-CHAIN depth (the decision-log `depth` field), which
is the index the serving calibrator queries — for the eagle (model) and suffix
groups (reuse fit_chain_hybrid_calib.load_samples), fit each of the 4 calibration
methods (reuse plot_calib_methods.fit_*) per (group, depth), and sample each
fitted predict() on a dense x-grid into a step lookup that
chain_hybrid_patch._ServingIsoCalibrator consumes (meta.per_position=true).

One JSON map per method: calib_pp_{histogram,isotonic,logistic,beta}.json with
  {"meta":{"per_position":true,"shrink":null,"method":...},
   "groups":{"eagle":{"<d>":{"x":[...],"y":[...]},...}, "suffix":{...}}}.
Depths with < --min-samples samples fall back to that group's GLOBAL fit.

suffix uses the raw count-ratio (non-Jeffreys), matching the serving comparison
(meta.shrink=null -> serving passes raw suffix_p).

Usage:
  python3 simulation/scripts/fit_chain_hybrid_calib_perpos.py \
      --decision-log .../decisions_select1_train.jsonl --out-dir .../
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_chain_hybrid_calib import load_samples  # noqa: E402
from plot_calib_methods import (  # noqa: E402
    binned, fit_beta, fit_histogram, fit_isotonic, fit_logistic,
)

METHODS = ("histogram", "isotonic", "logistic", "beta")
GRID = np.linspace(0.0, 1.0, 501)

# oracle_hit values: a HIT means the chosen token == gt (chain stays on the GT
# trajectory); a MISS means the chain drifted (none) or the GT ended (nogt).
ALIVE_HIT = ("both", "eagle", "suffix")
MISS_HIT = ("none", "nogt")


def _alive_depths(decision_log: str) -> set:
    """Set of (rid, decode_step, depth) decision rows on the still-accepting
    prefix: depth d is alive iff every earlier depth d'<d of the same
    (rid, decode_step) had oracle_hit in {both, eagle, suffix} (chosen token ==
    gt -> chain stayed on the GT trajectory). Depth 0 is always alive; a MISS
    (oracle_hit in {none, nogt}) kills aliveness for all deeper depths. Rows
    with no oracle_hit (non-oracle logs) never kill aliveness -> no-op filter.

    Used by --accept-conditioned. The default fitter pools every depth-d row,
    including dead-chain rows that can never occur in an accepted chain at
    serving time (verify rejects at the first miss), which biases the per-depth
    curve toward a survival signal instead of the conditional accept-given-
    previous-accepted that selection actually needs."""
    by_key = defaultdict(list)  # (rid, decode_step) -> [(depth, oracle_hit)]
    with open(decision_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail"):
                continue
            if (r.get("rid") is None or r.get("decode_step") is None
                    or r.get("depth") is None):
                continue
            by_key[(r["rid"], r["decode_step"])].append(
                (int(r["depth"]), r.get("oracle_hit")))
    alive = set()
    for (rid, ds), rows in by_key.items():
        rows.sort(key=lambda t: t[0])
        ok = True
        for depth, oh in rows:
            if ok:
                alive.add((rid, ds, depth))
            if oh in MISS_HIT:  # this depth missed -> deeper depths are dead
                ok = False
    return alive


def load_oracle_samples(decision_log: str, alive: set | None = None) -> dict:
    """ORACLE-trajectory calibration samples: walk an ORACLE decision log (which
    follows the GT/teacher-forced trajectory and records gt_token) and label EACH
    proposer by per-token correctness (token == gt_token) — NOT chain survival,
    and NOT conditioned on which proposer the rule chose. This is the correct,
    deployable calibration target (GT is needed only at this training-collection
    step; serving applies the frozen map). Returns the same shape as load_samples:
      {'eagle': [(eagle_p, depth, y)], 'suffix': [(suffix_p, depth, y, c, n)]}."""
    eagle, suffix = [], []
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
            depth = int(r["depth"])
            if alive is not None and (
                    r.get("rid"), r.get("decode_step"), depth) not in alive:
                continue
            if r.get("eagle_p") is not None and r.get("eagle_token") is not None:
                eagle.append((float(r["eagle_p"]), depth,
                              1.0 if r["eagle_token"] == gt else 0.0))
            if r.get("suffix_p") is not None and r.get("suffix_token") is not None:
                suffix.append((float(r["suffix_p"]), depth,
                               1.0 if r["suffix_token"] == gt else 0.0,
                               r.get("suffix_count"), r.get("suffix_total")))
    print(f"  oracle-label samples: eagle={len(eagle)} suffix={len(suffix)}",
          file=sys.stderr)
    return {"eagle": eagle, "suffix": suffix}


def load_target_prob_samples(decision_log: str, target_prob_file: str,
                             alive: set | None = None) -> dict:
    """CONTINUOUS-objective calibration samples: regress each proposer's score
    onto the TARGET model's softmax probability of the drafted token,
    q_target(token | GT prefix) — NOT the binary token==gt accept event. The
    q values are produced offline by capture_target_probs.py and joined to the
    decision log by (rid, decode_step, depth). Same row shape as
    load_oracle_samples, but y in [0,1] is CONTINUOUS:
      {'eagle': [(eagle_p, depth, q_eagle)],
       'suffix': [(suffix_p, depth, q_suffix, c, n)]}."""
    q = {}
    with open(target_prob_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q[(r["rid"], r["decode_step"], int(r["depth"]))] = r
    eagle, suffix = [], []
    n_e_miss = n_s_miss = 0
    with open(decision_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail"):
                continue
            tq = q.get((r["rid"], r["decode_step"], int(r["depth"])))
            if tq is None:
                continue
            depth = int(r["depth"])
            if alive is not None and (
                    r["rid"], r["decode_step"], depth) not in alive:
                continue
            if r.get("eagle_p") is not None and r.get("eagle_token") is not None:
                if tq.get("q_eagle") is not None:
                    eagle.append((float(r["eagle_p"]), depth,
                                  float(tq["q_eagle"])))
                else:
                    n_e_miss += 1
            if r.get("suffix_p") is not None and r.get("suffix_token") is not None:
                if tq.get("q_suffix") is not None:
                    suffix.append((float(r["suffix_p"]), depth,
                                   float(tq["q_suffix"]),
                                   r.get("suffix_count"), r.get("suffix_total")))
                else:
                    n_s_miss += 1
    print(f"  target-prob samples: eagle={len(eagle)} suffix={len(suffix)} "
          f"(rows w/o joined q: eagle={n_e_miss} suffix={n_s_miss})",
          file=sys.stderr)
    return {"eagle": eagle, "suffix": suffix}


def _fit_continuous(method: str, p: np.ndarray, y: np.ndarray):
    """Continuous-label replacements for the binary Platt/beta fits — sklearn
    LogisticRegression needs 0/1 labels, so for a continuous target y in [0,1]
    we regress with LinearRegression using the same feature map and clip to
    [0,1]. histogram/isotonic already handle continuous y natively."""
    from sklearn.linear_model import LinearRegression
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if method == "logistic":  # Platt analog: linear in p
        lr = LinearRegression().fit(p.reshape(-1, 1), y)
        return lambda x: np.clip(
            lr.predict(np.asarray(x, float).reshape(-1, 1)), 0.0, 1.0)
    # beta analog: linear in [ln p, ln(1-p)]
    eps = 1e-6
    pc = np.clip(p, eps, 1.0 - eps)
    lr = LinearRegression().fit(np.column_stack([np.log(pc), np.log(1.0 - pc)]), y)

    def predict(x):
        xc = np.clip(np.asarray(x, float), eps, 1.0 - eps)
        return np.clip(
            lr.predict(np.column_stack([np.log(xc), np.log(1.0 - xc)])), 0.0, 1.0)
    return predict


def _fit_depth_feature(method: str, p, y, d, continuous: bool):
    """ONE joint calibrator with DEPTH as an input feature (used by --mode
    depth_feature). Returns predict(p_grid, depth). logistic: features [p, d, p*d]
    (the interaction lets the curve's slope shift with depth); beta: logit-linear
    in [ln p, ln(1-p), d, d*ln p, d*ln(1-p)]. Continuous target_p (or a degenerate
    single-class pool) uses the LinearRegression analog. Shares statistical
    strength across depths -> robust where accept-conditioning starves deep depths.
    Only logistic/beta have a natural depth feature; histogram/isotonic fall back
    to the pooled global fit at the call site."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    p = np.asarray(p, float); y = np.asarray(y, float); d = np.asarray(d, float)
    eps = 1e-6

    def feats(pp, dd):
        pp = np.asarray(pp, float); dd = np.asarray(dd, float)
        if method == "logistic":
            return np.column_stack([pp, dd, pp * dd])
        pc = np.clip(pp, eps, 1.0 - eps)
        lp, l1 = np.log(pc), np.log(1.0 - pc)
        return np.column_stack([lp, l1, dd, dd * lp, dd * l1])

    X = feats(p, d)
    if continuous or np.unique(y).size < 2:
        lr = LinearRegression().fit(X, y)
        raw = lambda Xq: lr.predict(Xq)
    else:
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        raw = lambda Xq: lr.predict_proba(Xq)[:, 1]

    def predict(pp, depth):
        pp = np.asarray(pp, float)
        dd = np.full(pp.shape, float(depth))
        return np.clip(raw(feats(pp, dd)), 0.0, 1.0)
    return predict


def _predict_of(method: str, p: np.ndarray, y: np.ndarray,
                continuous: bool = False):
    if method == "histogram":  # per-bin mean of y — continuous-safe
        centers, rates, _ = binned(p, y)
        return fit_histogram(p, y, centers, rates)[3]
    if continuous and method in ("logistic", "beta"):
        return _fit_continuous(method, p, y)
    fn = {"isotonic": fit_isotonic, "logistic": fit_logistic,
          "beta": fit_beta}[method]  # isotonic handles continuous y natively
    return fn(p, y)[3]


def _sample(predict) -> dict:
    yv = np.clip(np.asarray(predict(GRID), dtype=float), 0.0, 1.0)
    return {"x": [round(float(x), 4) for x in GRID],
            "y": [round(float(v), 6) for v in yv]}


def _build_group_map(method, gdata, mode, continuous, min_samples):
    """{depth_str: {x,y}} lookup for one (group, method) under the chosen --mode.
    Returns (dd, how_str). All modes emit per_position grids the serving
    _ServingIsoCalibrator consumes unchanged:
      per_depth     - independent per-depth fit; sparse depths (< min_samples)
                      fall back to the group's pooled global fit.
      global        - the pooled global fit repeated at every depth 0..maxd.
      depth_feature - ONE joint f(p, depth) sampled per depth 0..maxd (logistic/
                      beta only); histogram/isotonic have no depth feature so they
                      use the pooled global fit at every depth.
    maxd = deepest OBSERVED depth (no extrapolation past the data)."""
    gp, gy = gdata["global"]
    global_samp = (_sample(_predict_of(method, gp, gy, continuous))
                   if len(gy) else {"x": [0.0, 1.0], "y": [0.0, 0.0]})
    present = sorted(gdata["depths"])
    if not present:
        return {}, "empty"
    maxd = present[-1]
    dd = {}
    if mode == "per_depth":
        n_fb = 0
        for d in present:
            p, y = gdata["depths"][d]
            if len(y) >= min_samples:
                dd[str(d)] = _sample(_predict_of(method, p, y, continuous))
            else:
                dd[str(d)] = global_samp
                n_fb += 1
        return dd, f"{len(present)} depths, {n_fb} sparse->global"
    if mode == "global":
        for d in range(maxd + 1):
            dd[str(d)] = global_samp
        return dd, f"pooled curve x {maxd + 1} depths"
    # depth_feature
    if method in ("logistic", "beta"):
        P, Y, D = [], [], []
        for d in present:
            p, y = gdata["depths"][d]
            P.append(p); Y.append(y); D.append(np.full(len(y), d, float))
        predict2d = _fit_depth_feature(method, np.concatenate(P),
                                       np.concatenate(Y), np.concatenate(D),
                                       continuous)
        for d in range(maxd + 1):
            dd[str(d)] = _sample(lambda x, _d=d: predict2d(x, _d))
        return dd, f"f(p,d) x {maxd + 1} depths"
    for d in range(maxd + 1):
        dd[str(d)] = global_samp
    return dd, f"pooled curve x {maxd + 1} (no depth feature)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decision-log", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-depth", type=int, default=64)
    ap.add_argument("--min-samples", type=int, default=500)
    ap.add_argument("--mode", choices=["per_depth", "global", "depth_feature"],
                    default="per_depth",
                    help="per_depth: independent calibrator per depth (sparse "
                         "depths fall back to the pooled global fit). global: ONE "
                         "pooled calibrator emitted at every depth (depth-agnostic, "
                         "robust). depth_feature: ONE joint f(p, depth) sampled "
                         "per depth (logistic/beta; histogram/isotonic -> global). "
                         "global/depth_feature avoid the high variance of per-depth "
                         "fits when --accept-conditioned starves deep depths.")
    ap.add_argument("--jeffreys", action="store_true",
                    help="suffix x = Jeffreys-shrunk (c+0.5)/(n+1) instead of raw "
                         "c/n; sets meta.shrink=jeffreys so the serving calibrator "
                         "applies the same shrink before the map lookup")
    ap.add_argument("--oracle-labels", action="store_true",
                    help="--decision-log is an ORACLE log; label each proposer by "
                         "per-token correctness (token==gt) along the GT trajectory "
                         "instead of chain-survival on the rule's self-rollout. "
                         "This is the correct calibration target.")
    ap.add_argument("--target-prob-labels", default=None,
                    help="CONTINUOUS objective: path to a target_probs.jsonl from "
                         "capture_target_probs.py. Regress each proposer's score "
                         "onto q_target(token|GT prefix) (joined by rid/step/depth) "
                         "instead of the binary token==gt event. Implies an ORACLE "
                         "--decision-log; mutually exclusive with --oracle-labels.")
    ap.add_argument("--accept-conditioned", action="store_true",
                    help="Fit each per-depth curve only on rows whose chain prefix "
                         "stayed on the GT trajectory (all earlier depths of the "
                         "same (rid, decode_step) had oracle_hit in "
                         "{both,eagle,suffix}). Drops dead-chain rows that can never "
                         "occur in an accepted chain at serving time. Requires an "
                         "ORACLE log (--oracle-labels or --target-prob-labels).")
    args = ap.parse_args()

    if args.target_prob_labels and args.oracle_labels:
        ap.error("--target-prob-labels and --oracle-labels are mutually exclusive")
    if args.accept_conditioned and not (args.oracle_labels
                                        or args.target_prob_labels):
        ap.error("--accept-conditioned needs an ORACLE log "
                 "(--oracle-labels or --target-prob-labels)")
    alive = None
    if args.accept_conditioned:
        alive = _alive_depths(args.decision_log)
        print(f"  accept-conditioned: {len(alive)} alive "
              f"(rid,step,depth) rows on the still-accepting prefix",
              file=sys.stderr)
    continuous = bool(args.target_prob_labels)
    if continuous:
        s = load_target_prob_samples(args.decision_log, args.target_prob_labels,
                                     alive=alive)
    elif args.oracle_labels:
        s = load_oracle_samples(args.decision_log, alive=alive)
    else:
        s = load_samples(args.decision_log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # group -> {"depths": {d: (p[],y[])}, "global": (p[],y[])}
    # eagle rows: (p, depth, y); suffix rows: (p, depth, y, c, n)
    per = {}
    for g, rows in (("eagle", s["eagle"]), ("suffix", s["suffix"])):
        # suffix rows are (p, depth, y, c, n); with --jeffreys use (c+.5)/(n+1)
        # as the calibration x (matches the serving shrink). eagle has no counts.
        use_jeff = args.jeffreys and g == "suffix"
        dp = defaultdict(list); dy = defaultdict(list); gp = []; gy = []
        for row in rows:
            depth, y = int(row[1]), float(row[2])
            if use_jeff:
                c, n = row[3], row[4]
                p = (c + 0.5) / (n + 1) if (c is not None and n) else float(row[0])
            else:
                p = float(row[0])
            gp.append(p); gy.append(y)
            if depth < args.max_depth:
                dp[depth].append(p); dy[depth].append(y)
        per[g] = {
            "depths": {d: (np.asarray(dp[d], float), np.asarray(dy[d], float))
                       for d in dp},
            "global": (np.asarray(gp, float), np.asarray(gy, float)),
        }
        print(f"  {g}: {len(rows)} samples, {len(per[g]['depths'])} depths",
              file=sys.stderr)

    for method in METHODS:
        groups = {}
        for g in ("eagle", "suffix"):
            dd, how = _build_group_map(method, per[g], args.mode, continuous,
                                       args.min_samples)
            groups[g] = dd
            print(f"    {method}/{g} [{args.mode}]: {how}", file=sys.stderr)
        blob = {"meta": {"per_position": True,
                         "shrink": ("jeffreys" if args.jeffreys else None),
                         "label": ("target_p" if continuous else
                                   "token_gt" if args.oracle_labels else
                                   "survival"),
                         "accept_conditioned": bool(args.accept_conditioned),
                         "mode": args.mode,
                         "method": method, "source": args.decision_log},
                "groups": groups}
        outp = out_dir / f"calib_pp_{method}.json"
        with open(outp, "w") as f:
            json.dump(blob, f)
        print(f"wrote {outp} (eagle depths={len(groups['eagle'])}, "
              f"suffix depths={len(groups['suffix'])})", file=sys.stderr)


if __name__ == "__main__":
    main()
