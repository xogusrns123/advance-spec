"""3-WAY selection ladder (gt-path, block-anchored) for the 6 selection methods,
generalized from the 2-way boundary study to 3 proposers via the unified rule:
each method produces a per-proposer estimate of P(proposer == gt), and select-1 =
ARGMAX over proposers (the 2-way "P(suffix==gt) > 0.5" binary boundary becomes a
3-way argmax). One-vs-rest: each proposer's correctness is estimated independently
(handles multi-hit rows naturally). All held-out via GroupKFold-by-rid OOF.

Methods (the estimator of P(P==gt) differs):
  raw       : the raw proposer prob (no fit)              -> argmax raw prob
  calib x4  : 1-D calibration of the raw prob (depth-pooled, "all-trained" style)
              histogram / isotonic / logistic(Platt) / beta
  mono      : calibration-CEILING = one-vs-rest monotone GBM P(P==gt | p_P up,
              max_other down, depth) -- best a monotone reweighting can do
  bayes     : unconstrained GBM P(P==gt | p_P, max_other, depth)  (the "0.5"->argmax)
  oracle    : true hit (ceiling)
+ single-proposer and 2-way(main+suffix) refs for context.

selacc = decisive (some-but-not-all available proposers hit) alive-conditioned;
MAT = block-anchored run-length of consecutive correct picks (offline-sim convention).

  python3 analyze_3way_ladder.py --cell 8b  --record-dir <dir> --exclude-loopy [--fig]
  python3 analyze_3way_ladder.py --cell 27b --record-dir <dir> --merge-dflash <dflash_proposals.jsonl> [--fig]
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

CELLS = {
    "8b": {"names": ("dflash", "e3", "suffix"),
           "tok": {"dflash": "eagle_token", "e3": "e3_token", "suffix": "suffix_token"},
           "p": {"dflash": "eagle_p", "e3": "e3_p", "suffix": "suffix_p"}, "model": "Qwen3-8B"},
    "27b": {"names": ("mtp", "dflash", "suffix"),
            "tok": {"mtp": "eagle_token", "dflash": "dflash_token", "suffix": "suffix_token"},
            "p": {"mtp": "eagle_p", "dflash": "dflash_p", "suffix": "suffix_p"}, "model": "Qwen3.5-27B"},
}
CALIBS = ("histogram", "isotonic", "logistic", "beta")
EPS = 1e-6


def load_chains(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if line:
            o = json.loads(line)
            if o.get("type") == "decision" and not o.get("tail"):
                chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains:
        chains[k].sort(key=lambda r: r["depth"])
    return chains


def merge_dflash(chains, path):
    idx = {(rid, ds, r["depth"]): r for (rid, ds), rs in chains.items() for r in rs}
    n = 0
    for line in open(path):
        o = json.loads(line)
        r = idx.get((o["rid"], o["decode_step"], o["depth"]))
        if r is not None:
            r["dflash_token"] = o["dflash_token"]; r["dflash_p"] = o.get("dflash_p"); n += 1
    print(f"merged {n} DFlash proposals")


def loopy_rids(record_dir, decisions_file, thresh=0.5, n=4):
    dd = Path(record_dir)
    reqs = {}
    for line in open(dd / decisions_file):
        o = json.loads(line)
        if o.get("type") == "req":
            reqs[o["rid"]] = tuple(o["input_ids"])
    gt = {}
    if (dd / "gt_tokens.jsonl").exists():
        for line in open(dd / "gt_tokens.jsonl"):
            r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
    bad = set()
    for rid, ids in reqs.items():
        out = gt.get(ids)
        if out and len(out) >= n + 1:
            g = [tuple(out[i:i + n]) for i in range(len(out) - n + 1)]
            if len(set(g)) / max(len(g), 1) < thresh:
                bad.add(rid)
    return bad


def row_info(r, names, tok, pk):
    gt = r.get("gt_token")
    toks = {P: r.get(tok[P]) for P in names}
    probs = {P: (r.get(pk[P]) if r.get(pk[P]) is not None else 0.0) for P in names}
    avail = [P for P in names if toks[P] is not None]
    hits = {P for P in avail if gt is not None and toks[P] == gt}
    return gt, avail, hits, probs


def collect_samples(chains, names, tok, pk):
    """One sample per (decisive-alive row, available proposer): features + label +
    a back-reference to the row dict so OOF estimates can be written onto it."""
    S = {P: {"x": [], "y": [], "rid": [], "ref": []} for P in names}
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt, avail, hits, probs = row_info(r, names, tok, pk)
            decisive = 0 < len(hits) < len(avail)
            if decisive:
                for P in avail:
                    mo = max(probs[q] for q in avail if q != P)
                    S[P]["x"].append([probs[P], mo, r["depth"]])
                    S[P]["y"].append(1 if P in hits else 0)
                    S[P]["rid"].append(rid)
                    S[P]["ref"].append(r)
            if gt is not None and len(hits) == 0:
                alive = False
    return S


def _beta_features(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.c_[np.log(p), -np.log(1 - p)]


def oof_estimates(S, names, method):
    """Annotate each sampled row: ref["_est"][method][P] = OOF P(P==gt) estimate."""
    for P in names:
        x = np.asarray(S[P]["x"], float); y = np.asarray(S[P]["y"], int)
        rid = np.asarray(S[P]["rid"]); refs = S[P]["ref"]
        n = len(y)
        if n == 0:
            continue
        pred = np.full(n, float(y.mean()))
        if method != "raw" and method != "oracle" and y.min() != y.max():
            ng = len(set(rid.tolist()))
            for tr, te in GroupKFold(min(5, ng)).split(x, y, rid):
                p_tr, p_te = x[tr, 0], x[te, 0]
                if method in CALIBS:
                    if method == "isotonic":
                        m = IsotonicRegression(out_of_bounds="clip").fit(p_tr, y[tr])
                        pred[te] = m.predict(p_te)
                    elif method == "logistic":
                        m = LogisticRegression(max_iter=1000).fit(p_tr[:, None], y[tr])
                        pred[te] = m.predict_proba(p_te[:, None])[:, 1]
                    elif method == "beta":
                        m = LogisticRegression(max_iter=1000).fit(_beta_features(p_tr), y[tr])
                        pred[te] = m.predict_proba(_beta_features(p_te))[:, 1]
                    elif method == "histogram":
                        edges = np.linspace(0, 1, 16)
                        bi_tr = np.clip(np.digitize(p_tr, edges) - 1, 0, len(edges) - 2)
                        rate = np.full(len(edges) - 1, float(y[tr].mean()))
                        for b in range(len(edges) - 1):
                            m = bi_tr == b
                            if m.sum() >= 5:
                                rate[b] = y[tr][m].mean()
                        bi_te = np.clip(np.digitize(p_te, edges) - 1, 0, len(edges) - 2)
                        pred[te] = rate[bi_te]
                else:  # mono / bayes : multivariate GBM on [p_P, max_other, depth]
                    cst = [1, -1, 0] if method == "mono" else None
                    clf = HistGradientBoostingClassifier(
                        max_depth=3, max_iter=200, learning_rate=0.05,
                        monotonic_cst=cst).fit(x[tr], y[tr])
                    pred[te] = clf.predict_proba(x[te])[:, 1]
        for i, r in enumerate(refs):
            r.setdefault("_est", {}).setdefault(method, {})[P] = float(pred[i])


def selacc_mat(chains, pick_of, names, tok, pk, subset):
    sset = set(subset); n = corr = 0; Ls = []
    for rs in chains.values():
        alive = True; Lp = 0; pol_alive = True
        for r in rs:
            gt, avail, hits, probs = row_info(r, names, tok, pk)
            av = [p for p in avail if p in sset]; hs = hits & sset
            row_alive = len(hs) > 0
            decisive = row_alive and (len(hs) < len(av))
            pick = pick_of(r, av, probs) if av else None
            ok = pick in hs
            if alive and decisive:
                n += 1; corr += int(ok)
            if pol_alive:
                if row_alive and ok:
                    Lp += 1
                else:
                    pol_alive = False
            if not row_alive:
                alive = False
        Ls.append(Lp)
    return corr / max(n, 1), sum(Ls) / max(len(Ls), 1)


def pick_raw(r, av, probs):
    return max(av, key=lambda P: probs[P]) if av else None


def pick_est(method):
    def f(r, av, probs):
        e = r.get("_est", {}).get(method, {})
        return max(av, key=lambda P: e.get(P, -1.0)) if av else None
    return f


def pick_oracle(order):
    def f(r, av, probs):
        gt, avail, hits, _ = r["_ri"]
        hs = hits & set(av)
        for P in order:
            if P in hs:
                return P
        for P in order:
            if P in av:
                return P
        return av[0] if av else None
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["8b", "27b"], required=True)
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--decisions-file", default="decisions_select1_oracle.jsonl")
    ap.add_argument("--merge-dflash", default=None)
    ap.add_argument("--exclude-loopy", action="store_true")
    ap.add_argument("--fig", action="store_true")
    args = ap.parse_args()

    c = CELLS[args.cell]; names, tok, pk = c["names"], c["tok"], c["p"]
    main_p, added_p, suffix_p = names
    chains = load_chains(Path(args.record_dir) / args.decisions_file)
    if args.exclude_loopy:
        bad = loopy_rids(args.record_dir, args.decisions_file)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        print(f"excluded {len(bad)} loopy reqs")
    if args.merge_dflash:
        merge_dflash(chains, args.merge_dflash)
        chains = {k: v for k, v in chains.items()
                  if any(r.get(tok[added_p]) is not None for r in v)}
    # cache row_info on each row for the oracle picker
    for rs in chains.values():
        for r in rs:
            r["_ri"] = row_info(r, names, tok, pk)
    print(f"cell={args.cell} proposers={names} blocks={len(chains)}")

    S = collect_samples(chains, names, tok, pk)
    for P in names:
        print(f"  decisive samples[{P}] = {len(S[P]['y'])} (hit-rate "
              f"{np.mean(S[P]['y']) if S[P]['y'] else float('nan'):.3f})")
    methods = ("raw",) + CALIBS + ("mono", "bayes")
    for m in methods:
        oof_estimates(S, names, m)

    # policies: single, 2-way(main+suffix) raw/oracle, then 3-way raw/4calib/mono/bayes/oracle
    base2 = (main_p, suffix_p)
    rows = []
    def add(lab, pk_fn, sub):
        sa, mt = selacc_mat(chains, pk_fn, names, tok, pk, sub)
        rows.append((lab, sa, mt)); print(f"{lab:26s} selacc={sa:.4f}  MAT={mt:.4f}")
    print()
    add(f"{main_p}-only", pick_raw, (main_p,))
    add(f"{added_p}-only", pick_raw, (added_p,))
    add(f"{suffix_p}-only", pick_raw, (suffix_p,))
    add(f"2way raw({main_p}+{suffix_p})", pick_raw, base2)
    add(f"2way oracle", pick_oracle(base2), base2)
    print("  --- 3-way ---")
    add("3way raw", pick_raw, names)
    for m in CALIBS:
        add(f"3way calib:{m}", pick_est(m), names)
    add("3way mono(ceiling)", pick_est("mono"), names)
    add("3way bayes", pick_est("bayes"), names)
    add("3way oracle", pick_oracle(names), names)

    if args.fig:
        import sys, matplotlib
        sys.path.insert(0, str(Path(__file__).parent)); matplotlib.use("Agg")
        from _ladder_style import ladder_bar
        figdir = Path(args.record_dir) / "figures"; figdir.mkdir(exist_ok=True)
        # show: single x3, 2way raw/oracle, 3way raw/4calib/mono/bayes/oracle
        labs = [r[0] for r in rows]
        C = {"raw": "#7f7f7f", "calib": "#e377c2", "mono": "darkorange",
             "bayes": "lime", "oracle": "#d62728"}
        cols = []
        for l in labs:
            if "oracle" in l: cols.append(C["oracle"])
            elif "calib" in l: cols.append(C["calib"])
            elif "mono" in l: cols.append(C["mono"])
            elif "bayes" in l: cols.append(C["bayes"])
            elif "only" in l: cols.append("#1f77b4")
            else: cols.append(C["raw"])
        names_str = "+".join(names)
        ladder_bar([r[1] for r in rows], labs, "decisive selection accuracy (alive-cond)",
                   f"3-way selection ladder ({names_str}, {c['model']}, gt-path, held-out OOF)",
                   f"{figdir}/ladder_selacc_3way_{args.cell}.png", fmt="{:.3f}", colors=cols)
        ladder_bar([r[2] for r in rows], labs, "MAT (block-anchored)",
                   f"3-way MAT ladder ({names_str}, {c['model']}, gt-path, held-out OOF)",
                   f"{figdir}/ladder_mat_3way_{args.cell}.png", fmt="{:.3f}", colors=cols)
        print(f"\nfigures -> {figdir}/ladder_{{selacc,mat}}_3way_{args.cell}.png")


if __name__ == "__main__":
    main()
