"""Fit a 3-way per-proposer SELECTOR bundle for SERVED realized calib/bayes (Phase 2).

The served 3-way selector (chain_hybrid_patch.py _dfa_enabled branch) currently does RAW
argmax + consensus only. This fits per-proposer P(correct) models {eagle/MTP, dflash, suffix}
from the existing 3-way ORACLE decision log, so the served selector can argmax CALIBRATED
scores (mirrors the offline picks()/bayes that gave block-anchored MAT ~5.57 for 27B 3-way).

Two model kinds per proposer, both emitted into one bundle JSON:
  - gbm:  HistGradientBoosting (same hyperparams as the offline bayes) -> base64 pickle
          (MUST be fit in the SAME sklearn as the serving docker; run this IN sglang-bench).
  - logistic: StandardScaler + LogisticRegression -> plain JSON coefs (version-safe).
Features: eagle/dflash = [prob, depth]; suffix = [prob, depth, match_len, log1p(count)].
Label: [proposer_token == gt]. loopy reqs excluded. Self-verifies OOF selacc + block-anchored
MAT against the unweighted offline baseline before dumping.

Run (IN docker):  docker exec sglang-bench python3 \
   /workspace/simulation/scripts/select1_ladder/fit_3way_selector.py \
   --dir simulation/results/chain_hybrid_perdepth/qwen35_27b_3way_real_full \
   --props mtp:eagle_token:eagle_p dflash:dflash_token:dflash_p suffix:suffix_token:suffix_p \
   --out  simulation/results/chain_hybrid_perdepth/qwen35_27b_3way_real_full/sel3_bundle.json
"""
import json, math, base64, pickle, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

def load_blocks(path, props):
    raw = defaultdict(list)
    for line in open(path):
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            raw[(o["rid"], o["decode_step"])].append(o)
    blocks = {}
    for k, rs in raw.items():
        rs.sort(key=lambda r: r["depth"]); pos = []
        for r in rs:
            e = {"depth": r["depth"], "gt": r.get("gt_token"), "rid": k[0], "P": {}}
            for nm, tk, pk in props:
                t = r.get(tk); p = r.get(pk)
                if t is not None and p is not None:
                    e["P"][nm] = dict(tok=t, prob=float(p), mlen=float(r.get("match_len") or 0),
                                      lcnt=math.log1p(float(r.get("suffix_count") or 0)))
            pos.append(e)
        blocks[k] = pos
    return blocks

def loopy(d):
    dd = Path(d); gtf = dd / "gt_tokens.jsonl"; reqs = {}
    for line in open(dd / "decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "req": reqs[o["rid"]] = tuple(o["input_ids"])
    bad = set()
    if gtf.exists():
        gt = {}
        for line in open(gtf):
            r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
        for rid, ids in reqs.items():
            out = gt.get(ids)
            if not out or len(out) < 5: continue
            g = [tuple(out[i:i+4]) for i in range(len(out)-3)]
            if len(set(g)) / max(len(g), 1) < 0.5: bad.add(rid)
    return bad

def featvec(nm, P, depth):
    v = [P["prob"], float(depth)]
    if nm == "suffix":
        v += [P["mlen"], P["lcnt"]]
    return v

def feat_names(nm):
    return ["prob", "depth"] + (["match_len", "lcnt"] if nm == "suffix" else [])

def collect(alive, nm):
    X, y, g = [], [], []
    for (rid, _, dep), e in alive:
        if nm not in e["P"]: continue
        X.append(featvec(nm, e["P"][nm], dep)); y.append(1 if e["P"][nm]["tok"] == e["gt"] else 0); g.append(rid)
    return np.array(X, float), np.array(y), np.array(g)

def oof_proba(method, X, y, g):
    pred = np.zeros(len(y))
    if len(y) < 10 or len(set(y)) < 2: return np.full(len(y), float(y.mean()))
    for tr, te in GroupKFold(min(5, len(set(g)))).split(X, y, g):
        if len(set(y[tr])) < 2: pred[te] = y[tr].mean(); continue
        if method == "gbm":
            m = HGB(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0).fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        else:
            mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
            m = LogisticRegression(max_iter=1000).fit((X[tr]-mu)/sd, y[tr])
            pred[te] = m.predict_proba((X[te]-mu)/sd)[:, 1]
    return pred

def fit_full(method, X, y):
    if method == "gbm":
        m = HGB(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0).fit(X, y)
        return dict(kind="gbm", model_b64=base64.b64encode(pickle.dumps(m)).decode())
    mu = X.mean(0); sd = X.std(0) + 1e-9
    m = LogisticRegression(max_iter=1000).fit((X-mu)/sd, y)
    return dict(kind="logistic", mean=mu.tolist(), std=sd.tolist(),
                coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]))

def run_length(blocks, pick_fn):
    tot = n = 0
    for k, pos in blocks.items():
        n += 1; run = 0
        for e in pos:
            gt = e["gt"]
            if gt is None or not e["P"]: break
            av = list(e["P"]); hits = [nm for nm in av if e["P"][nm]["tok"] == gt]
            if len(hits) == len(av): run += 1; continue
            if len(hits) == 0: break
            nm = pick_fn((k[0], k[1], e["depth"]), e)
            if nm is not None and e["P"][nm]["tok"] == gt: run += 1
            else: break
        tot += run
    return tot / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--props", nargs="+", required=True, help="name:token_field:prob_field")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    props = [tuple(p.split(":")) for p in args.props]
    names = [p[0] for p in props]
    log = f"{args.dir}/decisions_select1_oracle.jsonl"
    blocks = load_blocks(log, props)
    bad = loopy(args.dir); blocks = {k: v for k, v in blocks.items() if k[0] not in bad}
    alive = []
    for k, pos in blocks.items():
        al = True
        for e in pos:
            if not al: break
            if not e["P"]: continue
            av = list(e["P"]); hits = [nm for nm in av if e["P"][nm]["tok"] == e["gt"]]
            if 0 < len(hits) < len(av): alive.append(((k[0], k[1], e["depth"]), e))
            if e["gt"] is not None and len(hits) == 0: al = False
    print(f"blocks={len(blocks)} decisive={len(alive)} proposers={names}")

    bundle = {"meta": {"source": log, "props": names}, "models": {"gbm": {}, "logistic": {}}}
    # OOF scores for selacc/MAT verification
    oof = {"gbm": defaultdict(dict), "logistic": defaultdict(dict)}
    for nm in names:
        X, y, g = collect(alive, nm)
        if not len(y):
            print(f"  {nm}: no rows, skip"); continue
        print(f"  {nm}: n={len(y)} base_acc={y.mean():.4f} feats={feat_names(nm)}")
        for method in ("gbm", "logistic"):
            bundle["models"][method][nm] = dict(features=feat_names(nm), **fit_full(method, X, y))
            p = oof_proba(method, X, y, g)
            for (did, e), pp in zip([(d, e) for d, e in alive if nm in e["P"]], p):
                oof[method][did][nm] = float(pp)
    # verify selacc + block-anchored MAT
    def picker(method):
        return lambda did, e: (max(oof[method][did], key=lambda n: oof[method][did][n])
                               if oof[method].get(did) else None)
    for method in ("gbm", "logistic"):
        sa = np.mean([1.0 if (oof[method].get(d) and e["P"][max(oof[method][d], key=lambda n: oof[method][d][n])]["tok"] == e["gt"]) else 0.0
                      for d, e in alive])
        mat = run_length(blocks, picker(method))
        print(f"  VERIFY {method}: OOF selacc={sa:.4f}  block-anchored MAT={mat:.3f}")
    raw_mat = run_length(blocks, lambda did, e: max(e["P"], key=lambda n: e["P"][n]["prob"]) if e["P"] else None)
    orc_mat = run_length(blocks, lambda did, e: next((n for n in e["P"] if e["P"][n]["tok"] == e["gt"]), None))
    print(f"  (ref) raw block-anchored MAT={raw_mat:.3f}  oracle={orc_mat:.3f}")
    json.dump(bundle, open(args.out, "w"))
    print(f"WROTE {args.out}  (gbm+logistic per proposer)")

if __name__ == "__main__":
    main()
