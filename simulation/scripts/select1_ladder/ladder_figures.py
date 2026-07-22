"""Ladder per user spec:
  single proposers / calib(best-of-4, prob+depth) / calib(best-of-4, prob+depth+suffix(mlen,cnt))
  / bayes(prob+depth) / bayes(prob+depth+suffix(mlen,cnt)) / oracle
calib = best (max OOF selacc) of FOUR real calibration methods:
  histogram, isotonic, beta  -> 1-D probability calibrators (prob only)
  logistic                   -> multivariate LogisticRegression (uses the full feature set)
bayes = unconstrained HistGradientBoosting (GBM) on the full feature set.
suffix(mlen,cnt) = [match_len, log1p(suffix_count)]; cnt is informationally == tot given prob.
selacc on decisive positions; MAT = run-length on the gt trajectory. loopy reqs excluded.
LADDER_INSAMPLE=1 -> fit==eval (optimistic) instead of GroupKFold OOF.
"""
import json, math, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

ROOT = "simulation/results/chain_hybrid_perdepth"
OUT = "/tmp/claude-20051/-home-muchwater-advance-spec/65ec03a0-7cb7-4645-911b-7079f77dec72/scratchpad"
INSAMPLE = os.environ.get("LADDER_INSAMPLE") == "1"

CELLS = [
 dict(key="14b_2way", model="Qwen3-14B   ·   EAGLE3 + suffix", dir="qwen3_14b_ar",
      props=[("EAGLE3","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
 dict(key="27b_2way", model="Qwen3.5-27B   ·   MTP + suffix", dir="qwen35_27b_ar",
      props=[("MTP","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
 dict(key="8b_3way", model="Qwen3-8B   ·   DFlash + EAGLE3 + suffix", dir="qwen3_8b_dflash_e3_ceiling20",
      props=[("DFlash","eagle_token","eagle_p"),("EAGLE3","e3_token","e3_p"),("suffix","suffix_token","suffix_p")]),
 dict(key="27b_3way", model="Qwen3.5-27B   ·   MTP + DFlash + suffix", dir="qwen35_27b_3way_real_full",
      props=[("MTP","eagle_token","eagle_p"),("DFlash","dflash_token","dflash_p"),("suffix","suffix_token","suffix_p")]),
]

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
            if len(set(g))/max(len(g),1) < 0.5: bad.add(rid)
    return bad, len(reqs)

def fv(e, nm, featset):
    P = e["P"][nm]; v = [P["prob"], e["depth"]]
    if featset == "prob_depth_suffix" and nm == "suffix":
        v += [P["mlen"], P["lcnt"]]
    return v

def fit_predict(method, Xfull, prob, y, fit, pred):
    ytr = y[fit]
    if len(set(ytr)) < 2:
        return np.full(len(pred), ytr.mean())
    if method == "logistic":
        Xt = Xfull[fit]; m = Xt.mean(0); s = Xt.std(0) + 1e-9
        clf = LogisticRegression(max_iter=1000).fit((Xt - m) / s, ytr)
        return clf.predict_proba((Xfull[pred] - m) / s)[:, 1]
    if method == "gbm":
        clf = HGB(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0).fit(Xfull[fit], ytr)
        return clf.predict_proba(Xfull[pred])[:, 1]
    p = prob  # 1-D calibrators below use the proposer's own prob only
    if method == "iso":
        ir = IsotonicRegression(out_of_bounds="clip").fit(p[fit], ytr); return ir.predict(p[pred])
    if method == "beta":
        F = np.c_[np.log(np.clip(p, 1e-6, 1)), np.log(np.clip(1 - p, 1e-6, 1))]
        clf = LogisticRegression(max_iter=1000).fit(F[fit], ytr); return clf.predict_proba(F[pred])[:, 1]
    if method == "hist":
        edges = np.unique(np.quantile(p[fit], np.linspace(0, 1, 11)))
        if len(edges) < 3: return np.full(len(pred), ytr.mean())
        bt = np.clip(np.digitize(p[fit], edges[1:-1]), 0, len(edges)-2)
        means = np.array([ytr[bt == b].mean() if (bt == b).any() else ytr.mean() for b in range(len(edges)-1)])
        bp = np.clip(np.digitize(p[pred], edges[1:-1]), 0, len(edges)-2)
        return means[bp]
    raise ValueError(method)

def picks(alive, props, featset, method):
    names = [p[0] for p in props]
    R = {nm: {"X":[], "p":[], "y":[], "g":[], "d":[]} for nm in names}
    for did, e in alive:
        for nm in e["P"]:
            R[nm]["X"].append(fv(e, nm, featset)); R[nm]["p"].append(e["P"][nm]["prob"])
            R[nm]["y"].append(1 if e["P"][nm]["tok"] == e["gt"] else 0)
            R[nm]["g"].append(e["rid"]); R[nm]["d"].append(did)
    P_of = defaultdict(dict)
    for nm in names:
        X = np.array(R[nm]["X"], float); p = np.array(R[nm]["p"], float)
        y = np.array(R[nm]["y"]); g = np.array(R[nm]["g"]); ds = R[nm]["d"]
        if len(y) < 10 or len(set(y)) < 2:
            for d, yy in zip(ds, y): P_of[d][nm] = float(yy)
            continue
        prd = np.zeros(len(y))
        if INSAMPLE:
            prd = fit_predict(method, X, p, y, np.arange(len(y)), np.arange(len(y)))
        else:
            for tr, te in GroupKFold(min(5, len(set(g)))).split(X, y, g):
                prd[te] = fit_predict(method, X, p, y, tr, te)
        for d, pp in zip(ds, prd): P_of[d][nm] = float(pp)
    return {did: (max(P_of[did], key=lambda nm: P_of[did][nm]) if P_of[did] else None) for did, _ in alive}

def selacc(alive, pk):
    return np.mean([1.0 if (pk[d] is not None and e["P"][pk[d]]["tok"] == e["gt"]) else 0.0
                    for d, e in alive]) if alive else float("nan")

def run_length(blocks, pick_fn):
    tot = 0; n = 0
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

def best_calib(blocks, alive, props, featset):
    best = None
    for m in ("hist", "iso", "beta", "logistic"):
        pk = picks(alive, props, featset, m); sa = selacc(alive, pk)
        if best is None or sa > best[1]:
            best = (m, sa, pk)
    mat = run_length(blocks, lambda did, e, _p=best[2]: _p.get(did))
    return best[0], best[1], mat

def make_cell(cell):
    d = f"{ROOT}/{cell['dir']}"; props = cell["props"]; names = [p[0] for p in props]
    blocks = load_blocks(f"{d}/decisions_select1_oracle.jsonl", props)
    bad, n_tot = loopy(d); blocks = {k: v for k, v in blocks.items() if k[0] not in bad}
    used = len({k[0] for k in blocks})
    alive = []
    for k, pos in blocks.items():
        al = True
        for e in pos:
            if not al: break
            if not e["P"]: continue
            av = list(e["P"]); hits = [nm for nm in av if e["P"][nm]["tok"] == e["gt"]]
            if 0 < len(hits) < len(av): alive.append(((k[0], k[1], e["depth"]), e))
            if e["gt"] is not None and len(hits) == 0: al = False
    nd = len(alive)
    sfx = "suffix(mlen,cnt)"
    bars = []
    # single proposers
    for nm in names:
        sa = np.mean([1.0 if e["P"].get(nm, {}).get("tok") == e["gt"] else 0.0 for _, e in alive if nm in e["P"]]) if nd else float("nan")
        mat = run_length(blocks, lambda did, e, _n=nm: _n if _n in e["P"] else None)
        bars.append((f"{nm} only", sa, mat, "single"))
    # raw: argmax of raw probabilities (uncalibrated)
    def pick_raw(did, e):
        return max(e["P"], key=lambda nm: e["P"][nm]["prob"]) if e["P"] else None
    sa = selacc(alive, {did: pick_raw(did, e) for did, e in alive})
    bars.append(("raw  (prob)", sa, run_length(blocks, pick_raw), "raw"))
    # calib best-of-4 (two feature sets)
    for fs, lbl in [("prob_depth", "calib best-of-4\n(prob+depth)"),
                    ("prob_depth_suffix", f"calib best-of-4\n(prob+depth+{sfx})")]:
        m, sa, mat = best_calib(blocks, alive, props, fs)
        bars.append((f"{lbl}\n[{m}]", sa, mat, "calib"))
    # bayes (GBM) (two feature sets)
    for fs, lbl in [("prob_depth", "bayes\n(prob+depth)"),
                    ("prob_depth_suffix", f"bayes\n(prob+depth+{sfx})")]:
        pk = picks(alive, props, fs, "gbm"); sa = selacc(alive, pk)
        mat = run_length(blocks, lambda did, e, _p=pk: _p.get(did))
        bars.append((lbl, sa, mat, "bayes"))
    bars.append(("oracle", 1.0, run_length(blocks, lambda did, e: next((nm for nm in e["P"] if e["P"][nm]["tok"] == e["gt"]), None)), "oracle"))

    mode = "IN-SAMPLE (fit=eval, optimistic)" if INSAMPLE else "held-out (OOF)"
    sub = (f"bfcl_v4 web_search · requests used {used}/{n_tot} ({n_tot-used} runaway-repeat excluded) · "
           f"{nd} decisive · {mode}")
    colors = {"single":"#9aa7ad","raw":"#c2683a","calib":"#1f7a8c","bayes":"#6a4c93","oracle":"#2f7a57"}
    plots = [("Selection accuracy (decisive)", 1, [b for b in bars if b[3]!="single"], "selacc"),
             ("Selection accuracy (decisive)", 1, bars, "selacc_withsingle"),
             ("MAT", 2, bars, "mat")]
    for metric, idx, use, tag in plots:
        vals=[b[idx] for b in use]; labs=[b[0] for b in use]; ks=[b[3] for b in use]
        fig, ax = plt.subplots(figsize=(9.4, 6.6))
        xp = np.arange(len(use))
        ax.bar(xp, vals, color=[colors[k] for k in ks], edgecolor="white", width=0.72)
        for x, v in zip(xp, vals):
            ax.text(x, v + max(vals)*0.012, f"{v:.3f}" if idx==1 else f"{v:.2f}", va="bottom", ha="center", fontsize=8.5, fontfamily="monospace")
        ax.set_xticks(xp); ax.set_xticklabels(labs, rotation=28, ha="right", fontsize=8)
        ax.set_ylim(0, max(vals)*1.15); ax.set_ylabel(metric)
        ax.set_title(cell["model"], fontsize=13, fontweight="bold", loc="left")
        fig.text(0.012, 0.965, sub, fontsize=7.3, color="#5e6e78", ha="left")
        ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(rect=[0,0,1,0.93])
        out = f"{OUT}/lad_{cell['key']}_{tag}{'_insample' if INSAMPLE else ''}.png"
        fig.savefig(out, dpi=145); plt.close(fig)
    print(f"{cell['key']}: reqs {used}/{n_tot}, decisive {nd}")
    for b in bars: print(f"   {b[0].replace(chr(10),' '):52s} selacc={b[1]:.3f} MAT={b[2]:.3f}")

for c in CELLS:
    try: make_cell(c)
    except Exception as e:
        import traceback; print(f"ERR {c['key']}: {e}"); traceback.print_exc()
