"""Per-tree calibration then compare FIRST tokens (user's idea). local & global trees have
DIFFERENT reliability curves (local 0.5@n=2 != global 0.5@n=20), so comparing raw c/n
(strategy B) is apples-to-oranges. Fit a SEPARATE OOF calibrator per tree
(P(top==gt | prob, log n, match_len)) and pick the tree with the higher CALIBRATED
first-token prob. Compare first-token accept vs A(path-score), B(raw prob), G(global), oracle.

Run (IN docker): docker exec sglang-bench python3 /workspace/simulation/scripts/select1_ladder/global_tree_calib.py
"""
import json, math
import numpy as np
from collections import defaultdict
from arctic_inference.suffix_decoding.cache import SuffixDecodingCache, SuffixTree
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

GT="simulation/results/chain_hybrid_perdepth/qwen3_14b_ar/gt_tokens.jsonl"
MAXD=64; FACTOR=4.0; OFFSET=0.0; MINP=0.1

def spec(tree, ctx):
    try: return tree.speculate(ctx, MAXD, FACTOR, OFFSET, MINP, False)
    except Exception:
        return SuffixTree.speculate(tree, np.array(ctx, np.int32), MAXD, FACTOR, OFFSET, MINP, False)

def top(d):
    if not d.token_ids: return None
    c=int(d.counts[0]); p=float(d.probs[0]); tok=int(d.token_ids[0])
    n=int(round(c/p)) if p>0 else 0
    return dict(tok=tok, c=c, n=n, p=p, ml=int(d.match_len), score=float(d.score))

recs=[json.loads(l) for l in open(GT)]
sc=SuffixDecodingCache(max_tree_depth=MAXD)
rows=[]   # per suffix-proposing position
for ri,rec in enumerate(recs):
    rid=f"r{ri}"; prompt=list(rec.get("input_ids") or []); gen=list(rec.get("output_ids") or [])
    if not gen: continue
    sc.start_request(rid, prompt); lt=sc._local_trees[rid]; gtree=sc._global_tree
    seq=list(prompt)
    for tok in gen:
        ctx=seq[-MAXD:]
        dl=top(spec(lt,ctx)); dg=top(spec(gtree,ctx))
        if dl is not None:
            for d in (dl,dg):
                if d is not None: d["corr"]=int(d["tok"]==tok)
            rows.append(dict(rid=rid, l=dl, g=dg, half=(dl["n"]==2 and dl["c"]==1)))
        seq.append(tok);
    sc.add_active_response(rid, gen); sc.stop_request(rid)

def oof_cal(subset):
    """OOF P(corr) for rows in subset (list of feature dicts). returns list aligned."""
    X=np.array([[r["p"], math.log1p(r["n"]), r["ml"]] for r in subset], float)
    y=np.array([r["corr"] for r in subset]); g=np.array([r["rid"] for r in subset])
    pred=np.full(len(y), y.mean())
    if len(set(y.tolist()))>1 and len(y)>50:
        for tr,te in GroupKFold(5).split(X,y,g):
            if len(set(y[tr].tolist()))<2: pred[te]=y[tr].mean(); continue
            pred[te]=HGB(max_depth=3,max_iter=200,learning_rate=0.06,l2_regularization=1.0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return pred

# build local & global subsets with back-pointers
loc=[dict(r["l"], rid=r["rid"], _i=i) for i,r in enumerate(rows)]
glo=[dict(r["g"], rid=r["rid"], _i=i) for i,r in enumerate(rows) if r["g"] is not None]
cl=oof_cal(loc); cg=oof_cal(glo)
for r in rows: r["cal_l"]=None; r["cal_g"]=None
for r_,v in zip(loc,cl): rows[r_["_i"]]["cal_l"]=float(v)
for r_,v in zip(glo,cg): rows[r_["_i"]]["cal_g"]=float(v)

def acc_of(pick):  # pick(row)-> corr (0/1)
    return 100*np.mean([pick(r) for r in rows])
def acc_half(pick):
    hr=[r for r in rows if r["half"]]
    return 100*np.mean([pick(r) for r in hr]), len(hr)

def S_A(r):  # path-score (current)
    if r["g"] is None: return r["l"]["corr"]
    return r["l"]["corr"] if r["l"]["score"]>=r["g"]["score"] else r["g"]["corr"]
def S_B(r):  # raw first-token prob
    if r["g"] is None: return r["l"]["corr"]
    return r["l"]["corr"] if r["l"]["p"]>=r["g"]["p"] else r["g"]["corr"]
def S_CAL(r):  # per-tree calibrated first-token prob
    if r["g"] is None: return r["l"]["corr"]
    return r["l"]["corr"] if r["cal_l"]>=r["cal_g"] else r["g"]["corr"]
def S_G(r):  return r["g"]["corr"] if r["g"] is not None else r["l"]["corr"]
def S_O(r):  return max(r["l"]["corr"], r["g"]["corr"] if r["g"] is not None else 0)

print(f"suffix-proposing positions: {len(rows)}   local-1/2: {sum(r['half'] for r in rows)}")
print(f"\n{'strategy':34} {'ALL accept':>11} {'LOCAL-1/2 accept':>18}")
for name,fn in (("A current (max PATH-score)",S_A),("B raw first-token prob",S_B),
                ("CAL per-tree calibrated (user)",S_CAL),("G always global",S_G),
                ("O oracle (either tree)",S_O)):
    a=acc_of(fn); h,nh=acc_half(fn)
    print(f"{name:34} {a:>10.1f}% {h:>17.1f}%")
