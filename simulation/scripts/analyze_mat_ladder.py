"""MAT-loss ladder (vs oracle) for raw / calib / joint(ep,sp,d) / joint(all),
parallel to the decisive-accuracy ladder. Reconstructs L per policy on the eval
oracle log (calib = accept_rate cond beta; joint = OOF GBM picks)."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
CALIB = "simulation/results/chain_hybrid_perdepth/qwen3_14b_ar/calib_cond-trained/calib_pp_beta.json"
ALIVE = {"eagle", "suffix", "both"}
_M = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
          for d, v in dd.items()} for g, dd in json.load(open(CALIB))["groups"].items()}
def cal(grp, p, d):
    m = _M[grp]
    if d in m: xs, ys = m[d]
    else:
        le = [k for k in m if k <= d]; xs, ys = m[max(le)] if le else m[max(m)]
    return max(float(np.interp(p, xs, ys)), 1e-6)

chains = defaultdict(list); step_acc = {}
for line in open(f"{DIR}/decisions_select1_oracle.jsonl"):
    line = line.strip()
    if not line: continue
    r = json.loads(line); t = r.get("type")
    if t == "decision" and not r.get("tail"): chains[(r["rid"], r["decode_step"])].append(r)
    elif t == "step": step_acc[(r["rid"], r["decode_step"])] = r.get("accept_len")
for k in chains: chains[k].sort(key=lambda r: r["depth"])

# alive decisive rows for OOF joint classifiers
meta, feat, lab = [], [], []
for (rid, ds), rs in chains.items():
    alive = True
    for r in rs:
        if not alive: break
        h = r.get("oracle_hit")
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            meta.append((rid, ds, r["depth"]))
            feat.append([r["eagle_p"], r["suffix_p"], r["depth"], r["match_len"] or 0,
                         r["suffix_count"] or 0, r["suffix_total"] or 0])
            lab.append(1 if h == "suffix" else 0)
        if h not in ALIVE: alive = False
feat = np.array(feat); lab = np.array(lab); grp = np.array([m[0] for m in meta])
def oof(cols):
    pick = np.empty(len(lab), int)
    for tr, te in GroupKFold(5).split(feat, lab, grp):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)
        m.fit(feat[tr][:, cols], lab[tr]); pick[te] = (m.predict_proba(feat[te][:, cols])[:, 1] > 0.5)
    return {meta[i]: ("suffix" if pick[i] else "eagle") for i in range(len(lab))}
PJ = oof([0, 1, 2]); PA = oof([0, 1, 2, 3, 4, 5])

def is_correct(pk, hit): return hit == "both" or (hit == "eagle" and pk == "eagle") or (hit == "suffix" and pk == "suffix")
def raw_pick(r): return "eagle" if r["suffix_p"] is None else ("suffix" if (r["eagle_p"] is None or r["suffix_p"] > r["eagle_p"]) else "eagle")
def calib_pick(r):
    if r["suffix_p"] is None: return "eagle"
    if r["eagle_p"] is None: return "suffix"
    return "suffix" if cal("suffix", r["suffix_p"], r["depth"]) > cal("eagle", r["eagle_p"], r["depth"]) else "eagle"
def recon(key, rs, pickfn=None, pmap=None):
    served = step_acc.get(key)
    if served is None: return None
    for r in rs:
        hit = r.get("oracle_hit")
        if hit not in ALIVE: return r["depth"]
        if pmap is not None:
            pk = pmap.get((key[0], key[1], r["depth"]), "eagle") if hit in ("eagle", "suffix") else "eagle"
        else:
            pk = pickfn(r)
        if not is_correct(pk, hit): return r["depth"]
    return served
L = {p: [] for p in ["raw", "calib", "jointJ", "jointA", "oracle"]}
for k, rs in chains.items():
    if step_acc.get(k) is None: continue
    L["oracle"].append(step_acc[k])
    L["raw"].append(recon(k, rs, pickfn=raw_pick))
    L["calib"].append(recon(k, rs, pickfn=calib_pick))
    L["jointJ"].append(recon(k, rs, pmap=PJ))
    L["jointA"].append(recon(k, rs, pmap=PA))
mat = {p: float(np.mean(v)) for p, v in L.items()}
print("MAT:", {p: round(mat[p], 4) for p in mat})
mo = mat["oracle"]
print("MAT loss vs oracle:")
for p in ["raw", "calib", "jointJ", "jointA", "oracle"]:
    print(f"  {p:7s} loss={mo-mat[p]:.4f}")
