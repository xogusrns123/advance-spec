"""What MAT does the exact Bayes boundary give? Out-of-fold, same reconstruction
as analyze_calib_why.py so raw/calib/oracle/bayes are apples-to-apples.

Bayes policy = P(suffix right | features) > 0.5, trained on alive-conditioned
decisive rows, applied OUT-OF-FOLD (GroupKFold by rid -> no leakage). Two variants:
  bayes(ep,sp)  = the green contour in raw_at_ceiling.png (what the user pointed at)
  bayes(all)    = ep,sp,depth,match_len,count,total  (the 0.795 ceiling)
Reconstruction: walk main chain; survive while pick matches gt (oracle_hit); if it
clears all 16 main-chain depths it rides the same suffix tail = served accept_len.
"""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
ORACLE_LOG = f"{DIR}/decisions_select1_oracle.jsonl"
CALIB_MAP = f"{DIR}/calib_cond-trained/calib_pp_beta.json"
ALIVE = {"eagle", "suffix", "both"}
SERVED = {"raw": 1.318, "calib": 1.351, "oracle": 1.761}

# calib map
_mp = json.load(open(CALIB_MAP))["groups"]
_M = {g: {int(d): (np.asarray(v["x"], float), np.asarray(v["y"], float))
          for d, v in dd.items()} for g, dd in _mp.items()}
def _cal(grp, p, d):
    m = _M[grp]
    if d in m:
        xs, ys = m[d]
    else:
        le = [k for k in m if k <= d]
        xs, ys = m[max(le)] if le else m[max(m)]
    return max(float(np.interp(p, xs, ys)), 1e-6)

# load chains + served accept_len
chains = defaultdict(list)
step_acc = {}
for line in open(ORACLE_LOG):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    t = r.get("type")
    if t == "decision":
        if r.get("tail"):
            continue
        chains[(r["rid"], r["decode_step"])].append(r)
    elif t == "step":
        step_acc[(r["rid"], r["decode_step"])] = r.get("accept_len")
for k in chains:
    chains[k].sort(key=lambda r: r["depth"])

# ---- build ALIVE decisive matrix (for training the Bayes policy) ---------------
rows_meta = []   # (rid, ds, depth)
feat = []        # ep, sp, depth, match_len, count, total
lab = []
for (rid, ds), rs in chains.items():
    alive = True
    for r in rs:
        if not alive:
            break
        h = r.get("oracle_hit")
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            rows_meta.append((rid, ds, r["depth"]))
            feat.append([r["eagle_p"], r["suffix_p"], r["depth"],
                         r["match_len"] or 0, r["suffix_count"] or 0, r["suffix_total"] or 0])
            lab.append(1 if h == "suffix" else 0)
        if h not in ALIVE:
            alive = False
feat = np.array(feat, float); lab = np.array(lab);
groups = np.array([m[0] for m in rows_meta])
n = len(lab)
print(f"alive decisive n={n}")

# ---- out-of-fold Bayes predictions ---------------------------------------------
def oof_pick(cols):
    pick = np.empty(n, int)
    gk = GroupKFold(n_splits=5)
    accs = []
    for tr, te in gk.split(feat, lab, groups):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)
        m.fit(feat[tr][:, cols], lab[tr])
        p = m.predict_proba(feat[te][:, cols])[:, 1]
        pick[te] = (p > 0.5).astype(int)
        accs.append(((p > 0.5).astype(int) == lab[te]).mean())
    return pick, float(np.mean(accs))

pick_epsp, acc_epsp = oof_pick([0, 1])
pick_epspd, acc_epspd = oof_pick([0, 1, 2])           # same features as calib
pick_all, acc_all = oof_pick([0, 1, 2, 3, 4, 5])
print(f"OOF decisive acc:  bayes(ep,sp)={acc_epsp:.3f}  "
      f"bayes(ep,sp,depth)={acc_epspd:.3f}  bayes(all)={acc_all:.3f}")
PICK_EPSP = {rows_meta[i]: ("suffix" if pick_epsp[i] else "eagle") for i in range(n)}
PICK_EPSPD = {rows_meta[i]: ("suffix" if pick_epspd[i] else "eagle") for i in range(n)}
PICK_ALL = {rows_meta[i]: ("suffix" if pick_all[i] else "eagle") for i in range(n)}

# ---- reconstruction ------------------------------------------------------------
def is_correct(pick, hit):
    if hit == "both":
        return True
    if hit == "eagle":
        return pick == "eagle"
    if hit == "suffix":
        return pick == "suffix"
    return False

def raw_pick(r):
    if r["suffix_p"] is None:
        return "eagle"
    if r["eagle_p"] is None:
        return "suffix"
    return "suffix" if r["suffix_p"] > r["eagle_p"] else "eagle"

def calib_pick(r):
    ep, sp, d = r["eagle_p"], r["suffix_p"], r["depth"]
    if sp is None:
        return "eagle"
    if ep is None:
        return "suffix"
    return "suffix" if _cal("suffix", sp, d) > _cal("eagle", ep, d) else "eagle"

def reconstruct(rs, key, pickfn=None, pickmap=None):
    served = step_acc.get(key)
    if served is None:
        return None
    for r in rs:
        hit = r.get("oracle_hit")
        if hit not in ALIVE:
            return r["depth"]
        if pickmap is not None:
            mk = (key[0], key[1], r["depth"])
            if hit in ("eagle", "suffix") and mk in pickmap:
                pk = pickmap[mk]
            else:                                  # both / agreement -> any pick ok
                pk = "eagle"
        else:
            pk = pickfn(r)
        if not is_correct(pk, hit):
            return r["depth"]
    return served

L = {p: [] for p in ["oracle", "raw", "calib", "bayes_epsp", "bayes_epspd", "bayes_all"]}
for key, rs in chains.items():
    if step_acc.get(key) is None:
        continue
    L["oracle"].append(step_acc[key])
    L["raw"].append(reconstruct(rs, key, pickfn=raw_pick))
    L["calib"].append(reconstruct(rs, key, pickfn=calib_pick))
    L["bayes_epsp"].append(reconstruct(rs, key, pickmap=PICK_EPSP))
    L["bayes_epspd"].append(reconstruct(rs, key, pickmap=PICK_EPSPD))
    L["bayes_all"].append(reconstruct(rs, key, pickmap=PICK_ALL))

print("\n=== reconstructed MAT (mean accepted draft tokens) ===")
mraw = np.mean(L["raw"])
labels = {"raw": "raw          (ep,sp; 1 thr)", "calib": "calib monot. (ep,sp,depth)",
          "bayes_epsp": "bayes joint  (ep,sp)", "bayes_epspd": "bayes joint  (ep,sp,depth)",
          "bayes_all": "bayes joint  (all 6 feat)", "oracle": "oracle (GT)"}
for p in ["raw", "calib", "bayes_epsp", "bayes_epspd", "bayes_all", "oracle"]:
    m = float(np.mean(L[p]))
    print(f"  {labels[p]:30s} recon={m:.4f}  (over-raw {m-mraw:+.4f})")

# served-scale: shift reconstruction by the known raw recon->served offset
off = SERVED["raw"] - mraw
print(f"\n=== served-scale estimate (recon + offset {off:+.4f}; raw->{SERVED['raw']}) ===")
for p, name in [("raw", "raw"), ("calib", "calib"), ("bayes_epsp", "bayes(ep,sp)"),
                ("bayes_all", "bayes(all feats)"), ("oracle", "oracle")]:
    print(f"  {name:16s} ~ {np.mean(L[p])+off:.3f}")
mo = np.mean(L["oracle"])
print(f"\nfraction of raw->oracle MAT gap recovered:")
for p, name in [("calib", "calib"), ("bayes_epsp", "bayes(ep,sp)"), ("bayes_all", "bayes(all)")]:
    frac = (np.mean(L[p]) - mraw) / (mo - mraw)
    print(f"  {name:16s} {frac*100:5.1f}%")
