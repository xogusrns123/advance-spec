"""Faithful offline per-depth select-1 MAT test (14B EAGLE3 vs suffix).

Suffix ARM and RAW score are the REAL served ones from the oracle decisions log
(suffix_token, suffix_p per depth); eagle arm likewise. ONLY the suffix SCORE is
swapped: raw suffix_p  vs  a NEW confidence = GBM on pure-suffix-side features
(back-off of the served suffix_token at shorter contexts, local/global agreement,
token class, unigram freq) computed from tries fed the REAL prompt (matched from
gt_tokens) + committed responses. new_conf is trained on served suffix accept
(suffix_token==gt), GroupKFold by request. Per-depth pick: suffix iff score>=eagle_p,
accept while picked token==gt (valid: accepted prefix is gt-conditioned).
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, "/workspace")
from transformers import AutoTokenizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold

MAXL = 12


def _entropy(cnt):
    n = sum(cnt.values())
    if n <= 1:
        return 0.0
    import math as _m
    return -sum((c / n) * _m.log(c / n) for c in cnt.values())


class Trie:
    def __init__(self):
        self.count = [defaultdict(Counter) for _ in range(MAXL + 1)]
        self.seq = []

    def commit(self, t):
        for L in range(1, min(MAXL, len(self.seq)) + 1):
            self.count[L][tuple(self.seq[-L:])][t] += 1
        self.seq.append(t)

    def child(self, L):
        if L < 1 or L > MAXL or len(self.seq) < L:
            return None
        return self.count[L].get(tuple(self.seq[-L:]))

    def longest_argmax(self):
        for L in range(min(MAXL, len(self.seq)), 0, -1):
            c = self.count[L].get(tuple(self.seq[-L:]))
            if c:
                return c.most_common(1)[0][0]
        return None

    def topk(self, k):
        for L in range(min(MAXL, len(self.seq)), 0, -1):
            c = self.count[L].get(tuple(self.seq[-L:]))
            if c:
                return [t for t, _ in c.most_common(k)]
        return []


def tok_classer(tk):
    cache = {}
    def cls(t):
        if t in cache:
            return cache[t]
        s = tk.decode([int(t)]); st = s.strip()
        c = (float(s[:1].isspace() or s == ""), float("\n" in s),
             float(len(st) > 0 and all(not ch.isalnum() for ch in st)),
             float(st.isdigit()), float(st.isalpha()), float(len(s)))
        cache[t] = c
        return c
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cls = tok_classer(tk)
    _bnd = {}
    def bnd(t):
        if t not in _bnd:
            _bnd[t] = any(ch in ".\n,?!;:" for ch in tk.decode([int(t)]))
        return _bnd[t]

    # parse log
    dec = defaultdict(lambda: defaultdict(dict)); stp = defaultdict(dict); order = []
    for line in open(args.log):
        r = json.loads(line); t = r.get("type")
        if t == "req":
            order.append(r["rid"])
        elif t == "decision":
            dec[r["rid"]][r["decode_step"]][r["depth"]] = r
        elif t == "step":
            stp[r["rid"]][r["decode_step"]] = r["accept_len"]

    def recon(rid):
        out = []
        for s in sorted(stp[rid]):
            a = stp[rid][s]; row = dec[rid][s]
            for d in range(0, a + 1):
                if d in row and row[d].get("gt_token") is not None:
                    out.append(row[d]["gt_token"])
        return out

    # gt_tokens rows
    gtrows = [json.loads(l) for l in open(args.gt)]

    def match(out):
        for gr in gtrows:
            o = gr["output_ids"]
            if o[1:1 + len(out)] == out and len(out) >= 8:
                return list(gr["input_ids"]), list(o)
        return None

    # process in log order; global trie gets every request's response
    G = Trie()
    uni = Counter()
    feats_all, y_all, grp_all = [], [], []
    EP_all, EA_all = [], []   # eagle_p and eagle-accept, aligned to feats_all
    conf_pos = {}   # rid -> {output_index j -> feat row index}
    topk_pos = {}   # rid -> {output_index j -> my-trie top-8 tokens}
    rid_full = {}
    matched = []
    NAMES = ["suffix_p", "logc", "logt", "matchlen",
             "bo1", "am1", "bo2", "am2", "bo_ml", "g_argmax", "g_has", "uni",
             "cls_space", "cls_nl", "cls_punct", "cls_digit", "cls_alpha", "cls_len",
             "g_branch", "g_ent", "prose_frac", "suf_recurs", "mismatch",
             "prompt_anch", "dist_bound"]

    for rid in order:
        out = recon(rid)
        if len(out) < 8:
            # still feed to global for fidelity
            L0 = Trie()
            for t in out:
                G.commit(t); uni[t] += 1
            continue
        m = match(out)
        if m is None:
            for t in out:
                G.commit(t); uni[t] += 1
            continue
        prompt, full = m
        matched.append(rid)
        Lt = Trie()
        for t in prompt:
            Lt.commit(t)
        # prompt n-gram sets (for verbatim-quote / prompt-anchored detection)
        pset = [set() for _ in range(MAXL + 1)]
        for L in range(1, MAXL + 1):
            for i in range(len(prompt) - L + 1):
                pset[L].add(tuple(prompt[i:i + L]))
        BND = set()  # committed indices that are clause boundaries
        rid_full[rid] = full
        conf_pos[rid] = {}
        # walk full output; committed == full[1:]; log (step,depth<=accept) maps to j=1+cum+depth
        # build a lookup: output index j -> (served suffix_token, suffix_p, count, total, match_len, gt)
        jinfo = {}
        for s in sorted(stp[rid]):
            cum = 0
            # recompute cum as committed length before step s
            pass
        # compute cum per step
        cum = 0
        for s in sorted(stp[rid]):
            a = stp[rid][s]
            for d in range(0, a + 1):
                row = dec[rid][s].get(d)
                if row is None:
                    continue
                j = 1 + cum + d   # output index
                jinfo[j] = row
            cum += a + 1
        # dense replay over full output
        last_bnd = -1
        for j in range(len(full)):
            g = full[j]
            row = jinfo.get(j)
            if row is not None and row.get("suffix_token") is not None:
                st = row["suffix_token"]
                sp = float(row.get("suffix_p") or 0.0)
                cnt = float(row.get("suffix_count") or 0.0)
                tot = float(row.get("suffix_total") or 0.0)
                ml = int(row.get("match_len") or 0)
                def bo(L2):
                    c = Lt.child(L2)
                    if not c:
                        return 0.0, 0.0
                    nn = sum(c.values())
                    return c.get(st, 0) / nn, float(c.most_common(1)[0][0] == st)
                bo1, am1 = bo(1); bo2, am2 = bo(2)
                bo_ml, _ = bo(min(max(ml, 1), MAXL))
                gam = float(G.longest_argmax() == st)
                gc = G.child(1)
                g_has = float(bool(gc) and st in gc)
                un = uni.get(st, 0) / max(1, len(G.seq))
                cc = cls(st)
                # PATTERN features (regime / semantic-fit proxies, motivated by micro failures)
                g2 = G.child(2)
                g_branch = float(len(g2)) if g2 else 0.0     # global generality of bigram ctx
                g_ent = _entropy(g2) if g2 else 0.0
                recent = Lt.seq[-16:]
                prose_frac = (sum(cls(t)[4] for t in recent) / len(recent)) if recent else 0.0
                suf_recurs = float(st in set(Lt.seq[-32:]))   # topical coherence
                prevtok = Lt.seq[-1] if Lt.seq else st
                pcls = cls(prevtok)
                mismatch = float(pcls[4] == 1.0 and cc[2] == 1.0)  # prev-alpha -> suffix-punct
                prompt_anch = float(ml >= 1 and tuple(Lt.seq[-ml:]) in pset[min(ml, MAXL)])
                dist_bound = float(min(j - last_bnd, 40)) if last_bnd >= 0 else 40.0
                feats_all.append([sp, math.log1p(cnt), math.log1p(tot), float(ml),
                                  bo1, am1, bo2, am2, bo_ml, gam, g_has, un,
                                  cc[0], cc[1], cc[2], cc[3], cc[4], cc[5],
                                  g_branch, g_ent, prose_frac, suf_recurs, mismatch,
                                  prompt_anch, dist_bound])
                y_all.append(int(st == g)); grp_all.append(rid)
                ep_here = float(row.get("eagle_p") or 0.0)
                et_here = row.get("eagle_token")
                EP_all.append(ep_here); EA_all.append(int(et_here == g))
                conf_pos[rid][j] = len(feats_all) - 1
                topk_pos.setdefault(rid, {})[j] = Lt.topk(8)
            Lt.commit(g); G.commit(g); uni[g] += 1
            if bnd(g):
                last_bnd = j
        # note: committed==full[1:]; but we also fed full[0]; global gets full[0..], fine

    X = np.asarray(feats_all, np.float32); Y = np.asarray(y_all, np.int8); grp = np.array(grp_all)
    print(f"matched {len(matched)}/{len(order)} requests; positions={len(Y)}; served suffix accept={Y.mean()*100:.1f}%")

    # sanity: my feature suffix_p (col0) is the LOG's suffix_p (by construction) -> ok
    def oof(Xm, ym):
        c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, random_state=0)
        return cross_val_predict(c, Xm, ym, cv=GroupKFold(5), groups=grp, method="predict_proba")[:, 1]

    EP = np.asarray(EP_all, np.float32); EA = np.asarray(EA_all, np.int8)
    conf = oof(X, Y)                       # new suffix confidence (all new features)
    cal_e = oof(EP.reshape(-1, 1), EA)     # calibrated eagle P(accept) from eagle_p
    cal_s = oof(X[:, [0]], Y)              # calibrated suffix P(accept) from raw suffix_p only
    # JOINT competition model: estimate BOTH arms' P(correct) from [suffix feats + eagle_p]
    Xj = np.hstack([X, EP.reshape(-1, 1)])
    Ds = oof(Xj, Y)                        # P(suffix correct | suffix feats, eagle_p)
    De = oof(Xj, EA)                       # P(eagle correct  | suffix feats, eagle_p)
    def insample(Xm, ym):
        c = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_depth=6, random_state=0)
        c.fit(Xm, ym); return c.predict_proba(Xm)[:, 1]
    Ds_is = insample(Xj, Y); De_is = insample(Xj, EA)   # in-sample ceiling with FULL feature set

    # MAT sim
    def sim(mode):
        accs = []
        for rid in matched:
            cum = 0
            for s in sorted(stp[rid]):
                a = stp[rid][s]
                d = 0
                while True:
                    row = dec[rid][s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                    ep = float(row.get("eagle_p") or 0.0); sp = float(row.get("suffix_p") or 0.0)
                    j = 1 + cum + d
                    fi = conf_pos.get(rid, {}).get(j)
                    if mode == "eagle":
                        tok = et
                    elif mode == "suffix":
                        tok = st
                    elif mode == "oracle":
                        tok = g if (st == g or et == g) else None
                    elif mode == "joint":
                        use_suf = (st is not None) and (fi is not None) and (Ds[fi] > De[fi])
                        if fi is None:
                            use_suf = (st is not None) and (sp >= ep)
                        tok = st if use_suf else et
                    elif mode == "insample":
                        use_suf = (st is not None) and (fi is not None) and (Ds_is[fi] > De_is[fi])
                        if fi is None:
                            use_suf = (st is not None) and (sp >= ep)
                        tok = st if use_suf else et
                    else:
                        if mode == "raw" or fi is None:
                            ss, ec = sp, ep
                        elif mode == "new":
                            ss, ec = conf[fi], ep
                        elif mode == "new_both":
                            ss, ec = conf[fi], cal_e[fi]
                        elif mode == "cal_raw":
                            ss, ec = cal_s[fi], cal_e[fi]
                        use_suf = (st is not None) and (ss >= ec); tok = st if use_suf else et
                    if tok is not None and tok == g:
                        d += 1
                    else:
                        break
                accs.append(d)
                cum += a + 1
        return float(np.mean(accs)), len(accs)

    mr, n = sim("raw"); me, _ = sim("eagle"); msu, _ = sim("suffix"); mo, _ = sim("oracle")
    mn, _ = sim("new"); mcr, _ = sim("cal_raw"); mnb, _ = sim("new_both"); mj, _ = sim("joint"); mis, _ = sim("insample")
    print(f"\nper-depth select-1 MAT (n_steps={n}):")
    print(f"   eagle(EAGLE3)-only        : {me:.4f}")
    print(f"   suffix-only               : {msu:.4f}")
    print(f"   RAW      (suffix_p vs eagle_p)    : {mr:.4f}")
    print(f"   CAL_RAW  (cal suffix_p vs cal eagle): {mcr:.4f}   (delta {mcr-mr:+.4f})")
    print(f"   NEW      (new conf vs raw eagle_p): {mn:.4f}   (delta {mn-mr:+.4f})")
    print(f"   NEW_BOTH (new conf vs cal eagle)  : {mnb:.4f}   (delta {mnb-mr:+.4f})")
    print(f"   JOINT    (Ds>De, competition)     : {mj:.4f}   (delta {mj-mr:+.4f})")
    print(f"   INSAMPLE (full-feat ceiling)      : {mis:.4f}   (delta {mis-mr:+.4f})")
    print(f"   ORACLE (best per depth)   : {mo:.4f}")
    if mo - mr > 1e-6:
        for nm, mv in [("NEW", mn), ("NEW_BOTH", mnb), ("JOINT", mj), ("INSAMPLE", mis)]:
            print(f"   {nm} recovers {(mv-mr)/(mo-mr)*100:+.1f}% of raw->oracle gap")

    # gap composition: within the ORACLE run, per-depth categorize decision-sensitivity
    cat = Counter(); raw_ok = Counter(); new_ok = Counter()
    for rid in matched:
        cum = 0
        for s in sorted(stp[rid]):
            a = stp[rid][s]; d = 0
            while True:
                row = dec[rid][s].get(d)
                if row is None or row.get("gt_token") is None:
                    break
                g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                sc = (st == g); ec = (et == g)
                if not (sc or ec):
                    break  # oracle run ends
                ep = float(row.get("eagle_p") or 0.0); sp = float(row.get("suffix_p") or 0.0)
                fi = conf_pos.get(rid, {}).get(1 + cum + d)
                if sc and ec:
                    cat["both"] += 1
                elif sc:
                    cat["only_suffix"] += 1
                    raw_ok["only_suffix"] += int(sp >= ep)
                    new_ok["only_suffix"] += int((conf[fi] if fi is not None else sp) >= ep)
                else:
                    cat["only_eagle"] += 1
                    raw_ok["only_eagle"] += int(sp < ep)
                    new_ok["only_eagle"] += int((conf[fi] if fi is not None else sp) < ep)
                d += 1
            cum += a + 1
    tot = sum(cat.values())
    print(f"\ngap composition within oracle run (n={tot} depth-positions):")
    for k in ("both", "only_suffix", "only_eagle"):
        print(f"   {k:<12}: {cat[k]} ({cat[k]/tot*100:.1f}%)")
    for k in ("only_suffix", "only_eagle"):
        if cat[k]:
            print(f"   correct pick @ {k}: raw {raw_ok[k]/cat[k]*100:.1f}%  new {new_ok[k]/cat[k]*100:.1f}%")

    # sweep decision bias: pick suffix iff conf >= cal_e + b (find MAT-optimal operating point)
    def sim_bias(b):
        accs = []
        for rid in matched:
            cum = 0
            for s in sorted(stp[rid]):
                a = stp[rid][s]; d = 0
                while True:
                    row = dec[rid][s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                    ep = float(row.get("eagle_p") or 0.0); sp = float(row.get("suffix_p") or 0.0)
                    fi = conf_pos.get(rid, {}).get(1 + cum + d)
                    if fi is not None:
                        use_suf = (st is not None) and (conf[fi] >= cal_e[fi] + b)
                    else:
                        use_suf = (st is not None) and (sp >= ep)
                    tok = st if use_suf else et
                    if tok is not None and tok == g:
                        d += 1
                    else:
                        break
                accs.append(d); cum += a + 1
        return float(np.mean(accs))
    # which FEATURES separate the DECISION? only_suffix (Y=1,EA=0) vs only_eagle (Y=0,EA=1)
    os_m = (Y == 1) & (EA == 0); oe_m = (Y == 0) & (EA == 1)
    print(f"\nfeature separation only_suffix(n={os_m.sum()}) vs only_eagle(n={oe_m.sum()}) [std-diff, sorted]:")
    seps = []
    for i, nm in enumerate(NAMES):
        a = X[os_m, i]; b = X[oe_m, i]
        sd = (a.mean() - b.mean()) / (X[:, i].std() + 1e-9)
        seps.append((nm, a.mean(), b.mean(), sd))
    for nm, am, bm, sd in sorted(seps, key=lambda t: -abs(t[3]))[:12]:
        print(f"   {nm:<11} only_suf={am:6.3f} only_eag={bm:6.3f}  std-diff={sd:+.3f}")

    # RESIDUAL errors of NEW_BOTH: only_eagle but model still picks suffix (conf>=cal_e). Decode.
    print(f"\nRESIDUAL confident inversions NEW_BOTH still mis-picks (only_eagle, conf>=cal_e):")
    shown = 0
    for rid in matched:
        cum = 0
        for s in sorted(stp[rid]):
            a = stp[rid][s]; d = 0
            while True:
                row = dec[rid][s].get(d)
                if row is None or row.get("gt_token") is None:
                    break
                g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                sc = (st == g); ec = (et == g)
                if not (sc or ec):
                    break
                fi = conf_pos.get(rid, {}).get(1 + cum + d)
                if ec and not sc and fi is not None and conf[fi] >= cal_e[fi] and shown < 20:
                    full = rid_full[rid]; jj = 1 + cum + d
                    ctx = tk.decode(full[max(0, jj - 14):jj])
                    print(f"   conf={conf[fi]:.2f} cal_e={cal_e[fi]:.2f} ml={int(X[fi,3])} | ...{ctx[-46:]!r} "
                          f"SUF={tk.decode([st])!r} EAG={tk.decode([et])!r} GT={tk.decode([g])!r}")
                    shown += 1
                d += 1
            cum += a + 1

    # HYBRID: raw aggressiveness + conf veto -> suffix iff (suffix_p>=eagle_p) AND (conf>=v)
    def sim_hybrid(v):
        accs = []
        for rid in matched:
            cum = 0
            for s in sorted(stp[rid]):
                a = stp[rid][s]; d = 0
                while True:
                    row = dec[rid][s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                    ep = float(row.get("eagle_p") or 0.0); sp = float(row.get("suffix_p") or 0.0)
                    fi = conf_pos.get(rid, {}).get(1 + cum + d)
                    cf = conf[fi] if fi is not None else sp
                    use_suf = (st is not None) and (sp >= ep) and (cf >= v)
                    tok = st if use_suf else et
                    if tok is not None and tok == g:
                        d += 1
                    else:
                        break
                accs.append(d); cum += a + 1
        return float(np.mean(accs))
    print(f"\nHYBRID sweep (suffix iff suffix_p>=eagle_p AND conf>=v):")
    hb = (None, -1)
    for v in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        mv = sim_hybrid(v)
        if mv > hb[1]:
            hb = (v, mv)
        print(f"   v={v:.2f}: MAT {mv:.4f} ({mv-mr:+.4f} vs raw)")
    print(f"   BEST v={hb[0]:.2f} MAT {hb[1]:.4f} ({hb[1]-mr:+.4f} vs raw, {(hb[1]-mr)/(mo-mr)*100:+.1f}% of gap)")

    # ---- REALIZED TREE-select: suffix proposes top-w + eagle; accept if gt in union ----
    # width_fn(conf) -> w (# suffix candidates). cost = avg(1 + w) verify tokens/depth.
    def sim_tree(width_fn):
        accs = []; widths = []
        for rid in matched:
            cum = 0
            for s in sorted(stp[rid]):
                a = stp[rid][s]; d = 0
                while True:
                    row = dec[rid][s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                    j = 1 + cum + d; fi = conf_pos.get(rid, {}).get(j)
                    cf = conf[fi] if fi is not None else 0.5
                    w = max(1, min(8, int(width_fn(cf))))
                    widths.append(w)
                    sib = topk_pos.get(rid, {}).get(j, [])
                    cand = [st] + [c for c in sib if c != st]
                    cand = set(cand[:w])
                    if (g in cand) or (et == g):
                        d += 1
                    else:
                        break
                accs.append(d); cum += a + 1
        return float(np.mean(accs)), float(np.mean(widths))

    print(f"\nREALIZED tree-select MAT (suffix top-w + eagle; cost=avg(1+w)):")
    for W in (1, 2, 3, 4):
        mv, aw = sim_tree(lambda cf, W=W: W)
        print(f"   uniform  w={W}: MAT {mv:.4f}  avg_w={aw:.2f}  cost≈{1+aw:.2f}  ({mv-mr:+.4f} vs raw)")
    # adaptive: widen where conf low (top-1 likely wrong). tune to ~match a uniform avg budget.
    import numpy as _np
    med = float(_np.median(conf))
    for lo, hi in [(1, 3), (1, 5), (2, 4)]:
        mv, aw = sim_tree(lambda cf, lo=lo, hi=hi, med=med: hi if cf < med else lo)
        print(f"   adaptive lo={lo}/hi={hi}(@conf<med): MAT {mv:.4f}  avg_w={aw:.2f}  cost≈{1+aw:.2f}  ({mv-mr:+.4f} vs raw)")

    print(f"\ndecision-bias sweep (suffix iff conf >= cal_e + b):")
    best = (None, -1)
    for b in [-0.4, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]:
        mv = sim_bias(b)
        if mv > best[1]:
            best = (b, mv)
        print(f"   b={b:+.2f}: MAT {mv:.4f} ({mv-mr:+.4f} vs raw)")
    print(f"   BEST b={best[0]:+.2f} MAT {best[1]:.4f} ({best[1]-mr:+.4f} vs raw, {(best[1]-mr)/(mo-mr)*100:+.1f}% of gap)")


if __name__ == "__main__":
    main()
