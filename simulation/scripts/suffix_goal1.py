"""GOAL 1: close >=30% of the raw->oracle gap in CHAIN per-depth select-1, using
LOG-ONLY microscopic features (available for ALL requests, no gt_tokens match):
suffix_p, suffix_count, suffix_total, match_len, eagle_p, depth, suffix-token class.
Joint competition: Ds=P(suffix correct|suffix feats), De=P(eagle correct|eagle_p);
pick suffix iff Ds>De. Also a fully-joint variant (both scores in both models).
Plus a microscopic decomposition of the suffix_p=0.5 atom.
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


def entropy(cnt):
    n = sum(cnt.values())
    if n <= 1:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in cnt.values())


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="comma-sep result dirs")
    args = ap.parse_args()
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ccache = {}
    def cls(t):
        if t is None:
            return (0, 0, 0, 0, 0, 0)
        if t in ccache:
            return ccache[t]
        s = tk.decode([int(t)]); st = s.strip()
        c = (float(s[:1].isspace() or s == ""), float("\n" in s),
             float(len(st) > 0 and all(not ch.isalnum() for ch in st)),
             float(st.isdigit()), float(st.isalpha()), float(len(s)))
        ccache[t] = c
        return c

    X, Ys, Ye, grp = [], [], [], []
    pos_index = {}  # (di,rid,step,depth) -> feat row
    topk_index = {}  # (di,rid,step,depth) -> suffix top-k tokens (my trie)
    DEC = {}; STP = {}; gids = []; seen = set()
    for di, d0 in enumerate([x.strip() for x in args.dirs.split(",") if x.strip()]):
        dec = defaultdict(lambda: defaultdict(dict)); stp = defaultdict(dict); order = []
        for line in open(f"{d0}/decisions_select1_oracle.jsonl"):
            r = json.loads(line); t = r.get("type")
            if t == "req": order.append(r["rid"])
            elif t == "decision": dec[r["rid"]][r["decode_step"]][r["depth"]] = r
            elif t == "step": stp[r["rid"]][r["decode_step"]] = r["accept_len"]
        G = Trie()   # global trie (response-only), per serving-run (per dir)
        nuse = 0
        for rid in order:
            # committed recon + step->cum map
            recon = []; cum = {}
            for s in sorted(stp[rid]):
                cum[s] = len(recon); a = stp[rid][s]
                for dd in range(0, a + 1):
                    if dd in dec[rid][s] and dec[rid][s][dd].get("gt_token") is not None:
                        recon.append(dec[rid][s][dd]["gt_token"])
            if len(recon) < 8:
                for t in recon: G.commit(t)
                continue
            ck = tuple(recon[:16])
            if ck in seen:
                for t in recon: G.commit(t)
                continue
            seen.add(ck); nuse += 1
            gid = (di, rid); gids.append(gid); DEC[gid] = dec[rid]; STP[gid] = stp[rid]
            # committed index -> (s,d) for live positions
            pos2sd = {}
            for s in sorted(stp[rid]):
                a = stp[rid][s]
                for d in range(0, a + 1):
                    if d in dec[rid][s]:
                        pos2sd[cum[s] + d] = (s, d)
            Lt = Trie()
            for i in range(len(recon)):
                g = recon[i]
                sd = pos2sd.get(i)
                if sd is not None:
                    s, d = sd; row = dec[rid][s][d]
                    st = row.get("suffix_token")
                    if st is not None:
                        sp = float(row.get("suffix_p") or 0.0)
                        c = float(row.get("suffix_count") or 0.0); nn = float(row.get("suffix_total") or 0.0)
                        ml = int(row.get("match_len") or 0); ep = float(row.get("eagle_p") or 0.0)
                        def bo(L2):
                            cc2 = Lt.child(L2)
                            if not cc2: return 0.0, 0.0
                            tot2 = sum(cc2.values()); return cc2.get(st, 0) / tot2, float(cc2.most_common(1)[0][0] == st)
                        bo1, am1 = bo(1); bo2, am2 = bo(2); bo_ml, _ = bo(min(max(ml, 1), MAXL))
                        gam = float(G.longest_argmax() == st)
                        gc = G.child(1); g_has = float(bool(gc) and st in gc)
                        g2 = G.child(2); g_branch = float(len(g2)) if g2 else 0.0; g_ent = entropy(g2) if g2 else 0.0
                        recent = Lt.seq[-16:]
                        prose = (sum(cls(t)[4] for t in recent) / len(recent)) if recent else 0.0
                        cc = cls(st)
                        X.append([sp, math.log1p(c), math.log1p(nn), float(ml), ep, float(d),
                                  bo1, am1, bo2, am2, bo_ml, gam, g_has, g_branch, g_ent, prose,
                                  cc[0], cc[1], cc[2], cc[3], cc[4], cc[5]])
                        Ys.append(int(st == g)); Ye.append(int(row.get("eagle_token") == g)); grp.append(str(ck))
                        pos_index[(di, rid, s, d)] = len(X) - 1
                        topk_index[(di, rid, s, d)] = Lt.topk(4)   # suffix tree candidates (my trie)
                Lt.commit(g); G.commit(g)
        print(f"[{d0.split('/')[-1]}] unique {nuse}/{len(order)}")
    X = np.asarray(X, np.float32); Ys = np.asarray(Ys, np.int8); Ye = np.asarray(Ye, np.int8)
    grp = np.array(grp)
    print(f"POOLED requests={len(gids)} positions={len(Ys)} suffix_acc={Ys.mean()*100:.1f}% eagle_acc={Ye.mean()*100:.1f}%")

    def oof(Xm, ym):
        c = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06, max_depth=6,
                                           l2_regularization=1.0, random_state=0)
        return cross_val_predict(c, Xm, ym, cv=GroupKFold(5), groups=grp, method="predict_proba")[:, 1]

    # models
    SUF = [i for i in range(X.shape[1]) if i != 4]   # suffix-side feats (exclude eagle_p idx4)
    Ds_pure = oof(X[:, SUF], Ys)             # P(suffix correct | pure suffix feats)
    De_pure = oof(X[:, [4]], Ye)             # P(eagle correct | eagle_p)
    cal_s = oof(X[:, [0]], Ys)               # calibrated suffix_p -> P(suffix correct)  (for "calib" pick-1)
    Ds_j = oof(X, Ys)                        # joint (all feats incl eagle_p, depth)
    De_j = oof(X, Ye)
    # IN-SAMPLE ceiling: fit on all, predict all (optimistic upper bound = best this feature set can do)
    def insample(Xm, ym):
        # SAME model config as OOF (fair ceiling; deeper model would just overfit more)
        c = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_depth=4, random_state=0)
        c.fit(Xm, ym)
        return c.predict_proba(Xm)[:, 1]
    Ds_is = insample(X, Ys); De_is = insample(X, Ye)

    def sim(mode):
        accs = []
        for gid in gids:
            di, rid = gid; dec = {rid: DEC[gid]}; stp = {rid: STP[gid]}
            for s in sorted(stp[rid]):
                d = 0
                while True:
                    row = dec[rid][s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                    sp = float(row.get("suffix_p") or 0.0); ep = float(row.get("eagle_p") or 0.0)
                    fi = pos_index.get((di, rid, s, d))
                    if mode == "oracle":
                        tok = g if (st == g or et == g) else None
                    elif mode == "raw":
                        tok = st if (st is not None and sp >= ep) else et
                    elif mode == "pure":
                        us = (st is not None and fi is not None and Ds_pure[fi] > De_pure[fi])
                        tok = st if us else et
                    elif mode == "calib":
                        us = (st is not None and fi is not None and cal_s[fi] > De_pure[fi])
                        tok = st if us else et
                    elif mode == "single_suffix":
                        tok = st
                    elif mode == "single_eagle":
                        tok = et
                    elif mode == "joint":
                        us = (st is not None and fi is not None and Ds_j[fi] > De_j[fi])
                        tok = st if us else et
                    elif mode == "insample":
                        us = (st is not None and fi is not None and Ds_is[fi] > De_is[fi])
                        tok = st if us else et
                    else:
                        tok = et
                    if tok is not None and tok == g:
                        d += 1
                    else:
                        break
                accs.append(d)
        return float(np.mean(accs))

    mr = sim("raw"); mo = sim("oracle"); mp = sim("pure"); mj = sim("joint"); mis = sim("insample")
    gap = mo - mr
    print(f"\nCHAIN per-depth MAT:")
    print(f"   RAW (suffix_p vs eagle_p): {mr:.4f}")
    print(f"   PURE  (Ds[suffix feats] > De[eagle_p]): {mp:.4f}  ({(mp-mr)/gap*100:+.1f}% of gap)")
    print(f"   JOINT (Ds,De on all feats)            : {mj:.4f}  ({(mj-mr)/gap*100:+.1f}% of gap)")
    print(f"   INSAMPLE ceiling (fit=pred, all feats): {mis:.4f}  ({(mis-mr)/gap*100:+.1f}% of gap)")
    print(f"   ORACLE: {mo:.4f}   (gap {gap:.4f})")

    # ===================== GOAL 2: competition frame in a TREE =====================
    # Per depth we verify a SET of candidate tokens (cost = |set|). PURE competition
    # (Ds_pure vs De_pure, fusion-free) decides the set. Accept if gt in the set.
    def suf_cands(st, tk):
        c = [st] if st is not None else []
        for t in tk:
            if t != st:
                c.append(t)
        return c

    def tree_sim(polfn):
        accs = []; costs = []; nodes_per_step = []
        for gid in gids:
            di, rid = gid; dec = DEC[gid]; stp = STP[gid]
            for s in sorted(stp):
                d = 0; step_nodes = 0
                while True:
                    row = dec[s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                    fi = pos_index.get((di, rid, s, d))
                    tk = topk_index.get((di, rid, s, d), [])
                    ds = Ds_pure[fi] if fi is not None else float(row.get("suffix_p") or 0.0)
                    de = De_pure[fi] if fi is not None else float(row.get("eagle_p") or 0.0)
                    vset = polfn(st, et, ds, de, tk)
                    sz = max(1, len(vset))
                    costs.append(sz); step_nodes += sz
                    if g in vset:
                        d += 1
                    else:
                        break
                accs.append(d); nodes_per_step.append(step_nodes)
        A = np.mean(accs)
        return float(A), float(np.mean(costs)), float(np.mean(nodes_per_step))

    def p_chain(st, et, ds, de, tk):
        w = st if ds > de else et
        return {w} if w is not None else set()

    def p_union(w):
        def f(st, et, ds, de, tk):
            vs = set(suf_cands(st, tk)[:w])
            if et is not None:
                vs.add(et)
            return vs
        return f

    def p_hedge(tau, ws=1):
        def f(st, et, ds, de, tk):
            win_suffix = ds > de
            vs = set()
            sc = suf_cands(st, tk)
            if win_suffix:
                vs.update(sc[:1])
            elif et is not None:
                vs.add(et)
            if abs(ds - de) < tau:   # uncertain -> hedge with the loser (and extra suffix cand)
                if win_suffix and et is not None:
                    vs.add(et)
                else:
                    vs.update(sc[:ws])
            return vs
        return f

    # raw-competition hedge (NO learned model): hedge iff |suffix_p - eagle_p| < tau
    def p_hedge_raw(tau):
        def f(st, et, sp_, ep_, tk, spval, epval):
            win_suffix = spval >= epval
            vs = set([st]) if (win_suffix and st is not None) else (set([et]) if et is not None else set())
            if abs(spval - epval) < tau:
                los = et if win_suffix else st
                if los is not None:
                    vs.add(los)
            return vs
        return f

    def tree_sim_raw(polfn):   # variant passing raw suffix_p/eagle_p
        accs = []; costs = []
        for gid in gids:
            di, rid = gid; dec = DEC[gid]; stp = STP[gid]
            for s in sorted(stp):
                d = 0
                while True:
                    row = dec[s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                    spv = float(row.get("suffix_p") or 0.0); epv = float(row.get("eagle_p") or 0.0)
                    vset = polfn(st, et, None, None, [], spv, epv)
                    costs.append(max(1, len(vset)))
                    if g in vset:
                        d += 1
                    else:
                        break
                accs.append(d)
        return float(np.mean(accs)), float(np.mean(costs))

    # ---------- VERIFY-COST TABLE: draft tree size PER STEP (after pruning) ----------
    def tree_sim2(pol):
        accs = []; nps = []
        for gid in gids:
            di, rid = gid; dec = DEC[gid]; stp = STP[gid]
            for s in sorted(stp):
                d = 0; sn = 0
                while True:
                    row = dec[s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]
                    fi = pos_index.get((di, rid, s, d)); tk = topk_index.get((di, rid, s, d), [])
                    vs = pol(row, fi, tk)
                    sn += max(1, len(vs))
                    if g in vs:
                        d += 1
                    else:
                        break
                accs.append(d); nps.append(sn)
        return float(np.mean(accs)), float(np.mean(nps))

    def _ste(row):
        return (row.get("suffix_token"), row.get("eagle_token"),
                float(row.get("suffix_p") or 0.0), float(row.get("eagle_p") or 0.0))
    def po_single_suf(row, fi, tk):
        st = row.get("suffix_token"); return {st} if st is not None else set()
    def po_single_eag(row, fi, tk):
        et = row.get("eagle_token"); return {et} if et is not None else set()
    def po_raw_pick(row, fi, tk):
        st, et, sp, ep = _ste(row); w = st if (st is not None and sp >= ep) else et
        return {w} if w is not None else set()
    def po_calib_pick(row, fi, tk):
        st, et, sp, ep = _ste(row)
        w = (st if (st is not None and sp >= ep) else et) if fi is None else (st if cal_s[fi] > De_pure[fi] else et)
        return {w} if w is not None else set()
    def mg_raw(row, fi, sp, ep):
        return (sp >= ep, abs(sp - ep))
    def mg_comp(row, fi, sp, ep):
        return (sp >= ep, abs(sp - ep)) if fi is None else (Ds_pure[fi] > De_pure[fi], abs(Ds_pure[fi] - De_pure[fi]))
    def po_hedge(margin_fn, tau):
        def f(row, fi, tk):
            st, et, sp, ep = _ste(row)
            win_suf, m = margin_fn(row, fi, sp, ep)
            vs = ({st} if (win_suf and st is not None) else ({et} if et is not None else set()))
            if m < tau:
                los = et if win_suf else st
                if los is not None:
                    vs.add(los)
            return vs
        return f
    def po_union(w):
        def f(row, fi, tk):
            st = row.get("suffix_token"); et = row.get("eagle_token")
            vs = set(suf_cands(st, tk)[:w])
            if et is not None:
                vs.add(et)
            return vs
        return f

    METHODS = [
        ("single-suffix",                  po_single_suf),
        ("single-eagle",                   po_single_eag),
        ("RAW pick-1",                     po_raw_pick),
        ("CALIB pick-1",                   po_calib_pick),
        ("raw-hedge tau0.2",               po_hedge(mg_raw, 0.2)),
        ("raw-hedge tau0.5",               po_hedge(mg_raw, 0.5)),
        ("OURS comp-hedge tau0.2",         po_hedge(mg_comp, 0.2)),
        ("OURS comp-hedge tau0.5",         po_hedge(mg_comp, 0.5)),
        ("union top1 (=realizable ORACLE)", po_union(1)),
        ("union top4 (wide ORACLE)",        po_union(4)),
    ]
    res = {name: tree_sim2(pol) for name, pol in METHODS}
    mraw = res["RAW pick-1"][0]; morc = res["union top1 (=realizable ORACLE)"][0]
    gap = morc - mraw
    print(f"\n########## VERIFY-COST TABLE (tree regime, {len(gids)} req) [Model-B accept: LOWER-BOUND nodes] ##########")
    print(f"{'method':<34}{'MAT':>8}{'nodes/step':>12}{'nodes/tok':>11}{'%gap':>8}")
    for name, _ in METHODS:
        mat, nps = res[name]
        print(f"{name:<34}{mat:>8.3f}{nps:>12.2f}{nps/mat:>11.2f}{(mat-mraw)/gap*100:>7.0f}%")

    # Model-A: CONSISTENT single-pass backbone+leaf tree. Backbone = winner chain (continue only
    # while winner==gt); loser is a LEAF (if it matches gt at the backbone-miss -> +1 then STOP).
    # nodes = drafted winner-backbone-depth + hedge leaves (additive, single pass).
    def modelA(margin_fn, tau):
        accs = []; nps = []
        for gid in gids:
            di, rid = gid; dec = DEC[gid]; stp = STP[gid]
            for s in sorted(stp):
                d = 0; sn = 0
                while True:
                    row = dec[s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                    sp = float(row.get("suffix_p") or 0.0); ep = float(row.get("eagle_p") or 0.0)
                    fi = pos_index.get((di, rid, s, d))
                    win_suf, m = margin_fn(row, fi, sp, ep)
                    wtok = st if win_suf else et; ltok = et if win_suf else st
                    sn += 1                       # winner backbone node (drafted)
                    hedged = m < tau
                    if hedged:
                        sn += 1                   # loser leaf
                    if wtok is not None and wtok == g:
                        d += 1                    # backbone stays on gt, continue
                    else:
                        if hedged and ltok is not None and ltok == g:
                            d += 1                # accept loser leaf, then STOP (leaf has no child)
                        break
                accs.append(d); nps.append(sn)
        return float(np.mean(accs)), float(np.mean(nps))

    print(f"\n########## Model-A (CONSISTENT single-pass backbone+leaf) ##########")
    print(f"{'method':<34}{'MAT':>8}{'nodes/step':>12}{'nodes/tok':>11}{'%gap':>8}")
    for nm, mfn, tau in [("raw-hedge A tau0.5", mg_raw, 0.5), ("OURS comp-hedge A tau0.5", mg_comp, 0.5),
                         ("raw-hedge A tau1.0(union)", mg_raw, 9.9)]:
        mat, nps = modelA(mfn, tau)
        print(f"{nm:<34}{mat:>8.3f}{nps:>12.2f}{nps/mat:>11.2f}{(mat-mraw)/gap*100:>7.0f}%")

    # ---- microscopic 0.5 decomposition ----
    print(f"\n=== suffix_p ~ 0.5 decomposition (LOG) ===")
    b = defaultdict(lambda: [0, 0])   # (count,total) -> [suffix_correct, n]
    mlb = defaultdict(lambda: [0, 0])
    n05 = 0
    for rid in order:
        for s in dec[rid]:
            for d, row in dec[rid][s].items():
                sp = row.get("suffix_p"); st = row.get("suffix_token"); g = row.get("gt_token")
                if sp is None or st is None or g is None or abs(sp - 0.5) > 1e-6:
                    continue
                n05 += 1
                c = row.get("suffix_count"); nn = row.get("suffix_total"); ml = row.get("match_len")
                key = (c, nn)
                b[key][0] += int(st == g); b[key][1] += 1
                mlb[ml][0] += int(st == g); mlb[ml][1] += 1
    print(f"total suffix_p=0.5 positions: {n05}")
    print("by (count,total), top:")
    for key, (h, nn) in sorted(b.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"   count/total={key}: n={nn} ({nn/n05*100:.1f}%)  suffix_acc={h/nn*100:.1f}%")
    print("by match_len:")
    for ml, (h, nn) in sorted(mlb.items(), key=lambda kv: (kv[0] is None, kv[0]))[:10]:
        print(f"   ml={ml}: n={nn}  suffix_acc={h/max(1,nn)*100:.1f}%")


if __name__ == "__main__":
    main()
