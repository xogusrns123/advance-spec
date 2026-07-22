"""GOAL 1 (max data): close >=30% of raw->oracle gap in CHAIN select-1. To beat
the DATA-STARVATION (in-sample ceiling ~38% but OOF ~13% on ~66 matched reqs),
drop the gt_tokens prompt-matching entirely: build the suffix tries from the
COMMITTED RESPONSE (from the log) only -> EVERY request in EVERY dir is usable
(no matching loss). Suffix ARM + count/match_len/eagle_p stay the REAL served
values from the log; trie features (back-off, local/global agreement, prose,
token-class) are computed response-only. Joint competition Ds>De, LIVE positions,
content-deduped GroupKFold. Reports OOF + in-sample gap closure.
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


def tok_classer(tk):
    cache = {}
    def cls(t):
        if t is None:
            return (0., 0., 0., 0., 0., 0.)
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
    ap.add_argument("--dirs", required=True)
    args = ap.parse_args()
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cls = tok_classer(tk)

    feats, Ys, Ye, grp = [], [], [], []
    pos_index = {}; DEC = {}; STP = {}; gids = []; seen = set()
    for di, d in enumerate([x.strip() for x in args.dirs.split(",") if x.strip()]):
        dec = defaultdict(lambda: defaultdict(dict)); stp = defaultdict(dict); order = []
        for line in open(f"{d}/decisions_select1_oracle.jsonl"):
            r = json.loads(line); t = r.get("type")
            if t == "req": order.append(r["rid"])
            elif t == "decision": dec[r["rid"]][r["decode_step"]][r["depth"]] = r
            elif t == "step": stp[r["rid"]][r["decode_step"]] = r["accept_len"]
        G = Trie(); nuse = 0
        for rid in order:
            out = []
            for s in sorted(stp[rid]):
                a = stp[rid][s]; row = dec[rid][s]
                for dd in range(0, a + 1):
                    if dd in row and row[dd].get("gt_token") is not None:
                        out.append(row[dd]["gt_token"])
            if len(out) < 8:
                for t in out: G.commit(t)
                continue
            ck = tuple(out[:16])
            if ck in seen:
                for t in out: G.commit(t)
                continue
            seen.add(ck)
            gid = (di, rid); gids.append(gid); DEC[gid] = dec[rid]; STP[gid] = stp[rid]
            pos_index[gid] = defaultdict(dict); nuse += 1
            Lt = Trie(); jinfo = {}; cum = 0
            for s in sorted(stp[rid]):
                a = stp[rid][s]
                for dd in range(0, a + 1):
                    if dec[rid][s].get(dd) is not None:
                        jinfo[cum + dd] = (s, dd, dec[rid][s][dd])
                cum += a + 1
            for j in range(len(out)):
                g = out[j]; info = jinfo.get(j)
                if info is not None:
                    s, dd, row = info; st = row.get("suffix_token")
                    if st is not None:
                        sp = float(row.get("suffix_p") or 0.0)
                        cnt = float(row.get("suffix_count") or 0.0)
                        tot = float(row.get("suffix_total") or 0.0)
                        ml = int(row.get("match_len") or 0); ep = float(row.get("eagle_p") or 0.0)
                        def bo(L2):
                            c = Lt.child(L2)
                            if not c: return 0., 0.
                            nn = sum(c.values()); return c.get(st, 0) / nn, float(c.most_common(1)[0][0] == st)
                        bo1, am1 = bo(1); bo2, am2 = bo(2); bo_ml, _ = bo(min(max(ml, 1), MAXL))
                        gam = float(G.longest_argmax() == st)
                        gc = G.child(1); g_has = float(bool(gc) and st in gc)
                        g2 = G.child(2); g_branch = float(len(g2)) if g2 else 0.; g_ent = entropy(g2) if g2 else 0.
                        recent = Lt.seq[-16:]
                        prose = (sum(cls(t)[4] for t in recent) / len(recent)) if recent else 0.
                        cc = cls(st)
                        feats.append([sp, math.log1p(cnt), math.log1p(tot), float(ml), ep, float(dd),
                                      bo1, am1, bo2, am2, bo_ml, gam, g_has, g_branch, g_ent, prose,
                                      cc[0], cc[1], cc[2], cc[3], cc[4], cc[5]])
                        Ys.append(int(st == g)); Ye.append(int(row.get("eagle_token") == g))
                        grp.append(str(ck)); pos_index[gid][s][dd] = len(feats) - 1
                Lt.commit(g); G.commit(g)
        print(f"[{d.split('/')[-1]}] used {nuse}/{len(order)} unique")

    X = np.asarray(feats, np.float32); Ys = np.asarray(Ys, np.int8); Ye = np.asarray(Ye, np.int8)
    grp = np.array(grp)
    print(f"POOLED(no-prompt): unique_req={len(gids)} live-pos={len(Ys)} suffix_acc={Ys.mean()*100:.1f}%")

    def oof(ym):
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.07, max_depth=4, random_state=0)
        return cross_val_predict(c, X, ym, cv=GroupKFold(5), groups=grp, method="predict_proba")[:, 1]
    def ins(ym):
        c = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_depth=6, random_state=0)
        c.fit(X, ym); return c.predict_proba(X)[:, 1]
    Ds = oof(Ys); De = oof(Ye); Ds_is = ins(Ys); De_is = ins(Ye)

    def sim(mode):
        accs = []
        for gid in gids:
            dec = DEC[gid]; stp = STP[gid]
            for s in sorted(stp):
                dd = 0
                while True:
                    row = dec[s].get(dd)
                    if row is None or row.get("gt_token") is None: break
                    g = row["gt_token"]; st = row.get("suffix_token"); et = row.get("eagle_token")
                    sp = float(row.get("suffix_p") or 0.0); ep = float(row.get("eagle_p") or 0.0)
                    fi = pos_index[gid][s].get(dd)
                    if mode == "oracle":
                        tok = g if (st == g or et == g) else None
                    elif mode == "raw":
                        tok = st if (st is not None and sp >= ep) else et
                    else:
                        DD, DDe = (Ds, De) if mode == "joint" else (Ds_is, De_is)
                        us = (st is not None and fi is not None and DD[fi] > DDe[fi])
                        tok = st if (us or (fi is None and st is not None and sp >= ep)) else et
                    if tok is not None and tok == g: dd += 1
                    else: break
                accs.append(dd)
        return float(np.mean(accs))

    mr = sim("raw"); mo = sim("oracle"); mj = sim("joint"); mis = sim("insample")
    gap = mo - mr
    print(f"\nCHAIN per-depth MAT (no-prompt, {len(gids)} unique req):")
    print(f"   RAW: {mr:.4f}   ORACLE: {mo:.4f}   gap {gap:.4f}")
    print(f"   JOINT (OOF)      : {mj:.4f}  ({(mj-mr)/gap*100:+.1f}% of gap)")
    print(f"   INSAMPLE ceiling : {mis:.4f}  ({(mis-mr)/gap*100:+.1f}% of gap)")


if __name__ == "__main__":
    main()
