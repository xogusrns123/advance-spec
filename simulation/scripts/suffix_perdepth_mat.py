"""Does the new pure-suffix-side confidence lift per-depth select-1 MAT vs raw suffix_p?

Main arm (eagle_p/eagle_token = MTP on 27B) comes from the oracle decisions log.
Suffix score is either raw suffix_p (log) or a NEW confidence: a GBM on the new
features (back-off + local/global agreement + token-class), trained on suffix
accept (GroupKFold by request, out-of-fold). Per-depth selection: at each depth
pick suffix iff sscore>=eagle_p, accept while picked token==gt (valid from the
log because the accepted prefix is gt-conditioned). MAT = mean accepted length.
Isolation: ONLY the suffix score changes between raw and new (eagle_p stays raw).
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

    def longest(self):
        for L in range(min(MAXL, len(self.seq)), 0, -1):
            g = tuple(self.seq[-L:])
            if g in self.count[L]:
                return L, g
        return 0, None

    def child(self, L):
        if L < 1 or L > MAXL or len(self.seq) < L:
            return None
        return self.count[L].get(tuple(self.seq[-L:]))


def tok_classer(tokenizer):
    cache = {}
    def cls(tid):
        if tid in cache:
            return cache[tid]
        s = tokenizer.decode([int(tid)]); st = s.strip()
        c = (float(s[:1].isspace() or s == ""), float("\n" in s),
             float(len(st) > 0 and all(not ch.isalnum() for ch in st)),
             float(st.isdigit()), float(st.isalpha()), float(len(s)))
        cache[tid] = c
        return c
    return cls


def replay_features(prompt, out, cls, gtrie):
    """Dense: per abs position of `out`, return (feat vector, my_suffix_p, my_head, accept)."""
    lt = Trie()
    for t in prompt:
        lt.commit(t)
    feats, sp_list, head_list, acc_list = [], [], [], []
    for pos in range(len(out)):
        g = out[pos]
        L, key = lt.longest()
        if L >= 1 and key is not None:
            ch = lt.count[L][key]; n = sum(ch.values())
            mc = ch.most_common(2); head = mc[0][0]; c0 = mc[0][1]
            p_head = c0 / n
            def hf(Lx):
                c = lt.child(Lx)
                if not c:
                    return 0.0, 0.0
                nn = sum(c.values())
                return c.get(head, 0) / nn, float(c.most_common(1)[0][0] == head)
            hf1, am1 = hf(1); hf2, am2 = hf(2); hfm, amm = hf(max(1, L - 1))
            votes = sum(hf(Lx)[1] for Lx in range(1, L + 1)) / L
            gL, gkey = gtrie.longest()
            g_agree = 0.0; g_hf = 0.0
            if gL >= 1 and gkey is not None:
                gc = gtrie.count[gL][gkey]; gh = gc.most_common(1)[0][0]
                g_hf = gc[gh] / sum(gc.values()); g_agree = float(gh == head)
            hc = cls(head)
            feats.append([p_head, math.log1p(n), math.log1p(c0), float(L),
                          hf1, hf2, hfm, am1, am2, amm, votes, entropy(lt.child(2) or Counter()),
                          float(gL >= 1), g_agree, float(gL), g_hf,
                          hc[0], hc[1], hc[2], hc[3], hc[4], hc[5]])
            sp_list.append(p_head); head_list.append(head); acc_list.append(int(g == head))
        else:
            feats.append([0.0] * 22); sp_list.append(0.0); head_list.append(-1); acc_list.append(0)
        lt.commit(g); gtrie.commit(g)
    return feats, sp_list, head_list, acc_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cls = tok_classer(tokenizer)

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

    # per rid: reconstruct committed gt from the log (no gt_tokens / no prompt needed);
    # suffix arm + features come from a self-consistent local+global trie over committed
    # responses. eagle arm comes from the log at the same committed positions.
    gtrie = Trie()   # global across requests (committed responses)
    allX, allY, allG, allHead, allSP = [], [], [], [], []
    perpos = {}      # rid -> {committed_index: feat_index}
    rid_out = {}     # rid -> committed gt list
    rid_cum = {}     # rid -> {step: cum position}
    for rid in order:
        steps = sorted(stp[rid])
        out = []
        cum = {}
        for s in steps:
            cum[s] = len(out)
            a = stp[rid][s]; row0 = dec[rid][s]
            for d in range(0, a + 1):
                if d in row0 and row0[d].get("gt_token") is not None:
                    out.append(row0[d]["gt_token"])
        if len(out) < 8:
            continue
        feats, sp, head, acc = replay_features([], out, cls, gtrie)
        base = len(allX)
        perpos[rid] = {j: base + j for j in range(len(out))}
        allX.extend(feats); allY.extend(acc); allG.extend([rid] * len(out))
        allHead.extend(head); allSP.extend(sp)
        rid_out[rid] = out; rid_cum[rid] = cum

    X = np.asarray(allX, np.float32); Y = np.asarray(allY, np.int8)
    groups = np.array(allG); allHead = np.array(allHead); allSP = np.array(allSP)
    print(f"requests={len(rid_out)}/{len(order)}; positions={len(Y)}; my-suffix accept={Y.mean()*100:.1f}%")

    # train new suffix confidence, OOF (GroupKFold by request)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, random_state=0)
    conf = cross_val_predict(clf, X, Y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]

    # info: proxy (no-prompt) suffix vs served log suffix — head agreement at committed positions
    agree = tot = 0
    for rid in rid_out:
        for s in sorted(dec[rid]):
            cum = rid_cum[rid][s]; a = stp[rid][s]
            for d in range(0, a + 1):
                row = dec[rid][s].get(d); fidx = perpos[rid].get(cum + d)
                if row is not None and fidx is not None and row.get("suffix_token") is not None:
                    tot += 1; agree += int(int(allHead[fidx]) == row["suffix_token"])
    if tot:
        print(f"proxy check: my-head == served suffix_token at {agree/tot*100:.1f}% of committed positions (n={tot})")

    # per-depth selection sim
    def sarm(rid, cum, d):
        """my (no-prompt) suffix arm at committed position cum+d: (token, suffix_p, conf)."""
        fidx = perpos[rid].get(cum + d)
        if fidx is None:
            return None, 0.0, 0.0
        return int(allHead[fidx]), float(allSP[fidx]), float(conf[fidx])

    def simulate(score_mode):
        accs = []
        for rid in rid_out:
            for s in sorted(dec[rid]):
                cum = rid_cum[rid][s]
                d = 0
                while True:
                    row = dec[rid][s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    ep = row.get("eagle_p") or 0.0; g = row["gt_token"]; etok = row.get("eagle_token")
                    stok, sp_my, cf = sarm(rid, cum, d)
                    sscore = sp_my if score_mode == "raw" else (cf if score_mode == "new" else -1.0)
                    use_suffix = (stok is not None) and (sscore >= ep)
                    tok = stok if use_suffix else etok
                    if tok is not None and tok == g:
                        d += 1
                    else:
                        break
                accs.append(d)
        return float(np.mean(accs)), len(accs)

    def sim_special(mode):
        accs = []
        for rid in rid_out:
            for s in sorted(dec[rid]):
                cum = rid_cum[rid][s]
                d = 0
                while True:
                    row = dec[rid][s].get(d)
                    if row is None or row.get("gt_token") is None:
                        break
                    g = row["gt_token"]; etok = row.get("eagle_token")
                    stok, _, _ = sarm(rid, cum, d)
                    if mode == "oracle":
                        ok = (stok == g) or (etok == g)
                    elif mode == "eagle":
                        ok = (etok == g)
                    else:
                        ok = (stok == g)
                    if ok:
                        d += 1
                    else:
                        break
                accs.append(d)
        return float(np.mean(accs))

    mat_raw, n = simulate("raw")
    mat_new, _ = simulate("new")
    mat_eagle = sim_special("eagle")
    mat_suffix = sim_special("suffix")
    mat_oracle = sim_special("oracle")
    print(f"\nper-depth select-1 MAT (accepted draft tokens/step, n_steps={n}):")
    print(f"   eagle(MTP)-only        : {mat_eagle:.4f}")
    print(f"   suffix-only            : {mat_suffix:.4f}")
    print(f"   RAW  (score=suffix_p)  : {mat_raw:.4f}")
    print(f"   NEW  (score=new conf)  : {mat_new:.4f}   (delta vs raw {mat_new-mat_raw:+.4f})")
    print(f"   ORACLE (best per depth): {mat_oracle:.4f}")
    denom = (mat_oracle - mat_raw)
    if denom > 1e-6:
        print(f"   NEW recovers {(mat_new-mat_raw)/denom*100:+.1f}% of raw->oracle gap")


if __name__ == "__main__":
    main()
