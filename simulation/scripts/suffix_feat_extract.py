"""Extract a feature matrix for predicting suffix-decoding accept, pure suffix-side
(no model/fusion). Self-contained suffix model: per-request LOCAL trie (prompt+
response) gives the proposal (longest-match count-argmax head); label = head==gt.
Also a persistent GLOBAL trie (responses across requests) for cross-tree features.
Dumps X, y, feature names, and a p~0.5 mask to an npz for offline AUC/combination.
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, "/workspace")
from transformers import AutoTokenizer

MAXL = 12


def entropy(cnt):
    n = sum(cnt.values())
    if n <= 1:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in cnt.values())


class Trie:
    def __init__(self):
        self.count = [defaultdict(Counter) for _ in range(MAXL + 1)]
        self.recent = [dict() for _ in range(MAXL + 1)]   # gram -> most-recent follower
        self.lastpos = [dict() for _ in range(MAXL + 1)]  # gram -> most-recent occ index
        self.seq = []

    def commit(self, t, gpos):
        for L in range(1, min(MAXL, len(self.seq)) + 1):
            g = tuple(self.seq[-L:])
            self.count[L][g][t] += 1
            self.recent[L][g] = t
            self.lastpos[L][g] = gpos
        self.seq.append(t)

    def longest(self):
        for L in range(min(MAXL, len(self.seq)), 0, -1):
            g = tuple(self.seq[-L:])
            if g in self.count[L]:
                return L, g
        return 0, None

    def children_at(self, L):
        if L < 1 or L > MAXL or len(self.seq) < L:
            return None, None
        g = tuple(self.seq[-L:])
        return g, self.count[L].get(g)


def tok_class(tokenizer):
    cache = {}
    def cls(tid):
        if tid in cache:
            return cache[tid]
        s = tokenizer.decode([int(tid)])
        st = s.strip()
        c = dict(space=float(s[:1].isspace() or s == "" or (len(st) == 0)),
                 newline=float("\n" in s),
                 punct=float(len(st) > 0 and all(not ch.isalnum() for ch in st)),
                 digit=float(st.isdigit()),
                 alpha=float(st.isalpha()),
                 slen=float(len(s)))
        cache[tid] = c
        return c
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cls = tok_class(tokenizer)
    rows = [json.loads(l) for l in open(args.gt) if l.strip()]

    NAMES = ["headfrac", "logn", "logc", "matchlen", "nchildren", "entropy", "margin",
             "secondfrac", "hf_L1", "hf_L2", "hf_Lm1", "argmax_L1", "argmax_L2",
             "argmax_Lm1", "vote", "entropy_L2", "recency_agree", "dist_recent",
             "g_has", "g_agree", "g_matchlen", "g_headfrac", "g_logn",
             "logpos", "prev_accept", "streak", "recent_acc", "ml_extend",
             "head_space", "head_newline", "head_punct", "head_digit", "head_alpha",
             "head_len", "prev_space", "prev_punct",
             "agr_depth", "kn_max_hf", "head_uni", "branch_L1", "ml_gap"]
    X, Y, G = [], [], []
    gtrie = Trie()   # global: responses only, persists
    uni = Counter()  # global unigram frequency of committed response tokens

    for ridx, r in enumerate(rows):
        prompt = list(r["input_ids"]); gt = list(r["output_ids"])
        if not gt:
            continue
        lt = Trie()
        gpos = 0
        for t in prompt:
            lt.commit(t, gpos); gpos += 1
        acc_hist = []
        prev_L = 0
        streak = 0
        for pos in range(len(gt)):
            g = gt[pos]
            L, key = lt.longest()
            feat = None
            if L >= 1 and key is not None:
                ch = lt.count[L][key]; n = sum(ch.values())
                mc = ch.most_common(2)
                head = mc[0][0]; c_head = mc[0][1]
                p_head = c_head / n
                p_2nd = (mc[1][1] / n) if len(mc) > 1 else 0.0
                rec_head = lt.recent[L][key]
                lastp = lt.lastpos[L][key]
                # multi-length
                def hf_at(Lx):
                    _, c = lt.children_at(Lx)
                    if not c:
                        return 0.0, 0, 0.0
                    nn = sum(c.values())
                    am = c.most_common(1)[0][0]
                    return c.get(head, 0) / nn, int(am == head), entropy(c)
                hf1, am1, _ = hf_at(1)
                hf2, am2, e2 = hf_at(2)
                hfm1, amm1, _ = hf_at(max(1, L - 1))
                votes = sum(hf_at(Lx)[1] for Lx in range(1, L + 1)) / L
                # global trie
                gL, gkey = gtrie.longest()
                g_has = float(gL >= 1)
                g_head = None; g_hf = 0.0; g_n = 0
                if gL >= 1 and gkey is not None:
                    gc = gtrie.count[gL][gkey]; g_n = sum(gc.values())
                    g_head = gc.most_common(1)[0][0]; g_hf = gc[g_head] / g_n
                g_agree = float(g_head == head) if g_head is not None else 0.0
                hc = cls(head); pc = cls(key[-1])
                acc = int(g == head)
                ml_ext = float(L == prev_L + 1)
                recent_acc = (sum(acc_hist[-16:]) / len(acc_hist[-16:])) if acc_hist else 0.0
                # NEW batch-2 features
                agr_depth = 0
                for L2 in range(1, min(L, MAXL) + 1):
                    _, lc2 = lt.children_at(L2); _, gc2 = gtrie.children_at(L2)
                    if lc2 and gc2 and lc2.most_common(1)[0][0] == gc2.most_common(1)[0][0]:
                        agr_depth = L2
                kn_max_hf = max(hf_at(Lx)[0] for Lx in range(1, L + 1))
                head_uni = uni.get(head, 0) / max(1, len(gtrie.seq))
                _, c1 = lt.children_at(1); branch_L1 = float(len(c1)) if c1 else 0.0
                ml_gap = float(gL - L)
                feat = [p_head, math.log1p(n), math.log1p(c_head), float(L),
                        float(len(ch)), entropy(ch), p_head - p_2nd, p_2nd,
                        hf1, hf2, hfm1, float(am1), float(am2), float(amm1),
                        votes, e2, float(rec_head == head), math.log1p(max(0, pos - (lastp if lastp is not None else pos))),
                        g_has, g_agree, float(gL), g_hf, math.log1p(g_n),
                        math.log1p(pos), float(acc_hist[-1] if acc_hist else 0), float(streak),
                        recent_acc, ml_ext,
                        hc["space"], hc["newline"], hc["punct"], hc["digit"], hc["alpha"], hc["slen"],
                        pc["space"], pc["punct"],
                        float(agr_depth), kn_max_hf, head_uni, branch_L1, ml_gap]
                X.append(feat); Y.append(acc); G.append(ridx)
                acc_hist.append(acc)
                streak = streak + 1 if acc else 0
                prev_L = L
            # commit gt to both tries
            lt.commit(g, gpos)
            gtrie.commit(g, len(gtrie.seq))
            uni[g] += 1
            gpos += 1

    X = np.asarray(X, dtype=np.float32); Y = np.asarray(Y, dtype=np.int8)
    G = np.asarray(G, dtype=np.int32)
    np.savez_compressed(args.out, X=X, y=Y, names=np.array(NAMES), groups=G)
    print(f"positions={len(Y)}  features={X.shape[1]}  accept-rate={Y.mean()*100:.1f}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
