"""Would a suffix TREE (propose top-k children, not just top-1) rescue the MAT that
chain per-depth selection loses? Key measurement (14B, real prompt matched):
  - top-k suffix accept (gt in suffix's top-k children of the matched node)
  - TREE RESCUE: at positions where chain arms BOTH miss (suffix top-1 != gt AND
    eagle != gt) -> the chain-selection run ENDS -> is gt in suffix top-2/3/4/8?
  - tree-augmented oracle MAT (accept while gt in suffix-topk OR eagle==gt)
If rescue is substantial, a tree raises the ceiling and the reliability features
(where to widen) become meaningful.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, "/workspace")
MAXL = 12


class Trie:
    def __init__(self):
        self.count = [defaultdict(Counter) for _ in range(MAXL + 1)]
        self.seq = []
    def commit(self, t):
        for L in range(1, min(MAXL, len(self.seq)) + 1):
            self.count[L][tuple(self.seq[-L:])][t] += 1
        self.seq.append(t)
    def topk(self, k):
        for L in range(min(MAXL, len(self.seq)), 0, -1):
            c = self.count[L].get(tuple(self.seq[-L:]))
            if c:
                return [t for t, _ in c.most_common(k)], L
        return [], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    dec = defaultdict(lambda: defaultdict(dict)); stp = defaultdict(dict); order = []
    for line in open(args.log):
        r = json.loads(line); t = r.get("type")
        if t == "req": order.append(r["rid"])
        elif t == "decision": dec[r["rid"]][r["decode_step"]][r["depth"]] = r
        elif t == "step": stp[r["rid"]][r["decode_step"]] = r["accept_len"]
    gtrows = [json.loads(l) for l in open(args.gt)]

    def recon(rid):
        out = []
        for s in sorted(stp[rid]):
            a = stp[rid][s]; row = dec[rid][s]
            for d in range(0, a + 1):
                if d in row and row[d].get("gt_token") is not None:
                    out.append(row[d]["gt_token"])
        return out

    def match(out):
        for gr in gtrows:
            o = gr["output_ids"]
            if len(out) >= 8 and o[1:1 + len(out)] == out:
                return list(o)
        return None

    KS = [1, 2, 3, 4, 8]
    acc_at = Counter(); tot = 0
    neither = 0; rescue = Counter()
    G = Trie()
    # chain-oracle vs tree-oracle MAT (per-depth)
    chain_or = []; tree_or = {k: [] for k in KS}
    for rid in order:
        out = recon(rid); full = match(out)
        if full is None:
            for t in out: G.commit(t)
            continue
        Lt = Trie()
        # need prompt; match() gave full output; get prompt from same gt row
        prompt = None
        for gr in gtrows:
            if gr["output_ids"][1:1 + len(out)] == out:
                prompt = list(gr["input_ids"]); break
        for t in prompt:
            Lt.commit(t)
        cum = 0
        # position -> suffix topk (compute densely)
        topk_at = {}
        # replay to record topk per output index, then commit
        for j in range(len(full)):
            tks, _ = Lt.topk(8)
            topk_at[j] = tks
            Lt.commit(full[j]); G.commit(full[j])
        # tree candidates = SERVED suffix_token (top-1) + my-trie siblings (top-2+)
        def cands(row, j, k):
            st = row.get("suffix_token")
            base = [st] if st is not None else []
            for c in topk_at.get(j, []):
                if c != st and len(base) < k:
                    base.append(c)
            return base[:k]
        # per (step,depth): accept metrics + oracle runs
        for s in sorted(stp[rid]):
            a = stp[rid][s]
            # chain-oracle run (served suffix top1 | eagle)
            d = 0
            while True:
                row = dec[rid][s].get(d)
                if row is None or row.get("gt_token") is None: break
                g = row["gt_token"]; et = row.get("eagle_token"); st = row.get("suffix_token")
                if not (st == g or et == g): break
                d += 1
            chain_or.append(d)
            for k in KS:
                d = 0
                while True:
                    row = dec[rid][s].get(d)
                    if row is None or row.get("gt_token") is None: break
                    g = row["gt_token"]; et = row.get("eagle_token")
                    if not ((g in cands(row, 1 + cum + d, k)) or et == g): break
                    d += 1
                tree_or[k].append(d)
            for d in range(0, a + 1):
                row = dec[rid][s].get(d)
                if row is None or row.get("gt_token") is None: continue
                g = row["gt_token"]; et = row.get("eagle_token"); st = row.get("suffix_token")
                tot += 1
                for k in KS:
                    acc_at[k] += int(g in cands(row, 1 + cum + d, k))
                if st != g and et != g:  # chain arms both miss (run ends)
                    neither += 1
                    for k in KS:
                        if k > 1 and g in cands(row, 1 + cum + d, k):
                            rescue[k] += 1
            cum += a + 1

    print(f"positions={tot}")
    print("suffix accept @ top-k (gt in top-k children):")
    for k in KS:
        print(f"   k={k}: {acc_at[k]/tot*100:.1f}%")
    print(f"\nchain-arms-both-miss positions (run ends): {neither} ({neither/tot*100:.1f}%)")
    print("  of those, gt in suffix top-k (TREE RESCUE):")
    for k in KS:
        if k > 1:
            print(f"   k={k}: {rescue[k]/max(1,neither)*100:.1f}%")
    print(f"\nper-depth ORACLE MAT: chain(top1|eagle)={np.mean(chain_or):.4f}")
    for k in KS:
        if k > 1:
            print(f"   tree-oracle(top{k}|eagle)={np.mean(tree_or[k]):.4f}  (+{np.mean(tree_or[k])-np.mean(chain_or):.4f})")


if __name__ == "__main__":
    main()
