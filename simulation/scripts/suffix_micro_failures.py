"""Microscopic decoded-text look at the DECISION-SENSITIVE failure cases of
per-depth suffix-vs-eagle selection (14B), to find patterns statistics miss.
  A) confident suffix INVERSION: only_eagle-correct, but suffix_p high (we wrongly trust suffix)
  B) missed suffix WIN: only_suffix-correct, but suffix_p low (we wrongly drop suffix)
Prints decoded context + suffix/eagle/gt tokens for each, for manual pattern-finding.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
sys.path.insert(0, "/workspace")
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=22); ap.add_argument("--win", type=int, default=16)
    args = ap.parse_args()
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
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

    invert, missed = [], []
    for rid in order:
        out = recon(rid)
        full = match(out)
        if full is None:
            continue
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
                sp = float(row.get("suffix_p") or 0.0); ml = int(row.get("match_len") or 0)
                j = 1 + cum + d
                ctx = tk.decode(full[max(0, j - args.win):j])
                rec = (ml, round(sp, 2), repr(ctx[-52:]), repr(tk.decode([st])) if st is not None else "None",
                       repr(tk.decode([et])) if et is not None else "None", repr(tk.decode([g])))
                if ec and not sc and sp >= 0.6:
                    invert.append(rec)
                elif sc and not ec and sp <= 0.34:
                    missed.append(rec)
                d += 1
            cum += a + 1

    def show(title, items):
        print(f"\n{'='*100}\n{title} (showing {min(len(items),args.n)} of {len(items)})\n{'='*100}")
        print(f"{'ml':>2} {'sp':>4}  {'context':<52} {'SUF':>9} {'EAG':>9} {'GT':>9}")
        for ml, sp, c, su, eg, g in items[:args.n]:
            print(f"{ml:>2} {sp:>4}  {c[-52:]:<52} {su:>9} {eg:>9} {g:>9}")

    show("A) CONFIDENT SUFFIX INVERSION (suffix_p>=0.6, suffix WRONG, eagle right)", invert)
    show("B) MISSED SUFFIX WIN (suffix_p<=0.34, suffix RIGHT, eagle wrong)", missed)


if __name__ == "__main__":
    main()
