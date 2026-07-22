"""Rethink: with ONLY 'agreement' (token-identity match) admissible as an extra
feature, can we beat the trust-dominant ceiling on 3-way decisive positions?

Decompose decisive positions by agreement pattern and measure WHICH token is gt.
Then evaluate rule-based selectors that use ONLY {raw probs, agreement}:
  raw            argmax prob
  dominant       always the globally most-reliable proposer (= calibration limit)
  consensus>dom  if 2 proposers agree -> that token, else dominant
  consensus>raw  if 2 agree -> that token, else argmax prob
  dom-unless-WC  dominant, BUT if the two NON-dominant agree against it -> their token
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = {
    "8B 3-way": {"dir": "qwen3_8b_dflash_e3_ceiling20",
        "props": [("dflash", "eagle_token", "eagle_p"), ("eagle3", "e3_token", "e3_p"),
                  ("suffix", "suffix_token", "suffix_p")]},
    "27B 3-way": {"dir": "qwen35_27b_3way_real_full",
        "props": [("mtp", "eagle_token", "eagle_p"), ("dflash", "dflash_token", "dflash_p"),
                  ("suffix", "suffix_token", "suffix_p")]},
}


def load(path):
    ch = defaultdict(list)
    for line in open(path):
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            ch[(o["rid"], o["decode_step"])].append(o)
    for k in ch:
        ch[k].sort(key=lambda r: r["depth"])
    return ch


def loopy(d):
    dd = Path(d); gtf = dd / "gt_tokens.jsonl"
    if not gtf.exists():
        return set()
    reqs = {}
    for line in open(dd / "decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "req":
            reqs[o["rid"]] = tuple(o["input_ids"])
    gt = {}
    for line in open(gtf):
        r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
    bad = set()
    for rid, ids in reqs.items():
        out = gt.get(ids)
        if not out or len(out) < 5:
            continue
        g = [tuple(out[i:i+4]) for i in range(len(out)-3)]
        if len(set(g)) / max(len(g), 1) < 0.5:
            bad.add(rid)
    return bad


def decisive_positions(cell):
    d = f"{ROOT}/{cell['dir']}"
    ch = load(f"{d}/decisions_select1_oracle.jsonl")
    bad = loopy(d)
    props = cell["props"]
    out = []
    for (rid, ds), rs in ch.items():
        if rid in bad:
            continue
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            cand = [(nm, r.get(tk), (r.get(pk) if r.get(pk) is not None else 0.0))
                    for nm, tk, pk in props if r.get(tk) is not None and r.get(pk) is not None]
            hits = [nm for nm, t, p in cand if gt is not None and t == gt]
            if 0 < len(hits) < len(cand):
                out.append((cand, gt))
            if gt is not None and len(hits) == 0:
                alive = False
    return out


def dominant_name(pos):
    hit = Counter(); tot = Counter()
    for cand, gt in pos:
        for nm, t, p in cand:
            tot[nm] += 1; hit[nm] += (t == gt)
    return max(tot, key=lambda nm: hit[nm] / tot[nm]), {nm: hit[nm]/tot[nm] for nm in tot}


def consensus_token(cand):
    c = Counter(t for nm, t, p in cand)
    tok, n = c.most_common(1)[0]
    return (tok if n >= 2 else None)


def main():
    for name, cell in CELLS.items():
        pos = decisive_positions(cell)
        dom, rel = dominant_name(pos)
        print(f"\n### {name}  decisive={len(pos)}  dominant={dom} "
              f"(rel={ {k:round(v,3) for k,v in rel.items()} })")

        # agreement-pattern breakdown
        n_2agree = 0; pair_gt = 0; loner_gt = 0
        wc_against = 0; wc_right = 0; dom_right_in_wc = 0
        for cand, gt in pos:
            ct = consensus_token(cand)
            if ct is None:
                continue
            n_2agree += 1
            if ct == gt:
                pair_gt += 1
            else:
                loner_gt += 1
            # is the consensus the two NON-dominant agreeing against dominant?
            dom_tok = next((t for nm, t, p in cand if nm == dom), None)
            if dom_tok is not None and ct != dom_tok:
                wc_against += 1
                if ct == gt:
                    wc_right += 1
                if dom_tok == gt:
                    dom_right_in_wc += 1
        alld = len(pos) - n_2agree
        print(f"  patterns: all-distinct={alld} ({100*alld/len(pos):.0f}%)  "
              f"2-agree={n_2agree} ({100*n_2agree/len(pos):.0f}%)")
        if n_2agree:
            print(f"  when 2 agree: P(agreed==gt)={pair_gt/n_2agree:.3f}  "
                  f"P(loner==gt)={loner_gt/n_2agree:.3f}")
        if wc_against:
            print(f"  WEAK-consensus-vs-dominant: n={wc_against}  "
                  f"P(weak-consensus==gt)={wc_right/wc_against:.3f}  "
                  f"P(dominant==gt)={dom_right_in_wc/wc_against:.3f}  "
                  f"-> override {'HELPS' if wc_right>dom_right_in_wc else 'HURTS'}")

        # rule-based selector selacc (tokens only; probs for raw/tie)
        def selacc(policy):
            hit = 0
            for cand, gt in pos:
                hit += (policy(cand) == gt)
            return hit / len(pos)

        def pick_raw(cand):
            return max(cand, key=lambda c: c[2])[1]

        def pick_dom(cand):
            for nm, t, p in cand:
                if nm == dom:
                    return t
            return pick_raw(cand)

        def pick_cons_dom(cand):
            ct = consensus_token(cand)
            return ct if ct is not None else pick_dom(cand)

        def pick_cons_raw(cand):
            ct = consensus_token(cand)
            return ct if ct is not None else pick_raw(cand)

        def pick_dom_unless_wc(cand):
            ct = consensus_token(cand)
            dom_tok = next((t for nm, t, p in cand if nm == dom), None)
            if ct is not None and dom_tok is not None and ct != dom_tok:
                return ct          # the two non-dominant agree against dominant
            return pick_dom(cand)

        print(f"  selacc: raw={selacc(pick_raw):.4f}  dominant={selacc(pick_dom):.4f}  "
              f"cons>dom={selacc(pick_cons_dom):.4f}  cons>raw={selacc(pick_cons_raw):.4f}  "
              f"dom-unless-WC={selacc(pick_dom_unless_wc):.4f}  oracle=1.0000")


main()
