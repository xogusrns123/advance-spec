"""Probe: does the GLOBAL suffix tree carry information the LOCAL tree's 1/2 lacks?
ArcticInference runs local(prompt) + global(past responses) trees INDEPENDENTLY and
keeps the max-SCORE draft (the loser's evidence is discarded). Hypothesis (user): at
local-1/2 positions the global tree may have a sharper estimate, and MERGING counts
could fix the 0.5 problem.

Replays the eval gt trajectories in order, rebuilding both trees (start_request=prompt
-> local; add_active_response(gen) at finish -> global). At each decode position queries
BOTH trees' top-token estimate and compares to gt. Sizes:
  - at local-1/2 (n_l=2): how often global has info, its n_g, and accept(local-top) vs
    accept(global-top) vs accept(merge: combined-count top) vs accept(prefer-higher-n).
Run (IN docker): docker exec sglang-bench python3 /workspace/simulation/scripts/select1_ladder/global_tree_probe.py
"""
import json, sys
from collections import Counter, defaultdict
import numpy as np
from arctic_inference.suffix_decoding.cache import SuffixDecodingCache, SuffixTree

GT="simulation/results/chain_hybrid_perdepth/qwen3_14b_ar/gt_tokens.jsonl"
MAXD=64; FACTOR=4.0; OFFSET=0.0; MINP=0.1

def spec(tree, ctx):
    try:
        return tree.speculate(ctx, MAXD, FACTOR, OFFSET, MINP, False)
    except Exception:
        return SuffixTree.speculate(tree, np.array(ctx, np.int32), MAXD, FACTOR, OFFSET, MINP, False)

def top(d):
    """(tok, c, n, p, score) of the first speculated token, or None."""
    if not d.token_ids:
        return None
    c=int(d.counts[0]); p=float(d.probs[0]); tok=int(d.token_ids[0])
    n=int(round(c/p)) if p>0 else 0
    return (tok, c, n, p, float(d.score))

recs=[json.loads(l) for l in open(GT)]
sc=SuffixDecodingCache(max_tree_depth=MAXD)
# accumulators
N=0; n_local_half=0
g_has=0; g_same_tok=0
acc=defaultdict(int)            # at local-1/2: which strategy hits gt
ng_dist=Counter()
# strategy comparison (first-token accept): A=max path-score (current), B=max first-tok prob,
# C=max first-tok n (evidence), G=always-global-if-available. computed over suffix-proposing
# positions, and separately restricted to local-1/2 positions.
strat=defaultdict(int); strat_half=defaultdict(int)
pick_local_by_score=0
# overall (all suffix-proposing positions): accept of chosen(max-score) vs merge
ov=defaultdict(int); ov_n=0
def strat_tokens(dl,dg):
    """given local top tuple dl and global top tuple dg(or None), return dict of strategy->token."""
    if dg is None:
        return {k:dl[0] for k in ('A','B','C','G')}, True
    # dl/dg = (tok,c,n,p,score)
    A = dl[0] if dl[4] >= dg[4] else dg[0]          # current: max path score (>= -> local)
    B = dl[0] if dl[3] >= dg[3] else dg[0]          # max first-token prob
    C = dl[0] if dl[2] >= dg[2] else dg[0]          # max first-token evidence n
    G = dg[0]                                        # always global
    return {'A':A,'B':B,'C':C,'G':G}, (dl[4]>=dg[4])
for ri,rec in enumerate(recs):
    rid=f"r{ri}"; prompt=list(rec.get("input_ids") or []); gen=list(rec.get("output_ids") or [])
    if not gen: continue
    sc.start_request(rid, prompt)
    lt=sc._local_trees[rid]; gtree=sc._global_tree
    seq=list(prompt)
    for tok in gen:
        ctx=seq[-MAXD:]
        dl=top(spec(lt,ctx)); dg=top(spec(gtree,ctx))
        gtok=tok
        if dl is not None:
            ov_n+=1
            toks, a_is_local = strat_tokens(dl, dg)
            pick_local_by_score += a_is_local
            oracle = (dl[0]==gtok) or (dg is not None and dg[0]==gtok)
            for k,t in toks.items():
                strat[k]+= (t==gtok)
            strat['O']+= oracle
            is_half = (dl[2]==2 and dl[1]==1)
            if is_half:
                for k,t in toks.items():
                    strat_half[k]+= (t==gtok)
                strat_half['O']+= oracle
            # current behaviour ~ max-score; approximate chosen by higher path is complex,
            # so for OVERALL we compare local-top (the 1/2 source) vs a count-merge top.
            # merge: if same token, pooled; if different, pick higher combined count.
            if dg is not None and dg[0]==dl[0]:
                merged_tok=dl[0]
            elif dg is not None:
                merged_tok = dl[0] if dl[1] >= dg[1] else dg[0]   # higher absolute count
            else:
                merged_tok=dl[0]
            ov['local']+= (dl[0]==gtok)
            ov['merge']+= (merged_tok==gtok)
            ov['globalavail']+= (dg is not None)
            # LOCAL-1/2 slice
            if dl[2]==2 and dl[1]==1:
                n_local_half+=1
                acc['local_top']+= (dl[0]==gtok)
                if dg is not None:
                    g_has+=1; ng_dist[min(dg[2],20)]+=1
                    g_same_tok+= (dg[0]==dl[0])
                    acc['global_top']+= (dg[0]==gtok)
                    # merge: same tok -> local; diff -> higher combined count
                    mtok = dl[0] if dg[0]==dl[0] else (dl[0] if dl[1]>=dg[1] else dg[0])
                    acc['merge_top']+= (mtok==gtok)
                    acc['prefer_higher_n']+= ((dg[0] if dg[2]>dl[2] else dl[0])==gtok)
                else:
                    acc['global_top']+= 0
                    acc['merge_top']+= (dl[0]==gtok)
                    acc['prefer_higher_n']+= (dl[0]==gtok)
        seq.append(tok)
        N+=1
    sc.add_active_response(rid, gen); sc.stop_request(rid)

print(f"requests={len(recs)}  decode positions={N}")
print(f"suffix-proposing (local has a draft) positions={ov_n}")
print(f"\nLOCAL-1/2 positions (n_l=2,c_l=1): {n_local_half} ({100*n_local_half/max(ov_n,1):.1f}% of suffix-proposing)")
h=n_local_half
print(f"  global has info at these: {g_has} ({100*g_has/max(h,1):.1f}%)   "
      f"global top == local top: {g_same_tok} ({100*g_same_tok/max(g_has,1):.1f}% of those)")
print(f"  n_g distribution (cap 20) at local-1/2 (where global has info): "
      + ", ".join(f"{k}:{v}" for k,v in sorted(ng_dist.items())))
print(f"\n  ACCEPT rate at local-1/2 positions (==gt):")
for k in ('local_top','global_top','merge_top','prefer_higher_n'):
    print(f"    {k:16}: {100*acc[k]/max(h,1):.1f}%   ({acc[k]}/{h})")
print(f"\nOVERALL (suffix-proposing, n={ov_n}):  local-top accept={100*ov['local']/max(ov_n,1):.1f}%  "
      f"merge-top accept={100*ov['merge']/max(ov_n,1):.1f}%  global-available={100*ov['globalavail']/max(ov_n,1):.1f}%")

print(f"\n=== TREE-SELECTION CRITERION: first-token accept (current 'A' picks local {100*pick_local_by_score/max(ov_n,1):.0f}% by path-score) ===")
names={'A':'A current (max PATH-score)','B':'B max FIRST-TOKEN prob','C':'C max first-token n (evidence)','G':'G always global','O':'O oracle (either tree right)'}
print(f"  over ALL suffix-proposing (n={ov_n}):")
for k in ('A','B','C','G','O'):
    print(f"    {names[k]:30}: accept={100*strat[k]/max(ov_n,1):.1f}%")
print(f"  restricted to LOCAL-1/2 (n={n_local_half}):")
for k in ('A','B','C','G','O'):
    print(f"    {names[k]:30}: accept={100*strat_half[k]/max(n_local_half,1):.1f}%")
