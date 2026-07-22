#!/usr/bin/env python3
"""3-WAY chain-hybrid ceiling analysis (DFlash + EAGLE3 + suffix on Qwen3-8B).

Reads the gt-substituted CEILING arm log (decisions_select1_oracle.jsonl produced
with SGLANG_CHAIN_HYBRID_EAGLE3=1, mode=oracle: the block is teacher-forced to gt so
the target processes gt and every proposer's feature context is gt-CONSISTENT — the
"option B" measurement, no substituted-token feature artifact). Each decision row
carries all three proposers (eagle_token/eagle_p = DFlash, e3_token/e3_p = EAGLE3,
suffix_token/suffix_p = suffix) + gt_token, so EVERY selection policy's decisive
selection-accuracy and block-anchored MAT are computed OFFLINE from this one log,
all on the identical gt path.

selacc = decisive (some-but-not-all proposers hit) alive-conditioned selection
accuracy. MAT = block-anchored mean accept length = the run-length of consecutive
depths (from depth 0) where the policy's pick equals gt — the standard offline-sim
convention (matches analyze_boundary_ladder.py). The 3-way ORACLE MAT vs the 2-way
ORACLE MAT is the search-space-expansion ceiling; the e3-only unique hits are
EAGLE3's contribution that neither DFlash nor suffix can cover.

  python3 simulation/scripts/analyze_3way_ceiling.py --record-dir <dir> [--fig]
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

# Slot layouts. 8B: DFlash is the served main worker (eagle_token field), EAGLE3 is
# the in-process aux (e3_token). 27B: MTP is the served main worker (eagle_token),
# DFlash is the decoupled-on-gt aux merged from dflash_proposals.jsonl (dflash_token).
CELLS = {
    "8b": {"names": ("dflash", "e3", "suffix"),
           "tok": {"dflash": "eagle_token", "e3": "e3_token", "suffix": "suffix_token"},
           "p": {"dflash": "eagle_p", "e3": "e3_p", "suffix": "suffix_p"}},
    "27b": {"names": ("mtp", "dflash", "suffix"),
            "tok": {"mtp": "eagle_token", "dflash": "dflash_token", "suffix": "suffix_token"},
            "p": {"mtp": "eagle_p", "dflash": "dflash_p", "suffix": "suffix_p"}},
}
PROPOSERS = CELLS["8b"]["names"]
TOKKEY = CELLS["8b"]["tok"]
PKEY = CELLS["8b"]["p"]


def load_chains(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains:
        chains[k].sort(key=lambda r: r["depth"])
    return chains


def row_info(r):
    """Return (gt, available set, hits set, tokens dict, probs dict)."""
    gt = r.get("gt_token")
    toks = {p: r.get(TOKKEY[p]) for p in PROPOSERS}
    probs = {p: (r.get(PKEY[p]) if r.get(PKEY[p]) is not None else -1.0)
             for p in PROPOSERS}
    avail = {p for p in PROPOSERS if toks[p] is not None}
    hits = {p for p in avail if gt is not None and toks[p] == gt}
    return gt, avail, hits, toks, probs


def selacc_mat(chains, pick_of, subset=PROPOSERS):
    """pick_of(row, avail, probs) -> proposer name within `subset`.
    Returns (decisive selacc, block-anchored MAT). Decisive = within `subset`,
    some-but-not-all available proposers hit gt (the pick matters). Alive prefix =
    >=1 proposer in `subset` hits (the oracle chain can continue on gt)."""
    sset = set(subset)
    n = corr = 0
    Ls = []
    for rs in chains.values():
        alive = True
        Lp = 0
        pol_alive = True
        for r in rs:
            gt, avail, hits, toks, probs = row_info(r)
            av = avail & sset
            hs = hits & sset
            row_alive = len(hs) > 0
            decisive = row_alive and (len(hs) < len(av))
            pick = pick_of(r, av, probs) if av else None
            pick_correct = pick in hs
            if alive and decisive:
                n += 1
                corr += int(pick_correct)
            if pol_alive:
                if row_alive and pick_correct:
                    Lp += 1
                else:
                    pol_alive = False
            if not row_alive:
                alive = False
        Ls.append(Lp)
    return corr / max(n, 1), sum(Ls) / max(len(Ls), 1)


def pick_raw(r, av, probs):
    """argmax raw prob among available; PROPOSERS[0] (the served main) is the
    default/tiebreak (iteration order favors earlier proposers on ties)."""
    best = None
    best_p = -2.0
    for p in PROPOSERS:
        if p in av and probs[p] > best_p:
            best, best_p = p, probs[p]
    return best


def pick_oracle(hits_first):
    """A picker that always picks a hitting proposer when one exists (else dflash)."""
    def f(r, av, probs):
        gt, avail, hits, toks, _ = row_info(r)
        hs = hits & av
        for p in hits_first:
            if p in hs:
                return p
        for p in PROPOSERS:                       # no hit: fall back to main, in order
            if p in av:
                return p
        return next(iter(av)) if av else None
    return f


def ceiling_stats(chains, subset=PROPOSERS):
    """oracle-ceiling counts over decisive-alive positions: per-proposer UNIQUE
    hits (only that proposer covers gt) and any-of-subset coverage."""
    sset = set(subset)
    tot = alive_any = 0
    nogt = 0
    unique = defaultdict(int)
    by_combo = defaultdict(int)
    for rs in chains.values():
        oracle_alive = True
        for r in rs:
            gt, avail, hits, toks, probs = row_info(r)
            if gt is None:
                nogt += 1
                continue
            if not oracle_alive:
                continue
            tot += 1
            hs = hits & sset
            if hs:
                alive_any += 1
                by_combo["+".join(sorted(hs))] += 1
                if len(hs) == 1:
                    unique[next(iter(hs))] += 1
            else:
                oracle_alive = False        # 3-way oracle chain dies here
    return {"positions": tot, "any_hit": alive_any, "nogt": nogt,
            "unique": dict(unique), "by_combo": dict(by_combo)}


def loopy_rids(record_dir, decisions_file, thresh=0.5, n=4):
    """rids whose gt output is a degenerate repetition loop (distinct-n ratio <
    thresh) — e.g. a model reasoning loop that hits max_tokens. Such a request
    UNFAIRLY inflates suffix (its trie predicts the repeats) and the whole MAT, so
    we exclude it (the agent must be producing real tool-call trajectories for the
    ceiling to be trustworthy)."""
    dd = Path(record_dir)
    reqs = {}                              # rid -> input_ids
    for line in open(dd / decisions_file):
        o = json.loads(line)
        if o.get("type") == "req":
            reqs[o["rid"]] = tuple(o["input_ids"])
    gt = {}
    if (dd / "gt_tokens.jsonl").exists():
        for line in open(dd / "gt_tokens.jsonl"):
            r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
    bad = set()
    info = {}
    for rid, ids in reqs.items():
        out = gt.get(ids)
        if not out or len(out) < n + 1:
            continue
        g = [tuple(out[i:i + n]) for i in range(len(out) - n + 1)]
        dr = len(set(g)) / max(len(g), 1)
        info[rid] = (len(out), round(dr, 3))
        if dr < thresh:
            bad.add(rid)
    return bad, info


def merge_dflash(chains, path):
    """Inject decoupled DFlash proposals (dflash_offline.py --emit) into the served
    log rows by (rid, decode_step, depth) — for the 27B cell where DFlash runs on
    the gt trajectory, not in the served loop."""
    idx = {}
    for (rid, ds), rs in chains.items():
        for r in rs:
            idx[(rid, ds, r["depth"])] = r
    n = miss = 0
    for line in open(path):
        o = json.loads(line)
        key = (o["rid"], o["decode_step"], o["depth"])
        r = idx.get(key)
        if r is None:
            miss += 1; continue
        r["dflash_token"] = o["dflash_token"]; r["dflash_p"] = o.get("dflash_p")
        n += 1
    print(f"merged {n} DFlash proposals ({miss} unmatched)")


def main():
    global PROPOSERS, TOKKEY, PKEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--decisions-file", default="decisions_select1_oracle.jsonl")
    ap.add_argument("--cell", choices=["8b", "27b"], default="8b")
    ap.add_argument("--merge-dflash", default=None,
                    help="dflash_proposals.jsonl to merge (27b cell)")
    ap.add_argument("--exclude-loopy", action="store_true",
                    help="drop requests whose gt output is a degenerate repetition "
                         "loop (distinct-4 < --loopy-thresh) — they inflate suffix/MAT")
    ap.add_argument("--loopy-thresh", type=float, default=0.5)
    ap.add_argument("--fig", action="store_true", help="emit selacc/MAT bar figures")
    args = ap.parse_args()

    cell = CELLS[args.cell]
    PROPOSERS = cell["names"]; TOKKEY = cell["tok"]; PKEY = cell["p"]
    main_p, added_p, suffix_p = PROPOSERS         # (served-main, added-aux, suffix)

    log = Path(args.record_dir) / args.decisions_file
    chains = load_chains(log)
    print(f"cell={args.cell} proposers={PROPOSERS}  log: {log}  "
          f"chains(blocks)={len(chains)}  rows={sum(len(v) for v in chains.values())}")
    if args.exclude_loopy:
        bad, info = loopy_rids(args.record_dir, args.decisions_file, args.loopy_thresh)
        for rid in sorted(bad):
            print(f"  [loopy] drop rid {rid[:12]} len={info[rid][0]} distinct4={info[rid][1]}")
        before = len(chains)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        print(f"excluded {len(bad)} loopy reqs -> blocks {before} -> {len(chains)}")
    if args.merge_dflash:
        merge_dflash(chains, args.merge_dflash)
    # restrict to blocks where the added proposer is present (e.g. 27B reqs that
    # DFlash skipped — corrupted/too-long — would otherwise count as added-misses).
    if args.merge_dflash:
        keep = {k for k, rs in chains.items()
                if any(r.get(TOKKEY[added_p]) is not None for r in rs)}
        dropped = len(chains) - len(keep)
        chains = {k: chains[k] for k in keep}
        print(f"kept {len(chains)} blocks with {added_p} proposals ({dropped} dropped)")

    base2 = (main_p, suffix_p)
    policies = [
        (f"{main_p}-only", pick_raw, (main_p,)),
        (f"{added_p}-only", pick_raw, (added_p,)),
        (f"{suffix_p}-only", pick_raw, (suffix_p,)),
        (f"2way raw\n({main_p}+{suffix_p})", pick_raw, base2),
        (f"2way oracle\n({main_p}+{suffix_p})", pick_oracle(base2), base2),
        ("3way raw", pick_raw, PROPOSERS),
        ("3way oracle", pick_oracle(PROPOSERS), PROPOSERS),
    ]
    print(f"\n{'policy':28s} {'selacc':>8} {'MAT':>8}")
    rows = []
    for lab, pk, sub in policies:
        sa, mt = selacc_mat(chains, pk, sub)
        rows.append((lab, sa, mt))
        print(f"{lab.replace(chr(10),' '):28s} {sa:8.4f} {mt:8.4f}")

    print("\n=== oracle ceiling (decisive-alive positions) ===")
    cs = ceiling_stats(chains, PROPOSERS)
    cs2 = ceiling_stats(chains, base2)
    print(f"positions={cs['positions']} nogt={cs['nogt']}")
    print(f"any-of-3 hit = {cs['any_hit']}  ({cs['any_hit']/max(cs['positions'],1):.3f})")
    print(f"any-of-2 ({main_p}+{suffix_p}) hit = {cs2['any_hit']}  "
          f"({cs2['any_hit']/max(cs2['positions'],1):.3f})")
    exp = cs['any_hit'] - cs2['any_hit']
    print(f"  -> {added_p} search-space expansion: +{exp} positions "
          f"(+{100*exp/max(cs2['any_hit'],1):.1f}% over 2-way)")
    print(f"per-proposer UNIQUE hits (only that proposer covers gt): {cs['unique']}")
    print(f"hit-combo breakdown: {cs['by_combo']}")

    if args.fig:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from _ladder_style import ladder_bar
        except Exception as e:
            print(f"[fig] _ladder_style unavailable ({e}); skipping figures")
            return
        figdir = Path(args.record_dir) / "figures"
        figdir.mkdir(exist_ok=True)
        labels = [r[0] for r in rows]
        cols = ["#1f77b4", "#2ca02c", "#8c564b", "#7f7f7f", "#9467bd",
                "#7f7f7f", "#d62728"]
        names_str = "+".join(PROPOSERS)
        model = "Qwen3-8B" if args.cell == "8b" else "Qwen3.5-27B"
        ladder_bar([r[1] for r in rows], labels,
                   "decisive selection accuracy (alive-conditioned)",
                   f"3-way selection accuracy ({names_str}, SERVED, gt-path)\n"
                   f"{model}  bfcl_v4 web_search",
                   f"{figdir}/ceiling_selacc_3way_{args.cell}.png", fmt="{:.3f}", colors=cols)
        ladder_bar([r[2] for r in rows], labels,
                   "MAT (block-anchored accept length)",
                   f"3-way MAT ({names_str}, SERVED, gt-path)\n"
                   f"{model}  bfcl_v4 web_search",
                   f"{figdir}/ceiling_mat_3way_{args.cell}.png", fmt="{:.3f}", colors=cols)
        print(f"\nfigures -> {figdir}/ceiling_{{selacc,mat}}_3way_{args.cell}.png")


if __name__ == "__main__":
    main()
