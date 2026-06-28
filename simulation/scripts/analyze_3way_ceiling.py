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

PROPOSERS = ("dflash", "e3", "suffix")
TOKKEY = {"dflash": "eagle_token", "e3": "e3_token", "suffix": "suffix_token"}
PKEY = {"dflash": "eagle_p", "e3": "e3_p", "suffix": "suffix_p"}


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
    """argmax raw prob among available; dflash is the default/tiebreak order."""
    best = None
    best_p = -2.0
    for p in ("dflash", "e3", "suffix"):
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
        return "dflash" if "dflash" in av else (next(iter(av)) if av else None)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--decisions-file", default="decisions_select1_oracle.jsonl")
    ap.add_argument("--fig", action="store_true", help="emit selacc/MAT bar figures")
    args = ap.parse_args()

    log = Path(args.record_dir) / args.decisions_file
    chains = load_chains(log)
    print(f"log: {log}  | chains(blocks)={len(chains)}  "
          f"rows={sum(len(v) for v in chains.values())}")

    # ---- policies (label, picker, subset) --------------------------------- #
    policies = [
        ("dflash-only", pick_raw, ("dflash",)),
        ("e3-only", pick_raw, ("e3",)),
        ("suffix-only", pick_raw, ("suffix",)),
        ("2way raw\n(dflash+suffix)", pick_raw, ("dflash", "suffix")),
        ("2way oracle\n(dflash+suffix)", pick_oracle(("dflash", "suffix")),
         ("dflash", "suffix")),
        ("3way raw", pick_raw, PROPOSERS),
        ("3way oracle", pick_oracle(PROPOSERS), PROPOSERS),
    ]
    print(f"\n{'policy':28s} {'selacc':>8} {'MAT':>8}")
    rows = []
    for lab, pk, sub in policies:
        sa, mt = selacc_mat(chains, pk, sub)
        rows.append((lab, sa, mt))
        print(f"{lab.replace(chr(10),' '):28s} {sa:8.4f} {mt:8.4f}")

    print("\n=== oracle ceiling (decisive-alive positions, 3-way subset) ===")
    cs = ceiling_stats(chains, PROPOSERS)
    cs2 = ceiling_stats(chains, ("dflash", "suffix"))
    print(f"positions={cs['positions']} nogt={cs['nogt']}")
    print(f"any-of-3 hit = {cs['any_hit']}  ({cs['any_hit']/max(cs['positions'],1):.3f})")
    print(f"any-of-2 (dflash+suffix) hit = {cs2['any_hit']}  "
          f"({cs2['any_hit']/max(cs2['positions'],1):.3f})")
    exp = cs['any_hit'] - cs2['any_hit']
    print(f"  -> EAGLE3 search-space expansion: +{exp} positions "
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
        ladder_bar([r[1] for r in rows], labels,
                   "decisive selection accuracy (alive-conditioned)",
                   "3-way selection accuracy (DFlash+EAGLE3+suffix, SERVED, gt-path)\n"
                   "Qwen3-8B  bfcl_v4 web_search",
                   f"{figdir}/ceiling_selacc_3way.png", fmt="{:.3f}", colors=cols)
        ladder_bar([r[2] for r in rows], labels,
                   "MAT (block-anchored accept length)",
                   "3-way MAT (DFlash+EAGLE3+suffix, SERVED, gt-path)\n"
                   "Qwen3-8B  bfcl_v4 web_search",
                   f"{figdir}/ceiling_mat_3way.png", fmt="{:.3f}", colors=cols)
        print(f"\nfigures -> {figdir}/ceiling_{{selacc,mat}}_3way.png")


if __name__ == "__main__":
    main()
