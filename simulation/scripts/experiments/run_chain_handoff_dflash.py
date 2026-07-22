#!/usr/bin/env python3
"""Chain hand-off oracle dense table — DFlash main chain + suffix hand-off.

The DFlash analogue of run_chain_handoff_oracle.py (which uses an EAGLE3/MTP
main chain). It emits the SAME per-position row schema, so the downstream
analyzer (analyze_chain_handoff.py) is reused UNCHANGED — that script is
agnostic to whether the main-chain accept length `ae` came from EAGLE/MTP or
DFlash.

The two ingredients per gt position t:

  * A_d(t)    — DFlash block accept length: the DFlash block rooted at t
                predicts positions t..t+b-2 (b = draft block_size); A_d is the
                run-length of consecutive predictions that match the gt
                continuation. Produced OFFLINE by dflash_offline.py --emit
                --dense (one block re-speculated at EVERY gt position), since
                DFlash-27B cannot serve as the sglang main worker on the
                Qwen3.5 Mamba-hybrid arch. We read its dflash_proposals jsonl.
  * A_s(t, j) — for each hand-off depth j in 0..min(A_d, block_len): accept
                length of a suffix-decoding draft speculated from
                context + gt[:j]. IDENTICAL to run_chain_handoff_oracle.py:
                same SuffixDecodingCache, same trie discipline, same
                temporary_extension for j>0, same orc/grd variants.

GT trajectory: driven directly from the served record's gt_tokens.jsonl
(input_ids -> output_ids) plus the decision-log `req` rows (rid -> input_ids).
No oracle-vanilla capture is needed; the gt path is a fixed greedy trajectory
and both A_d and A_s are measured against it (self-consistent). Requests whose
input/output contains an OOB token (a corrupted capture row) are auto-skipped
exactly as dflash_offline.py skips them, so the position sets line up: we only
process rids for which DFlash proposals exist.

Trie discipline (mirrors run_chain_handoff_oracle.py / the live suffix pre-pass):
start_request(prompt) -> for position p=1..len(out)-1: add_active_response(
out[p-1]) so the trie holds prompt+out[:p] (the committed seed is in the trie,
the token being predicted is not) -> speculate at each j -> stop_request.
DFlash does not draft out[0] (the prefill bonus token), so positions start at 1.

Output: gzipped JSONL, one row per position, IDENTICAL schema to
run_chain_handoff_oracle.py:
  {"rid": str, "ci": int, "pos": int, "el": int, "ae": int, "gl": int,
   "sfx": [[j, orc, grd, match_len, greedy_chain_len, n_nodes, score], ...]}

Usage (inside sglang-bench container, from /workspace):
  python3 -m simulation.scripts.experiments.run_chain_handoff_dflash \
      --record-dir simulation/results/chain_hybrid_perdepth/specbench_qwen35_27b_mtp_2way \
      --decisions-file decisions_select1_oracle.jsonl \
      --dflash-proposals .../dflash_proposals_dense.jsonl \
      --output simulation/results/chain_handoff_oracle/qwen35_27b_dflash/specbench_dense.jsonl.gz \
      [--limit-requests 5] [--validate-greedy 2000]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_spec_decoding.suffix_decoding.suffix_tree import (  # noqa: E402
    SuffixDecodingCache,
)
# Reuse the SAME validated accept / tree-walk / greedy-chain helpers so the
# suffix side is bit-identical to the eagle/MTP hand-off table.
from simulation.scripts.experiments.run_chain_handoff_oracle import (  # noqa: E402
    _greedy_suffix_chain,
    _prefix_accept,
    _tree_walk_accept,
)


def _load_reqs(record_dir, decisions_file):
    """rid -> input_ids (from the decision-log `req` rows) and
    tuple(input_ids) -> output_ids (from gt_tokens.jsonl)."""
    dd = Path(record_dir)
    gt = {}
    gt_path = dd / "gt_tokens.jsonl"
    if not gt_path.exists():
        raise SystemExit(f"no gt_tokens.jsonl in {record_dir}")
    for line in open(gt_path):
        r = json.loads(line)
        gt[tuple(r["input_ids"])] = r["output_ids"]
    log = dd / decisions_file if decisions_file else None
    if log is None or not log.exists():
        hits = sorted(dd.glob("decisions_*.jsonl"))
        log = hits[0] if hits else None
    if log is None or not log.exists():
        raise SystemExit(f"no decisions log in {record_dir}")
    reqs = {}
    with open(log) as f:
        for line in f:
            # cheap pre-filter: only `req` rows carry input_ids; skip the
            # millions of `decision`/`step` rows without parsing them.
            if '"req"' not in line:
                continue
            r = json.loads(line)
            if r.get("type") == "req":
                reqs[r["rid"]] = r["input_ids"]
    return reqs, gt, log


def _load_dflash_chains(path):
    """(rid, decode_step) -> [token ordered by depth].

    A dense dflash_proposals jsonl row is
    {rid, decode_step, depth, dflash_token, dflash_p}; decode_step ds roots a
    block whose depth-d entry predicts output position ds+d (1-indexed into the
    output, i.e. out[ds+d]). So the block rooted at ds is the main chain for
    table position p == ds (it predicts out[p], out[p+1], ...)."""
    tmp = defaultdict(dict)  # (rid, ds) -> {depth: token}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tmp[(r["rid"], r["decode_step"])][r["depth"]] = r["dflash_token"]
    chains = {}
    for key, dmap in tmp.items():
        chains[key] = [dmap[d] for d in sorted(dmap)]
    return chains


def _new_cache():
    return SuffixDecodingCache(
        max_tree_depth=64, max_cached_requests=100000, enable_undo=True)


def _prefill_global(cache, reqs, gt, chain_rids, exclude=None):
    """Cache every request's COMPLETE response into the global tree (under
    throwaway __bg__ ids), optionally excluding one rid. Returns count cached.
    stop_request retains the response in the global tree, so after this the
    global tree holds the full corpus (minus `exclude`)."""
    n = 0
    for rid2, ids2 in reqs.items():
        if exclude is not None and rid2 == exclude:
            continue
        out2 = gt.get(tuple(ids2))
        if out2 is None or rid2 not in chain_rids:
            continue
        bg = f"__bg__{rid2}"
        cache.start_request(bg, np.asarray(list(ids2), dtype=np.int32))
        cache.add_active_response(bg, list(out2))
        cache.stop_request(bg)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record-dir", required=True,
                    help="served record dir with gt_tokens.jsonl + decisions log")
    ap.add_argument("--decisions-file", default=None,
                    help="decision log filename inside --record-dir "
                         "(default: first decisions_*.jsonl)")
    ap.add_argument("--dflash-proposals", required=True,
                    help="dense dflash_proposals jsonl from dflash_offline.py "
                         "--emit --dense")
    ap.add_argument("--output", required=True,
                    help="gzipped JSONL output path (.jsonl.gz)")
    ap.add_argument("--tail", type=int, default=64,
                    help="context tail length fed to speculate (ctx[-tail:])")
    ap.add_argument("--global-prefill", choices=["none", "full", "leaveoneout"],
                    default="none",
                    help="none: the global suffix tree accumulates online in "
                         "request order (cold-start). full: pre-populate the "
                         "global tree with EVERY request's complete response "
                         "(incl. the measured request itself) before measuring, "
                         "so speculate matches against the full corpus and can "
                         "match its own future continuation — an absolute upper "
                         "bound on the suffix arm. leaveoneout: same full "
                         "prefill, but evict the measured request's own complete "
                         "response before measuring it (its past still "
                         "accumulates incrementally), so the suffix sees the "
                         "full corpus of OTHER requests but never its own future "
                         "— a warm-cache bound without self-cheating.")
    ap.add_argument("--limit-requests", type=int, default=0,
                    help="smoke mode: keep only the first N distinct rids")
    ap.add_argument("--validate-greedy", type=int, default=0,
                    help="for the first N suffix evals, also call "
                         "speculate(use_tree_spec=False) and compare with the "
                         "derived greedy chain (stats in meta)")
    ap.add_argument("--full-handoff-depths", action="store_true",
                    help="compute suffix accept at every hand-off depth up to the "
                         "full DFlash block (for the tree experiment), not just up "
                         "to the linear chain accept ae")
    ap.add_argument("--max-spec-tokens", type=int, default=256)
    ap.add_argument("--max-spec-factor", type=float, default=4.0)
    ap.add_argument("--min-token-prob", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    reqs, gt, log = _load_reqs(args.record_dir, args.decisions_file)
    chains = _load_dflash_chains(args.dflash_proposals)
    chain_rids = {rid for (rid, _ds) in chains}
    print(f"reqs={len(reqs)} gt_rows={len(gt)} dflash_chains={len(chains)} "
          f"dflash_rids={len(chain_rids)} (load {time.time()-t0:.0f}s)",
          file=sys.stderr)

    spec_kwargs = dict(
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
        use_tree_spec=True,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = gzip.open(out_path, "wt", compresslevel=4)

    meta = {
        "record_dir": args.record_dir,
        "dflash_proposals": args.dflash_proposals,
        "main_chain": "dflash",
        "global_prefill": args.global_prefill,
        "spec_kwargs": spec_kwargs,
        "tail": args.tail,
        "limit_requests": args.limit_requests,
        "n_reqs_processed": 0,
        "n_reqs_skipped_no_dflash": 0,
        "n_reqs_skipped_no_gt": 0,
        "n_positions": 0,
        "n_positions_no_chain": 0,   # position with no DFlash block (end of seq)
        "n_spec_calls": 0,
        "n_spec_errors": 0,
        "validate_greedy": {"n": 0, "exact": 0, "prefix": 0},
    }
    vg_left = args.validate_greedy

    # Cache strategy:
    #   none        — one shared cache, global tree accumulates online in request
    #                 order (cold-start).
    #   full        — one shared cache, pre-populated with EVERY request's complete
    #                 response (incl. the measured one) so speculate can match its
    #                 own future continuation — absolute upper bound on suffix.
    #   leaveoneout — a FRESH cache per request, pre-populated with every OTHER
    #                 request's complete response. The measured request's future is
    #                 never inserted (only its past accumulates incrementally in the
    #                 loop), so there is no self-match and no need for the native
    #                 evict_cached_response (which segfaults on the large
    #                 accumulated tree). Warm full corpus, no self-cheating.
    shared_cache = None
    if args.global_prefill != "leaveoneout":
        shared_cache = _new_cache()
        if args.global_prefill == "full":
            npre = _prefill_global(shared_cache, reqs, gt, chain_rids)
            meta["n_global_prefill"] = npre
            print(f"global prefill (full, incl self): cached {npre} complete "
                  f"responses into the global tree", file=sys.stderr, flush=True)

    kept = 0
    n_pos = 0
    t1 = time.time()
    for rid, ids in reqs.items():
        if args.limit_requests and kept >= args.limit_requests:
            break
        out = gt.get(tuple(ids))
        if out is None:
            meta["n_reqs_skipped_no_gt"] += 1
            continue
        # Process only rids DFlash emitted (it skips OOB/corrupted requests);
        # this keeps the A_d and A_s position sets aligned.
        if rid not in chain_rids:
            meta["n_reqs_skipped_no_dflash"] += 1
            continue
        kept += 1
        prompt = list(ids)
        out = list(out)
        if args.global_prefill == "leaveoneout":
            cache = _new_cache()
            noth = _prefill_global(cache, reqs, gt, chain_rids, exclude=rid)
            meta["n_global_prefill"] = noth   # others per request (constant)
        else:
            cache = shared_cache
        cache_req_id = rid
        cache.start_request(cache_req_id, np.asarray(prompt, dtype=np.int32))
        # DFlash drafts from position p=1 (predicting out[p]); out[0] is the
        # prefill bonus token and is never drafted.
        for p in range(1, len(out)):
            # Trie now holds prompt + out[:p]: the committed seed out[p-1] is
            # added, the token out[p] being predicted is NOT.
            cache.add_active_response(cache_req_id, [out[p - 1]])

            chain = chains.get((rid, p))
            gt_future = out[p:]
            if not chain or not gt_future:
                meta["n_positions_no_chain"] += 1
                continue
            ae = _prefix_accept(chain, gt_future)
            ctx_tail = (prompt + out[:p])[-args.tail:]
            sfx_rows = []
            # --full-handoff-depths: compute suffix accept at EVERY hand-off depth
            # j up to the full DFlash block (not just up to the chain accept `ae`).
            # Needed for the TREE experiment, where the DFlash-tree head accept
            # A_d_tree can exceed the linear chain accept ae, so the tail may attach
            # at a depth j in (ae, block] that the ae-capped table never computed.
            jmax = len(chain) if args.full_handoff_depths else min(ae, len(chain))
            jmax = min(jmax, len(gt_future))
            for j in range(0, jmax + 1):
                ext = list(gt_future[:j])
                gt_j = gt_future[j:]
                draft = None
                meta["n_spec_calls"] += 1
                try:
                    if j == 0:
                        draft = cache.speculate(
                            cache_req_id,
                            np.asarray(ctx_tail, dtype=np.int32),
                            **spec_kwargs)
                    else:
                        with cache.temporary_extension(cache_req_id, ext):
                            draft = cache.speculate(
                                cache_req_id,
                                np.asarray(ctx_tail + ext, dtype=np.int32),
                                **spec_kwargs)
                except Exception:
                    meta["n_spec_errors"] += 1
                    draft = None
                if draft is None or not draft.token_ids:
                    sfx_rows.append([j, 0, 0,
                                     int(draft.match_len) if draft else 0,
                                     0, 0, 0.0])
                    continue
                orc = _tree_walk_accept(draft.token_ids, draft.parents, gt_j)
                gchain = _greedy_suffix_chain(
                    draft.token_ids, draft.parents,
                    draft.counts, draft.probs)
                grd = _prefix_accept(gchain, gt_j)
                if vg_left > 0:
                    vg_left -= 1
                    try:
                        if j == 0:
                            cdraft = cache.speculate(
                                cache_req_id,
                                np.asarray(ctx_tail, dtype=np.int32),
                                max_spec_tokens=args.max_spec_tokens,
                                max_spec_factor=args.max_spec_factor,
                                min_token_prob=args.min_token_prob,
                                use_tree_spec=False)
                        else:
                            with cache.temporary_extension(cache_req_id, ext):
                                cdraft = cache.speculate(
                                    cache_req_id,
                                    np.asarray(ctx_tail + ext, dtype=np.int32),
                                    max_spec_tokens=args.max_spec_tokens,
                                    max_spec_factor=args.max_spec_factor,
                                    min_token_prob=args.min_token_prob,
                                    use_tree_spec=False)
                        meta["validate_greedy"]["n"] += 1
                        ctoks = list(cdraft.token_ids)
                        if ctoks == gchain:
                            meta["validate_greedy"]["exact"] += 1
                        m = min(len(ctoks), len(gchain))
                        if ctoks[:m] == gchain[:m]:
                            meta["validate_greedy"]["prefix"] += 1
                    except Exception:
                        pass
                sfx_rows.append([
                    j, int(orc), int(grd), int(draft.match_len),
                    len(gchain), len(draft.token_ids),
                    round(float(draft.score), 4)])
            fh.write(json.dumps(
                {"rid": rid, "ci": 0, "pos": p,
                 "el": len(chain), "ae": int(ae), "gl": len(gt_future),
                 "sfx": sfx_rows},
                separators=(",", ":")) + "\n")
            meta["n_positions"] += 1
            n_pos += 1
            if n_pos % 50000 == 0:
                el = time.time() - t1
                print(f"  {n_pos} positions ({el:.0f}s, "
                      f"{n_pos/max(el,1e-9):.0f} pos/s)",
                      file=sys.stderr, flush=True)
        cache.stop_request(cache_req_id)
        meta["n_reqs_processed"] += 1
        if args.global_prefill == "leaveoneout":
            cache = None   # free the per-request cache (C++ tree) before the next

    fh.close()
    meta["elapsed_s"] = round(time.time() - t0, 1)
    with open(str(out_path) + ".meta.json", "w") as mf:
        json.dump(meta, mf, indent=2)
    print(f"DONE rows={meta['n_positions']} reqs={meta['n_reqs_processed']} "
          f"no_chain={meta['n_positions_no_chain']} "
          f"spec_errors={meta['n_spec_errors']} -> {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
