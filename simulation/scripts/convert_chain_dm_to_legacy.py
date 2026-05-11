"""Convert chain replay DM capture (agent_results_eagle3.json) to the
legacy ``draft_model_drafts.jsonl`` format consumed by the simulator.

The chain capture's logged step_idx starts at the first token AFTER SGLang's
spec-decode warmup (≈7 leading decode steps that get force-replayed but not
logged). The eagle3 capture's oracle entries also include warmup but with a
DIFFERENT lead — its `tokens[0][0]` field is offset from the response text
by ~6-8 tokens (per token-level alignment, varies by workload AND by
question).

The simulator's legacy ``draft_model_drafts.jsonl`` lookup is keyed by
``(bfcl_id, call_idx, step_idx)`` where step_idx is expected to match the
eagle3 capture's per-call oracle-entry index (i.e., its `pos`).  We therefore
auto-align each (q, call) by matching the first ~30 chain DM verified tokens
against the eagle3 capture's verified tokens, find the offset N where
``dm[0] == eagle3[N]``, and emit step_idx = local_idx + N for every dm
entry. Entries at positions 0..N-1 (warmup) get no chain DM record — sim
falls back to legacy/no-DM for those positions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import ijson


def _load_dm_capture(path):
    """Returns dict[bfcl_id][call_idx] = list[(verified_token, tree_dict)]."""
    with open(path) as f:
        d = json.load(f)
    out = defaultdict(dict)
    for q in d.get("questions", []):
        qid = (q.get("bfcl_id") or q.get("instance_id")
               or str(q.get("question_id", "")))
        if not qid:
            continue
        units = (q.get("agent_metrics") or {}).get("steps") or q.get("turns") or []
        for call_idx, u in enumerate(units):
            if not isinstance(u, dict):
                continue
            evs = (u.get("spec_decode") or {}).get("oracle_vanilla_entries") or []
            entries = []
            for e in evs:
                v = ((e.get("tokens") or [[None]])[0] or [None])[0]
                tree = e.get("eagle3_tree") or {}
                tids = tree.get("token_ids") or []
                pids = tree.get("parents") or []
                if v is None or not tids:
                    continue
                entries.append((int(v), list(tids), list(pids)))
            if entries:
                out[qid][call_idx] = entries
    return out


def _find_offset(dm_v_seq, e3_v_seq, max_off=30, probe_len=30):
    """Best offset N such that dm[i] == e3[i+N] over first probe_len tokens."""
    if not dm_v_seq or not e3_v_seq:
        return 0
    best_off, best_match = 0, -1
    n = min(probe_len, len(dm_v_seq))
    for off in range(-max_off, max_off + 1):
        m = 0
        for i in range(n):
            j = i + off
            if 0 <= j < len(e3_v_seq) and dm_v_seq[i] == e3_v_seq[j]:
                m += 1
        if m > best_match:
            best_match = m
            best_off = off
    return best_off, best_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm-capture", required=True,
                    help="agent_results_eagle3.json from chain replay")
    ap.add_argument("--eagle3-capture", required=True,
                    help="agent_results_eagle3.json (eagle3 reference)")
    ap.add_argument("--output", required=True,
                    help="draft_model_drafts.jsonl to write")
    ap.add_argument("--probe-len", type=int, default=30,
                    help="how many leading verified tokens to align on")
    args = ap.parse_args()

    print(f"Loading dm capture: {args.dm_capture}", file=sys.stderr)
    dm = _load_dm_capture(args.dm_capture)
    n_q = len(dm)
    n_calls = sum(len(v) for v in dm.values())
    print(f"  loaded {n_q} questions, {n_calls} calls", file=sys.stderr)

    print(f"Streaming eagle3: {args.eagle3_capture}", file=sys.stderr)
    n_written = 0
    n_aligned = 0
    n_skipped = 0
    offsets_seen = []

    with open(args.eagle3_capture, "rb") as ef, \
            open(args.output, "w") as outf:
        for q in ijson.items(ef, "questions.item"):
            qid = (q.get("bfcl_id") or q.get("instance_id")
                   or str(q.get("question_id", "")))
            if not qid or qid not in dm:
                continue
            dm_calls = dm[qid]
            units = (q.get("agent_metrics") or {}).get("steps") or q.get("turns") or []
            for call_idx, u in enumerate(units):
                if call_idx not in dm_calls:
                    continue
                e_evs = (u.get("spec_decode") or {}).get("oracle_vanilla_entries") or []
                e_v = [((e.get("tokens") or [[None]])[0] or [None])[0]
                       for e in e_evs]
                dm_entries = dm_calls[call_idx]
                dm_v = [v for v, _, _ in dm_entries]

                off, match = _find_offset(dm_v, e_v, probe_len=args.probe_len)
                if match < args.probe_len * 0.7:
                    print(f"  WARN: poor align for {qid} call={call_idx} "
                          f"off={off} matches={match}/{args.probe_len}",
                          file=sys.stderr)
                    n_skipped += 1
                    continue
                offsets_seen.append(off)
                n_aligned += 1

                for local_idx, (v, tids, pids) in enumerate(dm_entries):
                    eagle_pos = local_idx + off
                    if eagle_pos < 0:
                        continue  # before eagle3 capture starts
                    rec = {
                        "request_id": qid,
                        "call_idx": call_idx,
                        "step_idx": eagle_pos,
                        "token_ids": tids,
                        "parents": pids,
                    }
                    outf.write(json.dumps(rec) + "\n")
                    n_written += 1

    print(f"DONE: aligned {n_aligned}/{n_aligned + n_skipped} calls, "
          f"wrote {n_written} entries", file=sys.stderr)
    if offsets_seen:
        from collections import Counter
        oc = Counter(offsets_seen)
        print(f"  offset distribution: {dict(oc.most_common(10))}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
