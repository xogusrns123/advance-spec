"""Post-process MTP draft-collection JSONL into per-position accept-rate
stats with the same schema as compute_position_accepts.py:

  output["position_accepts"]["by_proposer"]["mtp"] = {
    "seq_accept":   [...],
    "ind_accept":   [...],
    "depth_ge":     [...],
    "cond_accept":  [...],
    "cond_denom":   [...],
  }

Per-step ground truth is reconstructed from each request's *bonus
tokens* (oracle-vanilla force-accept ⇒ each verify step appends exactly
one committed token). For step k of request r, ground_truth_future is
the bonus tokens of steps k, k+1, ..., k+max_position-1 (truncated at
the request's end).

Input JSONL format (from collect_mtp_drafts.py):
  {"request_id": str, "input_ids": [...], "output_ids": [...],
   "oracle_entries": [
     {"eagle3_tree": {"token_ids": [...], "parents": [...]},
      "tokens": [[bonus_token]],
      "req_id": "...", "proposer": "mtp"|"eagle3", ...},
     ...
   ]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.evaluation.tree_knapsack import position_accept_rates  # noqa: E402


def _empty_stats(max_pos: int) -> Dict[str, List[int]]:
    return {
        "seq_accept":  [0] * max_pos,
        "ind_accept":  [0] * max_pos,
        "depth_ge":    [0] * max_pos,
        "cond_accept": [0] * max_pos,
        "cond_denom":  [0] * max_pos,
    }


def _accumulate(stats, seq, ind, cond_acc, cond_dn, denom_depth):
    for d in range(denom_depth):
        stats["depth_ge"][d] += 1
        stats["seq_accept"][d] += seq[d]
        stats["ind_accept"][d] += ind[d]
        stats["cond_accept"][d] += cond_acc[d]
        stats["cond_denom"][d] += cond_dn[d]


def _entry_bonus_token(entry: dict) -> int | None:
    """Extract the single committed token for this oracle entry.
    Force-accept logs keep `tokens=[[bonus]]` per request — for our
    one-request-at-a-time client, this is just `tokens[0][0]`.
    """
    tokens = entry.get("tokens")
    if not tokens:
        return None
    first = tokens[0] if isinstance(tokens, list) else None
    if not first:
        return None
    if isinstance(first, list):
        return int(first[0]) if first else None
    return int(first)


def _entry_tree(entry: dict):
    """Return (token_ids, parents) for the draft tree. Prefer the
    BFS-ordered eagle3_tree dict (works for both EAGLE3 and MTP)."""
    e3 = entry.get("eagle3_tree")
    if e3 and e3.get("token_ids") is not None:
        return list(e3["token_ids"]), list(e3["parents"])
    # Fallback: flat draft from `eagle3` field (treated as chain).
    raw = entry.get("eagle3")
    if raw and raw[0]:
        flat = list(raw[0])
        # Each draft as a flat chain rooted at virtual root.
        parents = [-1] + list(range(len(flat) - 1))
        return flat, parents
    return [], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="JSONL from collect_mtp_drafts.py")
    ap.add_argument("--output", required=True,
                    help="JSON output (mirrors posacc_<wl>.json schema)")
    ap.add_argument("--max-position", type=int, default=64)
    ap.add_argument("--proposer-name", default="mtp",
                    help="Key under by_proposer (default 'mtp').")
    args = ap.parse_args()

    max_pos = args.max_position
    stats = _empty_stats(max_pos)
    n_reqs = 0
    n_steps = 0
    n_dropped_no_bonus = 0
    n_dropped_no_tree = 0

    with open(args.input) as fh:
        for line in fh:
            try:
                req = json.loads(line)
            except Exception:
                continue
            entries = req.get("oracle_entries") or []
            if not entries:
                continue
            n_reqs += 1

            # Ground-truth sequence: bonus tokens in step order.
            bonus_seq: List[int] = []
            valid_entries = []
            for e in entries:
                b = _entry_bonus_token(e)
                if b is None:
                    n_dropped_no_bonus += 1
                    continue
                bonus_seq.append(b)
                valid_entries.append(e)

            # For each step, the future ground truth is bonus_seq[k:].
            for k, e in enumerate(valid_entries):
                tids, pids = _entry_tree(e)
                if not tids and not pids:
                    n_dropped_no_tree += 1
                future = bonus_seq[k:]
                if not future:
                    continue
                seq, ind, ca, cd, dn = position_accept_rates(
                    tids, pids, future, max_pos)
                if dn > 0:
                    _accumulate(stats, seq, ind, ca, cd, dn)
                n_steps += 1

    out = {
        "metadata": {
            "input": args.input,
            "n_requests": n_reqs,
            "n_steps": n_steps,
            "n_dropped_no_bonus": n_dropped_no_bonus,
            "n_dropped_no_tree": n_dropped_no_tree,
            "max_position": max_pos,
            "proposer_name": args.proposer_name,
            "_doc": ("Computed from oracle-vanilla logs (force-accept "
                     "1-token-per-step). Each step's draft tree is "
                     "scored against future bonus tokens — same "
                     "definitions as compute_position_accepts.py: "
                     "seq = greedy walk, ind = any depth-d match, "
                     "cond = P(d | d-1)."),
        },
        "position_accepts": {
            "max_position": max_pos,
            "by_proposer": {args.proposer_name: stats},
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[mtp-postproc] wrote {args.output}", file=sys.stderr)
    print(f"  reqs={n_reqs} steps={n_steps} "
          f"(no_bonus={n_dropped_no_bonus} no_tree={n_dropped_no_tree})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
