"""Analyze a chain-hybrid PoC run produced by measure_chain_hybrid.py.

Joins the hybrid arm's decision records with per-step accept records on
(rid, decode_step) and prints:
  * per-arm headline: accept_length mean, committed/step, step_ms, draft_ms,
    tokens/s, draft overhead vs baseline
  * decision stats: suffix proposal/chosen/agreement rates (overall + by depth)
  * conditional accept rates by depth, split by chosen proposer.

Join semantics (see chain_hybrid_patch.py): the decision at depth d
(0-based) produced the (d+1)-th draft token of that step's chain, so with
accept_len = a:
    reached(d)  := a >= d      (verify walked through all shallower tokens)
    accepted(d) := a >= d + 1

Usage:
    python simulation/scripts/analyze_chain_hybrid.py \\
        --summary simulation/results/chain_hybrid_poc/poc.json \\
        [--skip-first-frac 0.2] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return records


def arm_headline(arm_row: dict) -> dict:
    out = {
        "wall_time_s": arm_row.get("wall_time_s"),
        "n_steps": arm_row.get("n_samples"),
        "accept_length_mean": arm_row.get("accept_length_mean"),
        "committed_tokens_mean": arm_row.get("committed_tokens_mean"),
        "step_ms": arm_row.get("step_ms"),
        "draft_cost_ms": arm_row.get("draft_cost_ms"),
        "target_cost_ms": arm_row.get("target_cost_ms"),
    }
    committed = out["committed_tokens_mean"]
    step_ms = out["step_ms"]
    if committed and step_ms:
        out["tokens_per_s"] = round(committed / step_ms * 1000.0, 2)
    return out


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    n = len(values)
    pick = lambda q: values[min(n - 1, int(q * n))]  # noqa: E731
    return {"p10": pick(0.10), "p50": pick(0.50), "p90": pick(0.90),
            "mean": round(statistics.mean(values), 4), "n": n}


def analyze_decisions(decision_path: str | Path,
                      skip_first_frac: float) -> dict:
    records = load_jsonl(decision_path)
    decisions = [r for r in records if r.get("type") == "decision"]
    steps = {(r["rid"], r["decode_step"]): r["accept_len"]
             for r in records if r.get("type") == "step"}

    if skip_first_frac > 0 and decisions:
        # Warmup exclusion: drop the earliest fraction (file order == time
        # order) so cold-trie steps don't dilute the suffix-side stats.
        cut = int(len(decisions) * skip_first_frac)
        decisions = decisions[cut:]

    # Tail-append records (depth >= S, chosen always "suffix", suffix_p is a
    # cumulative path prob) would pollute the head selection stats — analyze
    # them separately.
    tail = [d for d in decisions if d.get("tail")]
    decisions = [d for d in decisions if not d.get("tail")]

    n = len(decisions)
    out: dict = {"n_decisions": n, "n_step_records": len(steps),
                 "n_tail_records": len(tail)}
    if n == 0:
        return out

    proposed = [d for d in decisions if d.get("suffix_token") is not None]
    chosen_suffix = [d for d in decisions if d.get("chosen") == "suffix"]
    agreements = [d for d in proposed if d.get("agreement")]

    out["suffix_proposal_rate"] = round(len(proposed) / n, 4)
    out["suffix_chosen_rate"] = round(len(chosen_suffix) / n, 4)
    out["agreement_rate_among_proposals"] = (
        round(len(agreements) / len(proposed), 4) if proposed else None)
    out["suffix_p_when_chosen"] = quantiles(
        [d["suffix_p"] for d in chosen_suffix if d.get("suffix_p") is not None])
    out["match_len_when_chosen"] = quantiles(
        [float(d["match_len"]) for d in chosen_suffix
         if d.get("match_len") is not None])
    out["eagle_p_overall"] = quantiles(
        [d["eagle_p"] for d in decisions if d.get("eagle_p") is not None])

    # ---- per-depth table with accept join --------------------------------
    by_depth: dict[int, dict] = defaultdict(
        lambda: {"n": 0, "suffix_chosen": 0, "suffix_proposed": 0,
                 "reached": {"eagle3": [0, 0], "suffix": [0, 0]}})
    n_joined = 0
    for d in decisions:
        key = (d["rid"], d["decode_step"])
        depth = d["depth"]
        row = by_depth[depth]
        row["n"] += 1
        if d.get("suffix_token") is not None:
            row["suffix_proposed"] += 1
        if d.get("chosen") == "suffix":
            row["suffix_chosen"] += 1
        a = steps.get(key)
        if a is None:
            continue
        n_joined += 1
        if a >= depth:  # verify reached this depth
            acc, tot = row["reached"][d["chosen"]]
            row["reached"][d["chosen"]] = [acc + (1 if a >= depth + 1 else 0),
                                           tot + 1]

    # ---- tail block (route b: suffix run appended beyond S) --------------
    n_tail_joined = 0
    if tail:
        s_depth = min(d["depth"] for d in tail)  # = S (tail starts at S)
        by_pos: dict[int, dict] = defaultdict(
            lambda: {"n": 0, "reached": 0, "accepted": 0})
        per_step_len: dict[tuple, int] = defaultdict(int)
        accepted_total = 0
        for d in tail:
            key = (d["rid"], d["decode_step"])
            per_step_len[key] += 1
            pos = d["depth"] - s_depth
            row = by_pos[pos]
            row["n"] += 1
            a = steps.get(key)
            if a is None:
                continue
            n_tail_joined += 1
            if a >= d["depth"]:
                row["reached"] += 1
                if a >= d["depth"] + 1:
                    row["accepted"] += 1
                    accepted_total += 1
        out["tail"] = {
            "n_records": len(tail),
            "tail_start_depth": s_depth,
            "steps_with_tail_rate": (
                round(len(per_step_len) / len(steps), 4) if steps else None),
            "tail_len": quantiles([float(v) for v in per_step_len.values()]),
            "extra_committed_per_step": (
                round(accepted_total / len(steps), 4) if steps else None),
            "by_position": [
                {"pos": pos, "n": r["n"], "n_reached": r["reached"],
                 "cond_accept": (round(r["accepted"] / r["reached"], 4)
                                 if r["reached"] else None)}
                for pos, r in sorted(by_pos.items())
            ],
        }

    total = n + len(tail)
    out["join_rate"] = (
        round((n_joined + n_tail_joined) / total, 4) if total else None)

    table = []
    for depth in sorted(by_depth):
        row = by_depth[depth]
        e_acc, e_tot = row["reached"]["eagle3"]
        s_acc, s_tot = row["reached"]["suffix"]
        table.append({
            "depth": depth,
            "n": row["n"],
            "suffix_proposed_rate": round(row["suffix_proposed"] / row["n"], 4),
            "suffix_chosen_rate": round(row["suffix_chosen"] / row["n"], 4),
            "cond_accept_eagle3": round(e_acc / e_tot, 4) if e_tot else None,
            "n_reached_eagle3": e_tot,
            "cond_accept_suffix": round(s_acc / s_tot, 4) if s_tot else None,
            "n_reached_suffix": s_tot,
        })
    out["by_depth"] = table
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True,
                        help="poc.json from measure_chain_hybrid.py")
    parser.add_argument("--skip-first-frac", type=float, default=0.0,
                        help="Drop this leading fraction of decision records "
                             "as trie warmup (e.g. 0.2)")
    parser.add_argument("--json", default=None,
                        help="Also write the full analysis as JSON")
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    arms = summary.get("arms", {})

    analysis: dict = {"summary_file": args.summary, "arms": {}}

    print(f"\n=== chain-hybrid PoC: {summary.get('workload')} "
          f"S={summary.get('steps')} topk=1 n_tasks={summary.get('n_tasks')} ===")
    base = arms.get("baseline", {})
    for name, row in arms.items():
        head = arm_headline(row)
        analysis["arms"][name] = head
        if row.get("n_samples"):
            extra = ""
            if name != "baseline" and head.get("draft_cost_ms") \
                    and base.get("draft_cost_ms"):
                delta = head["draft_cost_ms"] - base["draft_cost_ms"]
                head["draft_overhead_vs_baseline_ms"] = round(delta, 3)
                extra = f"  draft_overhead={delta:+.2f}ms"
            print(f"[{name:8s}] accept_mean={head.get('accept_length_mean')} "
                  f"committed/step={head.get('committed_tokens_mean')} "
                  f"step_ms={head.get('step_ms')} "
                  f"tok/s={head.get('tokens_per_s')}{extra}")
        else:
            print(f"[{name:8s}] wall={head.get('wall_time_s')}s (reference, "
                  f"no per-step timing)")

    analysis["decisions"] = {}
    for name, row in arms.items():
        if not (isinstance(row, dict) and row.get("decision_log")):
            continue
        dec = analyze_decisions(row["decision_log"], args.skip_first_frac)
        analysis["decisions"][name] = dec
        print(f"\n--- decisions [{name}] (n={dec.get('n_decisions')}, "
              f"tail={dec.get('n_tail_records')}, "
              f"join_rate={dec.get('join_rate')}, "
              f"skip_first_frac={args.skip_first_frac}) ---")
        print(f"suffix proposal rate: {dec.get('suffix_proposal_rate')}   "
              f"chosen rate: {dec.get('suffix_chosen_rate')}   "
              f"agreement|proposed: {dec.get('agreement_rate_among_proposals')}")
        tbl = dec.get("by_depth") or []
        if tbl:
            hdr = (f"{'depth':>5} {'n':>7} {'sfx_prop':>9} {'sfx_chosen':>10} "
                   f"{'acc|eagle3':>11} {'(n)':>7} {'acc|suffix':>11} {'(n)':>7}")
            print(hdr)
            for r in tbl:
                print(f"{r['depth']:>5} {r['n']:>7} "
                      f"{r['suffix_proposed_rate']:>9} "
                      f"{r['suffix_chosen_rate']:>10} "
                      f"{str(r['cond_accept_eagle3']):>11} "
                      f"{r['n_reached_eagle3']:>7} "
                      f"{str(r['cond_accept_suffix']):>11} "
                      f"{r['n_reached_suffix']:>7}")
        t = dec.get("tail")
        if t:
            print(f"tail: start_depth={t['tail_start_depth']} "
                  f"steps_with_tail={t['steps_with_tail_rate']} "
                  f"len={t['tail_len']} "
                  f"extra_committed/step={t['extra_committed_per_step']}")
            print(f"{'pos':>5} {'n':>7} {'reached':>8} {'cond_accept':>12}")
            for r in t["by_position"]:
                print(f"{r['pos']:>5} {r['n']:>7} {r['n_reached']:>8} "
                      f"{str(r['cond_accept']):>12}")
    if not analysis["decisions"]:
        print("\n(no hybrid decision log in summary — decision analysis skipped)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
