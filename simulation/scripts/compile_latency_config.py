"""Compile measured target / draft / suffix costs into a Stage 6 latency_config.json.

Reads the three per-measurement JSONs produced by the ``measure_*_cost.py``
scripts and emits a single config that ``run_tree_oracle_sim.py`` (Stage 6)
consumes. Also emits richer ``_detailed`` nesting for analysis notebooks.

Aggregation rules:
  * target_forward_ms_by_topk[K][B]        median across (workload, steps) per (K, B)
  * eagle3_draft_ms_by_topk_steps[K][S][B] median across workloads per (K, S, B)
  * target_forward_ms[B]   — legacy flat: median across (workload, steps, topk)
  * eagle3_draft_ms[B]     — legacy flat: at --canonical-steps & --canonical-topk
  * eagle3_draft_ms_by_steps[S][B] — legacy flat: at --canonical-topk
  * draft_lm_tpot_ms       per_token_ms at --draft-ref-n (default 3)
                           from measure_draft_model_cost.json, median across workloads
  * suffix_speculate_ms    median across workloads
  * vanilla_step_ms        --vanilla-tpot-ms CLI override, else target_forward_ms[B_min]

Usage:
    python3 simulation/scripts/compile_latency_config.py \\
        --eagle3-cost   results/latency/eagle3_cost.json \\
        --draft-cost    results/latency/draft_model_cost.json \\
        --suffix-cost   results/latency/suffix_cost.json \\
        --canonical-steps 4 \\
        --draft-ref-n 3 \\
        --output        simulation/results/qwen3_14b/latency_config.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _median_by_budget(
    rows: list[dict], key: str, budget_field: str = "budget",
) -> dict[str, float]:
    """Group rows by budget, take median of ``rows[i][key]``."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if key in r and r[key] is not None and budget_field in r:
            try:
                grouped[int(r[budget_field])].append(float(r[key]))
            except (TypeError, ValueError):
                continue
    return {str(b): round(statistics.median(vs), 4)
            for b, vs in grouped.items() if vs}


def _filter_rows(rows: list[dict], **filters) -> list[dict]:
    """Keep rows where every filter field matches."""
    def ok(r):
        for k, v in filters.items():
            if r.get(k) != v:
                return False
        return True
    return [r for r in rows if ok(r) and "error" not in r]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eagle3-cost", required=True,
                   help="measure_eagle3_cost.py output JSON")
    p.add_argument("--draft-cost", default=None,
                   help="measure_draft_model_cost.py output JSON (optional)")
    p.add_argument("--suffix-cost", default=None,
                   help="measure_suffix_cost.py output JSON (optional)")
    p.add_argument("--canonical-steps", type=int, default=4,
                   help="Steps value used for the flat eagle3_draft_ms table")
    p.add_argument("--canonical-topk", type=int, default=16,
                   help="Topk value used for the flat legacy tables. "
                        "Only matters when the eagle3_cost.json has >1 topk.")
    p.add_argument("--draft-ref-n", type=int, default=3,
                   help="num_draft_tokens value used to derive per-token TPOT")
    p.add_argument("--vanilla-tpot-ms", type=float, default=None,
                   help="Explicit vanilla TPOT; if omitted, derived from "
                        "target_forward_ms at the smallest budget")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    eagle3 = json.loads(Path(args.eagle3_cost).read_text())
    eagle3_rows = eagle3.get("results", [])
    if not eagle3_rows:
        sys.exit("ERROR: no rows in eagle3_cost file")

    # Pull topk from row (new schema) or file-level (legacy). Default 16.
    file_level_topk = eagle3.get("topk")

    def _row_topk(r: dict) -> int:
        if "topk" in r and r["topk"] is not None:
            return int(r["topk"])
        if file_level_topk is not None:
            return int(file_level_topk)
        return 16

    # Stamp per-row topk for consistent downstream filtering.
    for r in eagle3_rows:
        if "error" in r:
            continue
        r.setdefault("topk", _row_topk(r))

    available_steps = sorted({int(r["steps"]) for r in eagle3_rows
                              if "steps" in r})
    available_budgets = sorted({int(r["budget"]) for r in eagle3_rows
                                if "budget" in r})
    available_topks = sorted({int(r["topk"]) for r in eagle3_rows
                              if "topk" in r})
    print(f"eagle3_cost: {len(eagle3_rows)} rows, topks={available_topks}, "
          f"steps={available_steps}, budgets={available_budgets}",
          file=sys.stderr)

    # Topk-aware tables
    # target_forward_ms_by_topk[K][B] — median across (workload, steps) per (K, B)
    target_forward_ms_by_topk: dict[str, dict[str, float]] = {}
    for K in available_topks:
        subset = _filter_rows(eagle3_rows, topk=K)
        target_forward_ms_by_topk[str(K)] = _median_by_budget(
            subset, "target_cost_ms")

    # eagle3_draft_ms_by_topk_steps[K][S][B] — median across workloads per (K, S, B)
    eagle3_draft_ms_by_topk_steps: dict[str, dict[str, dict[str, float]]] = {}
    for K in available_topks:
        per_step: dict[str, dict[str, float]] = {}
        for S in available_steps:
            subset = _filter_rows(eagle3_rows, topk=K, steps=S)
            per_step[str(S)] = _median_by_budget(subset, "draft_cost_ms")
        eagle3_draft_ms_by_topk_steps[str(K)] = per_step

    # Legacy flat target_forward_ms[B] — median across (workload, steps, topk)
    target_forward_ms = _median_by_budget(eagle3_rows, "target_cost_ms")

    # Canonical topk for legacy flat tables
    canon_topk = args.canonical_topk
    if canon_topk not in available_topks:
        closest_k = min(available_topks, key=lambda k: abs(k - canon_topk))
        print(f"WARN: canonical_topk={canon_topk} not measured; using closest={closest_k}",
              file=sys.stderr)
        canon_topk = closest_k

    # Legacy eagle3_draft_ms_by_steps[S][B] — at canonical topk
    eagle3_draft_ms_by_steps = eagle3_draft_ms_by_topk_steps.get(
        str(canon_topk), {})

    # Canonical flat eagle3_draft_ms[B] — from canonical_steps at canonical_topk
    canon = args.canonical_steps
    if canon not in available_steps:
        closest = min(available_steps, key=lambda s: abs(s - canon))
        print(f"WARN: canonical_steps={canon} not measured; using closest={closest}",
              file=sys.stderr)
        canon = closest
    eagle3_draft_ms = eagle3_draft_ms_by_steps.get(str(canon), {})

    # Vanilla TPOT
    if args.vanilla_tpot_ms is not None:
        vanilla_step_ms = float(args.vanilla_tpot_ms)
    elif target_forward_ms:
        smallest_b = min(int(b) for b in target_forward_ms.keys())
        vanilla_step_ms = target_forward_ms[str(smallest_b)]
        print(f"Derived vanilla_step_ms={vanilla_step_ms:.3f} "
              f"from target_forward_ms[{smallest_b}]", file=sys.stderr)
    else:
        sys.exit("ERROR: cannot derive vanilla_step_ms (no target data)")

    # Draft LM TPOT
    draft_lm_tpot_ms: Optional[float] = None
    draft_lm_tpot_by_n: dict[str, float] = {}
    draft_lm_src: Optional[str] = None
    if args.draft_cost:
        dm = json.loads(Path(args.draft_cost).read_text())
        dm_rows = dm.get("results", [])
        # per-N median across workloads
        by_n: dict[int, list[float]] = defaultdict(list)
        for r in dm_rows:
            if "error" in r or "per_token_ms" not in r:
                continue
            try:
                by_n[int(r["num_draft_tokens"])].append(float(r["per_token_ms"]))
            except (TypeError, ValueError):
                continue
        for n, vs in by_n.items():
            draft_lm_tpot_by_n[str(n)] = round(statistics.median(vs), 4)
        if args.draft_ref_n in by_n:
            draft_lm_tpot_ms = draft_lm_tpot_by_n[str(args.draft_ref_n)]
        elif by_n:
            # fall back to largest N (closest to asymptote)
            largest = max(by_n.keys())
            draft_lm_tpot_ms = draft_lm_tpot_by_n[str(largest)]
            print(f"WARN: draft_ref_n={args.draft_ref_n} not measured; "
                  f"using N={largest}", file=sys.stderr)
        draft_lm_src = dm.get("model")

    # Suffix speculate cost
    suffix_speculate_ms: Optional[float] = None
    suffix_by_workload: dict[str, float] = {}
    if args.suffix_cost:
        sf = json.loads(Path(args.suffix_cost).read_text())
        sf_rows = sf.get("results", [])
        vs = [float(r["speculate_ms"]) for r in sf_rows
              if r.get("speculate_ms") is not None]
        if vs:
            suffix_speculate_ms = round(statistics.median(vs), 4)
        for r in sf_rows:
            if r.get("speculate_ms") is not None:
                suffix_by_workload[r["workload"]] = round(
                    float(r["speculate_ms"]), 4)

    output = {
        "vanilla_step_ms": round(vanilla_step_ms, 4),
        "target_forward_ms": target_forward_ms,
        "eagle3_draft_ms": eagle3_draft_ms,
        "eagle3_draft_ms_by_steps": eagle3_draft_ms_by_steps,
        "target_forward_ms_by_topk": target_forward_ms_by_topk,
        "eagle3_draft_ms_by_topk_steps": eagle3_draft_ms_by_topk_steps,
        "draft_lm_tpot_ms": draft_lm_tpot_ms,
        "suffix_speculate_ms": suffix_speculate_ms,
        "_metadata": {
            "target_model": eagle3.get("model"),
            "draft_model": eagle3.get("draft_model"),
            "draft_lm": draft_lm_src,
            "canonical_steps": canon,
            "canonical_topk": canon_topk,
            "draft_ref_n": args.draft_ref_n,
            "available_topks": available_topks,
            "available_steps": available_steps,
            "available_budgets": available_budgets,
            "aggregation": "median across workloads (+steps for target_forward_ms)",
            "source_files": {
                "eagle3_cost": str(Path(args.eagle3_cost).resolve()),
                "draft_model_cost": (
                    str(Path(args.draft_cost).resolve()) if args.draft_cost else None),
                "suffix_cost": (
                    str(Path(args.suffix_cost).resolve()) if args.suffix_cost else None),
            },
        },
        "_detailed": {
            "draft_lm_tpot_ms_by_n": draft_lm_tpot_by_n,
            "suffix_speculate_ms_by_workload": suffix_by_workload,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Summary ---
    print("\n" + "=" * 68, file=sys.stderr)
    print("COMPILED LATENCY CONFIG", file=sys.stderr)
    print("=" * 68, file=sys.stderr)
    print(f"vanilla_step_ms:      {vanilla_step_ms:.3f}", file=sys.stderr)
    print(f"draft_lm_tpot_ms:     {draft_lm_tpot_ms}", file=sys.stderr)
    print(f"suffix_speculate_ms:  {suffix_speculate_ms}", file=sys.stderr)
    print(f"canonical_topk:       {canon_topk}", file=sys.stderr)
    print(f"canonical_steps:      {canon}", file=sys.stderr)
    print(f"{'budget':>7} | {'target_fwd_ms':>13} | {'e3_draft_ms':>12}",
          file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    for b in sorted(target_forward_ms.keys(), key=int):
        tfm = target_forward_ms.get(b, None)
        e3d = eagle3_draft_ms.get(b, None)
        print(f"{b:>7} | "
              f"{tfm if tfm is None else f'{tfm:>13.3f}'} | "
              f"{e3d if e3d is None else f'{e3d:>12.3f}'}",
              file=sys.stderr)

    # Per-topk target_forward_ms preview
    for K in available_topks:
        tbl = target_forward_ms_by_topk.get(str(K), {})
        if not tbl:
            continue
        print(f"\ntarget_forward_ms @ topk={K}:", file=sys.stderr)
        for b in sorted(tbl.keys(), key=int):
            print(f"  B={b:>4}  {tbl[b]:>7.3f} ms", file=sys.stderr)
    print(f"\nOutput: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
