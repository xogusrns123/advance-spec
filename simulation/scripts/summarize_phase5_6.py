"""Consolidate Stage 6 outputs across (workload, steps) artifacts.

Takes a list of run directories, reads each ``tree_oracle_sim.json``, and
emits:
  * A CSV table: (workload, steps, budget, method, mat, speedup_real, …)
  * A summary text report with per-method best budget and headline speedups

Usage:
    python3 simulation/scripts/summarize_phase5_6.py \\
        --runs simulation/results/qwen3_14b/specbench_steps{2,4,6,8} \\
        --output simulation/results/qwen3_14b/phase6_summary.csv \\
        --report simulation/results/qwen3_14b/phase6_report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

METHOD_RE = re.compile(r"_(mat|steps|speedup_r[0-9.]+|speedup_real)$")
LATENCY_FIELDS = {"budget", "target_forward_ms", "eagle3_draft_ms",
                  "draft_lm_tpot_ms"}


def parse_run_dir(path: Path) -> dict:
    """Extract (workload, steps) from a run directory like specbench_steps2."""
    name = path.name
    m = re.match(r"(.+)_steps(\d+)", name)
    if not m:
        return {"workload": name, "steps": None}
    return {"workload": m.group(1), "steps": int(m.group(2))}


def extract_methods(row: dict) -> dict[str, dict[str, float]]:
    """Turn a flat budget_sweep row into {method: {field: value}} form."""
    methods: dict[str, dict[str, float]] = {}
    for k, v in row.items():
        if k in LATENCY_FIELDS:
            continue
        m = METHOD_RE.search(k)
        if not m:
            continue
        field = m.group(1)
        method = k[: m.start()]
        methods.setdefault(method, {})[field] = v
    return methods


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run dirs containing tree_oracle_sim.json")
    p.add_argument("--output", required=True,
                   help="CSV output path")
    p.add_argument("--report", required=True,
                   help="Text report output path")
    p.add_argument("--speedup-ref", default="speedup_real",
                   help="Which speedup column to highlight in report "
                        "(speedup_real, speedup_r0.05, etc.)")
    args = p.parse_args()

    rows = []
    for run in args.runs:
        rd = Path(run)
        meta = parse_run_dir(rd)
        f = rd / "tree_oracle_sim.json"
        if not f.exists():
            print(f"SKIP: {f} not found")
            continue
        data = json.loads(f.read_text())
        sweep = data.get("latency", {}).get("budget_sweep", [])
        n_steps = data.get("metadata", {}).get("n_steps")
        for entry in sweep:
            B = entry.get("budget")
            for method, fields in extract_methods(entry).items():
                rows.append({
                    "workload": meta["workload"],
                    "steps": meta["steps"],
                    "budget": B,
                    "method": method,
                    "mat": fields.get("mat"),
                    "speedup_r0.05": fields.get("speedup_r0.05"),
                    "speedup_r0.1": fields.get("speedup_r0.1"),
                    "speedup_r0.2": fields.get("speedup_r0.2"),
                    "speedup_real": fields.get("speedup_real"),
                    "n_steps": n_steps,
                })

    # Write CSV
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out}")

    # Build report
    ref = args.speedup_ref
    methods = sorted({r["method"] for r in rows})
    configs = sorted({(r["workload"], r["steps"]) for r in rows})
    lines = []
    lines.append("=" * 78)
    lines.append(f"Phase 5/6 Summary — Qwen3-14B — speedup metric: {ref}")
    lines.append("=" * 78)
    lines.append(f"Configurations: {len(configs)} = {configs}")
    lines.append(f"Methods: {len(methods)} total")
    lines.append(f"Data rows: {len(rows)}")
    lines.append("")

    # Best-budget-per-method table per config
    for (workload, steps) in configs:
        lines.append("-" * 78)
        lines.append(f"[{workload} steps={steps}]  Best budget per method ({ref}):")
        lines.append(f"  {'method':<40} {'best_budget':>12} {'best_'+ref:>18} {'mat@best':>10}")
        for m in methods:
            sub = [r for r in rows
                   if r["workload"] == workload and r["steps"] == steps
                   and r["method"] == m and r[ref] is not None]
            if not sub:
                continue
            best = max(sub, key=lambda r: r[ref] or 0.0)
            s = best[ref]
            mat = best["mat"]
            lines.append(f"  {m:<40} {best['budget']:>12} "
                         f"{(f'{s:.3f}' if s else 'n/a'):>18} "
                         f"{(f'{mat:.3f}' if mat else 'n/a'):>10}")
        lines.append("")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Wrote report → {report_path}")

    # Brief stdout summary
    print()
    print("=" * 60)
    print(f"Brief top-methods for {ref}:")
    print("=" * 60)
    for (workload, steps) in configs:
        subset = [r for r in rows
                  if r["workload"] == workload and r["steps"] == steps
                  and r[ref] is not None]
        if not subset:
            continue
        top = sorted(subset, key=lambda r: r[ref] or 0.0, reverse=True)[:5]
        print(f"\n[{workload} steps={steps}] top 5 by {ref}:")
        for t in top:
            print(f"  {t['method']:<36} B={t['budget']:>4}  "
                  f"{ref}={t[ref]:.3f}  mat={t['mat']:.3f}")


if __name__ == "__main__":
    main()
