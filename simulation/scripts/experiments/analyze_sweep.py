#!/usr/bin/env python3
"""Aggregate sim sweep results — best-of-family per (capture, budget).

Reads tree_oracle_sim JSON files (any extension/hybrid sweep output) and emits
a sortable comparison table.

For each (capture, budget) combination:
  * MAT and spd_real for every method run
  * Best deployable extension_sfx (i.e. spd_real of best non-oracle variant)
  * Best oracle ceiling (extension*_oracle)
  * Best hybrid baseline (hybrid_e3:t)
  * Gap = best_extension / best_hybrid - 1 (positive = extension wins)

Usage:
    python3 simulation/scripts/experiments/analyze_sweep.py \
        [--in-dir simulation/results/explorations] [--csv out.csv]
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict


def find_methods_and_speedups(b: dict) -> dict[str, dict]:
    """Extract {method_prefix: {mat, spd_real, ext_size_avg}} for one budget row."""
    out: dict[str, dict] = {}
    for k in b:
        if not k.endswith("_speedup_real"):
            continue
        prefix = k[: -len("_speedup_real")]
        mat = b.get(f"{prefix}_mat")
        steps = b.get(f"{prefix}_steps", 0)
        target_tokens = b.get(f"{prefix}_total_target_tokens", 0)
        target_ms = b.get(f"{prefix}_total_target_ms", 0)
        draft_ms = b.get(f"{prefix}_total_draft_ms", 0)
        avg_ext = (target_tokens / steps) if steps else 0
        out[prefix] = {
            "mat": mat,
            "spd_real": b[k],
            "steps": steps,
            "avg_ext_size": avg_ext,
            "avg_target_ms": (target_ms / steps) if steps else 0,
            "avg_draft_ms": (draft_ms / steps) if steps else 0,
        }
    return out


def best_in_group(methods: dict, predicate) -> tuple[str, dict] | tuple[None, None]:
    """Return (name, stats) of method with highest spd_real matching predicate."""
    best_name, best_stats = None, None
    for name, stats in methods.items():
        if not predicate(name):
            continue
        if stats["spd_real"] is None:
            continue
        if best_stats is None or stats["spd_real"] > best_stats["spd_real"]:
            best_name, best_stats = name, stats
    return best_name, best_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",
                   default="/home/muchwater/advance-spec/simulation/results/explorations")
    p.add_argument("--csv", default=None,
                   help="Write per-(capture,budget,method) full table to this CSV")
    p.add_argument("--summary-csv", default=None,
                   help="Write per-(capture,budget) summary to this CSV")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.json"))
    if not files:
        sys.exit(f"No json found in {in_dir}")

    # Per-row records for full CSV
    rows: list[dict] = []
    # (capture, budget) -> [best_ext, best_ext_oracle, best_hybrid, eagle3, suffix, dm]
    summary_rows: list[dict] = []

    for f in files:
        d = json.loads(f.read_text())
        capture = f.stem  # e.g. qwen3_14b_specbench_steps8
        bs = d.get("latency", {}).get("budget_sweep", [])
        for b in bs:
            B = b.get("budget")
            methods = find_methods_and_speedups(b)

            for name, st in methods.items():
                rows.append({
                    "capture": capture, "budget": B, "method": name,
                    "mat": st["mat"], "spd_real": st["spd_real"],
                    "steps": st["steps"],
                    "avg_ext_size": round(st["avg_ext_size"], 1),
                    "avg_target_ms": round(st["avg_target_ms"], 2),
                    "avg_draft_ms": round(st["avg_draft_ms"], 2),
                })

            # Best deployable extension_sfx (non-oracle, all extension_sfx_*)
            best_sfx_dep, sfx_dep_st = best_in_group(
                methods,
                lambda n: n.startswith("extension_sfx_") and "oracle" not in n)
            # Best deployable extension (any extension_*, no oracle)
            best_ext_dep, ext_dep_st = best_in_group(
                methods,
                lambda n: n.startswith("extension") and "oracle" not in n)
            # Best extension oracle (ceiling)
            best_ext_or, ext_or_st = best_in_group(
                methods, lambda n: "extension" in n and "oracle" in n)
            # Best hybrid baseline
            best_hyb, hyb_st = best_in_group(
                methods, lambda n: n.startswith("hybrid_e3"))
            # eagle3 baseline
            _, eagle3_st = best_in_group(
                methods, lambda n: n == "eagle3")
            # suffix baseline
            _, suffix_st = best_in_group(
                methods, lambda n: n == "suffix")

            row = {
                "capture": capture, "budget": B,
                "best_ext_sfx_dep": best_sfx_dep,
                "best_ext_sfx_dep_spd": sfx_dep_st["spd_real"] if sfx_dep_st else None,
                "best_ext_sfx_dep_mat": sfx_dep_st["mat"] if sfx_dep_st else None,
                "best_ext_dep": best_ext_dep,
                "best_ext_dep_spd": ext_dep_st["spd_real"] if ext_dep_st else None,
                "best_ext_oracle": best_ext_or,
                "best_ext_oracle_spd": ext_or_st["spd_real"] if ext_or_st else None,
                "best_ext_oracle_mat": ext_or_st["mat"] if ext_or_st else None,
                "best_hybrid": best_hyb,
                "best_hybrid_spd": hyb_st["spd_real"] if hyb_st else None,
                "best_hybrid_mat": hyb_st["mat"] if hyb_st else None,
                "eagle3_spd": eagle3_st["spd_real"] if eagle3_st else None,
                "suffix_spd": suffix_st["spd_real"] if suffix_st else None,
            }
            # Gaps
            if row["best_ext_sfx_dep_spd"] and row["best_hybrid_spd"]:
                row["gap_sfx_vs_hybrid_pct"] = round(
                    (row["best_ext_sfx_dep_spd"] / row["best_hybrid_spd"] - 1) * 100, 2)
            if row["best_ext_dep_spd"] and row["best_hybrid_spd"]:
                row["gap_ext_vs_hybrid_pct"] = round(
                    (row["best_ext_dep_spd"] / row["best_hybrid_spd"] - 1) * 100, 2)
            if row["best_ext_oracle_spd"] and row["best_hybrid_spd"]:
                row["gap_oracle_vs_hybrid_pct"] = round(
                    (row["best_ext_oracle_spd"] / row["best_hybrid_spd"] - 1) * 100, 2)
            summary_rows.append(row)

    # --- Console summary ---
    print("\n" + "=" * 110)
    print("EXTENSION_SFX SWEEP SUMMARY — best-per-class spd_real per (capture, budget)")
    print("=" * 110)
    print(f"{'capture':<35} {'B':>4} {'sfx_dep':>10} {'ext_dep':>10} "
          f"{'ext_or':>10} {'hybrid':>10} {'gap_sfx%':>9} {'gap_or%':>9}")
    print("-" * 110)
    for r in summary_rows:
        cap_short = r["capture"][:34]
        sfx = f"{r['best_ext_sfx_dep_spd']:>10.3f}" if r['best_ext_sfx_dep_spd'] is not None else f"{'--':>10}"
        ext = f"{r['best_ext_dep_spd']:>10.3f}" if r['best_ext_dep_spd'] is not None else f"{'--':>10}"
        ora = f"{r['best_ext_oracle_spd']:>10.3f}" if r['best_ext_oracle_spd'] is not None else f"{'--':>10}"
        hyb = f"{r['best_hybrid_spd']:>10.3f}" if r['best_hybrid_spd'] is not None else f"{'--':>10}"
        g_sfx = f"{r.get('gap_sfx_vs_hybrid_pct', 0):>+9.2f}" if r.get('gap_sfx_vs_hybrid_pct') is not None else f"{'--':>9}"
        g_or = f"{r.get('gap_oracle_vs_hybrid_pct', 0):>+9.2f}" if r.get('gap_oracle_vs_hybrid_pct') is not None else f"{'--':>9}"
        print(f"{cap_short:<35} {r['budget']:>4} {sfx} {ext} {ora} {hyb} {g_sfx} {g_or}")

    # --- Best sfx config per (capture, budget) ---
    print("\n" + "=" * 110)
    print("BEST extension_sfx VARIANT per (capture, budget)")
    print("=" * 110)
    print(f"{'capture':<35} {'B':>4} {'best_variant':<35} {'spd':>8} {'mat':>6}")
    print("-" * 110)
    for r in summary_rows:
        if r["best_ext_sfx_dep"]:
            print(f"{r['capture'][:34]:<35} {r['budget']:>4} "
                  f"{r['best_ext_sfx_dep'][:34]:<35} "
                  f"{r['best_ext_sfx_dep_spd']:>8.3f} {r['best_ext_sfx_dep_mat']:>6.2f}")

    # --- Detailed per-method dump for cases where extension wins ---
    wins = [r for r in summary_rows
            if r.get("gap_sfx_vs_hybrid_pct", -100) > 0
            or r.get("gap_oracle_vs_hybrid_pct", -100) > 0]
    if wins:
        print("\n" + "=" * 110)
        print("WINS — capture/budget where any extension variant beats hybrid")
        print("=" * 110)
        for r in wins:
            print(f"  {r['capture']} B={r['budget']}: "
                  f"sfx_gap={r.get('gap_sfx_vs_hybrid_pct')}%, "
                  f"oracle_gap={r.get('gap_oracle_vs_hybrid_pct')}%, "
                  f"best_sfx={r.get('best_ext_sfx_dep')}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nFull table written to {args.csv}")
    if args.summary_csv:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Summary written to {args.summary_csv}")


if __name__ == "__main__":
    main()
