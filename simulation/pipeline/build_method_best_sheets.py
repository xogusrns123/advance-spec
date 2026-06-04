"""Build per-method best-config XLSX from sim outputs.

Reads sim outputs at {step_root}/_sims/{bench}_s{s}k{k}.json.
For each method, finds argmax over (s,k,B) by `speedup_real`.

Outputs `{step_root}/excel/method_best_config.xlsx` with sheets:
  basic_reference      — single:eagle3 best (s,k,B) per benchmark
  oracle_config        — extension_oracle best (s,k,B) per benchmark
  dns_config           — extension_dns best (s,k,B) per benchmark
  topk_config          — extension_topk best (s,k,B) per benchmark

Each sheet has:
  Summary section: per-benchmark best (s,k,B) + key metrics
  Sweep section: every (s,k,B) tested for that method (so user can see
                 the surface around the argmax)

Methods are identified by prefix match on the method-string key in the
sim JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xlsxwriter


METHOD_FAMILIES = {
    "basic_reference": "eagle3",
    "oracle_config": "extension_oracle_",
    "dns_config": "extension_dns_",
    "topk_config": "extension_topk_",
}

BENCHES = ["bfcl_v4", "specbench", "swebench_verified"]

# Fields we copy from each per-budget entry, prefixed with method name.
# These are stripped of the method prefix and become per-row columns.
PER_METHOD_FIELDS = [
    "mat",
    "speedup_real",
    "speedup_r0.05",
    "steps",
    "total_target_ms",
    "total_draft_ms",
    "total_target_tokens",
]


def load_sim(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_method_keys(sweep_entry: dict, family_prefix: str) -> List[str]:
    """Find method strings matching the family prefix.

    Sim normalizes method strings: ``extension_dns:0.5:0.8:4.0:0.0`` →
    ``extension_dns_l0.5_r0.8_f4.0_t0.0``. Keys in sweep_entry look like
    ``<method>_mat``, ``<method>_speedup_real``, etc.
    """
    methods = set()
    for k in sweep_entry.keys():
        for f in PER_METHOD_FIELDS:
            if k.endswith("_" + f):
                m = k[:-(len(f) + 1)]
                # Exact equality for `eagle3` (basic) — guard against
                # `eagle3_speedup_*` slipping into other matches.
                if family_prefix == "eagle3":
                    if m == "eagle3":
                        methods.add(m)
                # Family-prefix match, with guard: extension_topk_ should NOT
                # match extension_topk_match_*, extension_topk_cap_*.
                elif m.startswith(family_prefix):
                    rest = m[len(family_prefix):]
                    # rest must start with a param letter (l, k, f, r, t).
                    if rest and rest[0] in "lkfrt":
                        methods.add(m)
    return sorted(methods)


def gather_sweep_rows(
    sims_dir: Path, family_prefix: str,
) -> List[Dict[str, Any]]:
    """Across all (bench, s, k) sims, collect one row per (method, bench,
    s, k, B) with key metrics."""
    rows: List[Dict[str, Any]] = []
    for bench in BENCHES:
        for sim_path in sorted(sims_dir.glob(f"{bench}_s*k*.json")):
            sd = load_sim(sim_path)
            # Filename: bfcl_v4_s2k16.json
            stem = sim_path.stem  # bfcl_v4_s2k16
            sk_part = stem.split("_")[-1]  # s2k16
            s = int(sk_part.split("k")[0][1:])
            k = int(sk_part.split("k")[1])

            sweep = sd.get("latency", {}).get("budget_sweep", [])
            vanilla = sd.get("latency", {}).get("vanilla_step_ms")
            for entry in sweep:
                B = entry.get("budget")
                # Find methods of this family in this entry
                methods = get_method_keys(entry, family_prefix)
                for m in methods:
                    row = {
                        "benchmark": bench,
                        "method": m,
                        "s": s, "k": k, "B": B,
                        "vanilla_step_ms": vanilla,
                    }
                    for f in PER_METHOD_FIELDS:
                        row[f] = entry.get(f"{m}_{f}")
                    rows.append(row)
    return rows


def find_best_per_bench(
    rows: List[Dict[str, Any]], metric: str = "speedup_real",
) -> Dict[str, Dict[str, Any]]:
    """Per benchmark: pick the row with max metric."""
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        b = r["benchmark"]
        cur = best.get(b)
        v = r.get(metric)
        if v is None:
            continue
        if cur is None or (cur.get(metric) is None) or v > cur[metric]:
            best[b] = r
    return best


def write_sheet(
    workbook, sheet_name: str, rows: List[Dict[str, Any]],
    best: Dict[str, Dict[str, Any]],
) -> None:
    ws = workbook.add_worksheet(sheet_name[:31])
    bold = workbook.add_format({"bold": True})

    # Section 1: Summary (best per benchmark)
    ws.write_string(0, 0, "Best config per benchmark (argmax speedup_real)",
                    bold)
    cols = ["benchmark", "method", "s", "k", "B", "mat", "speedup_real",
            "speedup_r0.05", "vanilla_step_ms", "total_target_ms",
            "total_draft_ms", "steps", "total_target_tokens"]
    for j, c in enumerate(cols):
        ws.write_string(1, j, c, bold)
    r_idx = 2
    for b in BENCHES:
        row = best.get(b)
        if row is None:
            continue
        for j, c in enumerate(cols):
            v = row.get(c)
            if v is None:
                continue
            if isinstance(v, (int, float)):
                ws.write_number(r_idx, j, v)
            else:
                ws.write_string(r_idx, j, str(v))
        r_idx += 1

    # Section 2: Full sweep
    sweep_start = r_idx + 2
    ws.write_string(sweep_start, 0, "Full (s, k, B) sweep", bold)
    sweep_start += 1
    for j, c in enumerate(cols):
        ws.write_string(sweep_start, j, c, bold)
    sweep_start += 1
    for row in sorted(rows, key=lambda r: (
            r["benchmark"], r["s"], r["k"], r["B"], r["method"])):
        for j, c in enumerate(cols):
            v = row.get(c)
            if v is None:
                continue
            if isinstance(v, (int, float)):
                ws.write_number(sweep_start, j, v)
            else:
                ws.write_string(sweep_start, j, str(v))
        sweep_start += 1

    ws.freeze_panes(2, 0)
    ws.autofit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-root", required=True, type=Path)
    args = ap.parse_args()

    sims_dir = args.step_root / "_sims"
    out_dir = args.step_root / "excel"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "method_best_config.xlsx"

    wb = xlsxwriter.Workbook(str(out_path))

    for sheet_name, prefix in METHOD_FAMILIES.items():
        rows = gather_sweep_rows(sims_dir, prefix)
        best = find_best_per_bench(rows)
        summary = {b: f"s{best[b]['s']}/k{best[b]['k']}/B{best[b]['B']}@{best[b]['speedup_real']:.2f}x"
                   for b in best}
        print(f"{sheet_name}: {len(rows)} sweep rows, best per bench: {summary}")
        write_sheet(wb, sheet_name, rows, best)

    wb.close()
    sz = out_path.stat().st_size / 1024
    print(f"DONE → {out_path} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
