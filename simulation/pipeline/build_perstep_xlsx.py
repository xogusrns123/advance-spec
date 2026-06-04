"""Build per-method XLSX from per-step dumps.

Reads dumps at {step_root}/_perstep_dumps/{bench}_s{s}k16_B64.jsonl
and per-method best-config table (from sim sweep results).

For each method, picks the rows matching its best (s, k, B) per benchmark,
concatenates across benchmarks, writes one sheet per method to
{step_root}/excel/method_perstep_data.xlsx.

Sheets:
  00_RunInfo            — per-method best (s,k,B) and aggregate metrics
  basic_reference       — per-step rows for single:eagle3
  oracle_config         — per-step rows for extension_oracle
  dns_config            — per-step rows for extension_dns
  topk_config           — per-step rows for extension_topk
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import xlsxwriter


BENCHES = ["bfcl_v4", "specbench", "swebench_verified"]

# (sheet_name, sim_method_string_substring, best_s_per_bench)
METHOD_CONFIGS = [
    # basic = EAGLE3 backbone + suffix at every anchor (no selection).
    # Best (s,k) determined from summary after re-run.
    ("basic_reference", "extension:4.0:0.0", {
        "bfcl_v4": 2, "specbench": 2, "swebench_verified": 2}),
    ("oracle_config", "extension_oracle:4.0:0.0", {
        "bfcl_v4": 4, "specbench": 4, "swebench_verified": 2}),
    ("dns_config", "extension_dns:0.5:0.8:4.0:0.0", {
        "bfcl_v4": 2, "specbench": 2, "swebench_verified": 2}),
    ("topk_config", "extension_topk:0.5:8:4.0:0.0", {
        "bfcl_v4": 2, "specbench": 2, "swebench_verified": 2}),
]


def load_dump(path: Path, method: str, bench: str) -> List[dict]:
    """Filter rows for given method, tagging with benchmark."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("method") == method:
                r["benchmark"] = bench
                rows.append(r)
    return rows


def write_run_info_sheet(
    workbook, perstep_dir: Path,
) -> None:
    ws = workbook.add_worksheet("00_RunInfo")
    bold = workbook.add_format({"bold": True})
    ws.write_string(0, 0, "Best config per (method, benchmark)", bold)
    cols = ["sheet", "method", "benchmark", "s", "k", "B",
            "n_steps", "mat", "avg_total_step_ms", "avg_target_ms",
            "avg_draft_ms", "avg_ext_size"]
    for j, c in enumerate(cols):
        ws.write_string(1, j, c, bold)

    r_idx = 2
    for sheet_name, method, s_per_bench in METHOD_CONFIGS:
        for bench in BENCHES:
            s = s_per_bench.get(bench)
            if s is None:
                continue
            dump = perstep_dir / f"{bench}_s{s}k16_B64.jsonl"
            rows = load_dump(dump, method, bench)
            if not rows:
                continue
            n = len(rows)
            mat = sum(r["accepted"] for r in rows) / n
            avg_total = sum(r["total_step_ms"] or 0 for r in rows) / n
            avg_target = sum(r["target_ms"] or 0 for r in rows) / n
            avg_draft = sum(r["draft_ms"] or 0 for r in rows) / n
            avg_ext = sum(r["ext_size"] or 0 for r in rows) / n
            for j, c in enumerate(cols):
                v = {"sheet": sheet_name, "method": method,
                     "benchmark": bench, "s": s, "k": 16, "B": 64,
                     "n_steps": n, "mat": round(mat, 4),
                     "avg_total_step_ms": round(avg_total, 4),
                     "avg_target_ms": round(avg_target, 4),
                     "avg_draft_ms": round(avg_draft, 4),
                     "avg_ext_size": round(avg_ext, 2)}.get(c)
                if v is None:
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    ws.write_number(r_idx, j, v)
                else:
                    ws.write_string(r_idx, j, str(v))
            r_idx += 1
    ws.freeze_panes(2, 0)


def write_method_sheet(
    workbook, sheet_name: str, method: str,
    s_per_bench: Dict[str, int], perstep_dir: Path,
) -> int:
    ws = workbook.add_worksheet(sheet_name[:31])
    cols = ["benchmark", "sample_id", "call_idx", "step_id",
            "accepted", "ext_size", "target_ms", "draft_ms",
            "total_step_ms", "s", "k", "B",
            "tree_n_base",
            "tree_token_ids", "tree_parents", "tree_source",
            "tree_is_accepted"]
    bold = workbook.add_format({"bold": True})
    for j, c in enumerate(cols):
        ws.write_string(0, j, c, bold)
    ws.freeze_panes(1, 0)

    r_idx = 1
    for bench in BENCHES:
        s = s_per_bench.get(bench)
        if s is None:
            continue
        dump = perstep_dir / f"{bench}_s{s}k16_B64.jsonl"
        rows = load_dump(dump, method, bench)
        for r in rows:
            row_vals = {
                "benchmark": bench,
                "sample_id": r.get("request_id"),
                "call_idx": r.get("call_idx"),
                "step_id": r.get("step_id"),
                "accepted": r.get("accepted"),
                "ext_size": r.get("ext_size"),
                "target_ms": r.get("target_ms"),
                "draft_ms": r.get("draft_ms"),
                "total_step_ms": r.get("total_step_ms"),
                "s": s, "k": 16, "B": 64,
                "tree_n_base": r.get("tree_n_base"),
                # Tree arrays serialized as compact JSON strings (Excel cell
                # max 32k chars; typical tree ~40 nodes ≈ 250 chars).
                "tree_token_ids": (json.dumps(r["tree_token_ids"],
                                              separators=(",", ":"))
                                   if r.get("tree_token_ids") is not None
                                   else None),
                "tree_parents": (json.dumps(r["tree_parents"],
                                            separators=(",", ":"))
                                 if r.get("tree_parents") is not None
                                 else None),
                "tree_source": (json.dumps(r["tree_source"],
                                           separators=(",", ":"))
                                if r.get("tree_source") is not None
                                else None),
                "tree_is_accepted": (json.dumps(
                    [int(b) for b in r["tree_is_accepted"]],
                    separators=(",", ":"))
                                     if r.get("tree_is_accepted") is not None
                                     else None),
            }
            for j, c in enumerate(cols):
                v = row_vals.get(c)
                if v is None:
                    continue
                if isinstance(v, bool):
                    ws.write_boolean(r_idx, j, v)
                elif isinstance(v, (int, float)):
                    ws.write_number(r_idx, j, v)
                else:
                    # Excel cell max 32,767 chars — guard array columns.
                    if isinstance(v, str) and len(v) > 32760:
                        v = v[:32700] + "...[TRUNCATED]"
                    ws.write_string(r_idx, j, str(v))
            r_idx += 1
    return r_idx - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-root", required=True, type=Path)
    args = ap.parse_args()

    perstep_dir = args.step_root / "_perstep_dumps"
    out_dir = args.step_root / "excel"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "method_perstep_data.xlsx"

    wb = xlsxwriter.Workbook(
        str(out_path), {"constant_memory": True})

    write_run_info_sheet(wb, perstep_dir)

    for sheet_name, method, s_per_bench in METHOD_CONFIGS:
        n = write_method_sheet(
            wb, sheet_name, method, s_per_bench, perstep_dir)
        print(f"{sheet_name}: {n} rows")

    wb.close()
    sz = out_path.stat().st_size / 1024
    print(f"DONE → {out_path} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
