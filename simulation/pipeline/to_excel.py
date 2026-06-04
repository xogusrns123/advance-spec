"""Convert step-dataset files to per-benchmark XLSX workbooks.

For each benchmark, builds an .xlsx with these sheets:
  00_Run_Index            (1 row — this benchmark's run config)
  01_Step_Raw             (~10-32k rows per benchmark)
  02_Depth_Summary        (~30 rows)
  04_AnchorDepth_Summary  (~30 rows)
  05_FeatureBins_Summary  (~150 rows)
  06_Cost_Raw             (same as 01)

SKIPS: 03_Anchor_Raw, 06b_Node_Raw, 01b_Tree_Full — these have millions of
rows and won't fit in a single Excel sheet (Excel max = 1,048,576 rows).

Output: {output_dir}/excel/{benchmark}.xlsx

CLI:
    python -m simulation.pipeline.to_excel \\
        --root /workspace/simulation/results/step_dataset/qwen3_14b
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import xlsxwriter

BENCHES = ["bfcl_v4", "specbench", "swebench_verified"]
EXCEL_ROW_LIMIT = 1_048_576


def _read_jsonl_gz(path_glob: List[Path]) -> Iterator[dict]:
    for p in sorted(path_glob):
        if p.stat().st_size == 0:
            continue
        with gzip.open(p, "rb") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _read_tsv(path: Path) -> Tuple[List[str], List[List[str]]]:
    if not path.exists():
        return [], []
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, [])
        rows = list(reader)
    return header, rows


def _read_run_index_for(path: Path, benchmark: str) -> Tuple[List[str], List[List[str]]]:
    header, rows = _read_tsv(path)
    if not header:
        return header, []
    bench_col = header.index("benchmark") if "benchmark" in header else None
    if bench_col is None:
        return header, rows
    filtered = [r for r in rows if len(r) > bench_col and r[bench_col] == benchmark]
    return header, filtered


def _write_sheet_from_jsonl(
    workbook, sheet_name: str, source_files: List[Path],
    skip_keys: Optional[set] = None,
) -> int:
    """Write a sheet from JSONL.gz files. Returns row count."""
    skip_keys = skip_keys or set()
    ws = workbook.add_worksheet(sheet_name[:31])  # Excel sheet name max 31
    first = True
    header_cols: List[str] = []
    row_idx = 0
    truncated = False
    for r in _read_jsonl_gz(source_files):
        if first:
            header_cols = [k for k in r.keys() if k not in skip_keys]
            for col_idx, col_name in enumerate(header_cols):
                ws.write_string(0, col_idx, col_name)
            ws.freeze_panes(1, 0)
            row_idx = 1
            first = False
        if row_idx >= EXCEL_ROW_LIMIT:
            truncated = True
            break
        for col_idx, col_name in enumerate(header_cols):
            v = r.get(col_name)
            if v is None:
                continue
            if isinstance(v, bool):
                ws.write_boolean(row_idx, col_idx, v)
            elif isinstance(v, (int, float)):
                ws.write_number(row_idx, col_idx, v)
            else:
                ws.write_string(row_idx, col_idx, str(v))
        row_idx += 1
    if truncated:
        ws.write_string(EXCEL_ROW_LIMIT - 1, 0,
                        f"[TRUNCATED — Excel sheet limit {EXCEL_ROW_LIMIT}]")
    return row_idx - 1 if not first else 0


def _write_sheet_from_tsv(
    workbook, sheet_name: str, source_path: Path,
    filter_run_ids: Optional[set] = None,
) -> int:
    """Write a sheet from a TSV. Returns row count."""
    ws = workbook.add_worksheet(sheet_name[:31])
    header, rows = _read_tsv(source_path)
    if not header:
        ws.write_string(0, 0, "[FILE MISSING OR EMPTY]")
        return 0
    if filter_run_ids and "run_id" in header:
        run_col = header.index("run_id")
        rows = [r for r in rows if len(r) > run_col and r[run_col] in filter_run_ids]
    for col_idx, col_name in enumerate(header):
        ws.write_string(0, col_idx, col_name)
    ws.freeze_panes(1, 0)
    for r_idx, r in enumerate(rows, start=1):
        for c_idx, v in enumerate(r):
            if v == "":
                continue
            # Try numeric
            try:
                if "." in v or "e" in v.lower():
                    ws.write_number(r_idx, c_idx, float(v))
                else:
                    ws.write_number(r_idx, c_idx, int(v))
            except ValueError:
                ws.write_string(r_idx, c_idx, v)
    return len(rows)


def build_excel(root: Path, benchmark: str) -> Path:
    bench_dir = root / benchmark
    out_dir = root / "excel"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{benchmark}.xlsx"

    print(f"[xlsx] building {out_path}", file=sys.stderr)
    wb = xlsxwriter.Workbook(
        str(out_path),
        {"constant_memory": True, "default_date_format": "yyyy-mm-dd"})

    # 00_Run_Index — only this benchmark's row(s)
    run_index_path = root / "00_run_index.tsv"
    header, rows = _read_run_index_for(run_index_path, benchmark)
    ws = wb.add_worksheet("00_Run_Index")
    bench_run_ids: set = set()
    if header:
        rid_col = header.index("run_id") if "run_id" in header else None
        for c_idx, col in enumerate(header):
            ws.write_string(0, c_idx, col)
        ws.freeze_panes(1, 0)
        for r_idx, r in enumerate(rows, start=1):
            for c_idx, v in enumerate(r):
                if v == "":
                    continue
                try:
                    if "." in v or "e" in v.lower():
                        ws.write_number(r_idx, c_idx, float(v))
                    else:
                        ws.write_number(r_idx, c_idx, int(v))
                except ValueError:
                    ws.write_string(r_idx, c_idx, v)
            if rid_col is not None and len(r) > rid_col:
                bench_run_ids.add(r[rid_col])
    print(f"  00_Run_Index: {len(rows)} rows (run_ids={bench_run_ids})",
          file=sys.stderr)

    # 01_Step_Raw
    n = _write_sheet_from_jsonl(
        wb, "01_Step_Raw",
        list(bench_dir.glob("01_step_raw.shard*.jsonl.gz")))
    print(f"  01_Step_Raw: {n} rows", file=sys.stderr)

    # 02_Depth_Summary
    n = _write_sheet_from_tsv(
        wb, "02_Depth_Summary",
        bench_dir / "02_depth_summary.tsv",
        filter_run_ids=bench_run_ids)
    print(f"  02_Depth_Summary: {n} rows", file=sys.stderr)

    # 04_AnchorDepth_Summary
    n = _write_sheet_from_tsv(
        wb, "04_AnchorDepth_Summary",
        bench_dir / "04_anchor_depth_summary.tsv",
        filter_run_ids=bench_run_ids)
    print(f"  04_AnchorDepth_Summary: {n} rows", file=sys.stderr)

    # 05_FeatureBins_Summary
    n = _write_sheet_from_tsv(
        wb, "05_FeatureBins_Summary",
        bench_dir / "05_feature_bins_summary.tsv",
        filter_run_ids=bench_run_ids)
    print(f"  05_FeatureBins_Summary: {n} rows", file=sys.stderr)

    # 06_Cost_Raw
    n = _write_sheet_from_jsonl(
        wb, "06_Cost_Raw",
        list(bench_dir.glob("06_cost_raw.shard*.jsonl.gz")))
    print(f"  06_Cost_Raw: {n} rows", file=sys.stderr)

    wb.close()
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"[xlsx] DONE {out_path} ({sz:.1f} MB)", file=sys.stderr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="Root dir containing 00_run_index.tsv and per-benchmark subdirs")
    ap.add_argument("--benchmark", choices=BENCHES + ["all"], default="all")
    args = ap.parse_args()

    benches = BENCHES if args.benchmark == "all" else [args.benchmark]
    for b in benches:
        build_excel(args.root, b)


if __name__ == "__main__":
    main()
