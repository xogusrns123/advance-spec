"""Build single-file XLSX for BFCLv4 full per-step + tree-array analysis.

Reads 2 per-step JSONL dumps (one per unique config: s=2 and s=4 at k=16, B=64,
basic extension method) and emits one XLSX with sheets:

  00_Config         — capture config + sheet→config mapping + method-best table
  basic             — basic extension @ (s=2, k=16, B=64) — basic/dns/topk argmax
  oracle            — basic extension @ (s=4, k=16, B=64) — oracle argmax

Each data sheet has one row per (captured force-1) step with:
  step coords (sample_id, call_idx, step_id, ground_truth_token)
  metrics (accepted, ext_size, n_base, max_depth_extended,
           target_ms, draft_ms, total_step_ms)
  tree-wide arrays (JSON-stringified):
    tree_token_ids, tree_parents, tree_depth, tree_source,
    tree_path_prob, tree_is_accepted, tree_anchor_node_id, tree_anchor_depth
  category index arrays:
    idx_eagle, idx_basic_suffix,
    idx_ext_suffix_d1, ..., idx_ext_suffix_d{S}
  category counts:
    n_eagle, n_basic_suffix, n_ext_suffix_d1, ..., n_ext_suffix_d{S}

Cell-size guard: asserts every cell ≤ 32,000 chars; errors with row info if
exceeded (chunking not implemented since (s=2,4 + B=64) configs comfortably
fit; downstream can revise builder if larger configs added).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import xlsxwriter


CELL_LIMIT = 32_000  # safety margin under Excel's 32,767


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def latency_lookup(n: int, table: Dict[int, float]) -> Optional[float]:
    """Lookup target_forward_ms at next pow-2 >= n, clamp to largest entry."""
    if not table:
        return None
    n = max(1, int(n))
    b = 1
    while b < n:
        b <<= 1
    keys = sorted(table.keys())
    if b in table:
        return float(table[b])
    for k in keys:
        if k >= b:
            return float(table[k])
    return float(table[keys[-1]])


def compute_depth(parents: List[int]) -> List[int]:
    depth = [0] * len(parents)
    for i, p in enumerate(parents):
        depth[i] = 1 if p < 0 else depth[p] + 1
    return depth


MAX_NODE_DEPTH = 40  # depth columns d1..d40 (suffix can reach ~depth 38)


def build_sheet(
    workbook, sheet_name: str, dump_path: Path, S: int,
    target_forward_ms: Dict[int, float],
) -> int:
    """Build one data sheet from a per-step JSONL dump.

    Returns number of rows written.
    """
    ws = workbook.add_worksheet(sheet_name[:31])

    # Column order
    cols: List[str] = [
        "sample_id", "call_idx", "step_id",
        "ground_truth_token", "ground_truth_future",
        "accepted", "ext_size", "n_base", "max_depth_extended",
        "target_ms", "draft_ms", "total_step_ms",
        "target_ms_at_backbone_only",
        "gt_divergence_depth", "gt_remaining_after_accept",
        "running_context_len", "task_total_steps",
        # Accepted path trail
        "accepted_path_nodes", "accepted_path_tokens",
        "accepted_path_sources",
        # Tree-wide arrays
        "tree_token_ids", "tree_parents", "tree_depth",
        "tree_source", "tree_path_prob", "tree_is_accepted",
        "tree_anchor_node_id", "tree_anchor_depth",
        "tree_anchor_path_prob",
        # Suffix-specific arrays
        "tree_suffix_freq", "tree_suffix_cum_prob",
        "tree_suffix_edge_prob", "tree_match_len",
        # Category index arrays
        "idx_eagle", "idx_basic_suffix",
    ]
    # idx_ext_suffix_d{N} now means "ext suffix nodes at NODE TREE DEPTH N"
    # (not anchor depth). Extension chains can reach deep — cap at d40.
    for d in range(1, MAX_NODE_DEPTH + 1):
        cols.append(f"idx_ext_suffix_d{d}")
    # Category counts
    cols.extend(["n_eagle", "n_basic_suffix",
                 "n_eagle_accepted", "n_basic_suffix_accepted"])
    for d in range(1, MAX_NODE_DEPTH + 1):
        cols.append(f"n_ext_suffix_d{d}")
        cols.append(f"n_ext_suffix_d{d}_accepted")

    # Header
    bold = workbook.add_format({"bold": True})
    for j, c in enumerate(cols):
        ws.write_string(0, j, c, bold)
    ws.freeze_panes(1, 0)

    r_idx = 0
    if not dump_path.exists() or dump_path.stat().st_size == 0:
        return 0

    # First pass: count total steps per (rid, call_idx).
    task_total: Dict[tuple, int] = {}
    with open(dump_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("request_id"), rec.get("call_idx"))
            task_total[key] = task_total.get(key, 0) + 1

    # Running context length accumulator per (rid, call_idx).
    running_ctx: Dict[tuple, int] = {}

    with open(dump_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            r_idx += 1

            # Core fields
            tids = rec.get("tree_token_ids", []) or []
            parents = rec.get("tree_parents", []) or []
            source = rec.get("tree_source", []) or []
            pp = rec.get("tree_path_prob", []) or []
            is_acc_arr = rec.get("tree_is_accepted", []) or []
            anchor_ids = rec.get("tree_anchor_node_id", []) or []
            sfx_freq = rec.get("tree_suffix_freq", []) or []
            sfx_cum = rec.get("tree_suffix_cum_prob", []) or []
            match_len_arr = rec.get("tree_match_len", []) or []
            gt_future = rec.get("ground_truth_future") or []
            gt_token = rec.get("ground_truth_token")
            n_base = int(rec.get("tree_n_base") or 0)

            # Defensive: align arrays to same length
            n = len(tids)
            if any(len(x) != n for x in (parents, source, pp, is_acc_arr,
                                          anchor_ids)):
                # Pad / clip to n
                def _fit(lst, fill):
                    return (list(lst) + [fill] * (n - len(lst)))[:n]
                parents = _fit(parents, -1)
                source = _fit(source, "eagle")
                pp = _fit(pp, None)
                is_acc_arr = _fit(is_acc_arr, False)
                anchor_ids = _fit(anchor_ids, -2)

            depth = compute_depth(parents) if parents else []

            # Pad suffix-specific arrays to length n if missing.
            n = len(tids)
            def _fit_alt(lst, fill):
                return (list(lst) + [fill] * (n - len(lst)))[:n]
            sfx_freq = _fit_alt(sfx_freq, None)
            sfx_cum = _fit_alt(sfx_cum, None)
            match_len_arr = _fit_alt(match_len_arr, None)

            # Derived arrays
            # tree_anchor_path_prob: backbone → None; vroot suffix → 1.0;
            # backbone-anchored → path_prob[anchor_id]
            anchor_pp = [None] * n
            for i in range(n):
                aid = anchor_ids[i]
                if aid == -2:
                    anchor_pp[i] = None
                elif aid == -1:
                    anchor_pp[i] = 1.0
                elif 0 <= aid < n:
                    anchor_pp[i] = pp[aid] if aid < len(pp) else None

            # tree_suffix_edge_prob: cum_prob[i] / cum_prob[parent_in_graft]
            # if parent is suffix in same graft; else cum_prob[i] when parent
            # is anchor (eagle/vroot).
            edge_prob = [None] * n
            for i in range(n):
                if source[i] != "suffix":
                    continue
                p = parents[i]
                if p < 0 or source[p] != "suffix":
                    edge_prob[i] = sfx_cum[i]
                else:
                    pc = sfx_cum[p]
                    ci = sfx_cum[i]
                    if pc is not None and pc > 0 and ci is not None:
                        edge_prob[i] = float(ci) / float(pc)

            # Categories — depth grouping is by NODE's OWN tree depth.
            idx_eagle = list(range(n_base))
            idx_basic_suffix: List[int] = []
            idx_ext_by_depth: Dict[int, List[int]] = {
                d: [] for d in range(1, MAX_NODE_DEPTH + 1)
            }
            for i in range(n_base, n):
                aid = anchor_ids[i]
                if aid == -1:
                    idx_basic_suffix.append(i)
                elif aid >= 0:
                    node_d = int(depth[i]) if i < len(depth) else 0
                    if 1 <= node_d <= MAX_NODE_DEPTH:
                        idx_ext_by_depth[node_d].append(i)

            # is_accepted as 0/1
            is_acc_int = [1 if bool(b) else 0 for b in is_acc_arr]

            # Accepted path trail: walk from root via accepted children
            children: Dict[int, List[int]] = {}
            for i, p in enumerate(parents):
                children.setdefault(p, []).append(i)
            acc_path: List[int] = []
            cur = -1
            while True:
                pick = None
                for c in children.get(cur, []):
                    if is_acc_int[c]:
                        pick = c
                        break
                if pick is None:
                    break
                acc_path.append(pick)
                cur = pick
            acc_tokens = [tids[i] for i in acc_path]
            acc_sources = [source[i] for i in acc_path]
            # gt_divergence_depth: where accept_path diverges from gt_future
            gt_div = 0
            for d, n_idx in enumerate(acc_path, 1):
                if d - 1 < len(gt_future) and tids[n_idx] == gt_future[d - 1]:
                    gt_div = d
                else:
                    break
            gt_remaining = max(0, len(gt_future) - len(acc_path))

            # Per-category accepted counts.
            n_eagle_acc = sum(1 for i in range(min(n_base, n))
                              if is_acc_int[i])
            n_basic_acc = sum(1 for i in range(n_base, n)
                              if anchor_ids[i] == -1 and is_acc_int[i])
            # Accepted counts per NODE depth (matching idx_ext_suffix_dN).
            n_ext_acc_by_depth: Dict[int, int] = {
                d: 0 for d in range(1, MAX_NODE_DEPTH + 1)
            }
            for i in range(n_base, n):
                aid = anchor_ids[i]
                if aid >= 0 and is_acc_int[i]:
                    node_d = depth[i] if i < len(depth) else 0
                    if 1 <= node_d <= MAX_NODE_DEPTH:
                        n_ext_acc_by_depth[node_d] += 1

            # target_ms_at_backbone_only: latency lookup at n_base
            bb_target_ms = latency_lookup(n_base, target_forward_ms)

            # running_context_len: cumulative steps within (rid, call_idx)
            tkey = (rec.get("request_id"), rec.get("call_idx"))
            running_ctx[tkey] = running_ctx.get(tkey, 0) + 1
            ctx_len = running_ctx[tkey] - 1   # before this step

            # Row values
            row_vals = {
                "sample_id": rec.get("request_id"),
                "call_idx": rec.get("call_idx"),
                "step_id": rec.get("step_id"),
                "ground_truth_token": gt_token,
                "ground_truth_future": (
                    compact_json(gt_future) if gt_future else None),
                "accepted": rec.get("accepted"),
                "ext_size": rec.get("ext_size"),
                "n_base": n_base,
                "max_depth_extended": max(depth) if depth else 0,
                "target_ms": rec.get("target_ms"),
                "draft_ms": rec.get("draft_ms"),
                "total_step_ms": rec.get("total_step_ms"),
                "target_ms_at_backbone_only": bb_target_ms,
                "gt_divergence_depth": gt_div,
                "gt_remaining_after_accept": gt_remaining,
                "running_context_len": ctx_len,
                "task_total_steps": task_total.get(tkey, 0),
                "accepted_path_nodes": compact_json(acc_path),
                "accepted_path_tokens": compact_json(acc_tokens),
                "accepted_path_sources": compact_json(acc_sources),
                "tree_token_ids": compact_json(tids),
                "tree_parents": compact_json(parents),
                "tree_depth": compact_json(depth),
                "tree_source": compact_json(source),
                "tree_path_prob": compact_json(pp),
                "tree_is_accepted": compact_json(is_acc_int),
                "tree_anchor_node_id": compact_json(anchor_ids),
                "tree_anchor_depth": compact_json(
                    [depth[i] if i < n_base
                     else (0 if anchor_ids[i] == -1
                           else (depth[anchor_ids[i]]
                                 if anchor_ids[i] >= 0
                                    and anchor_ids[i] < len(depth)
                                 else -1))
                     for i in range(n)]),
                "tree_anchor_path_prob": compact_json(anchor_pp),
                "tree_suffix_freq": compact_json(sfx_freq),
                "tree_suffix_cum_prob": compact_json(sfx_cum),
                "tree_suffix_edge_prob": compact_json(edge_prob),
                "tree_match_len": compact_json(match_len_arr),
                "idx_eagle": compact_json(idx_eagle),
                "idx_basic_suffix": compact_json(idx_basic_suffix),
                "n_eagle": len(idx_eagle),
                "n_basic_suffix": len(idx_basic_suffix),
                "n_eagle_accepted": n_eagle_acc,
                "n_basic_suffix_accepted": n_basic_acc,
            }
            for d in range(1, MAX_NODE_DEPTH + 1):
                indices = idx_ext_by_depth.get(d, [])
                row_vals[f"idx_ext_suffix_d{d}"] = compact_json(indices)
                row_vals[f"n_ext_suffix_d{d}"] = len(indices)
                row_vals[f"n_ext_suffix_d{d}_accepted"] = (
                    n_ext_acc_by_depth.get(d, 0))

            # Write
            for j, c in enumerate(cols):
                v = row_vals.get(c)
                if v is None:
                    continue
                if isinstance(v, bool):
                    ws.write_boolean(r_idx, j, v)
                elif isinstance(v, (int, float)):
                    ws.write_number(r_idx, j, v)
                else:
                    s = str(v)
                    if len(s) > CELL_LIMIT:
                        raise RuntimeError(
                            f"Cell overflow: sheet={sheet_name} "
                            f"row={r_idx} col={c} len={len(s)} "
                            f"(rid={row_vals['sample_id']} step={row_vals['step_id']})")
                    ws.write_string(r_idx, j, s)
    return r_idx


def build_config_sheet(
    workbook, dump_dir: Path, sheets_info: List[Dict[str, Any]],
) -> None:
    ws = workbook.add_worksheet("00_Config")
    bold = workbook.add_format({"bold": True})
    rows: List[Any] = []
    rows.append(("Capture metadata", ""))
    rows.append(("model", "Qwen/Qwen3-14B"))
    rows.append(("target_model", "Qwen/Qwen3-14B"))
    rows.append(("benchmark", "bfcl_v4"))
    rows.append(("capture_steps (S)", 8))
    rows.append(("capture_topk (K)", 16))
    rows.append(("suffix_max_spec_factor (F)", 4.0))
    rows.append(("suffix_min_token_prob (T)", 0.0))
    rows.append(("oracle_force_1_capture", True))
    rows.append(("", ""))
    rows.append(("Sheet → config", ""))
    for info in sheets_info:
        rows.append((info["sheet"],
                     f"s={info['s']}, k={info['k']}, B={info['B']}, "
                     f"method=basic-extension(F=4.0,T=0.0)"))
    rows.append(("", ""))
    rows.append(("Method best (from earlier sweep)", ""))
    rows.append(("basic", "(s=2, k=16, B=64) speedup ≈ 1.74x"))
    rows.append(("oracle", "(s=4, k=16, B=64) speedup ≈ 3.61x"))
    rows.append(("dns", "(s=2, k=16, B=64) speedup ≈ 2.63x — shares config with basic"))
    rows.append(("topk", "(s=2, k=16, B=64) speedup ≈ 2.82x — shares config with basic"))
    rows.append(("", ""))
    rows.append(("Schema notes", ""))
    rows.append(("granularity", "per (sample, call, step) — force-1 advance"))
    rows.append(("tree arrays", "JSON-stringified, length = ext_size"))
    rows.append(("source values", "'eagle' (backbone) or 'suffix'"))
    rows.append(("anchor_node_id values",
                 "-2 = backbone, -1 = vroot suffix, >=0 = backbone-anchored suffix"))
    rows.append(("path_prob", "raw draft cumulative path probability"))
    rows.append(("idx_eagle", "node IDs for EAGLE3 backbone (= [0, n_base))"))
    rows.append(("idx_basic_suffix",
                 "node IDs for vroot suffix (anchor_node_id=-1)"))
    rows.append(("idx_ext_suffix_dN",
                 f"node IDs for backbone-anchored suffix at NODE TREE depth N "
                 f"(N=1..{MAX_NODE_DEPTH}); ext suffix can reach deep due to "
                 f"suffix chain extension"))

    for i, (k, v) in enumerate(rows):
        if k and not v:
            ws.write_string(i, 0, k, bold)
        else:
            ws.write_string(i, 0, str(k))
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                ws.write_number(i, 1, v)
            else:
                ws.write_string(i, 1, str(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    sheets_info = [
        {"sheet": "basic", "s": 2, "k": 16, "B": 64,
         "dump": "bfcl_v4_s2k16B64.jsonl"},
        {"sheet": "oracle", "s": 4, "k": 16, "B": 64,
         "dump": "bfcl_v4_s4k16B64.jsonl"},
    ]

    # Load latency_data for backbone-only cost lookups
    latency_path = (
        Path("/workspace/simulation/results/qwen3_14b/"
             "bfcl_v4_steps8_topk16_capture/latency_data.json"))
    target_forward_ms: Dict[int, float] = {}
    if latency_path.exists():
        with open(latency_path) as f:
            lat = json.load(f)
        target_forward_ms = {
            int(k): float(v)
            for k, v in (lat.get("target_forward_ms") or {}).items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(
        str(args.output), {"constant_memory": True})

    build_config_sheet(wb, args.dump_dir, sheets_info)

    for info in sheets_info:
        dump_path = args.dump_dir / info["dump"]
        n = build_sheet(wb, info["sheet"], dump_path, S=info["s"],
                        target_forward_ms=target_forward_ms)
        print(f"{info['sheet']}: {n} rows", file=sys.stderr)

    wb.close()
    sz = args.output.stat().st_size / 1024 / 1024
    print(f"DONE → {args.output} ({sz:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
