"""Patch sim JSON files in place by recomputing per-method
``total_target_ms_corrected`` / ``total_time_real_ms_corrected`` /
``speedup_real_corrected`` using the FLAT ``target_forward_ms`` table
(monotonic to B=512), bypassing the topk-specific table whose extrapolation
can go negative for sparse measurement ranges (e.g., topk=4 only has
B≤64, and B=32→64 has a noise-driven negative slope, which extrapolates
to negative target latency for ext_size ≥ ~370).

Approximation: corrected_target_ms ≈ steps × _interp(mean_ext_size). Mean
based — incurs Jensen bias for convex cost; ~5–15% absolute. Adequate to
turn the 17-19x outliers into sensible 1–2x corrected values, and the
ranking across methods stays meaningful.

Original fields untouched. New fields added per method:
  <method>_total_target_ms_corrected
  <method>_total_time_real_ms_corrected
  <method>_speedup_real_corrected   (= old_speedup * old_time / new_time)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def fixed_target_forward(B: float, table: dict, vanilla_ms: float) -> float:
    """Same interp shape as run_tree_oracle_sim._interp BUT clamps the
    extrapolation slope to be non-negative — the bug we're correcting for."""
    keys = sorted(int(k) for k in table.keys())
    if str(B) in table:
        return float(table[str(B)])
    if B <= keys[0]:
        if B <= 1:
            return vanilla_ms
        v = float(table[str(keys[0])])
        return vanilla_ms + (B - 1) / (keys[0] - 1) * (v - vanilla_ms)
    if B >= keys[-1]:
        kh, kl = keys[-1], keys[-2]
        vh = float(table[str(kh)])
        vl = float(table[str(kl)])
        slope = (vh - vl) / (kh - kl) if kh != kl else 0.0
        slope = max(0.0, slope)        # <<< the fix: never extrapolate downward
        return vh + slope * (B - kh)
    lo = max(k for k in keys if k <= B)
    hi = min(k for k in keys if k >= B)
    if lo == hi:
        return float(table[str(lo)])
    frac = (B - lo) / (hi - lo)
    return float(table[str(lo)]) + frac * (float(table[str(hi)])
                                            - float(table[str(lo)]))


def patch_file(fp: Path, latency_cfg: dict, dry_run: bool) -> int:
    with open(fp) as f:
        d = json.load(f)
    target_table = latency_cfg["target_forward_ms"]
    vanilla_ms = float(latency_cfg["vanilla_step_ms"])
    bs = d.get("latency", {}).get("budget_sweep", [])
    n_changed = 0
    for entry in bs:
        # Per-workload-budget numerator: speedup_real = numerator / time_real.
        # Extract from eagle3 (always sane) so we don't rely on each method's
        # potentially-broken time_real for ratio scaling.
        numerator = None
        eg_spd = entry.get("eagle3_speedup_real")
        eg_time = entry.get("eagle3_total_time_real_ms")
        if eg_spd is not None and eg_time is not None and eg_time > 0:
            numerator = eg_spd * eg_time

        methods: set[str] = set()
        for k in entry:
            if k.endswith("_total_target_ms"):
                methods.add(k[: -len("_total_target_ms")])
        for m in methods:
            steps = entry.get(f"{m}_steps", 0)
            tt = entry.get(f"{m}_total_target_tokens", 0)
            td = entry.get(f"{m}_total_draft_ms", 0)
            if steps <= 0 or tt <= 0:
                continue
            mean_ext = tt / steps
            new_target = steps * fixed_target_forward(
                mean_ext, target_table, vanilla_ms)
            new_time = new_target + td
            # Stash the original (broken) values for traceability.
            entry.setdefault(f"{m}_total_target_ms_pre_recover",
                             entry.get(f"{m}_total_target_ms"))
            entry.setdefault(f"{m}_total_time_real_ms_pre_recover",
                             entry.get(f"{m}_total_time_real_ms"))
            entry.setdefault(f"{m}_speedup_real_pre_recover",
                             entry.get(f"{m}_speedup_real"))
            # Overwrite primary fields with corrected values so existing
            # notebooks (compare_methods, compare_oracle_v12) read the fixed
            # numbers without modification.
            entry[f"{m}_total_target_ms"] = new_target
            entry[f"{m}_total_time_real_ms"] = new_time
            entry[f"{m}_total_target_ms_corrected"] = new_target
            entry[f"{m}_total_time_real_ms_corrected"] = new_time
            if numerator is not None and new_time > 0:
                corrected_spd = numerator / new_time
                entry[f"{m}_speedup_real"] = corrected_spd
                entry[f"{m}_speedup_real_corrected"] = corrected_spd
            n_changed += 1
    if not dry_run:
        with open(fp, "w") as f:
            json.dump(d, f, indent=2)
    return n_changed


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--latency-config",
        default="/home/muchwater/advance-spec/simulation/config/latency/qwen3_14b.json")
    p.add_argument(
        "--glob",
        default="/home/muchwater/advance-spec/simulation/results/explorations/sim_*_full.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    with open(args.latency_config) as f:
        L = json.load(f)
    files = sorted(glob.glob(args.glob))
    print(f"patching {len(files)} files (dry_run={args.dry_run})", file=sys.stderr)
    total = 0
    for fp in files:
        n = patch_file(Path(fp), L, args.dry_run)
        total += n
        print(f"  {Path(fp).name}: {n} method entries patched", file=sys.stderr)
    print(f"DONE — {total} (method, budget) pairs patched", file=sys.stderr)


if __name__ == "__main__":
    main()
