"""Merge dm-method delta sim JSONs into main sweep JSONs.

After a delta sweep produces sim_<wl>_<reslice>_dm.json containing only the
new draft_model-backbone methods, copy their per-budget keys into the
matching sim_<wl>_<reslice>_full.json so notebooks see one unified file.

Key prefixes copied from delta into main:
    draft_model_     (from single:draft_model)
    hybrid_dm_       (covers hybrid_dm:* and hybrid_dm_oracle:*)
    extension_dm_    (covers extension_dm:*, _oracle, _by_count, _by_score)

Existing keys with these prefixes in the main file are overwritten so a
re-run replaces stale numbers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DM_PREFIXES = ("draft_model_", "hybrid_dm_", "extension_dm_")


def merge_one(main_fp: Path, delta_fp: Path, dry_run: bool) -> tuple[int, int]:
    with open(main_fp) as f:
        main = json.load(f)
    with open(delta_fp) as f:
        delta = json.load(f)
    main_bs = main.get("latency", {}).get("budget_sweep", [])
    delta_bs = delta.get("latency", {}).get("budget_sweep", [])
    delta_by_budget = {e.get("budget"): e for e in delta_bs}
    n_keys = 0
    n_entries = 0
    for entry in main_bs:
        b = entry.get("budget")
        d = delta_by_budget.get(b)
        if d is None:
            continue
        for k, v in d.items():
            if k.startswith(DM_PREFIXES):
                entry[k] = v
                n_keys += 1
        n_entries += 1
    if not dry_run:
        with open(main_fp, "w") as f:
            json.dump(main, f, indent=2)
    return n_entries, n_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main-glob", required=True,
                   help="glob for main sweep JSONs, e.g. sim_bfcl_v4_*_full.json")
    p.add_argument("--delta-suffix", default="_dm.json",
                   help="suffix in delta filenames (default _dm.json)")
    p.add_argument("--main-suffix", default="_full.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import glob as _glob
    main_files = sorted(_glob.glob(args.main_glob))
    if not main_files:
        sys.exit(f"no files matched: {args.main_glob}")
    total_entries = 0
    total_keys = 0
    for mf in main_files:
        mp = Path(mf)
        delta_path = mp.with_name(mp.name.replace(args.main_suffix, args.delta_suffix))
        if not delta_path.exists():
            print(f"  SKIP (no delta): {mp.name}", file=sys.stderr)
            continue
        ne, nk = merge_one(mp, delta_path, args.dry_run)
        print(f"  {mp.name}: {ne} budgets, {nk} keys merged",
              file=sys.stderr)
        total_entries += ne
        total_keys += nk
    print(f"DONE — {total_entries} budget entries, {total_keys} keys merged "
          f"(dry_run={args.dry_run})", file=sys.stderr)


if __name__ == "__main__":
    main()
