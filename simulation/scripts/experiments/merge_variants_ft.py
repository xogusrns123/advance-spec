#!/usr/bin/env python3
"""Merge FT-swept variant results into the main `_full.json` sims.

For every reslice present in BOTH ``--variants-dir`` and ``--main-dir``:
  1. Load the variants-only JSON (which contains
     extension_by_count_r{r}_f{F}_t{T}_*, extension_by_score_t{th}_f{F}_t{T}_*,
     extension_prune_pt_t{pt}_f{F}_t{T}_*, and the dm equivalents).
  2. For every budget entry in the main file, copy the variant keys over.
  3. Drop the OLD non-FT variant keys (extension_by_count_r{r}_*,
     extension_by_score_t{th}_*, extension_prune_pt_t{pt}_*) so the notebook
     never sees a stale fixed-FT measurement next to the new FT-swept ones.

The main file is overwritten in place. The variants file is left untouched.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Pattern matchers (eagle3-base + dm-base symmetric).
_NEW_KEY = re.compile(
    r"^extension(?:_dm)?_(?:by_count_r|by_score_t|prune_pt_t)[\d.]+_f[\d.]+_t[\d.]+_"
)
_OLD_KEY = re.compile(
    r"^extension(?:_dm)?_(?:by_count_r|by_score_t|prune_pt_t)[\d.]+_(?!f[\d.]+_t[\d.]+_)"
)


def _is_new_variant_key(k: str) -> bool:
    return bool(_NEW_KEY.match(k))


def _is_old_variant_key(k: str) -> bool:
    return bool(_OLD_KEY.match(k))


def merge_one(main_path: Path, var_path: Path, dry_run: bool = False) -> dict:
    """Merge variant FT results into the main sim file. Returns stats."""
    main = json.loads(main_path.read_text())
    var = json.loads(var_path.read_text())

    main_bs = main["latency"]["budget_sweep"]
    var_bs = var["latency"]["budget_sweep"]

    main_by_b = {int(e["budget"]): e for e in main_bs}
    var_by_b = {int(e["budget"]): e for e in var_bs}

    added = 0
    overwritten = 0
    dropped = 0
    for b, ventry in var_by_b.items():
        if b not in main_by_b:
            continue
        mentry = main_by_b[b]
        # Drop OLD non-FT variant keys first.
        old_keys = [k for k in mentry if _is_old_variant_key(k)]
        for k in old_keys:
            del mentry[k]
            dropped += 1
        # Copy NEW FT-swept variant keys.
        for k, v in ventry.items():
            if not _is_new_variant_key(k):
                continue
            if k in mentry:
                overwritten += 1
            else:
                added += 1
            mentry[k] = v

    if not dry_run:
        main_path.write_text(json.dumps(main, indent=1, ensure_ascii=False) + "\n")

    return {"added": added, "overwritten": overwritten, "dropped_old": dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dir", required=True,
                    help="dir containing sim_<wl>_<reslice>_full.json (will be modified)")
    ap.add_argument("--variants-dir", required=True,
                    help="dir containing the variants-only sim JSONs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    main_dir = Path(args.main_dir)
    var_dir = Path(args.variants_dir)
    if not main_dir.is_dir() or not var_dir.is_dir():
        raise SystemExit(f"need two existing dirs; got {main_dir} / {var_dir}")

    files = sorted(var_dir.glob("sim_*_full.json"))
    if not files:
        raise SystemExit(f"no sim_*_full.json under {var_dir}")

    print(f"Merging {len(files)} variant files into {main_dir}/")
    total = {"added": 0, "overwritten": 0, "dropped_old": 0, "skipped": 0}
    for vp in files:
        mp = main_dir / vp.name
        if not mp.exists():
            print(f"  SKIP {vp.name}: no matching main file")
            total["skipped"] += 1
            continue
        st = merge_one(mp, vp, dry_run=args.dry_run)
        print(f"  {vp.name}: +{st['added']} new, ~{st['overwritten']} overwrote, "
              f"-{st['dropped_old']} old dropped")
        for k in st:
            total[k] = total.get(k, 0) + st[k]
    print(f"\nTotal: +{total['added']} new keys, "
          f"~{total['overwritten']} overwrites, "
          f"-{total['dropped_old']} old dropped, "
          f"skipped={total['skipped']}")
    if args.dry_run:
        print("(dry-run — no files written)")


if __name__ == "__main__":
    main()
