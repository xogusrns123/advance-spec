#!/usr/bin/env python3
"""Merge missing-method sim results into the existing main sim file.

Unlike merge_variants_ft.py (which only adds FT-swept variant keys), this
script adds ANY key from the source file that is not already present in
the target file. Existing target keys are NEVER overwritten.

Usage:
    python3 merge_missing_methods.py \\
        --target sim_swebench_verified_s2k16_full.json \\
        --source sim_swebench_verified_s2k16_missing.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(target_path: Path, source_path: Path, dry_run: bool = False) -> dict:
    target = json.loads(target_path.read_text())
    source = json.loads(source_path.read_text())

    t_bs = target["latency"]["budget_sweep"]
    s_bs = source["latency"]["budget_sweep"]

    t_by_b = {int(e["budget"]): e for e in t_bs}
    s_by_b = {int(e["budget"]): e for e in s_bs}

    added = 0
    skipped = 0
    for b, sentry in s_by_b.items():
        if b not in t_by_b:
            # Whole new budget entry — append to target
            t_bs.append(sentry)
            t_by_b[b] = sentry
            added += len(sentry) - 1  # minus 'budget' field
            continue
        tentry = t_by_b[b]
        for k, v in sentry.items():
            if k == "budget":
                continue
            if k in tentry:
                skipped += 1
            else:
                tentry[k] = v
                added += 1

    # Re-sort budget_sweep by budget
    t_bs.sort(key=lambda e: int(e["budget"]))

    if not dry_run:
        target_path.write_text(json.dumps(target, indent=1, ensure_ascii=False) + "\n")

    return {"added": added, "skipped_existing": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="existing sim file (modified in place)")
    ap.add_argument("--source", required=True, help="sim file with new methods to merge in")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    st = merge(Path(args.target), Path(args.source), dry_run=args.dry_run)
    print(f"target={args.target}")
    print(f"source={args.source}")
    print(f"  added: {st['added']} keys")
    print(f"  skipped (already present): {st['skipped_existing']}")
    if args.dry_run:
        print("(dry-run — no files written)")


if __name__ == "__main__":
    main()
