#!/usr/bin/env python3
"""Fit suffix-probability calibration maps from a chain-hybrid decision log.

Input : the ``decisions_select1.jsonl`` written by chain_hybrid_patch.py for
        the RAW (uncalibrated) select-1 arm. Two record types:
          {"type": "decision", rid, decode_step, depth, suffix_token,
           suffix_p, suffix_count(c), suffix_total(n), chosen, agreement, ...}
          {"type": "step", rid, decode_step, accept_len}
Output: two frozen isotonic maps consumable by chain_hybrid_patch's
        _ServingIsoCalibrator (== run_tree_oracle_sim._FrozenIsoCalibrator):
          --out-noshrink : x = raw count ratio   c/n        (meta.shrink=null)
          --out-jeffreys : x = Jeffreys-shrunk  (c+.5)/(n+1) (meta.shrink=set)

Sample construction (BOTH groups — calibrating only suffix would compare a
calibrated accept-prob against eagle's raw softmax, an unfair asymmetry that
over-suppresses suffix; the simulator calibrates both, so we do too):

  An edge contributes a (p, depth, y) sample iff its token was the realized
  chain token at that depth (only then is its acceptance observed):
    suffix group: chosen=="suffix" OR agreement (suffix_token==eagle_token)
                  -> x = suffix edge prob (raw c/n, or Jeffreys (c+.5)/(n+1))
    eagle  group: chosen=="eagle3" (the chain followed eagle; includes the
                  agreement + no-suffix-candidate cases) -> x = eagle_p
  y = 1 iff the step's accept_len >= depth + 1 (the chain-hybrid join rule).
  The losing side of a disagreement is censored (its token never entered the
  chain) — but that missing mass sits where the loser would stay the loser
  post-calibration, so the decision boundary stays well sampled. eagle_p has
  no trie counts, so the eagle group is identical across the two shrink maps.

Then per group/map, fit_iso_calibration.fit_group (quantile-bin -> per-bin
accept rate -> pool-adjacent-violators) -> monotone step map p -> P(accept).

Usage:
    python3 simulation/scripts/fit_chain_hybrid_calib.py \
        --decision-log .../decisions_select1.jsonl \
        --out-noshrink .../calib_noshrink.json \
        --out-jeffreys .../calib_jeffreys.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
from fit_iso_calibration import fit_group  # noqa: E402


def load_samples(decision_log: str) -> dict:
    """Return {'eagle': [(eagle_p, depth, y), ...],
              'suffix': [(suffix_p, depth, y, c, n), ...]} — one sample per
    edge whose token was the realized chain token at that depth."""
    accept_len: dict[tuple, int] = {}
    decisions: list[dict] = []
    with open(decision_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "step":
                accept_len[(rec["rid"], rec["decode_step"])] = rec["accept_len"]
            elif rec.get("type") == "decision":
                # Tail-append records carry cumulative path probs (not
                # single-edge count ratios) — exclude from calibration.
                if rec.get("tail"):
                    continue
                decisions.append(rec)

    eagle, suffix = [], []
    n_dec = n_unlabeled = 0
    for rec in decisions:
        key = (rec["rid"], rec["decode_step"])
        if key not in accept_len:
            n_unlabeled += 1
            continue
        n_dec += 1
        depth = int(rec["depth"])
        y = 1.0 if accept_len[key] >= depth + 1 else 0.0
        chosen = rec.get("chosen")
        agreement = rec.get("agreement") is True
        # eagle token is on the chain whenever suffix did not win.
        if chosen == "eagle3" and rec.get("eagle_p") is not None:
            eagle.append((float(rec["eagle_p"]), depth, y))
        # suffix token is on the chain when it won, or when it equals eagle's.
        if rec.get("suffix_token") is not None and rec.get("suffix_p") is not None \
                and (chosen == "suffix" or agreement):
            suffix.append((float(rec["suffix_p"]), depth, y,
                           rec.get("suffix_count"), rec.get("suffix_total")))

    print(f"  decisions joined to a step record: {n_dec} "
          f"(dropped {n_unlabeled} w/o step record)", file=sys.stderr)
    print(f"    eagle  samples (eagle on chain) : {len(eagle)}", file=sys.stderr)
    print(f"    suffix samples (suffix on chain): {len(suffix)}", file=sys.stderr)
    return {"eagle": eagle, "suffix": suffix}


def _fit_and_report(name: str, p: np.ndarray, y: np.ndarray, bins: int) -> dict:
    grp = fit_group(p, y, bins)
    xs, ys = np.asarray(grp["x"]), np.asarray(grp["y"])
    probe = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    cal = [float(ys[max(0, int(np.searchsorted(xs, q, side="right")) - 1)])
           for q in probe]
    print(f"    {name:7s} n={grp['n_samples']:6d} accept_rate={grp['accept_rate']:.4f} "
          f"steps={len(grp['x'])}", file=sys.stderr)
    print("            p->P(accept): "
          + ", ".join(f"{q:g}->{c:.3f}" for q, c in zip(probe, cal)),
          file=sys.stderr)
    return grp


def _write_map(path: str, groups: dict, shrink: str, source: str) -> None:
    out = {
        "meta": {
            "shrink": None if shrink == "none" else shrink,
            "source": source,
            "fitter": "fit_chain_hybrid_calib.py",
        },
        "groups": groups,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}  (shrink={shrink})", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decision-log", required=True)
    ap.add_argument("--out-noshrink", required=True)
    ap.add_argument("--out-jeffreys", required=True)
    ap.add_argument("--bins", type=int, default=512)
    args = ap.parse_args()

    s = load_samples(args.decision_log)
    if not s["suffix"] or not s["eagle"]:
        sys.exit("missing eagle or suffix samples in the decision log")

    # eagle group: raw softmax prob, identical for both shrink maps.
    ep = np.asarray([r[0] for r in s["eagle"]], dtype=np.float64)
    ey = np.asarray([r[2] for r in s["eagle"]], dtype=np.float64)
    print("== eagle (shared) ==", file=sys.stderr)
    eagle_grp = _fit_and_report("eagle", ep, ey, args.bins)

    # suffix group: raw c/n vs Jeffreys (c+.5)/(n+1).
    sp_raw = np.asarray([r[0] for r in s["suffix"]], dtype=np.float64)
    sy = np.asarray([r[2] for r in s["suffix"]], dtype=np.float64)
    have_counts = all(r[3] is not None and r[4] for r in s["suffix"])
    if have_counts:
        c = np.asarray([r[3] for r in s["suffix"]], dtype=np.float64)
        n = np.asarray([r[4] for r in s["suffix"]], dtype=np.float64)
        sp_jeff = (c + 0.5) / (n + 1.0)
    else:
        print("  WARNING: some suffix samples lack trie counts; Jeffreys map "
              "falls back to raw c/n for those rows", file=sys.stderr)
        sp_jeff = sp_raw.copy()

    src = args.decision_log
    print("== no-shrink map (eagle + raw c/n suffix) ==", file=sys.stderr)
    suffix_ns = _fit_and_report("suffix", sp_raw, sy, args.bins)
    _write_map(args.out_noshrink,
               {"eagle": eagle_grp, "suffix": suffix_ns}, "none", src)
    print("== Jeffreys map (eagle + (c+.5)/(n+1) suffix) ==", file=sys.stderr)
    suffix_jf = _fit_and_report("suffix", sp_jeff, sy, args.bins)
    _write_map(args.out_jeffreys,
               {"eagle": eagle_grp, "suffix": suffix_jf}, "jeffreys", src)


if __name__ == "__main__":
    main()
