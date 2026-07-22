#!/usr/bin/env python3
"""FIXED-threshold hand-off sweep (offline, fast).

For each FIXED a* on a grid, the deployable rule keeps the DFlash head while the
hazard dflash_p >= a* (policy_threshold), then hands off to the suffix tail.
Realized accept = val(A_d, A_s(k*), K) on the ground-truth trajectory. Reports
the trajectory MAT per a*, next to DFlash-only (a*=inf -> full head) and O3
(max_j val = the tree/oracle ceiling). Reuses the loaders/val/walk from
analyze_handoff_policy so it is identical machinery, just a fixed a* instead of
the per-position T/(1+T).

Usage:
  python3 simulation/scripts/handoff_fixed_astar_sweep.py \
      --table <dense.jsonl.gz> --dflash-proposals <dflash_proposals_dense.jsonl> \
      --label online --ks 8,16,inf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.scripts.analyze_handoff_policy import (  # noqa: E402
    load_table, load_haz, val, o3_accept, policy_threshold, walk,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--dflash-proposals", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--variant", default="grd", choices=["grd", "orc"])
    ap.add_argument("--ks", default="8,16,inf")
    args = ap.parse_args()

    ks = [10**9 if x == "inf" else int(x) for x in args.ks.split(",")]
    klabels = args.ks.split(",")
    seqs = load_table(args.table, args.variant)
    haz = load_haz(args.dflash_proposals)
    for (rid, ci), pm in seqs.items():
        for pos in pm:
            ae, asv, el, gcl0, score0 = pm[pos]
            pm[pos] = (ae, asv, el, haz.get((rid, pos), []))  # e[3]=hazards

    def mat(fn):
        tot = steps = 0
        for pm in seqs.values():
            t, s = walk(pm, fn)
            tot += t
            steps += s
        return tot / max(steps, 1)

    # FIXED T grid (methodology parameter). a* = T/(1+T). T=0 -> DFlash-only.
    T_grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
              12.0, 16.0]
    print(f"[{args.label}] {args.table}")
    for kv, kl in zip(ks, klabels):
        m_d = mat(lambda pos, e, kv=kv: min(e[0], kv))            # DFlash-only
        m_s = mat(lambda pos, e, kv=kv: val(e[0], e[1], 0, kv))   # suffix-only
        m_o3 = mat(lambda pos, e, kv=kv: o3_accept(e[0], e[1], kv))  # tree/oracle
        print(f"\n  k={kl}:  DFlash-only={m_d:.3f}  suffix-only={m_s:.3f}  "
              f"O3(tree ceiling)={m_o3:.3f}")
        best = (-1.0, None, None)
        for T in T_grid:
            a = T / (1.0 + T)
            m = mat(lambda pos, e, a=a, kv=kv:
                    val(e[0], e[1], policy_threshold(e[3], a), kv))
            if m > best[0]:
                best = (m, T, a)
            print(f"    T={T:<5} (a*={a:.3f}) -> MAT={m:.3f}  "
                  f"(vs DFlash {100*(m/m_d-1):+.1f}%)")
        print(f"    >>> best fixed T={best[1]} (a*={best[2]:.3f}) MAT={best[0]:.3f} "
              f"| DFlash {m_d:.3f} | O3 {m_o3:.3f}")


if __name__ == "__main__":
    main()
