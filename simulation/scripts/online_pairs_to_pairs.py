#!/usr/bin/env python3
"""Convert online_pairs_<arm>.jsonl (from a served online-calibration run) into
the gzipped pairs_*.jsonl.gz schema that calib_perposition.load_per_depth and
plot_calib_scatter_box consume — so the calib_verify PER-DEPTH graphs render with
ZERO changes to those scripts.

Schema produced (one row per (rid, decode_step) chain):
  {"rid": "<rid>#<step>", "m": [[eagle_p, q_eagle], ...], "s": [[suffix_p, q_suffix], ...]}
where the list INDEX == draft depth (load_per_depth keys per-depth by position).
y = q_target (continuous in [0,1]) for both groups (the online target_p label).

Eagle is proposed at every chain depth, so "m" is the full chain. Suffix may be
absent at some depths; to keep index==depth we truncate "s" at its first gap from
depth 0 (post-gap suffix samples are dropped — rare, only the chain tail).

Usage:
  python3 simulation/scripts/online_pairs_to_pairs.py \
    --in <dir>/online_pairs_select1_online_logistic.jsonl \
    --out <dir>/pairs_select1_online_logistic.jsonl.gz
"""
from __future__ import annotations
import argparse
import gzip
import json
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    groups: dict = defaultdict(dict)  # (rid, step) -> {depth: rec}
    n_in = 0
    with open(a.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") != "online_pair":
                continue
            n_in += 1
            groups[(r["rid"], r.get("decode_step"))][int(r["depth"])] = r

    n_rows = n_m = n_s = 0
    with gzip.open(a.out, "wt") as f:
        for (rid, step), depths in groups.items():
            m = []
            d = 0
            while d in depths:
                rec = depths[d]
                ep, qe = rec.get("eagle_p"), rec.get("q_eagle")
                if ep is None or qe is None:
                    break
                m.append([float(ep), float(qe)])
                d += 1
            s = []
            d = 0
            while d in depths:
                rec = depths[d]
                sp, qs = rec.get("suffix_p"), rec.get("q_suffix")
                if sp is None or qs is None:
                    break
                s.append([float(sp), float(qs)])
                d += 1
            if not m and not s:
                continue
            f.write(json.dumps({"rid": f"{rid}#{step}", "m": m, "s": s}) + "\n")
            n_rows += 1
            n_m += len(m)
            n_s += len(s)

    print(f"read {n_in} online_pair rows -> {n_rows} chains "
          f"(eagle/model samples={n_m}, suffix samples={n_s}) -> {a.out}")


if __name__ == "__main__":
    main()
