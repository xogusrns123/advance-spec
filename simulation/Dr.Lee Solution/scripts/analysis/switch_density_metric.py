#!/usr/bin/env python3
"""Symmetric switch-density metric for the Dr.Lee(multislot) / Kim(boundary) debate.

The incumbent candidates (slot100 / bound100) define warm/cold by thresholding the
SUFFIX copy depth s(p) alone (theta=4) and ignore DFlash's a(p). That bakes in
"suffix is the frame, everything else is a slot" -- which is exactly the asymmetry
that makes neither party fully happy.

This adds the arm-symmetric quantity both interpretations actually point to:

    best(t) = argmax( a[t], s[t] )            # which proposer wins position t
    S       = density of positions where best(t) != best(t+1)   (per 100 tok)

S counts a warm<->cold boundary AND a frame<->slot entry as the SAME event -- a
best-proposer handoff -- using BOTH proposers, so it is baseline-neutral:
  * Dr.Lee's slot = a dflash-winning run bracketed by suffix-winning runs
    -> each slot contributes ~2 flips, slot density ~= S/2.
  * Kim's boundary = a best-proposer handoff -> boundary count = S * L.
The only difference between the two parties is normalization by block length L
(raw S for the single-proposer baseline; S*L/100 = "flips per block" for the
suffix-hybrid baseline). We report both, plus depth-weighted variants that test
the REPORT.md finding that events are not equal (value ~ warm depth unlocked).

Variants:
  sw100    100 * (#argmax flips) / L                     (raw symmetric S; count)
  swblk    sw100 * NSPEC/100  = expected flips per draft block  (Kim normalization)
  swmass   100 * sum_flips( max(a,s) at the post-flip pos ) / L (depth-weighted S)
  swgain   100 * sum_flips( |a-s| at the flip )          / L    (margin-weighted)
  cmpl     10  * mean_t( max(a,s) - min(a,s) )   pointwise complementarity (no events)
  orgain   10  * ( mean_t max(a,s) - max(mean a, mean s) )  per-pos oracle-over-best-single

Reuses units.json gains + curves_*.jsonl.gz from the interp_validation study, and
reprints the incumbent winners (warm, bm100-analog) side by side.

  python3 scripts/switch_density_metric.py --names spider swebench bfcl specbench
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse, gzip, json, os
from collections import defaultdict

NSPEC = 32  # deployed draft block (num_spec) used across the study


def load_curves(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            out[r["rid"]] = r
    return out


def winner_seq(a, s):
    """best-proposer label per position: +1 dflash, -1 suffix, 0 tie.
    Ties carry the previous decisive label forward so a flat tie run is not
    miscounted as two flips."""
    lab, prev = [], 0
    for ai, si in zip(a, s):
        if ai > si:
            w = 1
        elif si > ai:
            w = -1
        else:
            w = prev  # tie -> inherit last decisive winner
        lab.append(w)
        if w != 0:
            prev = w
    return lab


def call_metrics(cu):
    a, s = cu["a"], cu["s"]
    L = len(s)
    if L < 2:
        return None
    lab = winner_seq(a, s)
    m = dict(L=L, flips=0, mass=0.0, gain=0.0, cmpl=0.0,
             omax=0.0, sum_a=0.0, sum_s=0.0)
    for t in range(L):
        ai, si = a[t], s[t]
        m["omax"] += max(ai, si)
        m["sum_a"] += ai
        m["sum_s"] += si
        m["cmpl"] += max(ai, si) - min(ai, si)
        if t + 1 < L and lab[t + 1] != lab[t] and lab[t + 1] != 0 and lab[t] != 0:
            m["flips"] += 1
            m["mass"] += max(a[t + 1], s[t + 1])
            m["gain"] += abs(a[t + 1] - s[t + 1])
    return m


def unit_metrics(curves, rids):
    tot = defaultdict(float)
    for rid in rids:
        m = call_metrics(curves[rid])
        if m:
            for k, v in m.items():
                tot[k] += v
    L = tot["L"] or 1.0
    sw100 = 100.0 * tot["flips"] / L
    return dict(
        sw100=sw100,
        swblk=sw100 * NSPEC / 100.0,
        swmass=100.0 * tot["mass"] / L,
        swgain=100.0 * tot["gain"] / L,
        cmpl=10.0 * tot["cmpl"] / L,
        orgain=10.0 * (tot["omax"] / L - max(tot["sum_a"], tot["sum_s"]) / L),
    )


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            for k in range(i, j):
                r[order[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


NEW = ["sw100", "swblk", "swmass", "swgain", "cmpl", "orgain"]
INC = ["warm", "bound100", "slot100", "bm100"]  # incumbents from units.json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    units = json.load(open(os.path.join(args.dir, "units.json")))
    tu = [u for u in units if u["task"] != "__all__"]

    rows = []
    for name in args.names:
        curves = load_curves(os.path.join(args.dir, f"curves_{name}.jsonl.gz"))
        by_task = defaultdict(list)
        for rid, cu in curves.items():
            by_task[cu.get("task") or "all"].append(rid)
        for u in tu:
            if u["wl"] != name:
                continue
            rids = by_task.get(u["task"])
            if not rids:
                continue
            met = unit_metrics(curves, rids)
            inc = {k: u.get(k, float("nan")) for k in INC}
            rows.append(dict(wl=name, task=u["task"], g_drlee=u["g_drlee"],
                             g_kim=u["g_kim"], g_struct=u["g_struct"], **inc, **met))

    cols = NEW + INC
    hdr = f"{'unit':<26}{'gDr':>7}{'gKim':>7} |" + "".join(f"{m:>8}" for m in cols)
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["g_drlee"]):
        print(f"{r['wl'] + ':' + r['task']:<26}{r['g_drlee']:>+7.2f}{r['g_kim']:>+7.2f} |"
              + "".join(f"{r[m]:>8.2f}" for m in cols))

    print(f"\n== Spearman across {len(rows)} task units (higher |rho| = better predictor) ==")
    print(f"{'gain':<24}" + "".join(f"{m:>8}" for m in cols))
    out_corr = {}
    for g, lab in (("g_drlee", "Lee:comp-bestsingle"), ("g_kim", "Kim:comp-switch"),
                   ("g_struct", "struct:hoOR-swOR")):
        ys = [r[g] for r in rows]
        line = {m: spearman([r[m] for r in rows], ys) for m in cols}
        out_corr[g] = line
        print(f"{lab:<24}" + "".join(f"{line[m]:>8.2f}" for m in cols))

    # min(|rho_Lee|,|rho_Kim|): the "both agree" score -- a metric both parties
    # endorse must predict BOTH gains, so the weaker of the two is what matters.
    print(f"\n== 'both-agree' score = min(|rho_Lee|, |rho_Kim|) ==")
    ld, lk = out_corr["g_drlee"], out_corr["g_kim"]
    for m in sorted(cols, key=lambda m: -min(abs(ld[m]), abs(lk[m]))):
        print(f"  {m:<10} min={min(abs(ld[m]), abs(lk[m])):.2f}"
              f"   (Lee {ld[m]:+.2f}, Kim {lk[m]:+.2f})")

    if args.out:
        json.dump({"rows": rows, "spearman": out_corr}, open(args.out, "w"), indent=1)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
