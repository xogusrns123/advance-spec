#!/usr/bin/env python3
"""Broad candidate bake-off: rank arm-independent workload statistics by their
correlation with the MAT quantities the compose method is judged on.

All candidates are computed from the per-position curves (s = suffix copy depth,
a = DFlash leading match) only -- no compose/arm run -- so any winner can classify
a workload up front. Targets (all in MAT / mean-accepted-token units, from
units.json):
    K_compose  absolute compose MAT
    g_drlee    compose - best single proposer   (Lee baseline)
    g_kim      compose - suffix-hybrid (switch)  (Kim baseline)

Candidate families
  mass/share : warm2/4/8 = P(s>=th) ; dwarm4 = P(a>=4) ; either4 = P(max>=4) ;
               both4 = P(min>=4) ; xor4 = P((s>=4) xor (a>=4)) ; win_s = P(s>a)
  depth      : mean_s ; mean_a ; mean_max = E[max(a,s)] (oracle per-pos accept) ;
               mean_min ; orgain = mean_max - max(mean_s,mean_a)
  complement : cmpl = E[max-min] ; anticorr = -pearson(a,s)
  event      : sw100 (argmax flip / 100) ; swgain (margin-wtd) ; swmass (depth-wtd)
  grafting   : meanG = E[G(p)] parameter-free grafting headroom, computed directly
               on the curves -- G(p)=max_{0<=k<=a(p)}[k+min(s(p+k),B-k)]-max(a(p),s(p)).
               (the sim-free form of the study's established winner C_graft)

  python3 scripts/metric_bakeoff.py --names spider swebench bfcl specbench
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse, gzip, json, math, os
from collections import defaultdict

B_BLOCK = 32  # num_spec, system block budget used across the study
TH = 4        # warm threshold used in the study


def load_curves(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            out[r["rid"]] = r
    return out


def graft_headroom(a, s, p, B=B_BLOCK):
    base = max(a[p], s[p])
    best = base
    kmax = min(a[p], B)
    L = len(s)
    for k in range(0, kmax + 1):
        if p + k >= L:
            tail = 0
        else:
            tail = min(s[p + k], B - k)
        best = max(best, k + tail)
    return best - base


def winner_lab(a, s):
    lab, prev = [], 0
    for ai, si in zip(a, s):
        w = 1 if ai > si else (-1 if si > ai else prev)
        lab.append(w)
        if w:
            prev = w
    return lab


def call_accum(cu, acc):
    a, s = cu["a"], cu["s"]
    L = len(s)
    if L == 0:
        return
    acc["L"] += L
    lab = winner_lab(a, s)
    for p in range(L):
        ai, si = a[p], s[p]
        mx, mn = (ai, si) if ai >= si else (si, ai)
        acc["sum_s"] += si
        acc["sum_a"] += ai
        acc["sum_max"] += mx
        acc["sum_min"] += mn
        acc["cmpl"] += mx - mn
        acc["w2"] += si >= 2
        acc["w4"] += si >= 4
        acc["w8"] += si >= 8
        acc["dw4"] += ai >= 4
        acc["either4"] += mx >= 4
        acc["both4"] += mn >= 4
        acc["xor4"] += (si >= 4) != (ai >= 4)
        acc["win_s"] += si > ai
        acc["G"] += graft_headroom(a, s, p)
        # pearson accumulators for anticorr(a,s)
        acc["saa"] += ai * ai
        acc["sss"] += si * si
        acc["sas"] += ai * si
        if p + 1 < L and lab[p + 1] and lab[p] and lab[p + 1] != lab[p]:
            acc["flips"] += 1
            acc["mass"] += max(a[p + 1], s[p + 1])
            acc["gain"] += abs(a[p + 1] - s[p + 1])


def unit_metrics(curves, rids):
    acc = defaultdict(float)
    for rid in rids:
        call_accum(curves[rid], acc)
    L = acc["L"] or 1.0
    ea, es = acc["sum_a"] / L, acc["sum_s"] / L
    # pooled pearson(a,s)
    cov = acc["sas"] / L - ea * es
    va = acc["saa"] / L - ea * ea
    vs = acc["sss"] / L - es * es
    pear = cov / math.sqrt(va * vs) if va > 0 and vs > 0 else 0.0
    return dict(
        warm2=acc["w2"] / L, warm4=acc["w4"] / L, warm8=acc["w8"] / L,
        dwarm4=acc["dw4"] / L, either4=acc["either4"] / L, both4=acc["both4"] / L,
        xor4=acc["xor4"] / L, win_s=acc["win_s"] / L,
        mean_s=es, mean_a=ea, mean_max=acc["sum_max"] / L, mean_min=acc["sum_min"] / L,
        orgain=acc["sum_max"] / L - max(ea, es),
        cmpl=acc["cmpl"] / L, anticorr=-pear,
        sw100=100.0 * acc["flips"] / L,
        swgain=100.0 * acc["gain"] / L,
        swmass=100.0 * acc["mass"] / L,
        meanG=acc["G"] / L,
    )


CANDS = ["warm2", "warm4", "warm8", "dwarm4", "either4", "both4", "xor4", "win_s",
         "mean_s", "mean_a", "mean_max", "mean_min", "orgain", "cmpl", "anticorr",
         "sw100", "swgain", "swmass", "meanG"]


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            for k in range(i, j):
                r[order[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys); n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    units = [u for u in json.load(open(os.path.join(args.dir, "units.json")))
             if u["task"] != "__all__"]

    rows = []
    for name in args.names:
        curves = load_curves(os.path.join(args.dir, f"curves_{name}.jsonl.gz"))
        by_task = defaultdict(list)
        for rid, cu in curves.items():
            by_task[cu.get("task") or "all"].append(rid)
        for u in units:
            if u["wl"] != name:
                continue
            rids = by_task.get(u["task"])
            if not rids:
                continue
            met = unit_metrics(curves, rids)
            rows.append(dict(wl=name, task=u["task"],
                             K_compose=u["K"]["compose"],
                             g_drlee=u["g_drlee"], g_kim=u["g_kim"], **met))

    TARGETS = [("K_compose", "compose MAT (absolute)"),
               ("g_drlee", "Lee gain (comp-best single)"),
               ("g_kim", "Kim gain (comp-hybrid)")]
    corr = {}
    for t, _ in TARGETS:
        ys = [r[t] for r in rows]
        corr[t] = {m: spearman([r[m] for r in rows], ys) for m in CANDS}

    print(f"== Spearman across {len(rows)} task units ==")
    print(f"{'candidate':<12}" + "".join(f"{lab.split()[0]:>10}" for _, lab in TARGETS)
          + f"{'min|Lee,Kim|':>14}")
    def bothscore(m):
        return min(abs(corr['g_drlee'][m]), abs(corr['g_kim'][m]))
    for m in sorted(CANDS, key=lambda m: -bothscore(m)):
        print(f"{m:<12}" + "".join(f"{corr[t][m]:>10.2f}" for t, _ in TARGETS)
              + f"{bothscore(m):>14.2f}")

    print("\n== ranked by 'both-agree' = min(|rho_Lee|,|rho_Kim|) ==")
    for i, m in enumerate(sorted(CANDS, key=lambda m: -bothscore(m))[:8], 1):
        print(f"  {i}. {m:<10} both={bothscore(m):.2f}  "
              f"(Lee {corr['g_drlee'][m]:+.2f}, Kim {corr['g_kim'][m]:+.2f}, "
              f"MAT {corr['K_compose'][m]:+.2f})")

    if args.out:
        json.dump({"rows": rows, "spearman": corr,
                   "targets": [t for t, _ in TARGETS]}, open(args.out, "w"), indent=1)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
