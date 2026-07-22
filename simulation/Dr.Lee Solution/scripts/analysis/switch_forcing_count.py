# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import os
#!/usr/bin/env python3
"""Kim's metric is a COUNT (boundaries crossed = verify steps the SD hybrid saves).
Test WHICH boundary count predicts the gain -- the issue may be that the current
count includes boundaries that DON'T force the hybrid to switch proposers.

A boundary forces the hybrid an extra verify step only if the winning proposer
genuinely CHANGES -- i.e. between a suffix-ONLY region (dflash fails) and a
dflash-ONLY region (suffix fails). Where BOTH work, the hybrid need not switch,
so that adjacency is a FREE (non-switch-forcing) boundary and should not count.

Candidates (all COUNTS, per 1K tokens):
  bnd_pri   traj_regions s<->d, suffix-priority (both lumped into s)  = TEAM metric
  sw_argmax pointwise argmax(a,s) flips (ties inherit)                 = every winner flip
  sw_excl   pointwise handoffs between suffix-ONLY and dflash-ONLY
            (both / neither positions are transparent -- skipped)      = switch-FORCING count

Correlated (Spearman, 21 units) with g_drlee / g_kim.
"""
import gzip, json, os, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "scripts")
from plot_traj_warmcold import run_cover, segments_from_hits

RUN_LEN, GAP, MIN_SEG, TH = 4, 12, 16, 4
WL = {"spider": None, "swebench": None, "bfcl": None, "specbench": None}
IDIR = os.environ.get("INTERP_DIR", "results/interp_validation")


def mask(segs, n):
    m = [False] * n
    for s, e in segs:
        for i in range(s, e):
            m[i] = True
    return m


def counts(S, A):
    n = len(S)
    # (1) team: suffix-priority smoothed regions, all s<->d adjacencies
    m_s = mask(segments_from_hits(run_cover(S, RUN_LEN), GAP, MIN_SEG), n)
    m_a = mask(segments_from_hits(run_cover(A, RUN_LEN), GAP, MIN_SEG), n)
    cat = ["s" if ms else ("d" if ma else "n") for ms, ma in zip(m_s, m_a)]
    runs, st = [], 0
    for i in range(1, n + 1):
        if i == n or cat[i] != cat[st]:
            runs.append(cat[st]); st = i
    bnd_pri = sum(1 for i in range(1, len(runs)) if {runs[i - 1], runs[i]} == {"s", "d"})

    # (2) pointwise argmax flips (tie inherits previous)
    lab, prev, argf = [], 0, 0
    for ai, si in zip(A, S):
        w = 1 if ai > si else (-1 if si > ai else prev)
        lab.append(w)
        if w:
            prev = w
    for i in range(n - 1):
        if lab[i] and lab[i + 1] and lab[i] != lab[i + 1]:
            argf += 1

    # (3) switch-FORCING (pointwise): handoffs between suffix-ONLY and dflash-ONLY,
    #     both/neither positions transparent
    excl = []
    for ai, si in zip(A, S):
        if si >= TH and ai < TH:
            excl.append("so")
        elif ai >= TH and si < TH:
            excl.append("do")
    sw_excl = sum(1 for i in range(1, len(excl)) if excl[i] != excl[i - 1])

    # (4) switch-FORCING (SMOOTHED, region-based -- consistent w/ traj_regions):
    #     exclusive regions from the SAME smoothed masks; SO = m_s & ~m_a,
    #     DO = m_a & ~m_s; both/neither transparent; count SO<->DO region handoffs.
    own = []
    for ms, ma in zip(m_s, m_a):
        if ms and not ma:
            own.append("SO")
        elif ma and not ms:
            own.append("DO")
    sw_excl_sm = sum(1 for i in range(1, len(own)) if own[i] != own[i - 1])

    # (5) THRESHOLD-FREE winner flip (user's definition): winner = argmax(a, s);
    #     ties (a==s: both fail=0, or both succeed equally) have NO winner ->
    #     transparent; a boundary is where the strict winner changes S<->D.
    win = []
    for ai, si in zip(A, S):
        if ai > si:
            win.append("D")
        elif si > ai:
            win.append("S")
        # a == s -> tie, skip (no winner)
    sw_win = sum(1 for i in range(1, len(win)) if win[i] != win[i - 1])
    return bnd_pri, argf, sw_excl, sw_excl_sm, sw_win, n


def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        od = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j < n and v[od[j]] == v[od[i]]:
                j += 1
            for k in range(i, j):
                r[od[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


units = [u for u in json.load(open(f"{IDIR}/units.json")) if u["task"] != "__all__"]
rows = []
wl_agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
for wl in WL:
    cur = {}
    with gzip.open(f"{IDIR}/curves_{wl}.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l); cur[r["rid"]] = r
    by_task = defaultdict(list)
    for rid, c in cur.items():
        by_task[c.get("task") or "all"].append(rid)
    for u in units:
        if u["wl"] != wl:
            continue
        rids = by_task.get(u["task"])
        if not rids:
            continue
        acc = [0, 0, 0, 0, 0, 0]
        for rid in rids:
            c = cur[rid]
            A = [max(v, 0) for v in c["a"]]
            res = counts(c["s"], A)
            for k, v in enumerate(res):
                acc[k] += v
        N = acc[5] or 1
        rows.append(dict(wl=wl, task=u["task"], g_drlee=u["g_drlee"], g_kim=u["g_kim"],
                         bnd_pri=1e3 * acc[0] / N, sw_argmax=1e3 * acc[1] / N,
                         sw_excl=1e3 * acc[2] / N, sw_excl_sm=1e3 * acc[3] / N,
                         sw_win=1e3 * acc[4] / N))
        wa = wl_agg[wl]
        for k in range(6):
            wa[k] += acc[k]

METS = ["bnd_pri", "sw_argmax", "sw_excl", "sw_excl_sm", "sw_win"]
print("== per-workload counts /1K ==")
print(f"{'wl':<10}{'bnd_pri':>9}{'sw_argmax':>11}{'sw_excl':>9}{'sw_excl_sm':>12}{'sw_win':>9}")
for wl in WL:
    wa = wl_agg[wl]; N = wa[5] or 1
    print(f"{wl:<10}{1e3*wa[0]/N:>9.1f}{1e3*wa[1]/N:>11.1f}{1e3*wa[2]/N:>9.1f}"
          f"{1e3*wa[3]/N:>12.1f}{1e3*wa[4]/N:>9.1f}")
print(f"\n== Spearman (21 units) ==\n{'metric':<11}{'vs Lee':>9}{'vs Kim':>9}")
for m in METS:
    print(f"{m:<11}{spearman([r[m] for r in rows], [r['g_drlee'] for r in rows]):>9.2f}"
          f"{spearman([r[m] for r in rows], [r['g_kim'] for r in rows]):>9.2f}")
json.dump(rows, open(f"{IDIR}/switch_forcing_count.json", "w"), indent=1)
