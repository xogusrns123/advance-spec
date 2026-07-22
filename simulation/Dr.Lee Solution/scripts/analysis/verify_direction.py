#!/usr/bin/env python3
"""VERIFY the hypothesis: compose = [DFlash head][Suffix tail], so within ONE verify
step it can only cross a DFlash->Suffix boundary (dflash predicts, then suffix copies).
A Suffix->DFlash handoff in one step is architecturally impossible.

Checks:
 (1) structural: every compose 'bridge' round (head>0 AND tail>0) is DFlash-then-
     Suffix; there is NO field/path for the reverse. (confirmed by construction;
     we just count realized bridges from the RAW-compose rounds.)
 (2) directional winner-change boundaries per 1K from the curves:
     D->S (dflash-wins region -> suffix-wins region)  = bridgeable by compose
     S->D (suffix-wins -> dflash-wins)                 = NOT bridgeable
     (they nearly alternate, so counts ~ equal -> only ~half are exploitable).
 (3) do realized bridge rounds line up with the D->S boundary count?

  python3 scripts/verify_direction.py
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import gzip, json
from collections import defaultdict

IDIR = "results/interp_validation"
RAW = "results/mat_raw"
WL = ["spider", "swebench", "bfcl", "specbench"]


def load(p):
    o = {}
    with gzip.open(p, "rt") as f:
        for l in f:
            r = json.loads(l)
            o.setdefault("_list", []).append(r)
    return o["_list"]


def winner_dirs(S, A):
    """carry-forward winner; count D->S and S->D transitions."""
    raw = [("d" if ai > si else "s" if si > ai else None) for si, ai in zip(S, A)]
    prev = next((w for w in raw if w), "s")
    seq = []
    for w in raw:
        prev = w or prev
        seq.append(prev)
    ds = sd = 0
    for i in range(1, len(seq)):
        if seq[i - 1] == "d" and seq[i] == "s":
            ds += 1
        elif seq[i - 1] == "s" and seq[i] == "d":
            sd += 1
    return ds, sd, len(seq)


units = [u for u in json.load(open(f"{IDIR}/units.json")) if u["task"] != "__all__"]
rows = []
print(f"{'workload':<10}{'D->S/1K':>9}{'S->D/1K':>9}{'ratio':>7} | "
      f"{'rounds':>7}{'bridge':>8}{'bridge%':>8}{'brdgTok%':>9}{'reverse':>8}")
for wl in WL:
    cur = {r["rid"]: r for r in load(f"{IDIR}/curves_{wl}.jsonl.gz")}
    by_task = defaultdict(list)
    for rid, c in cur.items():
        by_task[c.get("task") or "all"].append(rid)
    # per-task rows for correlation
    for u in units:
        if u["wl"] != wl:
            continue
        rids = by_task.get(u["task"])
        if not rids:
            continue
        DS = SD = NN = 0
        for rid in rids:
            c = cur[rid]
            A = [max(v, 0) for v in c["a"]]
            ds, sd, n = winner_dirs(c["s"], A)
            DS += ds; SD += sd; NN += n
        NN = NN or 1
        rows.append(dict(wl=wl, task=u["task"], g_drlee=u["g_drlee"], g_kim=u["g_kim"],
                         ds1k=1e3 * DS / NN, sd1k=1e3 * SD / NN))
    # workload totals + rounds
    DS = SD = NN = 0
    for rid, c in cur.items():
        A = [max(v, 0) for v in c["a"]]
        ds, sd, n = winner_dirs(c["s"], A)
        DS += ds; SD += sd; NN += n
    rnd = load(f"{RAW}/rounds_{wl}.jsonl.gz")
    nr = len(rnd)
    brdg = [r for r in rnd if r["head"] > 0 and r["tail"] > 0]  # D->S realized
    reverse = [r for r in rnd if r.get("head", 0) < 0]          # impossible
    tot_tok = sum(r["acc"] for r in rnd) or 1
    brdg_tok = sum(r["acc"] for r in brdg)
    print(f"{wl:<10}{1e3*DS/NN:>9.1f}{1e3*SD/NN:>9.1f}{DS/max(SD,1):>7.2f} | "
          f"{nr:>7}{len(brdg):>8}{100*len(brdg)/max(nr,1):>7.1f}%"
          f"{100*brdg_tok/tot_tok:>8.1f}%{len(reverse):>8}")


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


print(f"\n== Spearman (21 units) ==\n{'metric':<12}{'vs Lee':>9}{'vs Kim':>9}")
for m, lab in (("ds1k", "D->S /1K"), ("sd1k", "S->D /1K")):
    print(f"{lab:<12}{spearman([r[m] for r in rows], [r['g_drlee'] for r in rows]):>9.2f}"
          f"{spearman([r[m] for r in rows], [r['g_kim'] for r in rows]):>9.2f}")
