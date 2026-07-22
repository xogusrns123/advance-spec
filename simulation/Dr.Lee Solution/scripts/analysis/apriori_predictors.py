#!/usr/bin/env python3
"""A-PRIORI (output-text-only) predictors of compose gain.

The winning metrics so far (warm4, either4, C=sqrt(so*do)) all use the per-position
curves s (suffix copy depth) and a (DFlash leading match). a REQUIRES running the
DFlash neural draft model -> not knowable before deployment. This script asks: can
we predict compose gain from the OUTPUT TOKENS ALONE, without running either
speculative proposer?

Key fact: s (suffix copyability) is a deterministic repetition statistic of the
token stream -- no neural model. So we test model-free repetition proxies that need
NO proposer at all:

  rep_n   fraction of output positions whose trailing n-gram already occurred earlier
          in (prompt + output-so-far)  -- pure n-gram self-repetition
  gzipr   1 - gzip(output_bytes)/len(output_bytes)  -- generic redundancy
  warm4   P(s>=4) from the study curves (suffix-tree copyability; no neural model,
          shown here to be reproduced by rep_n)

Correlated (Spearman, 21 task units) with g_drlee (Lee gain), g_kim (Kim gain),
K_compose (abs MAT). Runs on host: only needs output_ids/prompt_ids + curves.

  python3 scripts/apriori_predictors.py --names spider swebench bfcl specbench
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse, gzip, json, math, os, struct
from collections import defaultdict

WL_SRC = {"bfcl": "perpos_bfcl_full/bfcl_v4_full",
          "specbench": "perpos_specbench_full/specbench",
          "swebench": "perpos_swebench_alleval/swebench_4way",
          "spider": "perpos_spider_alleval/spider_4way"}


def load_curves(p):
    o = {}
    with gzip.open(p, "rt") as f:
        for l in f:
            r = json.loads(l)
            o[r["rid"]] = r
    return o


def rep_rate(hist, out, n):
    """fraction of output positions i whose n-gram ending at i appeared earlier
    in hist+out[:i]. hist = prompt context (seed the n-gram table)."""
    seen = set()
    seq = hist + out
    h0 = len(hist)
    hit = tot = 0
    for i in range(len(seq)):
        if i + 1 >= n:
            g = tuple(seq[i - n + 1:i + 1])
            if i >= h0:  # scoring only output positions
                tot += 1
                if g in seen:
                    hit += 1
            seen.add(g)
    return hit / tot if tot else 0.0


def gzip_ratio(out):
    b = b"".join(struct.pack("<i", t) for t in out)
    if not b:
        return 0.0
    return 1.0 - len(gzip.compress(b, 6)) / len(b)


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
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
    rx, ry = rank(xs), rank(ys)
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
    for wl in args.names:
        cur = load_curves(os.path.join(args.dir, f"curves_{wl}.jsonl.gz"))
        tr = json.load(open(f"results/{WL_SRC[wl]}.traces.json"))
        ev = {t["rid"]: t for t in tr["eval_traces"]}
        by_task = defaultdict(list)
        for rid, c in cur.items():
            by_task[c.get("task") or "all"].append(rid)
        for u in units:
            if u["wl"] != wl:
                continue
            rids = [r for r in by_task.get(u["task"], []) if r in ev]
            if not rids:
                continue
            acc = defaultdict(float)
            for rid in rids:
                t = ev[rid]
                out = t["output_ids"]
                hist = t.get("prompt_ids", [])
                s = cur[rid]["s"]
                acc["L"] += len(s)
                acc["w4"] += sum(1 for v in s if v >= 4)
                acc["Lo"] += len(out)
                acc["r2"] += rep_rate(hist, out, 2) * len(out)
                acc["r4"] += rep_rate(hist, out, 4) * len(out)
                acc["r8"] += rep_rate(hist, out, 8) * len(out)
                acc["gz"] += gzip_ratio(out) * len(out)
            L = acc["L"] or 1.0
            Lo = acc["Lo"] or 1.0
            rows.append(dict(wl=wl, task=u["task"], g_drlee=u["g_drlee"],
                             g_kim=u["g_kim"], K=u["K"]["compose"],
                             warm4=acc["w4"] / L,
                             rep2=acc["r2"] / Lo, rep4=acc["r4"] / Lo,
                             rep8=acc["r8"] / Lo, gzipr=acc["gz"] / Lo))

    METS = ["warm4", "rep2", "rep4", "rep8", "gzipr"]
    hdr = f"{'unit':<26}{'gDr':>7}{'gKim':>7}{'MAT':>6} |" + "".join(f"{m:>8}" for m in METS)
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["g_drlee"]):
        print(f"{r['wl'] + ':' + r['task']:<26}{r['g_drlee']:>+7.2f}{r['g_kim']:>+7.2f}"
              f"{r['K']:>6.2f} |" + "".join(f"{r[m]:>8.3f}" for m in METS))

    print(f"\n== Spearman across {len(rows)} task units  (MODEL-FREE except warm4) ==")
    corr = {}
    print(f"{'target':<20}" + "".join(f"{m:>8}" for m in METS))
    for g, lab in (("g_drlee", "Lee gain"), ("g_kim", "Kim gain"), ("K", "abs MAT")):
        ys = [r[g] for r in rows]
        line = {m: spearman([r[m] for r in rows], ys) for m in METS}
        corr[g] = line
        print(f"{lab:<20}" + "".join(f"{line[m]:>8.2f}" for m in METS))

    if args.out:
        try:
            json.dump({"rows": rows, "spearman": corr}, open(args.out, "w"), indent=1)
            print("saved ->", args.out)
        except Exception as e:
            print("(no save:", e, ")")


if __name__ == "__main__":
    main()
