#!/usr/bin/env python3
"""WHICH workload statistic explains the compose-over-hybrid MAT gap, at ALL three
granularities (workload / subtask / segment-within-workload)?

Density (winner-flip count /1K) fails inside workloads (bfcl subtasks inverted,
think vs tool_call inverted everywhere). Candidates that price the VALUE of a
boundary, all per-position and therefore segment-attributable:

  dens    strict winner flips /1K                        (count -- the team metric)
  vflip   Σ over flips of max(a,s) at the flip, /1K      (depth-weighted count)
  meanG   mean_p G(p),  G = max_{0<=k<=a(p)}[k + min(s(p+k), B-k)] - max(a(p),s(p))
          (grafting headroom: what packing head+tail into ONE step adds over the
           better single proposer at p -- the compose-vs-switch mechanism quantity)
  C4      sqrt(P(s>=4 & a<4) * P(a>=4 & s<4))            (complementary coverage)
  warm4   P(s>=4)                                        (suffix-copyable share)

Gains re-used from density_vs_hybridgain.json (raw compose - hybrid @ tau*).
Extends that json with the candidate metrics + prints the correlation matrix.

  PYTHONPATH=/workspace python3 scripts/gapmetric_candidates.py
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
from replay_segments_5way import piece_offsets, tag

B = 32
IN = Path("/workspace/simulation/results/pipeline_4way/density_vs_hybridgain.json")
CURVES = {"specbench": "results/deck_curves/curves_specbench.jsonl.gz",
          "bfcl": "results/deck_curves/curves_bfcl.jsonl.gz",
          "swebench": "results/interp_validation/curves_swebench.jsonl.gz",
          "spider": "results/interp_validation/curves_spider.jsonl.gz",
          "tau2": "results/interp_validation/curves_tau2.jsonl.gz"}
TRACES = {"specbench": "results/perpos_specbench_alleval/specbench_4way.traces.json",
          "bfcl": "results/perpos_bfcl_alleval/bfcl_4way.traces.json",
          "swebench": "results/perpos_swebench_alleval/swebench_4way.traces.json",
          "spider": "results/perpos_spider_alleval/spider_4way.traces.json",
          "tau2": "results/perpos_tau2_alleval/tau2_4way.traces.json"}
METS = ["dens", "vflip", "meanG", "C4", "warm4"]


def graft_gain(S, A, p):
    a, s = A[p], S[p]
    base = max(a, s)
    best = base
    L = len(S)
    for k in range(0, min(a, B) + 1):
        tail = min(S[p + k], B - k) if p + k < L else 0
        v = k + tail
        if v > best:
            best = v
    return best - base


class Acc:
    __slots__ = ("n", "flips", "vflip", "G", "so", "do", "w4")
    def __init__(self):
        self.n = self.flips = self.so = self.do = self.w4 = 0
        self.vflip = self.G = 0.0
    def met(self):
        n = self.n or 1
        so, do = self.so / n, self.do / n
        return dict(dens=1e3 * self.flips / n, vflip=1e3 * self.vflip / n,
                    meanG=self.G / n, C4=(so * do) ** 0.5, warm4=self.w4 / n)


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
    rx, ry = rank(xs), rank(ys); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")

D = json.load(open(IN))
acc_wl, acc_task, acc_seg = {}, {}, {}
for ds in CURVES:
    cur = {}
    with gzip.open(CURVES[ds], "rt") as f:
        for l in f:
            r = json.loads(l)
            cur[r["rid"]] = r
    ev = {t["rid"]: t for t in json.load(open(TRACES[ds]))["eval_traces"]}
    for rid, c in cur.items():
        t = ev.get(rid)
        if t is None:
            continue
        S = c["s"]
        A = [max(v, 0) for v in c["a"]]
        offs, full = piece_offsets(t["output_ids"], tok)
        cats = tag(full, offs, ds)
        task = c.get("task") or "all"
        a_wl = acc_wl.setdefault(ds, Acc())
        a_tk = acc_task.setdefault((ds, task), Acc())
        last_w = None
        for i in range(len(S)):
            seg = cats[i + 1] if i + 1 < len(cats) else cats[-1]
            a_sg = acc_seg.setdefault((ds, seg), Acc())
            s, a = S[i], A[i]
            w = "d" if a > s else ("s" if s > a else None)
            flip = 1 if (w and last_w and w != last_w) else 0
            if w:
                last_w = w
            g = graft_gain(S, A, i)
            vf = max(a, s) if flip else 0
            so = 1 if (s >= 4 and a < 4) else 0
            do = 1 if (a >= 4 and s < 4) else 0
            w4 = 1 if s >= 4 else 0
            for ac in (a_wl, a_tk, a_sg):
                ac.n += 1; ac.flips += flip; ac.vflip += vf
                ac.G += g; ac.so += so; ac.do += do; ac.w4 += w4
    print(f"[{ds}] done", flush=True)

# attach metrics to the gain rows
for r in D["workload"]:
    r.update(acc_wl[r["wl"]].met())
for r in D["subtask"]:
    r.update(acc_task[(r["wl"], r["task"])].met())
for r in D["segment"]:
    r.update(acc_seg[(r["wl"], r["seg"])].met())
json.dump(D, open(IN.with_name("gapmetric_candidates.json"), "w"), indent=1)

# ---- correlation report ----
def corr_line(rows, lab):
    ys = [r["gain"] for r in rows]
    print(f"{lab:<26}" + "".join(f"{spearman([r[m] for r in rows], ys):>9.2f}" for m in METS)
          + f"  (n={len(rows)})")

print("\n== Spearman vs gain (compose_raw - hybrid) ==")
print(f"{'level':<26}" + "".join(f"{m:>9}" for m in METS))
corr_line(D["workload"], "workload (n=5)")
corr_line(D["subtask"], "subtask pooled")
for ds in CURVES:
    sub = [r for r in D["subtask"] if r["wl"] == ds]
    if len(sub) >= 4:
        corr_line(sub, f"  subtask within {ds}")
corr_line(D["segment"], "segment pooled")
segsub = [r for r in D["segment"]]
# within-workload segment sign check (n=2-3 each): report value ordering matches
print("\n== segment within-workload ordering check (does metric rank tool_call above think?) ==")
for ds in CURVES:
    sub = {r["seg"]: r for r in D["segment"] if r["wl"] == ds}
    if "think" in sub and "tool_call" in sub:
        gwin = "tool_call" if sub["tool_call"]["gain"] > sub["think"]["gain"] else "think"
        line = f"{ds:<11} gain-winner={gwin:<9}"
        for m in METS:
            mwin = "tool_call" if sub["tool_call"][m] > sub["think"][m] else "think"
            line += f"  {m}:{'OK ' if mwin == gwin else 'X  '}"
        print(line)
print("\nsaved -> results/pipeline_4way/gapmetric_candidates.json")
