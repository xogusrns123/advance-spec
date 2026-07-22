#!/usr/bin/env python3
"""Does boundary density (winner-change /1K, the adopted team metric) track the
MAT gain of compose over the SD-paper hybrid?  Checked at three granularities,
REPORT ONLY (no figures):

  1. per workload   (5)
  2. per subtask    (task labels inside each workload)
  3. per segment    (think / tool_call / final inside each workload — the same
                     tag() as the MAT_segments figures)

density  : strict winner flips of argmax(dflash_accept, suffix_accept) per 1K tok
           (ties transparent), from the (s, a) curves of the SAME deck records.
gain     : K(compose raw) - K(SD hybrid @ pooled-best tau), from the deck replay
           logs / fallback sweeps / per-segment jsonls (raw compose = default arm).

  PYTHONPATH=/workspace python3 scripts/density_vs_hybridgain.py
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
from replay_segments_5way import piece_offsets, tag  # same segment taxonomy as MAT_segments

RLOG = Path("readable_outputs/figures/replay_logs")
SEG = Path("/workspace/simulation/results/pipeline_4way/segments")

WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
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


def load_curves(p):
    o = {}
    with gzip.open(p, "rt") as f:
        for l in f:
            r = json.loads(l)
            o[r["rid"]] = r
    return o


def winner_seq(S, A):
    return [("d" if a > s else "s" if s > a else None) for s, a in zip(S, A)]


def flips_and_len(S, A):
    win = [w for w in winner_seq(S, A) if w]
    return sum(1 for i in range(1, len(win)) if win[i] != win[i - 1]), len(S)


def flips_by_seg(S, A, cats):
    """strict winner flips attributed to the segment of the later flip position;
    per-seg token counts too. curve idx i <-> output token i+1."""
    w = winner_seq(S, A)
    ntok = defaultdict(int)
    nfl = defaultdict(int)
    for i in range(len(S)):
        seg = cats[i + 1] if i + 1 < len(cats) else cats[-1]
        ntok[seg] += 1
    last_w, last_seg = None, None
    for i in range(len(S)):
        if w[i] is None:
            continue
        seg = cats[i + 1] if i + 1 < len(cats) else cats[-1]
        if last_w is not None and w[i] != last_w:
            nfl[seg] += 1
        last_w, last_seg = w[i], seg
    return nfl, ntok


def parse_bytask(path, prop):
    """by-task K + rounds for one prop from a replay log."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(rf"\s+\[(.+?)\] .*{prop}=([0-9.]+)\((\d+)\)", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), int(m.group(3)))
        m2 = re.match(rf"\s+{prop}: K=([0-9.]+)\s+\(rounds=(\d+)\)", line)
        if m2:
            out["__all__"] = (float(m2.group(1)), int(m2.group(2)))
    return out


def sweep_bytask(ds):
    d = next(iter(json.load(open(SEG / f"fallback_sweep_fresh_{ds}.json")).values()))
    tau, e = max(d.items(), key=lambda kv: kv[1]["K"])
    bt = {t: (kn[0], kn[1]) for t, kn in e.get("by_task", {}).items()}
    bt["__all__"] = (e["K"], e.get("rounds", 0))
    return float(tau), bt


def seg_k(ds, arm, tagsfx=""):
    agg = defaultdict(lambda: [0, 0.0])
    fp = SEG / f"seg_{ds}_{arm}{tagsfx}.jsonl"
    for l in open(fp):
        r = json.loads(l)
        a = agg[r["seg"]]
        a[0] += 1
        a[1] += r["acc"]
    return {s: (v[1] / v[0], v[0]) for s, v in agg.items() if v[0] >= 50}


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

wl_rows, task_rows, seg_rows = [], [], []
for ds in WLS:
    cur = load_curves(CURVES[ds])
    tr = json.load(open(TRACES[ds]))
    ev = {t["rid"]: t for t in tr["eval_traces"]}

    # density: workload + per task + per segment
    F = N = 0
    tF, tN = defaultdict(int), defaultdict(int)
    sF, sN = defaultdict(int), defaultdict(int)
    for rid, c in cur.items():
        t = ev.get(rid)
        if t is None:
            continue
        A = [max(v, 0) for v in c["a"]]
        f, n = flips_and_len(c["s"], A)
        F += f; N += n
        task = c.get("task") or "all"
        tF[task] += f; tN[task] += n
        offs, full = piece_offsets(t["output_ids"], tok)
        cats = tag(full, offs, ds)
        nf, nt = flips_by_seg(c["s"], A, cats)
        for s in nt:
            sF[s] += nf.get(s, 0); sN[s] += nt[s]

    # gains
    craw = parse_bytask(RLOG / f"mat_{ds}_4way_calib_raw.replay.txt", "calib")
    tau, hyb = sweep_bytask(ds)
    dens_wl = 1e3 * F / max(N, 1)
    gain_wl = craw["__all__"][0] - hyb["__all__"][0]
    wl_rows.append(dict(wl=ds, dens=dens_wl, gain=gain_wl,
                        compose=craw["__all__"][0], hybrid=hyb["__all__"][0]))

    for task in sorted(tF):
        if task in craw and task in hyb and tN[task] >= 2000:
            task_rows.append(dict(wl=ds, task=task, dens=1e3 * tF[task] / tN[task],
                                  gain=craw[task][0] - hyb[task][0],
                                  compose=craw[task][0], hybrid=hyb[task][0],
                                  ntok=tN[task]))

    ck = seg_k(ds, "calib", "_raw")
    fk = seg_k(ds, "fallback")
    for s in sN:
        if s in ck and s in fk and sN[s] >= 2000:
            seg_rows.append(dict(wl=ds, seg=s, dens=1e3 * sF[s] / sN[s],
                                 gain=ck[s][0] - fk[s][0],
                                 compose=ck[s][0], hybrid=fk[s][0], ntok=sN[s]))

# ---------------- report ----------------
def table(rows, keys, title):
    print(f"\n== {title} ==")
    hdr = "".join(f"{k:>12}" for k in keys)
    print(f"{'unit':<26}" + hdr)
    for r in rows:
        unit = r["wl"] + (":" + r.get("task", r.get("seg", "")) if ("task" in r or "seg" in r) else "")
        print(f"{unit:<26}" + "".join(
            f"{r[k]:>12.2f}" if isinstance(r[k], float) else f"{r[k]:>12}" for k in keys))


table(sorted(wl_rows, key=lambda r: -r["dens"]), ["dens", "compose", "hybrid", "gain"],
      "WORKLOAD level (density /1K vs gain = compose_raw - hybrid)")
print(f"Spearman(dens, gain) = {spearman([r['dens'] for r in wl_rows], [r['gain'] for r in wl_rows]):+.2f}  (n={len(wl_rows)})")

table(sorted(task_rows, key=lambda r: (r["wl"], -r["dens"])),
      ["dens", "compose", "hybrid", "gain", "ntok"], "SUBTASK level")
print(f"Spearman ACROSS all subtasks = {spearman([r['dens'] for r in task_rows], [r['gain'] for r in task_rows]):+.2f}  (n={len(task_rows)})")
for ds in WLS:
    sub = [r for r in task_rows if r["wl"] == ds]
    if len(sub) >= 3:
        print(f"  within {ds:<10} rho = {spearman([r['dens'] for r in sub], [r['gain'] for r in sub]):+.2f}  (n={len(sub)})")

table(sorted(seg_rows, key=lambda r: (r["wl"], -r["dens"])),
      ["dens", "compose", "hybrid", "gain", "ntok"], "SEGMENT level (within workload)")
for ds in WLS:
    sub = [r for r in seg_rows if r["wl"] == ds]
    if len(sub) >= 3:
        print(f"  within {ds:<10} rho = {spearman([r['dens'] for r in sub], [r['gain'] for r in sub]):+.2f}  (n={len(sub)})")

# violations: units in the wrong quadrant relative to medians
print("\n== VIOLATIONS (high density but gain <= 0, or bottom-half density with top gain) ==")
allr = ([("workload", r) for r in wl_rows] + [("subtask", r) for r in task_rows]
        + [("segment", r) for r in seg_rows])
for lvl, rows in (("workload", wl_rows), ("subtask", task_rows), ("segment", seg_rows)):
    if not rows:
        continue
    med_d = sorted(r["dens"] for r in rows)[len(rows) // 2]
    med_g = sorted(r["gain"] for r in rows)[len(rows) // 2]
    for r in rows:
        unit = r["wl"] + (":" + r.get("task", r.get("seg", "")) if ("task" in r or "seg" in r) else "")
        if r["dens"] >= med_d and r["gain"] <= 0:
            print(f"  [{lvl}] {unit:<24} dens={r['dens']:.1f} (>=med {med_d:.1f}) but gain={r['gain']:+.2f} <= 0")
        elif r["dens"] < med_d and r["gain"] >= med_g and r["gain"] > 0 and lvl != "workload":
            print(f"  [{lvl}] {unit:<24} dens={r['dens']:.1f} (<med {med_d:.1f}) but gain={r['gain']:+.2f} (top half)")

json.dump(dict(workload=wl_rows, subtask=task_rows, segment=seg_rows),
          open("/workspace/simulation/results/pipeline_4way/density_vs_hybridgain.json", "w"), indent=1)
print("\nsaved -> results/pipeline_4way/density_vs_hybridgain.json")
