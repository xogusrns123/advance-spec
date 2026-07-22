#!/usr/bin/env python3
"""MAT on the WARM SET vs the EVAL SET, per workload, with suffix-tracking
coverage underneath.

  eval set  = held-out odd-parity conversations (all previous MAT figures);
              the suffix corpus is the disjoint warm half.
  warm set  = the corpus conversations THEMSELVES (in-corpus / repeat regime,
              multislot-k0 analog: the tree contains the evaluated conversation's
              own outputs). Captured via capture_traj --eval-set warm.

Outputs (figures/mat/):
  MAT_per_workload_sets.png        deck 3 arms x {eval solid, warm hatched}
                                   + coverage strip underneath
  select_vs_composition_sets.png   all 6 arms, one panel per set + coverage strip

Coverage = pooled share of evaluated positions inside suffix-tracking regions
(copy-run >= 4, gaps <= 12 closed, islands < 16 dropped — the same criterion as
every warm/cold figure), computed from each set's own capture record and cached
in replay_logs/coverage_sets.json (--recompute to refresh).

  python3 scripts/plot_mat_sets.py [--recompute]
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_traj_warmcold import run_cover, segments_from_hits  # same criterion

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "readable_outputs" / "figures" / "replay_logs"
OUT = BASE / "readable_outputs" / "figures" / "mat"

PROPS = ["dflash", "suffix", "select", "select_calib", "chain", "calib"]
DECK = ["dflash", "suffix", "chain"]
LABELS = {"dflash": "DFlash only", "suffix": "Suffix only",
          "select": "Select (raw)", "select_calib": "Select (calibrated)",
          "chain": "Composition (raw)", "calib": "Composition (calibrated)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "select": "#E45756",
          "select_calib": "#B94A8C", "chain": "#54A24B", "calib": "#2CA0A0"}
ORDER = ["qa", "math_reasoning", "web_search", "memory", "swebench", "spider_dbt"]
POOL = {"memory": ["memory_kv", "memory_rec_sum", "memory_vector"]}
DISPLAY = {"qa": "QA\n(Spec-Bench)", "math_reasoning": "Math\n(Spec-Bench)",
           "web_search": "Web Search\n(BFCL v4)", "memory": "Memory\n(BFCL v4)",
           "swebench": "SWE-Bench\n(Verified)", "spider_dbt": "Agentic SQL\n(Spider2-DBT)"}

# per dataset: MAT-log files per set + capture records per set (for coverage)
DS = {
    "bfcl": dict(logs_eval=["mat_bfcl_wl", "mat_bfcl_select"],
                 logs_warm=["mat_bfcl_warmset"],
                 rec_eval="results/perpos_bfcl_full/bfcl_v4_sub3.jsonl",
                 rec_warm="results/perpos_bfcl_full/bfcl_warmset.jsonl"),
    "specbench": dict(logs_eval=["mat_specbench_wl", "mat_specbench_select"],
                      logs_warm=["mat_specbench_warmset"],
                      rec_eval="results/perpos_specbench_full/specbench.jsonl",
                      rec_warm="results/perpos_specbench_full/specbench_warmset.jsonl"),
    "swebench": dict(logs_eval=["mat_swebench", "mat_swebench_select"],
                     logs_warm=["mat_swebench_warmset"],
                     rec_eval="results/perpos_swebench/swebench_quick.jsonl",
                     rec_warm="results/perpos_swebench/swebench_warmset.jsonl",
                     pool="swebench"),
    "spider": dict(logs_eval=["mat_spider_wl", "mat_spider_select"],
                   logs_warm=["mat_spider_warmset"],
                   rec_eval="results/perpos_spider/spider.jsonl",
                   rec_warm="results/perpos_spider/spider_warmset.jsonl",
                   pool="spider_dbt"),
}


def parse_by_task(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s+\[(.+?)\]\s+(.*)", line)
        if m:
            vals = {p: (float(v), int(r) if r else None)
                    for p, v, r in re.findall(r"(\w+)=([0-9.]+)(?:\((\d+)\))?",
                                              m.group(2))}
            if set(vals) & set(PROPS):
                out.setdefault(m.group(1), {}).update(vals)
    return out


def parse_overall(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s+(\w+): K=([0-9.]+)\s+\(rounds=(\d+)\)", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), int(m.group(3)))
    return out


def mat_for_set(which):
    """{workload: {prop: K}} for one set ('eval' or 'warm')."""
    by_task, overall = {}, {}
    for ds, cfg in DS.items():
        texts = [(LOGS / f"{f}.replay.txt").read_text()
                 for f in cfg[f"logs_{which}"] if (LOGS / f"{f}.replay.txt").exists()]
        for t in texts:
            for task, vals in parse_by_task(t).items():
                by_task.setdefault(task, {}).update(vals)
            if "pool" in cfg:
                ov = {}
                for p, v in ((p, v) for t2 in [t] for p, v in parse_overall(t2).items()):
                    ov[p] = v
                if ov:
                    overall.setdefault(cfg["pool"], {}).update(ov)
    K = {t: {p: v[0] for p, v in d.items()} for t, d in by_task.items()}
    for wl, d in overall.items():
        K.setdefault(wl, {}).update({p: v for p, (v, _) in d.items()})
    for w, members in POOL.items():                 # round-weighted pool
        K[w] = {}
        for p in PROPS:
            ks = [by_task[t][p] for t in members if p in by_task.get(t, {})]
            if len(ks) == len(members) and all(r is not None for _, r in ks):
                K[w][p] = sum(k * r for k, r in ks) / sum(r for _, r in ks)
    return K


def coverage_of_record(rec_path):
    """{label: (warm_positions, total_positions)} pooled over units."""
    rp = BASE / rec_path
    traces = json.load(open(rp.with_suffix(".traces.json")))
    ev = traces["eval_traces"]
    sm = defaultdict(dict)
    for l in open(rp):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        sm[r["rid"]][r["pos"]] = int(r.get("suffix_match_warm", 0))
    units = defaultdict(lambda: {"label": None, "s": []})
    for t in ev:
        rid = t["rid"]
        if rid not in sm:
            continue
        u = units[t.get("conv", rid)]
        u["label"] = t.get("task", "all")
        by_pos = sm[rid]
        u["s"].extend(by_pos.get(p, 0) for p in range(1, max(by_pos) + 1))
    agg = defaultdict(lambda: [0, 0])
    for u in units.values():
        segs = segments_from_hits(run_cover(u["s"], 4), 12, 16)
        agg[u["label"]][0] += sum(e - s for s, e in segs)
        agg[u["label"]][1] += len(u["s"])
    return dict(agg)


def coverage_for_set(which, cache, recompute):
    key = which
    if key in cache and not recompute:
        return cache[key]
    per_label = {}
    for ds, cfg in DS.items():
        rec = cfg[f"rec_{which}"]
        if not (BASE / rec).exists():
            continue
        for lab, (w, n) in coverage_of_record(rec).items():
            lab = cfg.get("pool", lab)      # pooled datasets: merge ALL labels
            a = per_label.setdefault(lab, [0, 0])
            a[0] += w; a[1] += n
    cov = {}
    for wl in ORDER:
        members = POOL.get(wl, [wl])
        w = sum(per_label.get(m, [0, 0])[0] for m in members)
        n = sum(per_label.get(m, [0, 0])[1] for m in members)
        if n:
            cov[wl] = w / n
    cache[key] = cov
    return cov


def draw_cov_strip(axc, wls, cov_e, cov_w):
    for ti, w in enumerate(wls):
        for off, cov, hatch in ((-0.17, cov_e.get(w), None), (0.17, cov_w.get(w), "//")):
            if cov is None:
                continue
            b = axc.bar(ti + off, cov * 100, width=0.3, color="#8A97A5",
                        hatch=hatch, edgecolor="white" if hatch else None)
            axc.text(ti + off, cov * 100 + 3, f"{cov * 100:.0f}%", ha="center",
                     va="bottom", fontsize=7.5, color="#555555")
    axc.set_ylim(0, 115)
    axc.set_yticks([0, 50, 100])
    axc.set_ylabel("coverage %", fontsize=8.5)
    axc.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="recompute coverage from records instead of the cache")
    args = ap.parse_args()

    Ke, Kw = mat_for_set("eval"), mat_for_set("warm")
    cache_path = LOGS / "coverage_sets.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    cov_e = coverage_for_set("eval", cache, args.recompute)
    cov_w = coverage_for_set("warm", cache, args.recompute)
    json.dump(cache, open(cache_path, "w"))
    wls = [w for w in ORDER if Ke.get(w) and Kw.get(w)]
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- figure 1: deck arms, paired eval/warm bars + coverage strip --------
    arms = DECK
    n, g = 2 * len(arms), 0.86
    bw = g / n
    fig, (ax, axc) = plt.subplots(2, 1, figsize=(2.2 + 2.3 * len(wls), 6.4),
                                  gridspec_kw={"height_ratios": [4.2, 1.0]},
                                  sharex=True)
    for ai, p in enumerate(arms):
        for si, (K, hatch, tag) in enumerate(((Ke, None, "eval"), (Kw, "//", "warm"))):
            pi = ai * 2 + si
            xs = [ti - g / 2 + bw * (pi + 0.5) for ti in range(len(wls))]
            ys = [K[w].get(p, 0.0) for w in wls]
            ax.bar(xs, ys, width=bw * 0.92, color=COLORS[p], hatch=hatch,
                   edgecolor="white" if hatch else None,
                   label=f"{LABELS[p]} — {tag}")
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.06, f"{y:.2f}", ha="center", va="bottom",
                        fontsize=6.8, color="#555555")
    ymax = max(K[w].get(p, 0) for K in (Ke, Kw) for w in wls for p in arms)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel("mean accept length / round  (tokens)", fontsize=10)
    ax.set_title("MAT — eval set (held-out) vs warm set (in-corpus repeat)   [measured]\n"
                 "Qwen3.5-27B + DFlash + Suffix — suffix corpus = warm half in both; "
                 "3-way disjoint split, full-round replay", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    draw_cov_strip(axc, wls, cov_e, cov_w)
    axc.set_xticks(range(len(wls)))
    axc.set_xticklabels([DISPLAY.get(w, w) for w in wls], fontsize=9.5)
    axc.text(0.005, 0.92, "suffix-tracking coverage (copy-run≥4):  solid = eval set,  hatched = warm set",
             transform=axc.transAxes, fontsize=8, va="top", color="#555555")
    fig.tight_layout()
    fp = OUT / "MAT_per_workload_sets.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")

    # ---- figure 2: all 6 arms, one panel per set + coverage strip -----------
    n, g = len(PROPS), 0.82
    bw = g / n
    fig, (ax1, ax2, axc) = plt.subplots(
        3, 1, figsize=(2.2 + 2.2 * len(wls), 10.0),
        gridspec_kw={"height_ratios": [3.2, 3.2, 1.0]}, sharex=True)
    ymax = max(K[w].get(p, 0) for K in (Ke, Kw) for w in wls for p in PROPS)
    for ax_, K, tag in ((ax1, Ke, "EVAL set (held-out)"),
                        (ax2, Kw, "WARM set (in-corpus repeat)")):
        for pi, p in enumerate(PROPS):
            xs = [ti - g / 2 + bw * (pi + 0.5) for ti in range(len(wls))]
            ys = [K[w].get(p, 0.0) for w in wls]
            ax_.bar(xs, ys, width=bw * 0.90, color=COLORS[p],
                    label=LABELS[p] if ax_ is ax1 else None)
            for x, y in zip(xs, ys):
                ax_.text(x, y + 0.07, f"{y:.2f}", ha="center", va="bottom",
                         fontsize=6.5, color="#555555")
        ax_.set_ylim(0, ymax * 1.16)
        ax_.set_ylabel(f"MAT — {tag}", fontsize=9.5)
        ax_.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax_.spines[sp].set_visible(False)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)
    ax1.set_title("Per-depth competition & composition, raw vs calibrated — "
                  "eval set vs warm set   [measured]", fontsize=10.5)
    draw_cov_strip(axc, wls, cov_e, cov_w)
    axc.set_xticks(range(len(wls)))
    axc.set_xticklabels([DISPLAY.get(w, w) for w in wls], fontsize=9.5)
    axc.text(0.005, 0.92, "suffix-tracking coverage (copy-run≥4):  solid = eval set,  hatched = warm set",
             transform=axc.transAxes, fontsize=8, va="top", color="#555555")
    fig.tight_layout()
    fp = OUT / "select_vs_composition_sets.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")

    for w in wls:
        print(f"  [{w}] cov eval={cov_e.get(w, 0):.0%} warm={cov_w.get(w, 0):.0%}  | "
              + "  ".join(f"{p}={Ke[w].get(p, 0):.2f}/{Kw[w].get(p, 0):.2f}"
                          for p in PROPS))


if __name__ == "__main__":
    main()
