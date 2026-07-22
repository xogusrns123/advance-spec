#!/usr/bin/env python3
"""MAT per workload under the clean TWO-WAY protocol (user-directed canonical):

  suffix tree  <- warm set (even / label-rank-even conversations) ONLY
  test         <- the ENTIRE eval set (no calibrate split — calibration excluded)
  arms         <- RAW signals only: DFlash, Suffix, Select (per-depth
                  first-crossing), Composition (argmax) — no calibrated arms

Coverage strip underneath: per workload, share of eval-set positions inside
suffix-tracking regions (copy-run >= 4, same criterion as every warm/cold
figure), computed from each record and cached in replay_logs/coverage_2way.json.

Sources: replay_logs/mat_{bfcl,swebench,specbench,spider}_2way.replay.txt
(replay_extension WITHOUT --three-way -> full eval set; bfcl uses the FULL
record, 84 eval convs, not the sub3 subset).

  python3 scripts/plot_mat_2way.py [--full] [--recompute]
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_mat_sets import coverage_of_record  # same criterion + loader

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "readable_outputs" / "figures" / "replay_logs"
OUT = BASE / "readable_outputs" / "figures" / "mat"

PROPS = ["dflash", "suffix", "select", "chain"]
PROPS_CALIB = ["dflash", "suffix", "select", "select_calib", "chain", "calib"]
LABELS = {"dflash": "DFlash only (chain)", "suffix": "Suffix only (chain)",
          "select": "Select (per-depth, raw)", "chain": "Composition (raw)",
          "select_calib": "Select (calibrated)", "calib": "Composition (calibrated)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "select": "#E45756",
          "chain": "#54A24B", "select_calib": "#B94A8C", "calib": "#2CA0A0"}
ORDER = ["qa", "math_reasoning", "web_search", "memory", "swebench", "spider_dbt"]
ORDER_FULL = ["memory_kv", "memory_rec_sum", "memory_vector", "web_search",
              "swebench", "spider_dbt", "writing", "roleplay", "translation",
              "summarization", "qa", "math_reasoning", "rag"]
POOL = {"memory": ["memory_kv", "memory_rec_sum", "memory_vector"]}
DISPLAY = {"qa": "QA\n(Spec-Bench)", "math_reasoning": "Math\n(Spec-Bench)",
           "web_search": "Web Search\n(BFCL v4)", "memory": "Memory\n(BFCL v4)",
           "swebench": "SWE-Bench\n(Verified)", "spider_dbt": "Agentic SQL\n(Spider2-DBT)",
           "memory_kv": "Memory KV\n(BFCL v4)", "memory_rec_sum": "Memory RecSum\n(BFCL v4)",
           "memory_vector": "Memory Vector\n(BFCL v4)", "writing": "Writing\n(Spec-Bench)",
           "roleplay": "Roleplay\n(Spec-Bench)", "translation": "Translation\n(Spec-Bench)",
           "summarization": "Summarization\n(Spec-Bench)", "rag": "RAG\n(Spec-Bench)"}
DS = {
    "bfcl": dict(log="mat_bfcl_2way", rec="results/perpos_bfcl_full/bfcl_v4_full.jsonl"),
    "specbench": dict(log="mat_specbench_2way",
                      rec="results/perpos_specbench_full/specbench.jsonl"),
    "swebench": dict(log="mat_swebench_2way", pool="swebench",
                     rec="results/perpos_swebench/swebench_quick.jsonl"),
    "spider": dict(log="mat_spider_2way", pool="spider_dbt",
                   rec="results/perpos_spider/spider.jsonl"),
}


def parse_by_task(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s+\[(.+?)\]\s+(.*)", line)
        if m:
            vals = {p: (float(v), int(r) if r else None)
                    for p, v, r in re.findall(r"(\w+)=([0-9.]+)(?:\((\d+)\))?",
                                              m.group(2))}
            if set(vals) & set(PROPS_CALIB):
                out.setdefault(m.group(1), {}).update(vals)
    return out


def parse_overall(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s+(\w+): K=([0-9.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--calib", action="store_true",
                    help="add the IN-SAMPLE calibrated arms (calibrate = test = "
                         "full eval set; logs mat_*_2way_calib) -> *_calib.png")
    args = ap.parse_args()
    props = PROPS_CALIB if args.calib else PROPS

    by_task, K = {}, {}
    for ds, cfg in DS.items():
        logs = [cfg["log"]] + ([cfg["log"] + "_calib"] if args.calib else [])
        for lg in logs:
            text = (LOGS / f"{lg}.replay.txt").read_text()
            if "pool" in cfg:
                K.setdefault(cfg["pool"], {}).update(parse_overall(text))
            else:
                for t, vals in parse_by_task(text).items():
                    by_task.setdefault(t, {}).update(vals)
    for t, d in by_task.items():
        K.setdefault(t, {}).update({p: v[0] for p, v in d.items()})
    for w, members in POOL.items():
        K[w] = {}
        for p in props:
            ks = [by_task[t][p] for t in members if p in by_task.get(t, {})]
            if len(ks) == len(members) and all(r is not None for _, r in ks):
                K[w][p] = sum(k * r for k, r in ks) / sum(r for _, r in ks)

    cache_path = LOGS / "coverage_2way.json"
    cov_lab = json.load(open(cache_path)) if cache_path.exists() and not args.recompute else {}
    if not cov_lab:
        for ds, cfg in DS.items():
            for lab, (w, n) in coverage_of_record(cfg["rec"]).items():
                lab = cfg.get("pool", lab)
                a = cov_lab.setdefault(lab, [0, 0])
                a[0] += w; a[1] += n
        json.dump(cov_lab, open(cache_path, "w"))
    order = ORDER_FULL if args.full else ORDER
    cov = {}
    for wl in order:
        members = POOL.get(wl, [wl])
        w = sum(cov_lab.get(m, [0, 0])[0] for m in members)
        n = sum(cov_lab.get(m, [0, 0])[1] for m in members)
        if n:
            cov[wl] = w / n
    wls = [w for w in order if w in K and K[w]]

    n, g = len(props), (0.82 if args.calib else 0.76)
    bw = g / n
    per_w = 1.6 if args.full else (2.3 if args.calib else 2.0)
    fig, (ax, axc) = plt.subplots(2, 1, figsize=(2.0 + per_w * len(wls), 6.2),
                                  gridspec_kw={"height_ratios": [4.2, 1.0]},
                                  sharex=True)
    for pi, p in enumerate(props):
        xs = [ti - g / 2 + bw * (pi + 0.5) for ti in range(len(wls))]
        ys = [K[w].get(p, 0.0) for w in wls]
        ax.bar(xs, ys, width=bw * 0.90, color=COLORS[p], label=LABELS[p])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.05, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=7 if args.calib else 7.5, color="#555555")
    ymax = max(K[w].get(p, 0) for w in wls for p in props)
    mult_bar = "calib" if args.calib else "chain"
    for ti, w in enumerate(wls):                    # x vs best standalone
        best = max(K[w].get("dflash", 0), K[w].get("suffix", 0))
        comp = K[w].get(mult_bar, 0)
        if best > 0 and comp > 0:
            x = ti - g / 2 + bw * (props.index(mult_bar) + 0.5)
            ax.text(x, comp + ymax * 0.06, f"×{comp / best:.2f}", ha="center",
                    va="bottom", fontsize=10.5, fontweight="bold",
                    color=COLORS[mult_bar])
    ax.set_ylim(0, ymax * 1.20)
    ax.set_ylabel("mean accept length / round  (tokens)", fontsize=10)
    sig = ("raw + IN-SAMPLE-calibrated arms (calibrate = test = full eval set)"
           if args.calib else "raw signals only")
    mult_name = "calibrated composition" if args.calib else "composition"
    ax.set_title(f"Chain vs chain, same verification budget — two-way protocol, "
                 f"{sig}   [measured]\n"
                 f"tree = warm set, test = FULL eval set (no calibrate split); "
                 f"× = {mult_name} vs best standalone", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    for ti, w in enumerate(wls):
        c = cov.get(w)
        if c is None:
            continue
        axc.bar(ti, c * 100, width=0.42, color="#8A97A5")
        axc.text(ti, c * 100 + 3, f"{c * 100:.0f}%", ha="center", va="bottom",
                 fontsize=8, color="#555555")
    axc.set_ylim(0, 115)
    axc.set_yticks([0, 50, 100])
    axc.set_ylabel("coverage %", fontsize=8.5)
    axc.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    axc.set_xticks(range(len(wls)))
    axc.set_xticklabels([DISPLAY.get(w, w) for w in wls],
                        fontsize=9 if args.full else 9.5,
                        rotation=18 if args.full else 0,
                        ha="right" if args.full else "center")
    axc.text(0.005, 0.92, "suffix-tracking coverage of the eval set (copy-run≥4)",
             transform=axc.transAxes, fontsize=8, va="top", color="#555555")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    name = ("MAT_per_workload_2way_full" if args.full else "MAT_per_workload_2way") \
        + ("_calib" if args.calib else "")
    fp = OUT / f"{name}.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")
    for w in wls:
        best = max(K[w].get("dflash", 0), K[w].get("suffix", 0))
        print(f"  [{w}] cov={cov.get(w, 0):.0%}  "
              + "  ".join(f"{p}={K[w].get(p, 0):.2f}" for p in props)
              + (f"  x={K[w].get(mult_bar, 0) / best:.2f}" if best else ""))


if __name__ == "__main__":
    main()
