#!/usr/bin/env python3
"""REAL version of required_figures_synthesized_ver/MAT_per_workload.png, same
3-bar design as the synthetic (DFlash-only chain / Suffix-only chain /
Composition chain, x{mult} = composition vs best standalone) on a trimmed
trend-telling workload set, each tick naming its parent benchmark:

  QA (Spec-Bench) -> Math (Spec-Bench) -> Web Search (BFCL v4)
      -> Memory (BFCL v4, kv/rec_sum/vector round-weight pooled) -> SWE-Bench

which spans cold (composition = DFlash lower bound, x<1) to warm/agentic
(bridge synergy, x>1).

Numbers are parsed from the saved replay logs (same test split, full-round
replay, 3-way disjoint split; specbench uses the within-label convlabel split
because that dataset is subtask-interleaved):
  mat_specbench_wl.replay.txt    (by-task -> qa, math_reasoning)
  mat_bfcl_3prop.replay.txt      (by-task with (rounds) -> web_search + memory pool)
  mat_swebench.replay.txt        (overall -> swebench)

  python3 scripts/plot_mat_per_workload.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "readable_outputs" / "figures" / "replay_logs"    # replay-log inputs
OUT = BASE / "readable_outputs" / "figures" / "mat"            # figure outputs
PROPS_ALL = ["dflash", "suffix", "chain", "calib"]   # parse/pool superset
PROPS = ["dflash", "suffix", "chain"]                # default 3-bar design
LABELS = {"dflash": "DFlash only (chain)", "suffix": "Suffix only (chain)",
          "chain": "Composition (ours, chain)"}
LABELS_CALIB = {"dflash": "DFlash only (chain)", "suffix": "Suffix only (chain)",
                "chain": "Composition (raw)", "calib": "Composition (calibrated)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "chain": "#54A24B",
          "calib": "#B94A8C"}
ORDER = ["qa", "math_reasoning", "web_search", "memory", "swebench", "spider_dbt"]
POOL = {"memory": ["memory_kv", "memory_rec_sum", "memory_vector"]}
ORDER_FULL = ["memory_kv", "memory_rec_sum", "memory_vector", "web_search",
              "swebench", "spider_dbt", "writing", "roleplay", "translation",
              "summarization", "qa", "math_reasoning", "rag"]
DISPLAY = {"qa": "QA\n(Spec-Bench)", "math_reasoning": "Math\n(Spec-Bench)",
           "web_search": "Web Search\n(BFCL v4)", "swebench": "SWE-Bench\n(Verified)",
           "memory": "Memory\n(BFCL v4)", "spider_dbt": "Agentic SQL\n(Spider2-DBT)",
           "memory_kv": "Memory KV\n(BFCL v4)", "memory_rec_sum": "Memory RecSum\n(BFCL v4)",
           "memory_vector": "Memory Vector\n(BFCL v4)", "writing": "Writing\n(Spec-Bench)",
           "roleplay": "Roleplay\n(Spec-Bench)", "translation": "Translation\n(Spec-Bench)",
           "summarization": "Summarization\n(Spec-Bench)", "rag": "RAG\n(Spec-Bench)"}


def parse_by_task(text):
    """{task: {prop: (K, rounds|None)}} — rounds present in newer replay logs."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s+\[(.+?)\]\s+(.*)", line)
        if m:
            vals = {p: (float(v), int(r) if r else None)
                    for p, v, r in re.findall(r"(\w+)=([0-9.]+)(?:\((\d+)\))?",
                                              m.group(2))}
            if set(vals) & set(PROPS_ALL):
                out[m.group(1)] = vals
    return out


def resolve(by_task, overall):
    """Flatten to {workload: {prop: K}}: singles + round-weighted POOL groups."""
    K = {t: {p: v[0] for p, v in d.items()} for t, d in by_task.items()}
    K.update(overall)
    for w, members in POOL.items():
        got = [by_task[t] for t in members if t in by_task]
        if len(got) != len(members):
            continue
        K[w] = {}
        for p in PROPS_ALL:
            ks = [d[p] for d in got if p in d]
            if len(ks) != len(members):
                continue
            if all(r is not None for _, r in ks):        # round-weighted pool
                K[w][p] = sum(k * r for k, r in ks) / sum(r for _, r in ks)
            else:                                        # old logs: plain mean
                K[w][p] = sum(k for k, _ in ks) / len(ks)
    return K


def parse_overall(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s+(\w+): K=([0-9.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="all 12 workloads (bfcl | swebench | specbench, family "
                         "separators) -> MAT_per_workload_full.png")
    ap.add_argument("--calib", action="store_true",
                    help="add the calibrated-composition bar (raw vs calibrated "
                         "signals, same argmax hand-off machine) -> *_calib.png")
    args = ap.parse_args()

    by_task = parse_by_task((FIG / "mat_specbench_wl.replay.txt").read_text())
    for cand in ("mat_bfcl_wl.replay.txt", "mat_bfcl_3prop.replay.txt",
                 "mat_bfcl_fulltraj.replay.txt"):
        bfcl_log = FIG / cand
        if bfcl_log.exists():
            break
    by_task.update(parse_by_task(bfcl_log.read_text()))
    overall = {"swebench": parse_overall((FIG / "mat_swebench.replay.txt").read_text())}
    spider_log = FIG / "mat_spider_wl.replay.txt"    # single-label -> overall lines only
    if spider_log.exists():
        overall["spider_dbt"] = parse_overall(spider_log.read_text())
    K = resolve(by_task, overall)
    wls = [w for w in (ORDER_FULL if args.full else ORDER) if w in K]

    props = PROPS_ALL if args.calib else PROPS
    labels = LABELS_CALIB if args.calib else LABELS
    mult_bar = "calib" if args.calib else "chain"    # x annotated on this bar
    n, g = len(props), (0.78 if args.calib else 0.72)
    bw = g / n
    per_w = 1.45 if args.full else 1.9
    fig, ax = plt.subplots(figsize=(1.9 + per_w * len(wls), 4.8))
    for pi, p in enumerate(props):
        xs = [ti - g / 2 + bw * (pi + 0.5) for ti in range(len(wls))]
        ys = [K[w].get(p, 0.0) for w in wls]
        ax.bar(xs, ys, width=bw * 0.90, color=COLORS[p], label=labels[p])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=7.5 if args.calib else 8, color="#555555")
    ymax = max(K[w].get(p, 0) for w in wls for p in props)
    for ti, w in enumerate(wls):                    # xMult vs best standalone
        best_single = max(K[w].get("dflash", 0), K[w].get("suffix", 0))
        comp = K[w].get(mult_bar, 0)
        if best_single > 0 and comp > 0:
            x = ti - g / 2 + bw * (props.index(mult_bar) + 0.5)
            ax.text(x, comp + ymax * 0.06, f"×{comp / best_single:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=COLORS[mult_bar])
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels([DISPLAY.get(w, w) for w in wls],
                       fontsize=8.5 if args.full else 10)
    if args.full:                                  # bfcl | swe+spider | specbench
        for xb in (3.5, 5.5):
            ax.axvline(xb, color="#BBBBBB", lw=0.8, ls=":")
    ax.set_ylabel("mean accept length / round  (tokens)", fontsize=10)
    mult_name = "calibrated composition" if args.calib else "composition"
    ax.set_title("Chain vs chain, same verification budget   [measured]\n"
                 "Qwen3.5-27B + DFlash + Suffix — 3-way disjoint split, full-round "
                 f"replay;  × = {mult_name} vs best standalone", fontsize=10)
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, frameon=False, loc="upper left",
              ncol=2 if args.calib else 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    name = ("MAT_per_workload_full" if args.full else "MAT_per_workload") \
        + ("_calib" if args.calib else "")
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"{name}.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")
    for w in wls:
        bs = max(K[w].get("dflash", 0), K[w].get("suffix", 0))
        print(f"  [{w}] " + "  ".join(f"{p}={K[w].get(p, 0):.2f}" for p in props)
              + f"  x={K[w].get(mult_bar, 0) / bs:.2f}")


if __name__ == "__main__":
    main()
