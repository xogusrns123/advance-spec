#!/usr/bin/env python3
"""Dr.Lee's prediction, made measurable: a per-workload K-SLOT metric that maps
onto the multislot bench's k (novel slots per response).

Definition (same as the interp_validation structure stats, pointwise theta=4):
  slot = interior cold gap of length <= W (=15) between two suffix-warm runs
         on the arm-independent s(p) curve (suffix copy depth along gt).
  k-analogue per call  = slots per call (a call = one response, the toy's unit)
  k-analogue per 96tok = slots per 96 output tokens (the toy's response length,
                         for cross-workload comparability — real calls vary
                         30x in length, the toy is fixed at 96)

Calibration: the SAME measurement applied to the multislot records gives the
mapping true-k -> measured slots (it compresses: adjacent slots merge, theta=4
fragments), written to _toy_calibration.json and drawn as reference lines.

Outputs (one folder, one file per workload):
  kslot_metric/{workload}.json      overall + per-task metric
  kslot_metric/_toy_calibration.json
  kslot_metric/summary.md
  kslot_metric/kslot_per_workload.png

  PYTHONPATH=/workspace python3 scripts/extract_kslot_metric.py \
      --dir results/interp_validation --names spider swebench bfcl specbench
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import argparse
import gzip
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(__file__))
from validate_interpretations import structure_of  # noqa: E402

WL_COLOR = {"spider": "#4C78A8", "swebench": "#F58518", "bfcl": "#54A24B",
            "specbench": "#E45756"}
W_BLOCK, THETA = 15, 4


def load_curves(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            out[r["rid"]] = r
    return out


def call_stats(curves):
    """per rid: slots (interior gaps <= W), positions, bridgeable count."""
    rows = {}
    for rid, cu in curves.items():
        st = structure_of(cu, W_BLOCK, THETA)
        if st is None:
            continue
        rows[rid] = dict(task=cu.get("task") or "all", L=st["L"],
                         slots=st["n_slot"], bridgeable=st["bridgeable"],
                         gaps=st["gaps"])
    return rows


def agg(rows):
    n = len(rows)
    if n == 0:
        return {}
    sl = sorted(r["slots"] for r in rows)
    L = sum(r["L"] for r in rows)
    slots = sum(r["slots"] for r in rows)
    gaps = [g for r in rows for g in r["gaps"]]
    return dict(
        n_calls=n,
        mean_call_len=L / n,
        slots_per_call=dict(mean=slots / n, median=sl[n // 2],
                            p90=sl[int(0.9 * (n - 1))]),
        slots_per_96tok=96.0 * slots / L,
        slots_per_100tok=100.0 * slots / L,
        bridgeable_share=(sum(r["bridgeable"] for r in rows) / slots
                          if slots else 0.0),
        gap_hist={"1-3": sum(1 for g in gaps if g <= 3),
                  "4-15": sum(1 for g in gaps if 4 <= g <= 15),
                  ">15": sum(1 for g in gaps if g > 15)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="default: <dir>/kslot_metric")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.dir, "kslot_metric")
    os.makedirs(out_dir, exist_ok=True)

    # ---- toy calibration: true k -> measured slots (same definition)
    toy = {}
    for k in (0, 1, 2, 4, 8):
        p = os.path.join(args.dir, f"curves_ms_k{k}.jsonl.gz")
        if not os.path.exists(p):
            continue
        a = agg(list(call_stats(load_curves(p)).values()))
        toy[k] = dict(slots_per_call=a["slots_per_call"]["mean"],
                      slots_per_96tok=a["slots_per_96tok"])
    json.dump(toy, open(os.path.join(out_dir, "_toy_calibration.json"), "w"),
              indent=1)

    # ---- per workload
    results = {}
    for name in args.names:
        curves = load_curves(os.path.join(args.dir, f"curves_{name}.jsonl.gz"))
        rows = call_stats(curves)
        by_task = defaultdict(list)
        for r in rows.values():
            by_task[r["task"]].append(r)
        res = dict(workload=name, theta=THETA, W=W_BLOCK,
                   definition="interior cold gap <= W between suffix-warm runs "
                              "(s(p) >= theta) — the multislot 'novel slot' analogue",
                   overall=agg(list(rows.values())),
                   by_task={t: agg(v) for t, v in sorted(by_task.items())},
                   toy_calibration=toy)
        results[name] = res
        json.dump(res, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)

    # ---- summary.md
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("# k-slot metric per workload (Dr.Lee's axis, measured)\n\n")
        f.write("slot = suffix-warm 흐름을 끊는 interior cold gap ≤ W(15tok), "
                "s(p)≥4 기준 — multislot bench의 'novel slot'과 동일 정의.\n\n")
        f.write("| workload | calls | mean len | slots/call (med, p90) | "
                "slots/96tok | bridgeable | gaps 1-3 / 4-15 / >15 |\n|---|---|---|---|---|---|---|\n")
        for name in args.names:
            a = results[name]["overall"]
            g = a["gap_hist"]
            f.write(f"| {name} | {a['n_calls']} | {a['mean_call_len']:.0f} | "
                    f"{a['slots_per_call']['mean']:.1f} ({a['slots_per_call']['median']}, "
                    f"{a['slots_per_call']['p90']}) | {a['slots_per_96tok']:.2f} | "
                    f"{a['bridgeable_share']:.0%} | {g['1-3']} / {g['4-15']} / {g['>15']} |\n")
        f.write("\n**toy calibration (true k → measured, same definition):**\n\n")
        f.write("| true k | slots/call | slots/96tok |\n|---|---|---|\n")
        for k, v in sorted(toy.items()):
            f.write(f"| {k} | {v['slots_per_call']:.1f} | {v['slots_per_96tok']:.2f} |\n")
        f.write("\n측정은 k를 압축한다(인접 slot 병합·θ 단편화): true k=1,2,4,8이 "
                "slots/call 기준 각각 위 표의 값으로 측정됨 — k≥4에서 포화. "
                "따라서 real workload 값은 서수(ordinal)로 읽을 것.\n")

    # ---- figure: two normalizations side by side, toy k reference lines
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    names = args.names
    for ax, key, ylab, toy_key in (
            (axes[0], "slots_per_96tok", "slots per 96 output tokens\n(toy response length)",
             "slots_per_96tok"),
            (axes[1], None, "slots per call (mean; a call = one response)",
             "slots_per_call")):
        vs = []
        for n in names:
            a = results[n]["overall"]
            vs.append(a["slots_per_96tok"] if key else a["slots_per_call"]["mean"])
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)
        ax.bar(range(len(names)), vs, color=[WL_COLOR.get(n, "#888") for n in names],
               width=0.62)
        for i, v in enumerate(vs):
            ax.text(i, v + max(vs) * 0.015, f"{v:.2f}" if key else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=10)
        # toy reference: merge k-levels whose measured value coincides; on the
        # per-call panel (huge specbench bar) draw the k=1..8 range as one band
        lv = {}
        for k, tv in sorted(toy.items()):
            if k == 0:
                continue
            lv.setdefault(round(tv[toy_key], 2), []).append(k)
        if key is None:
            lo, hi = min(lv), max(lv)
            ax.axhspan(lo, hi, color="#666", alpha=0.12, lw=0)
            ax.text(len(names) - 0.42, hi, f" toy k=1–8 range\n ({lo:.1f}–{hi:.1f}; "
                    "toy call = 96 tok)", va="bottom", fontsize=8, color="#444")
        else:
            for i, (yv, ks) in enumerate(sorted(lv.items())):
                ax.axhline(yv, color="#666", lw=0.9, ls=":", alpha=0.8)
                ax.text(len(names) - 0.42, yv + 0.03,
                        " toy k=" + "·".join(map(str, ks)), va="bottom",
                        fontsize=8, color="#444")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=10.5)
        ax.set_ylabel(ylab, fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Dr.Lee's k-slot axis measured on real workloads — "
                 "suffix-warm flow interruptions (gap ≤ 15 tok), dotted = same "
                 "measurement on the multislot bench at true k", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fp = os.path.join(out_dir, "kslot_per_workload.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")
    for name in args.names:
        a = results[name]["overall"]
        print(f"  {name}: slots/call {a['slots_per_call']['mean']:.1f} "
              f"(median {a['slots_per_call']['median']}), /96tok "
              f"{a['slots_per_96tok']:.2f}, bridgeable {a['bridgeable_share']:.0%}")


if __name__ == "__main__":
    main()
