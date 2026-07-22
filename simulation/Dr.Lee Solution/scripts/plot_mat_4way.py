#!/usr/bin/env python3
"""Dr.Lee calibrated-version MAT bars over the four full-trajectory workloads
(Spec-Bench / BFCLv4 / SWE-bench / Spider2-DBT), 2026-07-05 recaptures.

Four arms, fixed colors (user-directed):
  dflash  DFlash single proposer                      blue    #4C78A8
  suffix  Suffix single proposer                      orange  #F58518
  calib   Compose (DFlash hazard = BETA calibration,
          suffix tail = ISOTONIC score->E[accept])    green   #54A24B
  oracle  handoff oracle (always hands off at the
          realized-best position, suffix extension)   red     #E45756

Protocol: TWO-WAY IN-SAMPLE (user canonical) — suffix tree = warm set only,
test = ENTIRE eval set; the compose calibrators are fit on the same full eval
set (replay_extension --calib-insample --hazard-fit beta, no --three-way).

Sources: readable_outputs/figures/replay_logs/mat_{ds}_4way_{grp}.replay.txt
for grp in (singles, calib, oracle) — produced by run_4way_pipeline.sh.

Outputs: figures/mat/MAT_per_workload_4way.png (4 pooled workload columns)
         figures/mat/MAT_per_subtask_4way.png  (subtask detail: specbench
         subtasks + bfcl categories; swe/spider single columns)

  python3 scripts/plot_mat_4way.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "readable_outputs" / "figures" / "replay_logs"
OUT = BASE / "readable_outputs" / "figures" / "mat" / "mat_bars"

PROPS = ["dflash", "suffix", "fallback", "calib", "oracle"]
LABELS = {"dflash": "DFlash (single)", "suffix": "Suffix (single)",
          "fallback": "SD-paper hybrid (fallback)",
          "calib": "Compose (logistic head + isotonic tail)",
          "oracle": "Oracle (best handoff)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "fallback": "#9467BD",
          "calib": "#54A24B", "oracle": "#E45756"}
GROUPS = ["singles", "calib", "oracle"]

DS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
# consistent label: line1 = dataset · harness (specbench has none);
# line2 = tasks we used (of the original) · subtask grouping.
DS_NAME = {"specbench": "Spec-Bench\n480 tasks (all) · 6 subtasks",
           "bfcl": "BFCL v4 · bfcl_eval (prompt-mode FC)\n753 tasks · 5 categories",
           "swebench": "SWE-bench Verified · mini-swe-agent\n60 tasks (250-step) · 12 repos",
           "spider": "Spider2-DBT · spider-agent-dbt\n68 tasks (all) · 68 databases",
           "tau2": "τ²-bench · tau2 official sim\n64 tasks · 3 domains"}
# subtask panel: specbench subtasks + bfcl categories; swe/spider pooled.
# the 8 MT-bench per-question categories are pooled into one mt_bench column
# (standard Spec-Bench 6-subtask view).
MT_CATS = {"writing", "roleplay", "reasoning", "math", "coding",
           "extraction", "stem", "humanities"}
SUB_ORDER = {"specbench": ["mt_bench", "translation", "summarization", "qa",
                           "math_reasoning", "rag"],
             "bfcl": ["web_search", "web_search_no_snippet", "memory_kv",
                      "memory_rec_sum", "memory_vector"],
             "tau2": ["airline", "retail", "telecom"]}


LOG_SUFFIX = ""      # set via --log-suffix: read mat_{ds}_4way_{grp}{sfx}.replay.txt
COMPOSE_SUFFIX = ""  # set via --compose-suffix: override ONLY the calib(compose) group
OUT_SUFFIX = ""      # set via --out-suffix: appended to output filename
NOTE = ""            # set via --note: appended to figure titles
SWEEP_DIR = ""       # set via --sweep-dir: fallback_sweep_fresh_{ds}.json live here
FB_TAUS = {}         # ds -> pooled-best tau (filled by _load_fallback)
FB_GRID = []         # swept tau grid (filled by _load_fallback)
FB_SPLIT = False     # sweep json was a deployable calib/test split


def _load_fallback():
    """SD-paper hybrid (round-level score-threshold fallback -> DFlash) at its
    pooled-best swept tau, from replay_fallback_sweep json. Returns
    {ds: (best_tau, entry)} where entry has K, rounds, by_task. A deployable
    split sweep ({'calib': {tau:...}, 'test': {tau:...}}) picks tau* on the
    CALIB half and reports the TEST-half entry at that tau."""
    global FB_SPLIT
    out = {}
    grid = set()
    if not SWEEP_DIR:
        return out
    for ds in DS:
        fp = Path(SWEEP_DIR) / f"fallback_sweep_fresh_{ds}.json"
        if not fp.exists():
            continue
        taus = next(iter(json.load(open(fp)).values()))   # {tau_str: {...}}
        if set(taus) == {"calib", "test"}:
            FB_SPLIT = True
            cal, tst = taus["calib"], taus["test"]
            grid.update(float(t) for t in tst)
            best_tau = max(cal, key=lambda t: cal[t]["K"])
            out[ds] = (float(best_tau), tst[best_tau])
            continue
        grid.update(float(t) for t in taus)
        best_tau, entry = max(taus.items(), key=lambda kv: kv[1]["K"])
        out[ds] = (float(best_tau), entry)
    FB_GRID[:] = sorted(grid)
    return out


def parse_log(path: Path):
    """-> (overall {prop: (K, rounds)}, by_task {task: {prop: (K, rounds)}})"""
    overall, by_task = {}, {}
    if not path.exists():
        return overall, by_task
    for line in path.read_text().splitlines():
        m = re.match(r"\s+(\w+): K=([0-9.]+)\s+\(rounds=(\d+)\)", line)
        if m and m.group(1) in PROPS:
            overall[m.group(1)] = (float(m.group(2)), int(m.group(3)))
            continue
        m = re.match(r"\s+\[(.+?)\]\s+(.*)", line)
        if m:
            for p, v, r in re.findall(r"(\w+)=([0-9.]+)\((\d+)\)", m.group(2)):
                if p in PROPS:
                    by_task.setdefault(m.group(1), {})[p] = (float(v), int(r))
    return overall, by_task


def load():
    K, T = {}, {}                       # K[ds][prop]=(mat, rounds); T[ds][task][prop]
    if "raw" in LOG_SUFFIX:
        # raw-cals compose (identity calibrators) is the DEFAULT arm: plain label,
        # no calibration mention (unlabeled figures are understood as no-calib)
        LABELS["calib"] = "Compose"
    for ds in DS:
        K[ds], T[ds] = {}, {}
        for grp in GROUPS:
            sfx = COMPOSE_SUFFIX if (grp == "calib" and COMPOSE_SUFFIX) else LOG_SUFFIX
            o, bt = parse_log(LOGS / f"mat_{ds}_4way_{grp}{sfx}.replay.txt")
            K[ds].update(o)
            for t, d in bt.items():
                T[ds].setdefault(t, {}).update(d)
    # inject the SD-paper hybrid (fallback @ pooled-best tau) as a 5th arm,
    # BEFORE the mt_bench pooling so its per-subtask values pool identically.
    for ds, (tau, entry) in _load_fallback().items():
        FB_TAUS[ds] = tau
        K[ds]["fallback"] = (entry["K"], entry.get("rounds", 0))
        for t, kn in entry.get("by_task", {}).items():
            T[ds].setdefault(t, {})["fallback"] = (kn[0], kn[1])
    # pool the 8 MT-bench categories into one rounds-weighted mt_bench column
    mt = {t: d for t, d in T["specbench"].items() if t in MT_CATS}
    if mt:
        pooled = {}
        for p in PROPS:
            pairs = [d[p] for d in mt.values() if p in d]
            r = sum(n for _, n in pairs)
            if r:
                pooled[p] = (sum(k * n for k, n in pairs) / r, r)
        T["specbench"] = {t: d for t, d in T["specbench"].items()
                          if t not in MT_CATS}
        T["specbench"]["mt_bench"] = pooled
    return K, T


def bars(ax, cols, K_of, tau_of=None, width_group=0.80, fs_val=7.5):
    n = len(PROPS)
    bw = width_group / n
    ymax = 0.0
    fb_labels = []
    for pi, p in enumerate(PROPS):
        xs = [ci - width_group / 2 + bw * (pi + 0.5) for ci in range(len(cols))]
        ys = [K_of(c).get(p, (0.0, 0))[0] for c in cols]
        ymax = max(ymax, max(ys) if ys else 0)
        ax.bar(xs, ys, width=bw * 0.9, color=COLORS[p], label=LABELS[p])
        for x, y in zip(xs, ys):
            if y > 0:
                ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom",
                        fontsize=fs_val, color="#555555")
        if p == "fallback" and tau_of is not None:
            for c, x, y in zip(cols, xs, ys):
                t = tau_of(c)
                if y > 0 and t is not None:
                    fb_labels.append((x, y, t))
    # per-bar best tau for the SD-paper hybrid (fallback) arm
    for x, y, t in fb_labels:
        ax.text(x, y + ymax * 0.055, f"τ*={t:g}", ha="center", va="bottom",
                fontsize=fs_val + 0.5, fontweight="bold", color=COLORS["fallback"])
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ymax


def main():
    import argparse
    global LOG_SUFFIX, COMPOSE_SUFFIX, OUT_SUFFIX, NOTE, SWEEP_DIR, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-suffix", default="",
                    help="read mat_{ds}_4way_{grp}<sfx>.replay.txt (e.g. _pre)")
    ap.add_argument("--compose-suffix", default="",
                    help="override ONLY the compose(calib) group log, e.g. _raw for "
                         "the no-calibration compose bar")
    ap.add_argument("--out-suffix", default="",
                    help="appended to output filename, e.g. _nocalib")
    ap.add_argument("--out-dir", default="",
                    help="override the output directory (e.g. the mat(deployable) folder)")
    ap.add_argument("--note", default="", help="appended to figure titles")
    ap.add_argument("--sweep-dir",
                    default="/workspace/simulation/results/pipeline_4way/segments",
                    help="dir with fallback_sweep_fresh_{ds}.json (SD-paper hybrid arm); "
                         "empty string disables the fallback bar")
    args = ap.parse_args()
    LOG_SUFFIX, NOTE, SWEEP_DIR = args.log_suffix, args.note, args.sweep_dir
    COMPOSE_SUFFIX, OUT_SUFFIX = args.compose_suffix, args.out_suffix
    if args.out_dir:
        OUT = Path(args.out_dir)
    if COMPOSE_SUFFIX == "_raw":
        LABELS["calib"] = "Compose (raw, no calib)"

    K, T = load()
    OUT.mkdir(parents=True, exist_ok=True)
    corner = ""
    if FB_GRID:
        corner = "τ swept: {" + ", ".join(f"{t:g}" for t in FB_GRID) + "}\n" + \
                 ("τ* picked on the calib half; bar = MAT on the test half"
                  if FB_SPLIT else "τ* on each SD-paper hybrid bar = best per column")

    # ---- figure 1: 4 pooled workload columns --------------------------------
    wls = [ds for ds in DS if K[ds]]
    fig, ax = plt.subplots(figsize=(2.4 + 2.5 * len(wls), 5.4))
    bars(ax, wls, lambda ds: K[ds], tau_of=lambda ds: FB_TAUS.get(ds))
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels([DS_NAME[ds] for ds in wls], fontsize=8.5)
    ax.set_ylabel("mean accept length  (tokens)", fontsize=11)
    ax.set_title("Dr.Lee extension — MAT per workload"
                 + (f"\n{NOTE}" if NOTE else ""), fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=2)
    if corner:
        ax.text(0.99, 0.99, corner, transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color="#555555")
    fig.tight_layout()
    fp = OUT / f"MAT_per_workload_4way{OUT_SUFFIX}.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")
    for ds in wls:
        print(f"  [{ds}] " + "  ".join(
            f"{p}={K[ds].get(p, (0, 0))[0]:.2f}({K[ds].get(p, (0, 0))[1]})"
            for p in PROPS))

    # subtask detail is now ONE figure PER workload — see plot_mat_subtask.py
    # (the old combined MAT_per_subtask_4way figure was retired per user request).


if __name__ == "__main__":
    main()
