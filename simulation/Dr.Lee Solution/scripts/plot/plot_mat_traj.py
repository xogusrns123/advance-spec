#!/usr/bin/env python3
"""MAT ladder figure for a full-trajectory capture (capture_traj.py output):
runs the CPU replay on the 3-way disjoint conversation split (NO LOO, exact
conv ids), parses overall + per-task K, renders the 6-bar family ladder plus a
per-task grouped panel when the record spans >1 task label.

Unlike the multislot/bfcl-clean figures (--max-rounds 8), full-trajectory
replays consume EVERY round of each call (--max-rounds 4096) — calls here are
hundreds of tokens, so an 8-round cap would only score their openings.

Run inside sglang-bench (figures dir is root-owned):
  cd "/workspace/simulation/Dr.Lee Solution" && python3 scripts/plot_mat_traj.py \
    --record results/perpos_bfcl_full/bfcl_v4_full.jsonl \
    --title "BFCLv4 full-traj (169 tasks, thinking-ON)" --outname mat_bfcl_full.png
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
PROPS = ["dflash", "suffix", "select", "calib", "chain", "oracle"]
LABELS = {"dflash": "single\n(DFlash)", "suffix": "single\n(Suffix)",
          "select": "per-depth\nselect (raw)", "calib": "per-depth\nselect (calib)",
          "chain": "composition", "oracle": "oracle"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "select": "#E45756",
          "calib": "#B94A8C", "chain": "#54A24B", "oracle": "#B279A2"}


def run_replay(record, props, group_mode, max_rounds, log_path):
    cmd = ["python3", "scripts/replay_extension.py", "--record", record,
           "--props", *props, "--three-way", "--group-mode", group_mode,
           "--max-rounds", str(max_rounds)]
    print("replay:", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"replay failed:\n{out.stdout[-2000:]}\n{out.stderr[-4000:]}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(out.stdout)
    return out.stdout


def parse_replay(text, props):
    K, by_task, split = {}, {}, ""
    m = re.search(r"3-way: calibrate on (\d+) calls \((\d+) convs\), "
                  r"test on (\d+) calls \((\d+) convs\)", text)
    if m:
        split = (f"calibrate {m.group(2)} convs/{m.group(1)} calls, "
                 f"test {m.group(4)} convs/{m.group(3)} calls")
    for line in text.splitlines():
        m = re.match(r"\s+(\w+): K=([0-9.]+)\s+\(rounds=(\d+)\)", line)
        if m and m.group(1) in props:
            K[m.group(1)] = float(m.group(2))
            continue
        m = re.match(r"\s+\[(.+?)\]\s+(.*)", line)
        if m:
            vals = dict(re.findall(r"(\w+)=([0-9.]+)", m.group(2)))
            if set(vals) & set(props):
                by_task[m.group(1)] = {p: float(v) for p, v in vals.items() if p in props}
    return K, by_task, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--outname", required=True, help="figure filename (png)")
    ap.add_argument("--props", nargs="+", default=PROPS)
    ap.add_argument("--group-mode", default="conv")
    ap.add_argument("--max-rounds", type=int, default=4096)
    ap.add_argument("--from-log", action="store_true",
                    help="reuse the saved replay log instead of re-running")
    args = ap.parse_args()

    fig_dir = BASE / "readable_outputs" / "figures" / "ladders"
    log_dir = BASE / "readable_outputs" / "figures" / "replay_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (Path(args.outname).stem + ".replay.txt")
    if args.from_log and log_path.exists():
        text = log_path.read_text()
    else:
        text = run_replay(args.record, args.props, args.group_mode,
                          args.max_rounds, log_path)
    K, by_task, split = parse_replay(text, args.props)
    missing = [p for p in args.props if p not in K]
    if missing:
        raise SystemExit(f"replay output lacked {missing}; log: {log_path}")

    tasks = sorted(by_task)
    two_panel = len(tasks) > 1
    if two_panel:
        w = max(7.4, 1.5 + 1.05 * len(tasks))
        fig, (ax, axt) = plt.subplots(2, 1, figsize=(w, 8.0),
                                      gridspec_kw={"height_ratios": [1.15, 1.0]})
    else:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))

    vals = [K[p] for p in args.props]
    bars = ax.bar(range(len(args.props)), vals, width=0.62,
                  color=[COLORS[p] for p in args.props])
    if "oracle" in args.props:
        b = bars[args.props.index("oracle")]
        b.set_hatch("//"); b.set_edgecolor("white")
    for x, v in enumerate(vals):
        ax.text(x, v + max(vals) * 0.015, f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(args.props)))
    ax.set_xticklabels([LABELS[p] for p in args.props], fontsize=9.5)
    ax.set_ylabel("MAT  (accepted tokens / round)", fontsize=10)
    ax.set_title(f"MAT — {args.title}  (3-way disjoint conv split, NO LOO)\n"
                 f"Qwen3.5-27B + DFlash + Suffix   {split}", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    if two_panel:
        n, g = len(args.props), 0.80
        bw = g / n
        for pi, p in enumerate(args.props):
            xs = [ti - g / 2 + bw * (pi + 0.5) for ti in range(len(tasks))]
            ys = [by_task[t].get(p, 0.0) for t in tasks]
            bb = axt.bar(xs, ys, width=bw * 0.92, color=COLORS[p], label=LABELS[p].replace("\n", " "))
            if p == "oracle":
                for b in bb:
                    b.set_hatch("//"); b.set_edgecolor("white")
        axt.set_xticks(range(len(tasks)))
        axt.set_xticklabels(tasks, fontsize=8.5,
                            rotation=30 if max(len(t) for t in tasks) > 8 else 0,
                            ha="right" if max(len(t) for t in tasks) > 8 else "center")
        axt.set_ylabel("MAT by task", fontsize=10)
        axt.grid(axis="y", alpha=0.3)
        tmax = max(v for t in tasks for v in by_task[t].values())
        axt.set_ylim(0, tmax * 1.32)                 # headroom so the legend clears the bars
        axt.legend(fontsize=8, ncol=3, frameon=False, loc="upper right")
        for s in ("top", "right"):
            axt.spines[s].set_visible(False)

    fig.tight_layout()
    fp = fig_dir / args.outname
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}: " + "  ".join(f"{p}={K[p]:.2f}" for p in args.props))
    if tasks:
        for t in tasks:
            print(f"  [{t}] " + "  ".join(f"{p}={by_task[t].get(p, 0):.2f}"
                                          for p in args.props))


if __name__ == "__main__":
    main()
