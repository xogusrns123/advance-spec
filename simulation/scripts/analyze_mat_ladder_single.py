"""MAT ladder with SINGLE-PROPOSER reference bars. Produces BOTH a clean
(runaway 42/44 excluded) and a raw (runaway included = contaminated) version.

8 bars: draft-only, suffix-only (single proposers — counterfactual per-step
accept length simulated from the oracle decision log) then raw / calib(histogram,
isotonic,logistic,beta) / oracle (select-1; MAT = served step accept_len).

Colors (no overlap): EAGLE-3 blue / MTP purple / Suffix orange / raw gray /
calib histogram green, isotonic cyan, logistic pink, beta olive / oracle red+★."""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_style import ladder_bar, ROLE  # noqa: E402

FIGDIR = "simulation/results/calib_why_analysis/figures"
ROOT = "simulation/results/chain_hybrid_perdepth"
RUNAWAY_MIN = 5000
CALIB = ["histogram", "isotonic", "logistic", "beta"]

# INCLUDE-ALL: tasks 42/44 are now legitimate fixed-agent results (the residual
# 42 reasoning loop is genuine model behavior, not a protocol bug) -> no runaway
# exclusion; all 20 eval tasks are plotted.
CELLS = {
    "27b": {"dir": f"{ROOT}/qwen35_27b_ar", "draft": "MTP",
            "title": "Qwen3.5-27B MTP", "out": "mtp_mat_compare.png"},
    "14b": {"dir": f"{ROOT}/qwen3_14b_ar", "draft": "EAGLE3",
            "title": "Qwen3-14B EAGLE3", "out": "mat_compare.png"},
}


def served_mat(path, clean):
    """mean per-step accept_len. Runaway rids are computed PER-ARM from this
    file's own decisions (rid hashes differ across serving runs)."""
    acc = {}; dec_n = defaultdict(int)
    for line in open(path):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            dec_n[str(o["rid"])] += 1
        elif o.get("type") == "step":
            acc[(str(o["rid"]), o["decode_step"])] = o.get("accept_len")
    run = {r for r, c in dec_n.items() if c > RUNAWAY_MIN} if clean else set()
    Ls = [v for (rid, ds), v in acc.items() if v is not None and rid not in run]
    return sum(Ls) / max(len(Ls), 1)


def single_proposer_mat(oracle_path, clean):
    """Simulate draft-only & suffix-only accept length (nogt/none terminate).
    Runaway rids computed from this (oracle) log's own decisions."""
    chains = defaultdict(list); dec_n = defaultdict(int)
    for line in open(oracle_path):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o); dec_n[str(o["rid"])] += 1
    exclude = {r for r, c in dec_n.items() if c > RUNAWAY_MIN} if clean else set()
    for k in chains: chains[k].sort(key=lambda r: r["depth"])
    Ld, Ls = [], []
    for (rid, ds), rs in chains.items():
        if str(rid) in exclude: continue
        ad = asf = 0; aD = aS = True
        for r in rs:
            h = r.get("oracle_hit")
            if aD and h in ("eagle", "both"): ad += 1
            else: aD = False
            if aS and h in ("suffix", "both"): asf += 1
            else: aS = False
            if not (aD or aS): break
        Ld.append(ad); Ls.append(asf)
    return sum(Ld) / max(len(Ld), 1), sum(Ls) / max(len(Ls), 1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["27b", "14b", "both"], default="both")
    args = ap.parse_args()
    items = CELLS.items() if args.cell == "both" else [(args.cell, CELLS[args.cell])]
    for key, c in items:
        d = c["dir"]
        draft_color = ROLE[c["draft"]]
        colors = [draft_color, ROLE["suffix"], ROLE["raw"], ROLE["histogram"],
                  ROLE["isotonic"], ROLE["logistic"], ROLE["beta"], ROLE["oracle"]]
        labels = [f"{c['draft']}\nonly", "suffix\nonly", "raw\n(sp>ep)",
                  "calib\nhistogram", "calib\nisotonic", "calib\nlogistic",
                  "calib\nbeta", "oracle\n(GT)"]
        cl = False  # include all tasks
        draft_mat, suffix_mat = single_proposer_mat(f"{d}/decisions_select1_oracle.jsonl", cl)
        raw = served_mat(f"{d}/decisions_select1.jsonl", cl)
        calibs = [served_mat(f"{d}/decisions_select1_calib_{m}_cond-trained.jsonl", cl) for m in CALIB]
        oracle = served_mat(f"{d}/decisions_select1_oracle.jsonl", cl)
        vals = [draft_mat, suffix_mat, raw] + calibs + [oracle]
        print(f"=== {key} {c['title']} (all 20 tasks) ===  " +
              "  ".join(f"{l.replace(chr(10),' ')}={v:.3f}" for l, v in zip(labels, vals)))
        ladder_bar(vals, labels, "MAT (per-step accept length)",
                   f"MAT: single-proposer vs select-1 (raw / calibration ×4 / oracle)\n"
                   f"({c['title']}, all 20 tasks incl. 42/44; single-proposer = sim)",
                   f"{FIGDIR}/{c['out']}", fmt="{:.3f}", colors=colors, star_idx=7)


if __name__ == "__main__":
    main()
