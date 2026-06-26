"""Decisive SELECTION ACCURACY ladder (raw / calib x4 / oracle) for both the
27B MTP and 14B EAGLE3 served cells. INCLUDE-ALL (no runaway exclusion — 42/44
are legitimate fixed-agent results). Same color scheme + oracle star
(_ladder_style). Outputs mtp_selacc_compare.png + eagle3_selacc_compare.png."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_style import ladder_bar, SELECT1_COLORS  # noqa: E402

ROOT = "simulation/results/chain_hybrid_perdepth"
FIGDIR = "simulation/results/calib_why_analysis/figures"
ALIVE = {"eagle", "suffix", "both"}
METHODS = ("histogram", "isotonic", "logistic", "beta")
XLABELS = ["raw\n(sp>ep)", "calib\nhistogram", "calib\nisotonic",
           "calib\nlogistic", "calib\nbeta", "oracle\n(GT)"]
CELLS = {
    "27b": {"dir": f"{ROOT}/qwen35_27b_ar", "title": "Qwen3.5-27B MTP", "out": "mtp_selacc_compare.png"},
    "14b": {"dir": f"{ROOT}/qwen3_14b_ar", "title": "Qwen3-14B EAGLE3", "out": "eagle3_selacc_compare.png"},
}


def _has_oracle_hit(p):
    if not os.path.exists(p):
        return False
    for line in open(p):
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            return "oracle_hit" in o
    return False


def _calib_arm(d, m):
    """cond-trained calib log if it carries oracle_hit, else the plain (no-suffix)
    log — the 14B cond-trained arms predate the oracle_hit-logging patch and only
    the plain decisions_select1_calib_{m}.jsonl have it."""
    cond = f"{d}/decisions_select1_calib_{m}_cond-trained.jsonl"
    plain = f"{d}/decisions_select1_calib_{m}.jsonl"
    return cond if _has_oracle_hit(cond) else plain


def _arms(d):
    return ([f"{d}/decisions_select1.jsonl"] +
            [_calib_arm(d, m) for m in METHODS] +
            [f"{d}/decisions_select1_oracle.jsonl"])


def selacc(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains: chains[k].sort(key=lambda r: r["depth"])
    n = correct = 0
    for rs in chains.values():
        alive = True
        for r in rs:
            if not alive: break
            h = r.get("oracle_hit")
            if h in ("eagle", "suffix"):
                n += 1
                ps = (r.get("chosen") == "suffix")
                correct += (ps and h == "suffix") or ((not ps) and h == "eagle")
            if h not in ALIVE: alive = False
    return correct / max(n, 1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["27b", "14b", "both"], default="both")
    args = ap.parse_args()
    items = CELLS.items() if args.cell == "both" else [(args.cell, CELLS[args.cell])]
    for key, c in items:
        sels = [selacc(p) for p in _arms(c["dir"])]
        print(f"=== {key} {c['title']} selacc (all tasks) ===  " +
              "  ".join(f"{l.replace(chr(10),' ')}={v:.3f}" for l, v in zip(XLABELS, sels)))
        ladder_bar(sels, XLABELS, "decisive selection accuracy (alive-conditioned)",
                   "Selection accuracy: raw vs calibration (4 methods) vs oracle\n"
                   f"({c['title']}, served, all 20 tasks incl. 42/44)",
                   f"{FIGDIR}/{c['out']}", fmt="{:.3f}", colors=SELECT1_COLORS, star_idx=5)


if __name__ == "__main__":
    main()
