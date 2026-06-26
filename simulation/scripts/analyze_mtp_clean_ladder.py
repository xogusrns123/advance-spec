"""27B MTP served DECISIVE SELECTION ACCURACY ladder (raw / calib x4 / oracle),
both clean (runaway 42/44 excluded) and raw (included). MAT lives in
analyze_mat_ladder_single.py. Same color scheme (_ladder_style)."""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_style import ladder_bar, SELECT1_COLORS  # noqa: E402

DIR = "simulation/results/chain_hybrid_perdepth/qwen35_27b_ar"
FIGDIR = "simulation/results/calib_why_analysis/figures"
ALIVE = {"eagle", "suffix", "both"}
RUNAWAY_MIN = 5000
XLABELS = ["raw\n(sp>ep)", "calib\nhistogram", "calib\nisotonic",
           "calib\nlogistic", "calib\nbeta", "oracle\n(GT)"]
ARMS = ["decisions_select1.jsonl"] + \
       [f"decisions_select1_calib_{m}_cond-trained.jsonl" for m in
        ("histogram", "isotonic", "logistic", "beta")] + \
       ["decisions_select1_oracle.jsonl"]


def selacc(fname, exclude_runaway):
    chains = defaultdict(list); dec_n = defaultdict(int)
    for line in open(f"{DIR}/{fname}"):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o); dec_n[str(o["rid"])] += 1
    run = {r for r, c in dec_n.items() if c > RUNAWAY_MIN} if exclude_runaway else set()
    for k in chains: chains[k].sort(key=lambda r: r["depth"])
    n = correct = 0
    for (rid, ds), rs in chains.items():
        if str(rid) in run: continue
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
    for mode, out, note in [("clean", "mtp_selacc_compare_clean.png", "runaway 42/44 excluded"),
                            ("raw", "mtp_selacc_compare_raw.png", "runaway 42/44 included — contaminated")]:
        sels = [selacc(fn, mode == "clean") for fn in ARMS]
        print(f"=== 27B MTP selacc [{mode}] ===  " +
              "  ".join(f"{l.replace(chr(10),' ')}={v:.3f}" for l, v in zip(XLABELS, sels)))
        ladder_bar(sels, XLABELS, "decisive selection accuracy (alive-conditioned)",
                   "Selection accuracy: raw vs calibration (4 methods) vs oracle\n"
                   f"(Qwen3.5-27B MTP, served, {note})",
                   f"{FIGDIR}/{out}", fmt="{:.3f}", colors=SELECT1_COLORS, star_idx=5)


if __name__ == "__main__":
    main()
