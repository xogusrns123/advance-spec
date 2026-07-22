"""REALIZED per-depth decomposition of the 2-way ladder (clean: pinned + held-out +
all-native), with three by-depth plots per cell:
  (1) sel_acc(d)       decisive selection accuracy at depth d, per arm
  (2) decisive_prob(d) P(position is decisive | reached at d) = dec_reach/reach
  (3) mat_loss(d)      survival_oracle(d) - survival_arm(d)  (per-depth MAT loss; sums to MAT gap)
Each arm reads its OWN realized pinned decision log (it follows its own picks; every arm is
token-pinned to the same gt trajectory). survival(d)=fraction of blocks alive at d; MAT=sum_d survival(d).

Run: python3 simulation/scripts/select1_ladder/realized_perdepth.py
"""
import json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "simulation/results/chain_hybrid_perdepth"
OUT = "/tmp/claude-20051/-home-muchwater-advance-spec/65ec03a0-7cb7-4645-911b-7079f77dec72/scratchpad"
MAXD = 16
MIN_N = 40

CELLS = [
    dict(key="14b_2way", title="Qwen3-14B  ·  EAGLE3 + suffix  (2-way)", dir="qwen3_14b_ar"),
    dict(key="27b_2way", title="Qwen3.5-27B  ·  MTP + suffix  (2-way)", dir="qwen35_27b_ar"),
]
# (label, color, candidate logs)  bare first (14B pinned) then _all-trained (27B)
ARMS = [
    ("raw",            "#c2683a", ["decisions_select1.jsonl"]),
    ("calib-logistic", "#1f7a8c", ["decisions_select1_calib_logistic.jsonl", "decisions_select1_calib_logistic_all-trained.jsonl"]),
    ("bayes",          "#6a4c93", ["decisions_select1_bayes.jsonl"]),
    ("oracle",         "#2f7a57", ["decisions_select1_oracle.jsonl"]),
]

def picked_token(o):
    return o.get("suffix_token") if o.get("chosen") == "suffix" else o.get("eagle_token")

def analyze(path):
    blocks = defaultdict(list)
    try:
        f = open(path)
    except FileNotFoundError:
        return None
    for line in f:
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            blocks[(o["rid"], o["decode_step"])].append(o)
    n = len(blocks)
    if not n: return None
    reach = np.zeros(MAXD); surv = np.zeros(MAXD)
    dec_reach = np.zeros(MAXD); dec_hit = np.zeros(MAXD)
    runlen = []
    for pos in blocks.values():
        pos.sort(key=lambda r: r["depth"]); run = 0
        for d, o in enumerate(pos):
            if d >= MAXD: break
            gt = o.get("gt_token")
            if gt is None: break
            reach[d] += 1
            et, st = o.get("eagle_token"), o.get("suffix_token")
            avail = [t for t in (et, st) if t is not None]
            hits = [t for t in avail if t == gt]
            if len(avail) >= 2 and 0 < len(hits) < len(avail):
                dec_reach[d] += 1
                if picked_token(o) == gt: dec_hit[d] += 1
            if picked_token(o) == gt:
                surv[d] += 1; run += 1
            else:
                break
        runlen.append(run)
    return dict(n=n, mat=float(np.mean(runlen)),
                survival=reach / n,
                dec_acc=np.where(dec_reach >= MIN_N, dec_hit / np.maximum(dec_reach, 1), np.nan),
                dec_prob=np.where(reach >= MIN_N, dec_reach / np.maximum(reach, 1), np.nan))

def lineplot(cell, series, ylabel, subtitle, tag, ylim=None):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for lbl, color, y in series:
        xs = [i for i in range(MAXD) if not np.isnan(y[i])]
        ys = [y[i] for i in xs]
        lw = 2.4 if lbl in ("bayes", "oracle") else 1.8
        ls = "--" if lbl == "oracle" else "-"
        ax.plot(xs, ys, ls, color=color, lw=lw, marker="o", ms=3.5, label=lbl)
    ax.set_xlabel("draft depth d"); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlim(-0.3, MAXD - 1)
    ax.set_title(cell["title"], fontsize=13, fontweight="bold", loc="left")
    ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=6.8, color="#5e6e78")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout()
    out = f"{OUT}/realized_{cell['key']}_{tag}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"   -> {out}")

def make_cell(cell):
    d = f"{ROOT}/{cell['dir']}"
    R = {}
    for lbl, color, cands in ARMS:
        res = None
        for fn in cands:
            res = analyze(f"{d}/{fn}")
            if res is not None: break
        if res is not None: R[lbl] = (color, res)
    print(f"\n### {cell['title']}  (n_blocks≈{R['raw'][1]['n']})  MAT: " +
          " ".join(f"{l}={R[l][1]['mat']:.3f}" for l in R))
    sub = "REALIZED served · held-out · token-pinned · all-native · greedy · bfcl_v4 web_search"

    # (1) sel_acc by depth — exclude oracle (=1.0 trivially)
    lineplot(cell, [(l, R[l][0], R[l][1]["dec_acc"]) for l in R if l != "oracle"],
             "decisive selection accuracy", sub + " · decisive positions", "selacc_by_depth", ylim=(0, 1.0))
    # (2) decisive prob by depth
    lineplot(cell, [(l, R[l][0], R[l][1]["dec_prob"]) for l in R],
             "P(decisive | reached at d)", sub + " · fraction of reached positions that are decisive",
             "decisive_prob_by_depth", ylim=(0, 1.0))
    # (3) MAT loss by depth = survival_oracle - survival_arm
    so = R["oracle"][1]["survival"]
    loss = [(l, R[l][0], np.clip(so - R[l][1]["survival"], 0, None)) for l in R if l != "oracle"]
    lineplot(cell, loss, "per-depth MAT loss  (survival_oracle - survival_arm)",
             sub + " · sums over depth to the raw/calib/bayes -> oracle MAT gap", "matloss_by_depth")
    # text summary
    for l in R:
        s = R[l][1]
        print(f"   {l:14s} selacc(d0..7): " +
              " ".join(f"{s['dec_acc'][i]:.2f}" if not np.isnan(s['dec_acc'][i]) else " - " for i in range(8)))

if __name__ == "__main__":
    for c in CELLS:
        try: make_cell(c)
        except Exception as ex:
            import traceback; print(f"ERR {c['key']}: {ex}"); traceback.print_exc()
