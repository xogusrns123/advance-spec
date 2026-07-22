"""Per-depth SUFFIX selection rate in DECISIVE situations (realized 2-way).

At a decisive position exactly one of {eagle/MTP, suffix} == gt, so the selector picks
either eagle or suffix. This plots, per depth d, the fraction of decisive-reached positions
where each arm PICKED suffix:
  - raw / calib-logistic / bayes  = the arm's actual suffix-pick rate(d)
  - oracle                        = the IDEAL rate = P(suffix == gt | decisive at d)
                                    (oracle picks suffix iff suffix is the correct one)
An arm above the oracle line OVER-picks suffix at that depth; below = UNDER-picks. Shows how
selection preference shifts with depth and where each method mis-allocates between proposers.

Run: python3 simulation/scripts/select1_ladder/realized_suffix_rate.py
"""
import json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "simulation/results/chain_hybrid_perdepth"
OUT = "/tmp/claude-20051/-home-muchwater-advance-spec/65ec03a0-7cb7-4645-911b-7079f77dec72/scratchpad"
MAXD = 16
MIN_N = 40   # suppress a depth's point if fewer than this many decisive positions

CELLS = [
    dict(key="14b_2way", title="Qwen3-14B  ·  EAGLE3 + suffix  (2-way)", dir="qwen3_14b_ar"),
    dict(key="27b_2way", title="Qwen3.5-27B  ·  MTP + suffix  (2-way)", dir="qwen35_27b_ar"),
]
# (label, color, candidate logs)  — bare log first (14B pinned), _all-trained fallback (27B)
ARMS = [
    ("raw",            "#c2683a", ["decisions_select1.jsonl"]),
    ("calib-logistic", "#1f7a8c", ["decisions_select1_calib_logistic.jsonl", "decisions_select1_calib_logistic_all-trained.jsonl"]),
    ("bayes (GBM)",    "#6a4c93", ["decisions_select1_bayes.jsonl"]),
    ("oracle (ideal)", "#2f7a57", ["decisions_select1_oracle.jsonl"]),
]

def picked_suffix(o):
    return o.get("chosen") == "suffix"

def suffix_rate_by_depth(path):
    pick = np.zeros(MAXD); dec = np.zeros(MAXD)
    try:
        f = open(path)
    except FileNotFoundError:
        return None
    for line in f:
        o = json.loads(line)
        if o.get("type") != "decision" or o.get("tail"): continue
        d = o.get("depth")
        if d is None or d >= MAXD: continue
        gt = o.get("gt_token")
        if gt is None: continue
        et, st = o.get("eagle_token"), o.get("suffix_token")
        avail = [t for t in (et, st) if t is not None]
        hits = [t for t in avail if t == gt]
        if len(avail) >= 2 and 0 < len(hits) < len(avail):   # decisive
            dec[d] += 1
            if picked_suffix(o): pick[d] += 1
    return pick, dec

def make_cell(cell):
    d = f"{ROOT}/{cell['dir']}"
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    print(f"\n### {cell['title']} — suffix selection rate by depth (decisive)")
    for lbl, color, cands in ARMS:
        res = None
        for fn in cands:
            res = suffix_rate_by_depth(f"{d}/{fn}")
            if res is not None: break
        if res is None: continue
        pick, dec = res
        rate = np.where(dec >= MIN_N, pick / np.maximum(dec, 1), np.nan)
        xs = [i for i in range(MAXD) if not np.isnan(rate[i])]
        ys = [rate[i] for i in xs]
        ls = "--" if lbl.startswith("oracle") else "-"
        lw = 2.4 if lbl.startswith(("bayes", "oracle")) else 1.8
        ax.plot(xs, ys, ls, color=color, lw=lw, marker="o", ms=3.5, label=lbl)
        print(f"   {lbl:16s} d0..7: " + " ".join(f"{rate[i]:.2f}" if not np.isnan(rate[i]) else " - " for i in range(8)))
    ax.set_xlabel("draft depth d"); ax.set_ylabel("P(pick suffix | decisive at d)")
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.3, MAXD - 1)
    ax.set_title(cell["title"], fontsize=13, fontweight="bold", loc="left")
    ax.text(0.0, 1.012, "REALIZED served · decisive positions · suffix-vs-dominant pick rate · "
            "oracle dashed = ideal (P(suffix==gt)) · held-out · pinned · greedy",
            transform=ax.transAxes, fontsize=6.8, color="#5e6e78")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout()
    out = f"{OUT}/realized_{cell['key']}_suffix_rate.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"   -> {out}")

if __name__ == "__main__":
    for c in CELLS:
        try: make_cell(c)
        except Exception as ex:
            import traceback; print(f"ERR {c['key']}: {ex}"); traceback.print_exc()
