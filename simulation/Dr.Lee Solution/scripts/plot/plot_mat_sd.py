#!/usr/bin/env python3
"""Standard MAT bar chart over the four full-trajectory workloads, now WITH the
Suffix-Decoding baseline arm (the paper's hybrid = per-step binary switch
between the suffix tree and the draft model; here with calibrated signals, so
it is a GENEROUS version of the SD baseline).

Arms (house colors, plot_mat_4way.py convention; SD baseline = purple):
  dflash  DFlash single proposer                       #4C78A8
  suffix  Suffix single proposer                       #F58518
  sd      Suffix Decoding hybrid (per-step switch)     #B279A2
  compose Compose (beta hazard + isotonic tail)        #54A24B
  oracle  Handoff oracle (best k per round)            #E45756

  PYTHONPATH=/workspace python3 scripts/plot_mat_sd.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
D = os.environ.get("INTERP_DIR", "results/interp_validation")
ARMS = ["dflash", "suffix", "switch_real", "compose", "handoff_oracle"]
LABELS = {"dflash": "DFlash (single)", "suffix": "Suffix (single)",
          "switch_real": "Suffix Decoding hybrid\n(per-step switch, SD baseline)",
          "compose": ("Compose\n(head+tail chain)" if "raw" in D
                      else "Compose (calibrated\nhead+tail chain)"),
          "handoff_oracle": "Handoff oracle\n(best k per round)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "switch_real": "#B279A2",
          "compose": "#54A24B", "handoff_oracle": "#E45756"}
WLS = ["spider", "swebench", "bfcl", "specbench"]
WL_NAME = {"spider": "Agentic SQL\n(Spider2-DBT)", "swebench": "SWE-bench Verified\n(mini-swe-agent)",
           "bfcl": "BFCL v4", "specbench": "Spec-Bench"}


def main():
    units = json.load(open(f"{D}/units.json"))
    rows = {u["wl"]: u["K"] for u in units if u["task"] == "__all__"}

    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    na = len(ARMS)
    bw = 0.8 / na
    ymax = 0.0
    for j, arm in enumerate(ARMS):
        xs = [i + (j - (na - 1) / 2) * bw for i in range(len(WLS))]
        vs = [rows[w][arm] for w in WLS]
        ymax = max(ymax, max(vs))
        hatch = "//" if arm == "handoff_oracle" else None
        face = "white" if arm == "handoff_oracle" else COLORS[arm]
        ax.bar(xs, vs, width=bw * 0.9, color=face, edgecolor=COLORS[arm],
               linewidth=1.3, hatch=hatch, label=LABELS[arm])
        for x, v in zip(xs, vs):
            ax.text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(WLS)))
    ax.set_xticklabels([WL_NAME[w] for w in WLS], fontsize=10.5)
    ax.set_ylabel("MAT (mean accepted draft tokens / round)", fontsize=11)
    ax.set_title("MAT with the Suffix-Decoding baseline — compose must be judged "
                 "against per-step switching, not only against singles\n"
                 "(Qwen3.5-27B DFlash + Suffix, offline replay, two-way in-sample; "
                 + ("SD hybrid uses the same raw signals as compose)" if "raw" in D
                    else "SD hybrid uses the same calibrated signals as compose)"),
                 fontsize=11.5, pad=12)
    ax.set_ylim(0, ymax * 1.14)
    ax.legend(fontsize=9, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = f"{D}/figures/mat_sd_baseline.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
