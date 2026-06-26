"""Per-draft-position plot, one per drafter (EAGLE3 / MTP):
  x  = draft position (depth)
  line  = selection accuracy at that position (raw select-1, decisive,
          conditioned on the oracle chain reaching that depth)
  bars  = MAT loss at that position = the per-position contribution to
          (oracle MAT - raw MAT), attributed to the depth where raw first
          diverges from the oracle chain. NON-cumulative.

Both quantities are simulated from each cell's oracle decision log (ep, sp,
oracle_hit per depth), so raw and oracle share the same chains. All tasks
included. EAGLE3 uses existing data (not re-collected)."""
from __future__ import annotations
import json, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "simulation/results/chain_hybrid_perdepth"
FIGDIR = "simulation/results/calib_why_analysis/figures"
ALIVE = {"eagle", "suffix", "both"}
CELLS = {
    "mtp": {"dir": f"{ROOT}/qwen35_27b_ar", "title": "Qwen3.5-27B MTP", "out": "perpos_mtp.png"},
    "eagle3": {"dir": f"{ROOT}/qwen3_14b_ar", "title": "Qwen3-14B EAGLE3", "out": "perpos_eagle3.png"},
}


def compute(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains: chains[k].sort(key=lambda r: r["depth"])
    n = defaultdict(int); correct = defaultdict(int); loss = defaultdict(float)
    alive_tot = defaultdict(int); nsteps = 0
    for rs in chains.values():
        nsteps += 1
        raw_alive = orc_alive = True; Lraw = Lorc = 0
        for d, r in enumerate(rs):
            h = r.get("oracle_hit"); ep = r.get("eagle_p"); sp = r.get("suffix_p")
            pick_s = (sp is not None and ep is not None and sp > ep)
            # alive decisions WITH gt reached at d (denominator for P(decisive))
            if orc_alive and h in ("eagle", "suffix", "both", "none"):
                alive_tot[d] += 1
            if orc_alive and h in ("eagle", "suffix"):          # decisive, on alive prefix
                n[d] += 1
                ok = (pick_s and h == "suffix") or ((not pick_s) and h == "eagle")
                correct[d] += int(ok)
            raw_ok = (h == "both") or (h == "eagle" and not pick_s) or (h == "suffix" and pick_s)
            if raw_alive and raw_ok: Lraw += 1
            else: raw_alive = False
            if orc_alive and h in ALIVE: Lorc += 1
            else: orc_alive = False
            if not raw_alive and not orc_alive: break
        if Lorc > Lraw:
            loss[Lraw] += (Lorc - Lraw)
    maxd = max(n) if n else 0
    depths = list(range(maxd + 1))
    selacc = [(correct[d] / n[d]) if n[d] > 0 else np.nan for d in depths]
    matloss = [loss[d] / max(nsteps, 1) for d in depths]
    decprob = [(n[d] / alive_tot[d]) if alive_tot[d] > 0 else np.nan for d in depths]
    return depths, selacc, matloss, decprob


def plot(depths, selacc, matloss, decprob, title, out):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax2 = ax.twinx()
    b = ax2.bar(depths, matloss, color="#2ca02c", alpha=0.65, width=0.82,
                label="MAT loss / position")
    ln, = ax.plot(depths, selacc, "o-", color="#1f77b4", lw=2, ms=6,
                  label="selection accuracy / position (raw)")
    dp, = ax.plot(depths, decprob, "s--", color="#ff7f0e", lw=2, ms=5,
                  label="P(decisive) / position")
    ax.set_xlabel("draft position (depth)", color="black")
    ax.set_ylabel("selection accuracy  /  P(decisive)", color="black")
    ax.tick_params(axis="both", labelcolor="black", colors="black")
    ax2.set_ylabel("MAT loss (per position)", color="black")
    ax2.tick_params(axis="y", labelcolor="black", colors="black")
    ax.set_xticks(depths); ax.set_ylim(0, 1.02)
    ax2.set_ylim(0, (max(matloss) * 1.15) or 1)
    ax.set_title(title, color="black")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.legend([ln, dp, b], [ln.get_label(), dp.get_label(), b.get_label()],
              loc="upper right", fontsize=9, labelcolor="black")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["mtp", "eagle3", "both"], default="both")
    args = ap.parse_args()
    items = CELLS.items() if args.cell == "both" else [(args.cell, CELLS[args.cell])]
    for key, c in items:
        depths, selacc, matloss, decprob = compute(f"{c['dir']}/decisions_select1_oracle.jsonl")
        print(f"=== {key} {c['title']} ===")
        for d in depths:
            print(f"  d={d:2d} selacc={selacc[d]:.3f} matloss={matloss[d]:.4f} P(dec)={decprob[d]:.3f}")
        plot(depths, selacc, matloss, decprob, c["title"], f"{FIGDIR}/{c['out']}")


if __name__ == "__main__":
    main()
