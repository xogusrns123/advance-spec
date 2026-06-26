"""Per-DEPTH root cause: why the joint discriminator (higher offline AUC) still
loses to raw on MAT — the same shallow-win / deep-loss mechanism as calibration.

On the disc run's oracle decision log (GT-labeled, test slice) we apply, OFFLINE,
the raw rule and the fitted logistic/beta discriminators to the SAME decisions,
and report per composed-chain depth:
  n(d)              decisive+contested decisions at depth d
  suffix-right      fraction where suffix is the GT proposer (optimal share)
  raw pick/acc      raw rule (suffix_p>eagle_p)
  disc pick/acc     discriminator rule (P(pick suffix)>0.5), per algo

MAT is a multiplicative survival integral (you must win depths 0..d-1 to reach
length d), so it is dominated by DEEP-depth wins. AUC weights every decision
equally and is dominated by the high-mass SHALLOW depths. A discriminator that
maximizes overall accuracy can improve shallow picks (higher AUC, higher
P(len>=1)) yet lose the deep picks that actually extend the accepted run ->
higher AUC but lower MAT. This script shows that crossover.

Usage (container):
  python3 simulation/scripts/analyze_o4_disc_perdepth.py \
      --dir simulation/results/o4_perdepth/qwen3_14b_disc --model-label EAGLE3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/workspace/simulation/oracle")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
from chain_hybrid_patch import _ServingDiscriminator  # noqa: E402

ALGOS = ["logistic", "beta"]
CMAP = {"logistic": "#2ca02c", "beta": "#17becf"}


def load(path):
    sp, ep, ml, cnt, tot, dep, corr = [], [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail"):
                continue
            if r.get("oracle_hit") not in ("eagle", "suffix"):
                continue
            if r.get("eagle_p") is None or r.get("suffix_p") is None:
                continue
            sp.append(float(r["suffix_p"])); ep.append(float(r["eagle_p"]))
            ml.append(r.get("match_len")); cnt.append(r.get("suffix_count"))
            tot.append(r.get("suffix_total"))
            dep.append(int(r["depth"]))
            corr.append(1 if r["oracle_hit"] == "suffix" else 0)
    return (np.asarray(sp), np.asarray(ep), ml, cnt, tot,
            np.asarray(dep, int), np.asarray(corr, int))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    ap.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args()
    D = Path(args.dir)
    sp, ep, ml, cnt, tot, dep, corr = load(D / "decisions_select1_oracle.jsonl")
    n = len(corr)
    raw_pick = (sp > ep).astype(int)
    print(f"decisive+contested n={n}  suffix-right={corr.mean():.3f}  "
          f"raw picks_suffix={raw_pick.mean():.3f} acc={(raw_pick==corr).mean():.3f}")

    discs, dpick = {}, {}
    for algo in ALGOS:
        disc = _ServingDiscriminator.load(str(D / f"disc_{algo}.json"))
        discs[algo] = disc
        p = np.array([disc.predict(sp[i], ep[i], ml[i], cnt[i], tot[i])
                      for i in range(n)])
        dpick[algo] = (p > 0.5).astype(int)
        print(f"  {algo:9s} picks_suffix={dpick[algo].mean():.3f} "
              f"acc={(dpick[algo]==corr).mean():.3f}")

    depths = sorted(set(dep.tolist()))
    rows = {}
    print(f"\n{'d':>2s} {'n':>5s} {'suf-rt':>6s} {'raw_ps':>6s} {'raw_ac':>6s}"
          + "".join(f" {a[:4]+'_ps':>8s} {a[:4]+'_ac':>8s}" for a in ALGOS))
    for d in depths:
        m = dep == d
        nd = int(m.sum())
        rec = dict(n=nd, sr=float(corr[m].mean()),
                   raw_ps=float(raw_pick[m].mean()),
                   raw_ac=float((raw_pick[m] == corr[m]).mean()))
        line = (f"{d:2d} {nd:5d} {rec['sr']:6.3f} {rec['raw_ps']:6.3f} "
                f"{rec['raw_ac']:6.3f}")
        for a in ALGOS:
            ps = float(dpick[a][m].mean()); ac = float((dpick[a][m] == corr[m]).mean())
            rec[f"{a}_ps"] = ps; rec[f"{a}_ac"] = ac
            line += f" {ps:8.3f} {ac:8.3f}"
        rows[d] = rec
        print(line)

    pd = [d for d in depths if rows[d]["n"] >= args.min_n]
    xs = np.array(pd)
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.5]})

    ax0.plot(xs, [rows[d]["sr"] for d in pd], "k-o", lw=2.2, ms=4,
             label="optimal share (suffix-right)", zorder=5)
    ax0.plot(xs, [rows[d]["raw_ps"] for d in pd], color="#1f77b4", lw=2.0,
             marker="s", ms=3, label="raw picks_suffix")
    for a in ALGOS:
        ax0.plot(xs, [rows[d][f"{a}_ps"] for d in pd], color=CMAP[a], lw=1.5,
                 ls="--", marker=".", ms=3, label=f"disc_{a} picks_suffix")
    ax0.set_xlabel("composed-chain depth d"); ax0.set_ylabel("suffix-pick rate")
    ax0.set_title("disc picks ~ raw (slightly less suffix,\n"
                  "marginally closer to optimal)", fontsize=10)
    ax0.set_ylim(0, 1.02); ax0.grid(alpha=0.3); ax0.legend(fontsize=7)

    ax1.plot(xs, [rows[d]["raw_ac"] for d in pd], color="#1f77b4", lw=2.2,
             marker="s", ms=3, label="raw acc")
    for a in ALGOS:
        ax1.plot(xs, [rows[d][f"{a}_ac"] for d in pd], color=CMAP[a], lw=1.5,
                 ls="--", marker=".", ms=3, label=f"disc_{a} acc")
    ax1.set_xlabel("composed-chain depth d")
    ax1.set_ylabel("selection accuracy vs GT")
    ax1.set_title("OFFLINE (GT trajectory): disc >= raw at most depths\n"
                  "(yet serving MAT is NOT higher — offline/on-policy gap)",
                  fontsize=10)
    ax1.set_ylim(0.4, 1.02); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    ax2.bar(xs, [rows[d]["n"] for d in pd], color="#bbbbbb")
    ax2.set_xlabel("depth d"); ax2.set_ylabel("# decisive decisions")
    ax2.set_title("decision mass", fontsize=10); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"O4 discriminator per-depth — higher OFFLINE accuracy than raw "
                 f"at most depths, yet serving MAT not higher "
                 f"(offline<->on-policy gap) — {args.model_label}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = D / "figures" / "o4_disc_perdepth.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
