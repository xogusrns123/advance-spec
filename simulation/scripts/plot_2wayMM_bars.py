"""Bar charts for the 2-way MODEL-vs-MODEL block-anchored study.

Two figures, each with an 8B (DFlash vs EAGLE3) and a 27B (MTP vs DFlash) panel:
  fig1: MAT     fig2: decisive selection accuracy
Arms (exactly): prop1-only, prop2-only, raw competition, calib isotonic,
                calib beta, calib mono, oracle.

single-proposer selacc = the proposer's hit-rate on the 2-way decisive rows
(i.e. "always commit to this proposer" accuracy) -- selacc_mat returns 0 for a
1-proposer subset because a row can't be decisive with one proposer, so we read
the decisive hit-rate directly from the collected samples.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from analyze_3way_ladder import (  # noqa: E402
    CELLS, load_chains, loopy_rids, row_info, collect_samples, oof_estimates,
    selacc_mat, pick_raw, pick_est, pick_oracle,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUTDIR = RESULTS / "twoway_modelmodel" / "figures"

# (cell, proposer pair, pretty labels, record dir, extra loopy exclude?)
CONFIGS = [
    dict(cell="8b", names=("dflash", "e3"), pretty=("DFlash", "EAGLE3"),
         model="Qwen3-8B", dir="qwen3_8b_dflash_e3_ceiling20", exclude_loopy=True),
    dict(cell="27b", names=("mtp", "dflash"), pretty=("MTP", "DFlash"),
         model="Qwen3.5-27B", dir="qwen35_27b_3way_real_full", exclude_loopy=False),
]

# display order -> (label, group) ; group drives colour
ARMS = [
    ("single_a", "single"), ("single_b", "single"),
    ("raw", "raw"),
    ("isotonic", "calib"), ("beta", "calib"), ("mono", "mono"),
    ("oracle", "oracle"),
]
COLORS = {"single": "#9aa7ad", "raw": "#c2683a", "calib": "#1f7a8c",
          "mono": "#e08a1e", "oracle": "#2f7a57"}


def compute(cfg):
    c = CELLS[cfg["cell"]]
    names = cfg["names"]
    tok = {P: c["tok"][P] for P in names}
    pk = {P: c["p"][P] for P in names}
    d = RESULTS / "chain_hybrid_perdepth" / cfg["dir"]
    chains = load_chains(d / "decisions_select1_oracle.jsonl")
    if cfg["exclude_loopy"]:
        bad = loopy_rids(str(d), "decisions_select1_oracle.jsonl")
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
    # keep blocks where both model proposers appear
    chains = {k: v for k, v in chains.items()
              if any(r.get(tok[names[0]]) is not None for r in v)
              and any(r.get(tok[names[1]]) is not None for r in v)}
    for rs in chains.values():
        for r in rs:
            r["_ri"] = row_info(r, names, tok, pk)
    S = collect_samples(chains, names, tok, pk)
    for m in ("raw", "isotonic", "beta", "mono", "oracle"):
        oof_estimates(S, names, m)

    a, b = names
    # single-proposer selacc = decisive hit-rate; MAT = 1-proposer block MAT
    hit = {P: (float(np.mean(S[P]["y"])) if S[P]["y"] else 0.0) for P in names}
    _, mat_a = selacc_mat(chains, pick_raw, names, tok, pk, (a,))
    _, mat_b = selacc_mat(chains, pick_raw, names, tok, pk, (b,))
    sa_raw, mat_raw = selacc_mat(chains, pick_raw, names, tok, pk, names)
    sa_iso, mat_iso = selacc_mat(chains, pick_est("isotonic"), names, tok, pk, names)
    sa_beta, mat_beta = selacc_mat(chains, pick_est("beta"), names, tok, pk, names)
    sa_mono, mat_mono = selacc_mat(chains, pick_est("mono"), names, tok, pk, names)
    sa_orc, mat_orc = selacc_mat(chains, pick_oracle(names), names, tok, pk, names)

    selacc = {"single_a": hit[a], "single_b": hit[b], "raw": sa_raw,
              "isotonic": sa_iso, "beta": sa_beta, "mono": sa_mono, "oracle": sa_orc}
    mat = {"single_a": mat_a, "single_b": mat_b, "raw": mat_raw,
           "isotonic": mat_iso, "beta": mat_beta, "mono": mat_mono, "oracle": mat_orc}
    labels = {"single_a": f"{cfg['pretty'][0]}\nonly",
              "single_b": f"{cfg['pretty'][1]}\nonly",
              "raw": "raw\ncompete", "isotonic": "calib\nisotonic",
              "beta": "calib\nbeta", "mono": "calib\nmono", "oracle": "oracle"}
    return dict(cfg=cfg, selacc=selacc, mat=mat, labels=labels,
                n_blocks=len(chains), n_dec=len(S[a]["y"]))


def draw(metric, ylabel, title, fname, data):
    fig, axes = plt.subplots(1, len(data), figsize=(7.2 * len(data), 5.4))
    if len(data) == 1:
        axes = [axes]
    for ax, D in zip(axes, data):
        vals = [D[metric][k] for k, _ in ARMS]
        cols = [COLORS[g] for _, g in ARMS]
        xp = np.arange(len(ARMS))
        ax.bar(xp, vals, color=cols, edgecolor="white", width=0.72)
        raw_v = D[metric]["raw"]; orc_v = D[metric]["oracle"]
        ax.axhline(raw_v, color=COLORS["raw"], ls=":", lw=1, alpha=0.55)
        ax.axhline(orc_v, color=COLORS["oracle"], ls=":", lw=1, alpha=0.55)
        for x, v in zip(xp, vals):
            ax.text(x, v + max(vals) * 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, fontfamily="monospace")
        ax.set_xticks(xp)
        ax.set_xticklabels([D["labels"][k] for k, _ in ARMS], fontsize=9)
        ax.set_ylim(0, max(vals) * 1.16)
        ax.set_ylabel(ylabel, fontsize=10)
        cfg = D["cfg"]
        ax.set_title(f"{cfg['model']}  ·  {cfg['pretty'][0]} vs {cfg['pretty'][1]}  "
                     f"(model-vs-model, block-anchored)\n"
                     f"{D['n_blocks']} blocks, {D['n_dec']} decisive rows",
                     fontsize=10.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / fname, dpi=140, bbox_inches="tight")
    print(f"wrote {OUTDIR / fname}")


def main():
    data = [compute(cfg) for cfg in CONFIGS]
    for D in data:
        cfg = D["cfg"]
        print(f"\n{cfg['model']} {cfg['names']}: blocks={D['n_blocks']} decisive={D['n_dec']}")
        for k, _ in ARMS:
            print(f"  {k:10s} selacc={D['selacc'][k]:.4f}  MAT={D['mat'][k]:.4f}")
    draw("mat", "MAT (block-anchored)",
         "2-way model-vs-model  —  MAT", "bars_mat_2wayMM.png", data)
    draw("selacc", "decisive selection accuracy",
         "2-way model-vs-model  —  selection accuracy", "bars_selacc_2wayMM.png", data)


if __name__ == "__main__":
    main()
