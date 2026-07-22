"""Figures for the leverage-weighted (depth-weighting) selection study.

Three images, matching the existing select1_ladder bar/per-depth style:
  1. lev_mat.png       MAT by weight scheme x selector, per cell (2x2), raw+oracle refs
  2. lev_selacc.png    selection accuracy, same layout
  3. lev_perdepth.png  per-depth raw selacc / per-depth MAT-loss share / per-depth
                       decisive rate, 4 cells overlaid (3 panels)

Bar numbers (selacc, MAT) are parsed from the saved rerun log so we don't refit the
GBM OOF; per-depth quantities are computed fresh from decisions_select1_oracle.jsonl.

Run: python3 simulation/scripts/select1_ladder/plot_leverage_weighted.py
"""
import json, math, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "simulation/results/chain_hybrid_perdepth"
LOG = f"{ROOT}/leverage_weighted_rerun.log"
OUT = f"{ROOT}/leverage_figures"
Path(OUT).mkdir(exist_ok=True)

# (log-name, model-label, dir, props[(name, tok_key, prob_key)])  -- main proposer first
CELLS = [
    ("14B 2-way", "Qwen3-14B   ·   EAGLE3 + suffix", "qwen3_14b_ar",
     [("EAGLE3", "eagle_token", "eagle_p"), ("suffix", "suffix_token", "suffix_p")]),
    ("27B 2-way", "Qwen3.5-27B   ·   MTP + suffix", "qwen35_27b_ar",
     [("MTP", "eagle_token", "eagle_p"), ("suffix", "suffix_token", "suffix_p")]),
    ("8B 3-way", "Qwen3-8B   ·   DFlash + EAGLE3 + suffix", "qwen3_8b_dflash_e3_ceiling20",
     [("DFlash", "eagle_token", "eagle_p"), ("EAGLE3", "e3_token", "e3_p"), ("suffix", "suffix_token", "suffix_p")]),
    ("27B 3-way", "Qwen3.5-27B   ·   MTP + DFlash + suffix", "qwen35_27b_3way_real_full",
     [("MTP", "eagle_token", "eagle_p"), ("DFlash", "dflash_token", "dflash_p"), ("suffix", "suffix_token", "suffix_p")]),
]

SELECTORS = ["gbm(p+d)", "gbm(p+d+sfx)", "logistic(p+d)"]
SCHEMES = ["none", "leverage", "inv", "linear"]
SCHEME_LABEL = {"none": "none", "leverage": "leverage(d)", "inv": "1/(d+1)", "linear": "maxd−d"}
SCHEME_COLOR = {"none": "#9aa7ad", "leverage": "#1f7a8c", "inv": "#c2683a", "linear": "#6a4c93"}
CELL_COLOR = {"14B 2-way": "#1f77b4", "27B 2-way": "#2ca02c",
              "8B 3-way": "#d62728", "27B 3-way": "#9467bd"}
FLT = re.compile(r"[-+]?\d+\.\d+")


# ---------------------------------------------------------------- parse bar table
def parse_log(path):
    bars = {c[0]: {s: {} for s in SELECTORS} for c in CELLS}
    refs = {c[0]: {} for c in CELLS}
    cell = sel = None
    for line in open(path):
        s = line.strip()
        m = re.match(r"### (.+?)\s+blocks=", s)
        if m:
            cell = m.group(1).strip(); sel = None; continue
        if cell is None:
            continue
        m = re.match(r"-- (.+?) --", s)
        if m:
            sel = m.group(1).strip(); continue
        if s.startswith("raw (prob argmax)"):
            f = [float(x) for x in FLT.findall(s)]
            refs[cell]["raw_sa"], refs[cell]["raw_mat"] = f[0], f[-1]; continue
        if s.startswith("oracle"):
            f = [float(x) for x in FLT.findall(s)]
            refs[cell]["orc_sa"], refs[cell]["orc_mat"] = f[0], f[-1]; continue
        tok = s.split()[0] if s else ""
        if sel and tok in SCHEMES:
            f = [float(x) for x in FLT.findall(s)]
            bars[cell][sel][tok] = (f[0], f[3])  # (selacc, MAT)
    return bars, refs


# ---------------------------------------------------------------- per-depth compute
def load_blocks(d, props):
    raw = defaultdict(list)
    for line in open(f"{d}/decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            raw[(o["rid"], o["decode_step"])].append(o)
    blocks = {}
    for k, rs in raw.items():
        rs.sort(key=lambda r: r["depth"]); pos = []
        for r in rs:
            e = {"depth": int(r["depth"]), "gt": r.get("gt_token"), "P": {}}
            for nm, tk, pk in props:
                t = r.get(tk); p = r.get(pk)
                if t is not None and p is not None:
                    e["P"][nm] = (t, float(p))
            pos.append(e)
        blocks[k] = pos
    return blocks


def loopy(d):
    dd = Path(d); gtf = dd / "gt_tokens.jsonl"; reqs = {}
    for line in open(dd / "decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "req":
            reqs[o["rid"]] = tuple(o["input_ids"])
    bad = set()
    if gtf.exists():
        gt = {}
        for line in open(gtf):
            r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
        for rid, ids in reqs.items():
            out = gt.get(ids)
            if not out or len(out) < 5:
                continue
            g = [tuple(out[i:i + 4]) for i in range(len(out) - 3)]
            if len(set(g)) / max(len(g), 1) < 0.5:
                bad.add(rid)
    return bad


def perdepth(blocks, maxd=16):
    sel_ok = np.zeros(maxd); sel_n = np.zeros(maxd)        # raw selacc on decisive
    dec = np.zeros(maxd); reach = np.zeros(maxd)           # decisive rate
    loss = np.zeros(maxd)                                  # raw->oracle MAT loss by first-miss depth
    raw_runs = []; orc_runs = []
    for pos in blocks.values():
        raw_run = orc_run = 0; raw_alive = orc_alive = True; fmiss = None
        for e in pos:
            gt = e["gt"]
            if gt is None or not e["P"]:
                break
            av = list(e["P"]); hits = [nm for nm in av if e["P"][nm][0] == gt]
            allhit = len(hits) == len(av); nohit = len(hits) == 0
            d = e["depth"]
            if orc_alive and not nohit and d < maxd:        # decision-stats while oracle alive
                reach[d] += 1
                if not allhit:
                    dec[d] += 1; sel_n[d] += 1
                    pick = max(e["P"], key=lambda nm: e["P"][nm][1])
                    if e["P"][pick][0] == gt:
                        sel_ok[d] += 1
            if orc_alive:
                if nohit: orc_alive = False
                else: orc_run += 1
            if raw_alive:
                if nohit:
                    raw_alive = False
                elif allhit:
                    raw_run += 1
                else:
                    pick = max(e["P"], key=lambda nm: e["P"][nm][1])
                    if e["P"][pick][0] == gt:
                        raw_run += 1
                    else:
                        raw_alive = False
                        fmiss = d
            if not raw_alive and not orc_alive:
                break
        if fmiss is not None and fmiss < maxd:
            loss[fmiss] += (orc_run - raw_run)
        raw_runs.append(raw_run); orc_runs.append(orc_run)
    nb = len(blocks)
    return dict(
        selacc=np.where(sel_n > 0, sel_ok / np.maximum(sel_n, 1), np.nan),
        dec_rate=np.where(reach > 0, dec / np.maximum(reach, 1), np.nan),
        loss_share=loss / max(loss.sum(), 1e-9),
        loss_avg=loss / max(nb, 1),
        reach=reach,
        raw_mat=float(np.mean(raw_runs)), orc_mat=float(np.mean(orc_runs)),
        d2_share=float(loss[:3].sum() / max(loss.sum(), 1e-9)))


# ---------------------------------------------------------------- figures
CELL_KEY = {"14B 2-way": "14b_2way", "27B 2-way": "27b_2way",
            "8B 3-way": "8b_3way", "27B 3-way": "27b_3way"}
SEL_KEY = {"gbm(p+d)": "gbm_pd", "gbm(p+d+sfx)": "gbm_pdsfx", "logistic(p+d)": "logistic_pd"}


def fig_one_bar(cell, model, sel, bars, refs, metric_idx, ylabel, fmt, fname):
    """One image: 4 weight-scheme bars + raw/oracle refs, for a single (model, method)."""
    vals = [bars[cell][sel].get(s, (np.nan, np.nan))[metric_idx] for s in SCHEMES]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    xp = np.arange(len(SCHEMES))
    ax.bar(xp, vals, color=[SCHEME_COLOR[s] for s in SCHEMES],
           edgecolor="white", width=0.66)
    vmax = np.nanmax(vals)
    for x, v in zip(xp, vals):
        if not np.isnan(v):
            ax.text(x, v + vmax * 0.012, fmt(v), va="bottom", ha="center",
                    fontsize=9.5, fontfamily="monospace")
    rk = "sa" if metric_idx == 0 else "mat"
    for key, col, lab in [("raw", "#c2683a", "raw (prob argmax)"), ("orc", "#2f7a57", "oracle")]:
        rv = refs[cell][f"{key}_{rk}"]
        ax.axhline(rv, ls="--", lw=1.8, color=col, alpha=0.9, label=f"{lab} ({fmt(rv)})")
        vmax = max(vmax, rv)
    ax.set_xticks(xp); ax.set_xticklabels([SCHEME_LABEL[s] for s in SCHEMES], fontsize=10)
    ax.set_xlabel("weight scheme  (sample_weight)")
    ax.set_ylim(0, vmax * 1.16); ax.set_ylabel(ylabel)
    ax.set_title(f"{model}\n{sel}", fontsize=12.5, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.6, framealpha=0.9, loc="lower right")
    fig.text(0.012, 0.022, "bfcl_v4 web_search · leverage-weighted selection · offline, "
             "block-anchored, GroupKFold-OOF · loopy reqs excluded",
             fontsize=7.2, color="#5e6e78", ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = f"{OUT}/{fname}"; fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def fig_perdepth_one(cell, model, pd):
    DMAX = 13
    xs = np.arange(DMAX)
    valid = pd["reach"][:DMAX] >= 30
    x = xs[valid]
    col = CELL_COLOR[cell]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    series = [
        ("Per-depth raw selection accuracy\n(decisive positions)", "selection accuracy",
         pd["selacc"], (0.55, 1.0), False),
        ("Per-depth MAT-loss share\n(raw→oracle, by first-miss depth)", "share of total MAT loss",
         pd["loss_share"], None, True),
        ("Per-depth decisive rate\n(P[real choice arises | oracle alive])", "decisive rate",
         pd["dec_rate"], (0, 1.0), False),
    ]
    for ax, (title, ylab, arr, ylim, shade) in zip(axes, series):
        if shade:
            ax.axvspan(-0.5, 2.5, color="#cccccc", alpha=0.30, zorder=0,
                       label=f"d≤2 = {pd['d2_share']*100:.0f}% of loss")
        ax.plot(x, np.asarray(arr)[:DMAX][valid], "-o", color=col, lw=2.2, ms=4.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("composed-chain depth d"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
        if ylim: ax.set_ylim(*ylim)
        if shade: ax.legend(fontsize=9, loc="upper right")
    fig.suptitle(f"{model}  —  per-depth: selacc capped low at shallow, yet MAT loss "
                 f"concentrates there (offline)", fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = f"{OUT}/lev_perdepth_{CELL_KEY[cell]}.png"; fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def main():
    # drop the earlier combined figures (superseded by per-model×method split)
    for old in ("lev_mat.png", "lev_selacc.png", "lev_perdepth.png"):
        p = Path(OUT) / old
        if p.exists(): p.unlink()
    bars, refs = parse_log(LOG)
    pd_all = {}
    for cell, model, dr, props in CELLS:
        d = f"{ROOT}/{dr}"
        blocks = load_blocks(d, props)
        bad = loopy(d); blocks = {k: v for k, v in blocks.items() if k[0] not in bad}
        pd = perdepth(blocks); pd_all[cell] = pd
        print(f"{cell:10s} blocks={len(blocks):5d} raw_mat={pd['raw_mat']:.3f} "
              f"orc_mat={pd['orc_mat']:.3f}  d<=2 loss share={pd['d2_share']*100:.0f}%  "
              f"(log raw/orc = {refs[cell]['raw_mat']:.3f}/{refs[cell]['orc_mat']:.3f})")
    # MAT + selacc: one image per (model, method)
    for cell, model, _dr, _props in CELLS:
        ck = CELL_KEY[cell]
        for sel in SELECTORS:
            sk = SEL_KEY[sel]
            fig_one_bar(cell, model, sel, bars, refs, 1, "MAT",
                        lambda v: f"{v:.2f}", f"lev_mat_{ck}_{sk}.png")
            fig_one_bar(cell, model, sel, bars, refs, 0, "Selection accuracy (decisive)",
                        lambda v: f"{v:.3f}", f"lev_selacc_{ck}_{sk}.png")
    # per-depth: one image per model
    for cell, model, _dr, _props in CELLS:
        fig_perdepth_one(cell, model, pd_all[cell])


if __name__ == "__main__":
    main()
