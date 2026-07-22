"""Reliability diagrams: raw prob vs OOF-calibrated prob, with per-bin accept COUNT.

Per cell, per proposer, two panels:
  LEFT  x = raw prob          y = empirical accept rate (token==gt), +diagonal, +/-SE
  RIGHT x = OOF-calibrated prob (isotonic P(accept|prob), GroupKFold by rid)
Each panel overlays, on a twin axis, the per-bin COUNT (total light + accepted dark)
-- the "accept된 count" the user asked for. ECE printed per panel.

Population = accept-conditioned proposing pool (walk each chain, stop once no
available proposer's token==gt; the same population the cond-trained calib maps
are fit on). label = token==gt.

The point this figure makes: calibration warps the x-axis so the curve hugs the
diagonal (reliability achieved) -- yet for the info-capped cells the calibrated
ranges of the competing proposers COLLAPSE/OVERLAP, so the cross-proposer
comparison that selection needs gains almost nothing.

Run (host): python3 simulation/scripts/select1_ladder/reliability_count.py
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

ROOT = "simulation/results/chain_hybrid_perdepth"
OUTDIR = Path("simulation/results/calib_reliability/figures")
NB = 20
EDGES = np.linspace(0.0, 1.0, NB + 1)
CENT = 0.5 * (EDGES[:-1] + EDGES[1:])

CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", [
        ("EAGLE3", "eagle_token", "eagle_p", "#1f77b4"),
        ("Suffix", "suffix_token", "suffix_p", "#d62728")]),
    "qwen35_27b_2way": ("qwen35_27b_ar", [
        ("MTP", "eagle_token", "eagle_p", "#1f77b4"),
        ("Suffix", "suffix_token", "suffix_p", "#d62728")]),
    "qwen3_8b_3way": ("qwen3_8b_dflash_e3_ceiling20", [
        ("DFlash", "eagle_token", "eagle_p", "#2ca02c"),
        ("EAGLE3", "e3_token", "e3_p", "#1f77b4"),
        ("Suffix", "suffix_token", "suffix_p", "#d62728")]),
    "qwen35_27b_3way": ("qwen35_27b_3way_real_full", [
        ("MTP", "eagle_token", "eagle_p", "#1f77b4"),
        ("DFlash", "dflash_token", "dflash_p", "#2ca02c"),
        ("Suffix", "suffix_token", "suffix_p", "#d62728")]),
}


def load_chains(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains:
        chains[k].sort(key=lambda r: r["depth"])
    return chains


def loopy_rids(d, thresh=0.5, n=4):
    dd = Path(d); gtf = dd / "gt_tokens.jsonl"
    if not gtf.exists():
        return set()
    reqs = {}
    for line in open(dd / "decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "req":
            reqs[o["rid"]] = tuple(o["input_ids"])
    gt = {}
    for line in open(gtf):
        r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
    bad = set()
    for rid, ids in reqs.items():
        out = gt.get(ids)
        if not out or len(out) < n + 1:
            continue
        g = [tuple(out[i:i + n]) for i in range(len(out) - n + 1)]
        if len(set(g)) / max(len(g), 1) < thresh:
            bad.add(rid)
    return bad


def collect(chains, props):
    """accept-conditioned per-proposer pools: name -> (prob[], y[], rid[])."""
    names = [p[0] for p in props]
    pool = {nm: ([], [], []) for nm in names}
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            tok = {p[0]: r.get(p[1]) for p in props}
            prob = {p[0]: r.get(p[2]) for p in props}
            avail = [nm for nm in names if tok[nm] is not None and prob[nm] is not None]
            if not avail:
                continue
            for nm in avail:
                pool[nm][0].append(float(prob[nm]))
                pool[nm][1].append(1 if gt is not None and tok[nm] == gt else 0)
                pool[nm][2].append(rid)
            hits = [nm for nm in avail if gt is not None and tok[nm] == gt]
            if gt is not None and len(hits) == 0:
                alive = False
    return {nm: (np.asarray(v[0]), np.asarray(v[1]), np.asarray(v[2]))
            for nm, v in pool.items()}


def oof_isotonic(p, y, g):
    pred = np.full(len(y), float(y.mean()))
    if len(y) < 50 or len(set(y)) < 2:
        return pred
    ng = len(set(g))
    for tr, te in GroupKFold(min(5, ng)).split(p, y, g):
        ir = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
        pred[te] = ir.predict(p[te])
    return pred


def oof_atom(p, y, g):
    """OOF per-ATOM calibration: a proposal's calibrated score = mean accept of TRAIN
    proposals sharing its EXACT prob value (the discrete c/n atom). Unseen test atoms
    (continuous probs) fall back to the train isotonic prediction. Finer than isotonic
    (per-atom, not monotone-pooled) -> maps each suffix atom to its own accept rate."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    pr = np.round(p, 4)
    pred = np.full(len(y), float(y.mean()))
    if len(y) < 50 or len(set(y.tolist())) < 2:
        return pred
    for tr, te in GroupKFold(min(5, len(set(g.tolist())))).split(p, y, g):
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
        prtr = pr[tr]; ytr = y[tr]
        order = np.argsort(prtr, kind="stable")
        uv, idx = np.unique(prtr[order], return_index=True)
        sums = np.add.reduceat(ytr[order], idx)
        cnts = np.add.reduceat(np.ones_like(ytr[order]), idx)
        rate = dict(zip(uv.tolist(), (sums / cnts).tolist()))
        iso_te = iso.predict(p[te])
        for j, i in enumerate(te):
            pred[i] = rate.get(float(pr[i]), float(iso_te[j]))
    return pred


def bin_stats(x, y):
    idx = np.clip(np.digitize(x, EDGES) - 1, 0, NB - 1)
    n = np.zeros(NB); acc = np.full(NB, np.nan); se = np.full(NB, np.nan); nacc = np.zeros(NB)
    for b in range(NB):
        m = idx == b
        n[b] = m.sum()
        if n[b] > 0:
            nacc[b] = y[m].sum()
            acc[b] = y[m].mean()
            se[b] = y[m].std() / np.sqrt(max(n[b], 1))
    return n, nacc, acc, se


def ece(x, y):
    n, _, acc, _ = bin_stats(x, y)
    tot = n.sum()
    val = 0.0
    for b in range(NB):
        if n[b] > 0:
            val += n[b] / tot * abs(acc[b] - CENT[b])
    return val


def panel(ax, x, y, color, title, min_n=10, count_ymax=None):
    n, nacc, acc, se = bin_stats(x, y)
    e = ece(x, y)
    # count bars on twin axis (right)
    axc = ax.twinx()
    axc.bar(CENT, n, width=0.045, color="#cccccc", alpha=0.55, zorder=1,
            label="total count")
    axc.bar(CENT, nacc, width=0.045, color=color, alpha=0.35, zorder=2,
            label="accepted count")
    axc.set_ylabel("count / bin", fontsize=8, color="#777")
    axc.tick_params(axis="y", labelsize=7, colors="#777")
    if count_ymax:  # shared count scale across proposers -> bars comparable
        axc.set_ylim(0, count_ymax * 1.05)
    # reliability line on left axis (rate)
    ok = n >= min_n
    ax.plot([0, 1], [0, 1], "k:", lw=0.9, alpha=0.6, zorder=3)
    ax.errorbar(CENT[ok], acc[ok], yerr=se[ok], fmt="o-", color=color, ms=4,
                lw=1.6, capsize=2, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zorder(axc.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel("P(token==gt) / accept rate", fontsize=8)
    ax.set_title(f"{title}   ECE={e:.3f}", fontsize=9)
    ax.grid(alpha=0.25)
    handles = [
        Line2D([0], [0], color=color, marker="o", ms=4, lw=1.6,
               label="accept rate = accepted/total"),
        Line2D([0], [0], color="k", ls=":", lw=0.9,
               label="perfect calibration (y=x)"),
        Patch(facecolor="#cccccc", alpha=0.55, label="total count / bin (all proposals)"),
        Patch(facecolor=color, alpha=0.35, label="accepted count / bin (token==gt)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.2,
              framealpha=0.9, handlelength=1.4, borderpad=0.4)
    return e


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, (dirname, props) in CELLS.items():
        d = f"{ROOT}/{dirname}"
        chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
        bad = loopy_rids(d)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        pool = collect(chains, props)
        nprop = len(props)
        fig, axes = plt.subplots(nprop, 3, figsize=(16, 3.1 * nprop), squeeze=False)
        print(f"\n=== {tag} (dir={dirname}, loopy-excl={len(bad)}) ===")
        # pre-pass: OOF-calibrate each proposer (isotonic + per-atom) + find the
        # figure-wide max bin count so the count axis is SHARED (bars comparable).
        prepped, count_ymax = [], 0.0
        for nm, _, _, color in props:
            p, y, g = pool[nm]
            if not len(y):
                continue
            cal = oof_isotonic(p, y, g)
            # atom-calibration only for the DISCRETE suffix score; continuous
            # proposers (eagle/mtp/dflash) keep isotonic (per-value lookup overfits
            # their ~unique floats, ECE blew up to .10-.18).
            acal = oof_atom(p, y, g) if nm.lower() == "suffix" else cal
            count_ymax = max(count_ymax, bin_stats(p, y)[0].max(),
                             bin_stats(cal, y)[0].max(), bin_stats(acal, y)[0].max())
            prepped.append((nm, color, p, y, cal, acal))
        print(f"  shared count axis ymax = {int(count_ymax)}")
        for r, (nm, color, p, y, cal, acal) in enumerate(prepped):
            e_raw = panel(axes[r][0], p, y, color,
                          f"{nm}: RAW prob  (n={len(y)}, base={y.mean():.3f})",
                          count_ymax=count_ymax)
            e_cal = panel(axes[r][1], cal, y, color,
                          f"{nm}: CALIBRATED prob (OOF isotonic)",
                          count_ymax=count_ymax)
            atom_note = "per-atom OOF" if nm.lower() == "suffix" else "= isotonic (continuous)"
            e_atom = panel(axes[r][2], acal, y, color,
                           f"{nm}: ATOM-CALIBRATED prob ({atom_note})",
                           count_ymax=count_ymax)
            axes[r][0].set_xlabel("raw prob", fontsize=8)
            axes[r][1].set_xlabel("calibrated prob", fontsize=8)
            axes[r][2].set_xlabel("atom-calibrated prob", fontsize=8)
            # how much does calibration spread or collapse the score range?
            print(f"  {nm:8} n={len(y):6} base_acc={y.mean():.3f}  "
                  f"ECE raw={e_raw:.3f} -> iso={e_cal:.3f} -> atom={e_atom:.3f}")
        fig.suptitle(f"Reliability + accept-count: raw vs OOF-calibrated prob — {tag}\n"
                     f"(accept-conditioned pool, label=token==gt; dotted=perfect calibration y=x)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        outp = OUTDIR / f"reliability_count_{tag}.png"
        fig.savefig(outp, dpi=140); plt.close(fig)
        print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
