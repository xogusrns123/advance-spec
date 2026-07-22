"""2-way select-1 ladder with the SHALLOW-WEIGHTED (leverage) selector family.

Replaces the earlier "depthwise calib" arm with the leverage-weighted experiment
(Candidate A): the SAME gbm(prob+depth) selector, trained with sample_weight that
deliberately upweights shallow (high-MAT-loss) depths. All 3 weight schemes shown:
  leverage(d)  (measured oracle-remaining curve) | 1/(d+1) | maxd−d.
calib = same selector, sample_weight=none (the apples-to-apples baseline). raw = argmax
prob. So calib≈leverage≈1/(d+1)≈maxd−d visually ⇒ shallow weighting does nothing.

Cells: 14B EAGLE3+suffix, 27B MTP+suffix. Offline, block-anchored, GroupKFold-OOF.

Figures:
  mat_ladder.png / selacc_ladder.png   single×2, raw, calib, shallow-wt×3, oracle
  perdepth_raw.png / perdepth_calib.png / perdepth_shallowwt_{leverage,inv,linear}.png
        x=depth, MAT-loss bars + selacc & decisive-rate lines.

Run: python3 simulation/scripts/select1_ladder/plot_twoway_shallow.py
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

ROOT = "simulation/results/chain_hybrid_perdepth"
OUT = f"{ROOT}/twoway_figures"
Path(OUT).mkdir(exist_ok=True)

CELLS = [
    ("14b_2way", "Qwen3-14B   ·   EAGLE3 + suffix", "qwen3_14b_ar",
     [("EAGLE3", "eagle_token", "eagle_p"), ("suffix", "suffix_token", "suffix_p")]),
    ("27b_2way", "Qwen3.5-27B   ·   MTP + suffix", "qwen35_27b_ar",
     [("MTP", "eagle_token", "eagle_p"), ("suffix", "suffix_token", "suffix_p")]),
]
SCHEMES = ["none", "leverage", "inv", "linear"]
SCHEME_LABEL = {"none": "calib\n(no wt)", "leverage": "shallow-wt\nleverage(d)",
                "inv": "shallow-wt\n1/(d+1)", "linear": "shallow-wt\nmaxd−d"}
COL = {"single": "#9aa7ad", "raw": "#c2683a", "none": "#1f7a8c",
       "leverage": "#6a4c93", "inv": "#9575b8", "linear": "#b9a4d6", "oracle": "#2f7a57"}
# per-depth image method order/labels
PD_METHODS = [
    ("raw", "raw (argmax prob)", "#c2683a", "perdepth_raw.png"),
    ("none", "calib (gbm prob+depth, no weight)", "#1f7a8c", "perdepth_calib.png"),
    ("leverage", "shallow-wt: sample_weight = leverage(d)", "#6a4c93", "perdepth_shallowwt_leverage.png"),
    ("inv", "shallow-wt: sample_weight = 1/(d+1)", "#9575b8", "perdepth_shallowwt_inv.png"),
    ("linear", "shallow-wt: sample_weight = maxd−d", "#b9a4d6", "perdepth_shallowwt_linear.png"),
]
DMAX = 13


# ----------------------------------------------------------------- data
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
            e = {"depth": int(r["depth"]), "gt": r.get("gt_token"), "rid": k[0], "P": {}}
            for nm, tk, pk in props:
                t = r.get(tk); p = r.get(pk)
                if t is not None and p is not None:
                    e["P"][nm] = dict(tok=t, prob=float(p))
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


def decisive_positions(blocks):
    alive = []
    for k, pos in blocks.items():
        al = True
        for e in pos:
            if not al:
                break
            if not e["P"]:
                continue
            hits = [nm for nm in e["P"] if e["P"][nm]["tok"] == e["gt"]]
            if 0 < len(hits) < len(e["P"]):
                alive.append(((k[0], k[1], e["depth"]), e))
            if e["gt"] is not None and len(hits) == 0:
                al = False
    return alive


def leverage_curve(blocks, maxd=64):
    lev_sum = np.zeros(maxd); reach = np.zeros(maxd)
    for pos in blocks.values():
        D = len(pos)
        for i, e in enumerate(pos):
            gt = e["gt"]
            if gt is None or not e["P"]:
                D = i; break
            if not any(v["tok"] == gt for v in e["P"].values()):
                D = i; break
        for i in range(min(D, maxd)):
            reach[i] += 1; lev_sum[i] += (D - i)
    return np.where(reach > 0, lev_sum / np.maximum(reach, 1), 0.0)


def weight_of(depth, scheme, lev, maxd):
    if scheme == "none":     return 1.0
    if scheme == "leverage": return float(lev[depth]) if depth < len(lev) and lev[depth] > 0 else 1e-3
    if scheme == "inv":      return 1.0 / (depth + 1)
    if scheme == "linear":   return float(max(maxd - depth, 1))
    raise ValueError(scheme)


# --------------------------------------------- gbm(prob+depth) selector w/ weights
def gbm_picks(alive, props, scheme, lev, maxd):
    names = [p[0] for p in props]
    R = {nm: {"X": [], "y": [], "g": [], "dep": [], "did": []} for nm in names}
    for did, e in alive:
        for nm in e["P"]:
            R[nm]["X"].append([e["P"][nm]["prob"], e["depth"]])
            R[nm]["y"].append(1 if e["P"][nm]["tok"] == e["gt"] else 0)
            R[nm]["g"].append(did[0]); R[nm]["dep"].append(e["depth"]); R[nm]["did"].append(did)
    P_of = defaultdict(dict)
    for nm in names:
        X = np.array(R[nm]["X"], float); y = np.array(R[nm]["y"])
        g = np.array(R[nm]["g"]); dids = R[nm]["did"]
        w = np.array([weight_of(dp, scheme, lev, maxd) for dp in R[nm]["dep"]], float)
        w = w * (len(w) / w.sum()) if w.sum() > 0 else np.ones_like(w)
        if len(y) < 10 or len(set(y)) < 2:
            for did, yy in zip(dids, y):
                P_of[did][nm] = float(yy)
            continue
        prd = np.zeros(len(y))
        for tr, te in GroupKFold(min(5, len(set(g)))).split(X, y, g):
            if len(set(y[tr])) < 2:
                prd[te] = y[tr].mean(); continue
            clf = HGB(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0)
            clf.fit(X[tr], y[tr], sample_weight=w[tr])
            prd[te] = clf.predict_proba(X[te])[:, 1]
        for did, pp in zip(dids, prd):
            P_of[did][nm] = float(pp)
    return {did: (max(P_of[did], key=lambda nm: P_of[did][nm]) if P_of[did] else None)
            for did, _ in alive}


# ----------------------------------------------------------------- metrics
def run_length(blocks, pickfn):
    tot = n = 0
    for k, pos in blocks.items():
        n += 1; run = 0
        for e in pos:
            gt = e["gt"]
            if gt is None or not e["P"]:
                break
            hits = [nm for nm in e["P"] if e["P"][nm]["tok"] == gt]
            if len(hits) == len(e["P"]):
                run += 1; continue
            if not hits:
                break
            nm = pickfn((k[0], k[1], e["depth"]), e)
            if nm is not None and e["P"][nm]["tok"] == gt:
                run += 1
            else:
                break
        tot += run
    return tot / max(n, 1)


def selacc(alive, pk):
    return float(np.mean([1.0 if (pk.get(d) is not None and e["P"][pk[d]]["tok"] == e["gt"]) else 0.0
                          for d, e in alive])) if alive else float("nan")


def perdepth_method(blocks, pickfn, maxd=16):
    sel_ok = np.zeros(maxd); sel_n = np.zeros(maxd)
    dec = np.zeros(maxd); reach = np.zeros(maxd); loss = np.zeros(maxd)
    m_runs = []; o_runs = []
    for k, pos in blocks.items():
        m_run = o_run = 0; m_alive = o_alive = True; fmiss = None
        for e in pos:
            gt = e["gt"]
            if gt is None or not e["P"]:
                break
            hits = [nm for nm in e["P"] if e["P"][nm]["tok"] == gt]
            allhit = len(hits) == len(e["P"]); nohit = not hits; d = e["depth"]
            if o_alive and not nohit and d < maxd:
                reach[d] += 1
                if not allhit:
                    dec[d] += 1; sel_n[d] += 1
                    nm = pickfn((k[0], k[1], d), e)
                    if nm is not None and e["P"][nm]["tok"] == gt:
                        sel_ok[d] += 1
            if o_alive:
                if nohit: o_alive = False
                else: o_run += 1
            if m_alive:
                if nohit:
                    m_alive = False
                elif allhit:
                    m_run += 1
                else:
                    nm = pickfn((k[0], k[1], d), e)
                    if nm is not None and e["P"][nm]["tok"] == gt:
                        m_run += 1
                    else:
                        m_alive = False; fmiss = d
            if not m_alive and not o_alive:
                break
        if fmiss is not None and fmiss < maxd:
            loss[fmiss] += (o_run - m_run)
        m_runs.append(m_run); o_runs.append(o_run)
    nb = len(blocks)
    return dict(selacc=np.where(sel_n > 0, sel_ok / np.maximum(sel_n, 1), np.nan),
                dec_rate=np.where(reach > 0, dec / np.maximum(reach, 1), np.nan),
                loss_avg=loss / max(nb, 1), reach=reach,
                mat=float(np.mean(m_runs)), orc_mat=float(np.mean(o_runs)),
                d2_share=float(loss[:3].sum() / max(loss.sum(), 1e-9)))


# ----------------------------------------------------------------- per-cell
def compute(cell):
    key, model, dr, props = cell
    d = f"{ROOT}/{dr}"
    blocks = load_blocks(d, props)
    bad = loopy(d); blocks = {k: v for k, v in blocks.items() if k[0] not in bad}
    alive = decisive_positions(blocks)
    names = [p[0] for p in props]
    lev = leverage_curve(blocks)
    maxd = max((e["depth"] for pos in blocks.values() for e in pos), default=16) + 1

    pick_raw = lambda did, e: (max(e["P"], key=lambda nm: e["P"][nm]["prob"]) if e["P"] else None)
    pk = {sch: gbm_picks(alive, props, sch, lev, maxd) for sch in SCHEMES}
    pickfns = {"raw": pick_raw}
    for sch in SCHEMES:
        pickfns[sch] = (lambda did, e, _p=pk[sch]: _p.get(did))

    bars = []
    for nm in names:
        sa = float(np.mean([1.0 if (nm in e["P"] and e["P"][nm]["tok"] == e["gt"]) else 0.0
                            for _, e in alive if nm in e["P"]])) if alive else float("nan")
        mat = run_length(blocks, lambda did, e, _n=nm: _n if _n in e["P"] else None)
        bars.append((f"{nm} only", sa, mat, "single"))
    bars.append(("raw\n(prob)", selacc(alive, {did: pick_raw(did, e) for did, e in alive}),
                 run_length(blocks, pick_raw), "raw"))
    for sch in SCHEMES:
        bars.append((SCHEME_LABEL[sch], selacc(alive, pk[sch]),
                     run_length(blocks, pickfns[sch]), sch))
    orc_mat = run_length(blocks, lambda did, e: next((nm for nm in e["P"]
                                                      if e["P"][nm]["tok"] == e["gt"]), None))
    bars.append(("oracle", 1.0, orc_mat, "oracle"))

    pd = {m: perdepth_method(blocks, fn) for m, fn in pickfns.items()}
    return dict(key=key, model=model, nblocks=len(blocks), ndecisive=len(alive),
                lev=lev, bars=bars, pd=pd)


# ----------------------------------------------------------------- figures
def fig_ladder(data, metric_idx, ylabel, fname, fmt):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    for ax, dat in zip(axes, data):
        bars = dat["bars"]
        labs = [b[0] for b in bars]; vals = [b[metric_idx] for b in bars]; ks = [b[3] for b in bars]
        xp = np.arange(len(bars)); vmax = max(vals)
        ax.bar(xp, vals, color=[COL[k] for k in ks], edgecolor="white", width=0.74)
        for x, v in zip(xp, vals):
            ax.text(x, v + vmax * 0.012, fmt(v), va="bottom", ha="center",
                    fontsize=8.3, fontfamily="monospace")
        ax.set_xticks(xp); ax.set_xticklabels(labs, rotation=24, ha="right", fontsize=8.0)
        ax.set_ylim(0, vmax * 1.16); ax.set_ylabel(ylabel)
        ax.set_title(dat["model"], fontsize=12, fontweight="bold", loc="left")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
        ax.text(0.0, -0.30, f"{dat['ndecisive']} decisive · {dat['nblocks']} blocks",
                transform=ax.transAxes, fontsize=7.2, color="#5e6e78")
    fig.suptitle(f"2-way select-1 ladder — {ylabel}   (calib & shallow-wt = gbm(prob+depth); "
                 "shallow-wt upweights shallow depths · offline, block-anchored, OOF)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{OUT}/{fname}"; fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def fig_perdepth(data, method, mlabel, barcol, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    xs = np.arange(DMAX)
    for ax, dat in zip(axes, data):
        pd = dat["pd"][method]
        valid = pd["reach"][:DMAX] >= 30
        x = xs[valid]
        ax.bar(x, pd["loss_avg"][:DMAX][valid], width=0.66, color=barcol, alpha=0.5,
               edgecolor="white", label="MAT loss (→oracle), tokens/block", zorder=2)
        ax.set_xlabel("composed-chain depth d")
        ax.set_ylabel("MAT loss  (tokens forfeited / block)")
        ax.set_ylim(0, max(pd["loss_avg"][:DMAX].max() * 1.18, 1e-3))
        ax.set_title(dat["model"], fontsize=12, fontweight="bold", loc="left")
        ax.set_xticks(x); ax.grid(axis="y", alpha=0.2); ax.spines[["top"]].set_visible(False)
        ax2 = ax.twinx()
        ax2.plot(x, pd["selacc"][:DMAX][valid], "-o", color="#16324f", lw=2.2, ms=4.5,
                 label="selection accuracy", zorder=4)
        ax2.plot(x, pd["dec_rate"][:DMAX][valid], "--s", color="#8a8a8a", lw=1.8, ms=3.5,
                 label="decisive rate", zorder=3)
        ax2.set_ylabel("selection accuracy / decisive rate"); ax2.set_ylim(0, 1.02)
        ax2.spines[["top"]].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8.2, loc="upper right", framealpha=0.92)
        ax.text(0.0, -0.16, f"MAT {pd['mat']:.3f} → oracle {pd['orc_mat']:.3f}  ·  "
                f"d≤2 = {pd['d2_share']*100:.0f}% of total MAT loss",
                transform=ax.transAxes, fontsize=7.6, color="#5e6e78")
    fig.suptitle(f"Per-depth diagnostics — {mlabel}   "
                 "(offline, block-anchored; bars = MAT loss, lines = selacc & decisive rate)",
                 fontsize=11.8, y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = f"{OUT}/{fname}"; fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def main():
    # drop the superseded depthwise-calib per-depth image
    p = Path(OUT) / "perdepth_depthwise.png"
    if p.exists(): p.unlink()
    data = [compute(c) for c in CELLS]
    for dat in data:
        print(f"\n### {dat['model']}  blocks={dat['nblocks']} decisive={dat['ndecisive']}  "
              f"lev(d0..3)={' '.join(f'{x:.2f}' for x in dat['lev'][:4])}")
        for lab, sa, mat, _k in dat["bars"]:
            print(f"   {lab.replace(chr(10),' '):24s} selacc={sa:.4f}  MAT={mat:.3f}")
    fig_ladder(data, 2, "MAT", "mat_ladder.png", lambda v: f"{v:.2f}")
    fig_ladder(data, 1, "Selection accuracy (decisive)", "selacc_ladder.png", lambda v: f"{v:.3f}")
    for method, mlabel, barcol, fname in PD_METHODS:
        fig_perdepth(data, method, mlabel, barcol, fname)


if __name__ == "__main__":
    main()
