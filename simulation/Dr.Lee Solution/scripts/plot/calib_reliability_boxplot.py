#!/usr/bin/env python3
"""Reliability boxplots, per (workload x function x proposer). Each proposer is
plotted on the axes its ACTUAL calibrator uses:

  dflash head:  x = raw prob (model conf, [0,1]) -> y = conditional accept RATE
                P(match | conf).  fits: raw / logistic / beta / isotonic / affine
                (affine = Dr.Lee 0.69*raw+0.29, dflash-only)
  suffix tail:  x = arctic path SCORE            -> y = realized ACCEPT TOKENS
                (count; budget-aware, accept-conditioned -- exactly the pairs the
                controller's cal_t isotonic is fit on). fits: raw (identity, the
                3-6x over-estimate) / linear / isotonic (the deployed tail).

x binned; box = per-request distribution in the bin; dark dots = pooled empirical
mean; RED line = the fitted function. dflash boxes/dots BLUE, suffix ORANGE.

  # extract (container):  --extract [--wl NAME]
  # render (from cache):  --render
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse, gzip, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")

REC = {"specbench": "results/perpos_specbench_alleval/specbench_4way.jsonl",
       "bfcl": "results/perpos_bfcl_alleval/bfcl_4way.jsonl",
       "swebench": "results/perpos_swebench_alleval/swebench_4way.jsonl",
       "spider": "results/perpos_spider_alleval/spider_4way.jsonl",
       "tau2": "results/perpos_tau2_alleval/tau2_4way.jsonl"}
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench Verified",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
CACHE = Path("/workspace/simulation/results/pipeline_4way/calib_pairs")
OUTDIR = Path("readable_outputs/figures/mat/calibration/calib_curves")
MINN = 8
BLUE, ORANGE, RED = "#4C78A8", "#F58518", "#D62728"
DARK = {"dflash": "#2f4d6e", "suffix": "#9c5410"}
COL = {"dflash": BLUE, "suffix": ORANGE}
DFLASH_FUNCS = ["raw", "linear", "logistic", "beta", "isotonic"]
SUFFIX_FUNCS = ["raw", "linear", "isotonic"]


# ---------- extraction ----------
def extract(wl):
    import numpy as np
    from replay_extension import _ad, ArcticSuffix, adaptive_nhead, greedy_tree_walk_path
    from fusion_tree import build_extension_chain
    rec = REC[wl]
    traces = json.load(open(Path(rec).with_suffix(".traces.json")))
    warm = traces["warm_traces"]
    ev = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)
    recs = defaultdict(dict)
    for l in open(rec):
        l = l.strip()
        if l:
            r = json.loads(l); recs[r["rid"]][r["pos"]] = r

    # dflash head: (conf, match) accept-conditioned, grouped by rid
    dfl = {}
    for rid, rby in recs.items():
        pp = []
        for r in rby.values():
            conf, match = r["dflash_conf"], r["dflash_match"]
            ad = _ad(match)
            for d in range(min(ad + 1, len(conf))):
                pp.append((float(conf[d]), int(match[d])))
        if pp:
            dfl[rid] = pp

    # suffix tail: (arctic score, realized accept count) accept-conditioned,
    # budget-aware, along the deployed chain policy -- exactly _fit_tail_iso's pairs
    suffix = ArcticSuffix(); suffix.fit(warm)
    suf = {}
    for rid, rby in recs.items():
        t = ev.get(rid)
        if t is None:
            continue
        gt, pids = t["output_ids"], t["prompt_ids"]
        suffix.new_eval(pids)
        pp = []
        m = 0
        for _ in range(4096):
            r = rby.get(m + 1)
            if r is None or m >= len(gt):
                break
            block_full, cf, mt = r["dflash_tok"], r["dflash_conf"], r["dflash_match"]
            root = gt[m]
            ctx = pids + gt[:m] + [root]
            for kk in range(min(_ad(mt), r["W"]) + 1):
                budget = num_spec - kk
                if budget <= 0:
                    break
                tk, sc = suffix._spec(ctx + block_full[:kk], budget)
                r_acc = 0
                for tok, g in zip(tk[:budget], gt[m + 1 + kk:]):
                    if tok != g:
                        break
                    r_acc += 1
                pp.append((float(sc), float(r_acc)))
            _, T = suffix.probe(ctx, num_spec)
            k = adaptive_nhead(cf, T=T, num_spec=r["W"])
            tk = suffix.speculate(ctx + block_full[:k], num_spec)
            trd = build_extension_chain(block_full[:k], tk[:max(0, num_spec - k)])
            pth = greedy_tree_walk_path(list(trd.tokens), list(trd.parents), gt[m + 1:])
            acc = [trd.tokens[i] for i in pth]
            nxt = [root] + acc + ([gt[m + 1 + len(pth)]] if m + 1 + len(pth) < len(gt) else [])
            suffix.add_response(nxt)
            m += 1 + len(pth) + 1
        if pp:
            suf[rid] = pp
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, dat in (("dflash", dfl), ("suffix", suf)):
        with gzip.open(CACHE / f"{wl}_{name}.json.gz", "wt") as f:
            json.dump({str(k): v for k, v in dat.items()}, f)
        print(f"[{wl}] {name}: {len(dat)} rids, {sum(len(v) for v in dat.values())} pairs",
              flush=True)


# ---------- fits ----------
def dflash_fit(name, xs, ys):
    if name == "raw":
        return lambda x: x
    if name == "linear":                       # FITTED affine a*conf+b (not fixed)
        import numpy as np
        a, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
        return lambda x, _a=a, _b=b: min(1.0, max(0.0, _a * x + _b))
    if name == "logistic":
        from replay_extension import _fit_logistic
        return _fit_logistic(xs, ys)
    if name == "beta":
        from replay_extension import _fit_beta
        return _fit_beta(xs, ys)
    if name == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        import numpy as np
        ir = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(xs, float), np.asarray(ys, float))
        return lambda x, _i=ir: float(_i.predict([x])[0])
    raise ValueError(name)


def suffix_fit(name, xs, ys):
    import numpy as np
    if name == "raw":
        return lambda x: x                         # identity: score AS the estimate
    if name == "linear":
        a, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
        return lambda x, _a=a, _b=b: float(_a * x + _b)
    if name == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(xs, float), np.asarray(ys, float))
        return lambda x, _i=ir: float(_i.predict([max(0.0, x)])[0])
    raise ValueError(name)


# ---------- render ----------
def render():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for wl in REC:
        for proposer in ("dflash", "suffix"):
            fp = CACHE / f"{wl}_{proposer}.json.gz"
            if not fp.exists():
                print(f"[skip] no cache {fp}"); continue
            with gzip.open(fp, "rt") as f:
                data = json.load(f)
            xs = [p[0] for v in data.values() for p in v]
            ys = [p[1] for v in data.values() for p in v]

            if proposer == "dflash":
                edges = [i / 10 for i in range(11)]
                xlab = "raw prob  (DFlash conf)"
                ylab = "conditional accept rate  P(match | raw prob)"
                funcs, fitter = DFLASH_FUNCS, dflash_fit
                xmax, ymax = 1.0, 1.0
            else:
                hi = float(np.percentile(xs, 99)) if xs else 32.0
                hi = max(4.0, hi)
                nb = 12
                edges = [hi * i / nb for i in range(nb + 1)]
                xlab = "arctic path score"
                ylab = "realized accept tokens  E[accept | score]"
                funcs, fitter = SUFFIX_FUNCS, suffix_fit
                xmax = hi
                ymax = max(ys) * 1.05 if ys else 32.0
            nb = len(edges) - 1
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(nb)]
            width = (edges[1] - edges[0]) * 0.6

            def binof(x):
                for i in range(nb):
                    if x < edges[i + 1] or i == nb - 1:
                        return i
                return nb - 1
            perbin = [[] for _ in range(nb)]
            pool_s = [0.0] * nb; pool_n = [0] * nb
            for v in data.values():
                cell = defaultdict(lambda: [0.0, 0])
                for xx, yy in v:
                    b = binof(xx)
                    cell[b][0] += yy; cell[b][1] += 1
                    pool_s[b] += yy; pool_n[b] += 1
                for b, (s, n) in cell.items():
                    if n >= MINN:
                        perbin[b].append(s / n)
            pooled = [pool_s[b] / pool_n[b] if pool_n[b] else np.nan for b in range(nb)]

            pcol, dark = COL[proposer], DARK[proposer]
            for fn_name in funcs:
                fn = fitter(fn_name, xs, ys)
                fig, ax = plt.subplots(figsize=(7.2, 5.4))
                ax.boxplot([b if b else [np.nan] for b in perbin], positions=centers,
                           widths=width, showfliers=False,
                           medianprops=dict(color=dark),
                           boxprops=dict(color=pcol), whiskerprops=dict(color=pcol),
                           capprops=dict(color=pcol))
                ax.scatter(centers, pooled, s=34, c=dark, zorder=5,
                           label="pooled empirical mean")
                gx = [xmax * i / 200 for i in range(201)]
                gy = [max(0.0, fn(x)) for x in gx]
                ax.plot(gx, gy, color=RED, lw=2.4, zorder=6, label=f"fitted: {fn_name}")
                if proposer == "dflash":
                    ax.plot([0, 1], [0, 1], color="#ccc", lw=1.0, ls="--", zorder=1)
                else:
                    ax.plot([0, min(xmax, ymax)], [0, min(xmax, ymax)],
                            color="#ccc", lw=1.0, ls="--", zorder=1)  # identity ref
                ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
                ax.set_xticks(edges)
                ax.set_xticklabels([f"{e:.1f}" if proposer == "dflash" else f"{e:.0f}"
                                    for e in edges])
                ax.set_xlabel(xlab, fontsize=10.5)
                ax.set_ylabel(ylab, fontsize=10.5)
                pl = "DFlash head" if proposer == "dflash" else "Suffix tail"
                sub = ("raw prob → conditional accept rate" if proposer == "dflash"
                       else "arctic score → realized accept tokens")
                ax.set_title(f"{WL_NAME[wl]} — {pl} — fit: {fn_name}\n"
                             f"{sub}  (boxes = per-request spread)",
                             fontsize=11, fontweight="bold")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=9, loc="upper left")
                out = OUTDIR / f"reliability_{wl}_{proposer}_{fn_name}.png"
                fig.savefig(out, dpi=140, bbox_inches="tight")
                plt.close(fig)
            print(f"[{wl}/{proposer}] rendered {len(funcs)} funcs", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--wl", default="")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.extract:
        for wl in ([args.wl] if args.wl else list(REC)):
            extract(wl)
    if args.render:
        render()
