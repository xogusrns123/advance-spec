#!/usr/bin/env python3
"""RAW-PROB compose calibration diagnostic (motivation for correction).

Replays the compose selector on the RAW signals (cal_h = raw DFlash conf,
cal_t = raw arctic suffix score — the identity `--raw-cals` arm), and at the
CHOSEN handoff k_hat of every round records the estimated vs. realized value
of each term in the composition objective

    a_hat_k = 1 + G_hat_k + S_hat_k * T_hat_k                (replay_extension.py)

  * G_hat_k = sum_{i<k} prod_{j<i} conf[j]   expected accepted DFlash tokens
  * S_hat_k = prod_{j<k} conf[j]             P(head survives all k tokens)
  * T_hat_k = raw arctic score at handoff k  expected suffix tail accept
  * a_hat_k                                  expected total accept length

and the realized twins along the teacher-forced (greedy) trajectory:

  * G   = leading match run of the head block_full[:k]  (accepted DFlash tokens)
  * S   = 1{head fully survived to k}                   (realized survival)
  * T   = acc - G  when S==1 (tail reached), else unobserved
  * a   = 1 + acc   (the +1 is the bonus token, not the root) (realized total)

Everything is CPU replay off an existing capture_perpos record — no GPU, no
model forward. Dumps the per-round pairs to json and renders the 4-panel
figure (plot 1 = stacked bars a_hat vs a; plots 2/3/4 = est-vs-real 2D
histograms with a red y=x target). DFlash terms blue, Suffix terms orange.

  python3 scripts/analysis/diag_compose_calibration.py \
      --record results/perpos_specbench_full/specbench_4way.jsonl --tag specbench
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # .../scripts
sys.path.insert(0, "/workspace")
from fusion_tree import build_extension_chain        # noqa: E402
from measure_k_fusion import ArcticSuffix            # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402


def _ad(match):
    i = 0
    while i < len(match) and match[i] == 1:
        i += 1
    return i


def collect_head(record):
    """Per-token DFlash head calibration pairs (conf, match), ACCEPT-CONDITIONED:
    for each captured block include depths 0..a_d (leading match run + the first
    miss) — the positions actually verified along a matching prefix, where the
    DFlash token is generated from the correct context (beyond the first miss the
    chain is off-trajectory). Straight from the record jsonl (no arctic/replay)."""
    pairs = []
    for l in open(record):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        conf, match = r["dflash_conf"], r["dflash_match"]
        for d in range(min(_ad(match) + 1, len(conf))):
            pairs.append((float(conf[d]), int(match[d])))
    print(f"collected {len(pairs)} accept-conditioned head (conf, match) pairs")
    return pairs


def collect(record, max_rounds, head_w=1.0, head_clip=None, tail_w=1.0,
            tail_w_by_cat=None, three_way=False, group_mode="convlabel",
            handoff=False, ho_tw_by_cat=None):
    """Replay the compose over every eval trace; return per-round dicts recorded
    at the chosen k_hat. head_w/head_clip/tail_w apply the deployed weights to
    BOTH the selection objective and the recorded estimates:
      cal_h(c) = min(head_clip, head_w*c) ;  cal_t(s) = tail_w*s
    Defaults (1.0, None, 1.0) = the raw-prob compose (plots 1-5); the two-scalar
    weights (1.4, 1.0, 0.125) give the weighted compose (plots 10-14).
    tail_w_by_cat: {task -> tail_w} overrides tail_w per trace category (head raw
    + per-category-optimal tail weight). three_way: eval only on the test half
    (convlabel split, matching the method-family sweep)."""
    def cal_h(c):
        v = head_w * float(c)
        return min(head_clip, v) if head_clip is not None else v

    traces = json.load(open(Path(record).with_suffix(".traces.json")))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)

    recs = defaultdict(dict)
    for l in open(record):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r

    test_rids = None
    if three_way:
        from replay_extension import split_parity
        _, split_par = split_parity(eval_traces, group_mode)
        test_rids = {rid for rid in recs if split_par.get(rid, 0) == 1}
        print(f"three-way {group_mode}: eval on {len(test_rids)}/{len(recs)} test convs")

    suffix = ArcticSuffix()
    suffix.fit(warm_traces)                           # global warm tree
    rows = []
    KMAX = num_spec
    agg = {kk: [0.0] * (KMAX + 1) for kk in
           ("cnt", "rhead", "rtail", "ehead", "etot_raw", "etot_w")} if handoff else None
    n_tr = 0
    for rid in sorted(recs):
        tr = eval_traces.get(rid)
        if tr is None or (test_rids is not None and rid not in test_rids):
            continue
        n_tr += 1
        task = tr.get("task", "")
        tw = (tail_w_by_cat.get(task, tail_w_by_cat.get("__all__", tail_w))
              if tail_w_by_cat else tail_w)      # SELECTION tail weight (deployed arm)
        hw = (ho_tw_by_cat.get(task, ho_tw_by_cat.get("__all__", tw))
              if ho_tw_by_cat else tw)           # handoff "tail-weight expectation" line weight
        def cal_t(s, _w=tw):
            return _w * float(s)
        gt, pids, rby = tr["output_ids"], tr["prompt_ids"], recs[rid]
        suffix.new_eval(pids)
        m = 0
        for _ in range(max_rounds):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            block_full = rec["dflash_tok"]
            conf = rec["dflash_conf"]
            root = gt[m]
            ctx_list = pids + gt[:m] + [root]
            W_ = min(num_spec, len(conf))

            # per-candidate budgeted suffix tail score at every handoff kk
            tails = []
            for kk in range(W_ + 1):
                budget = num_spec - kk
                tails.append(suffix._spec(ctx_list + block_full[:kk], budget)
                             if budget > 0 else ([], 0.0))

            # compose argmax_k [1 + G_k + S_k * T_k] on the (weighted) signals
            S_k, G_k = 1.0, 0.0
            Gk_list, Sk_list = [0.0] * (W_ + 1), [1.0] * (W_ + 1)   # per-k head expectations
            best_val, k = 1.0 + cal_t(tails[0][1]), 0
            for j in range(W_):
                S_k *= cal_h(conf[j]); G_k += S_k
                Gk_list[j + 1], Sk_list[j + 1] = G_k, S_k
                val = 1.0 + G_k + S_k * cal_t(tails[j + 1][1])
                if val > best_val:
                    best_val, k = val, j + 1

            # estimated components at the chosen k (same weighting as selection)
            Shat = 1.0
            for j in range(k):
                Shat *= cal_h(conf[j])
            Ghat = 0.0
            acc_s = 1.0
            for j in range(k):
                acc_s *= cal_h(conf[j]); Ghat += acc_s
            That = cal_t(tails[k][1])
            ahat = 1.0 + Ghat + Shat * That

            # realized components (teacher-forced greedy against gt)
            tail_toks = tails[k][0][:max(0, num_spec - k)]
            tree = build_extension_chain(block_full[:k], tail_toks)
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            acc = len(path)
            G_real = 0
            for j in range(k):
                if m + 1 + j < len(gt) and block_full[j] == gt[m + 1 + j]:
                    G_real += 1
                else:
                    break
            S_real = 1 if G_real == k else 0          # k==0 -> survives vacuously
            T_real = (acc - G_real) if S_real == 1 else None
            a_real = 1 + acc          # +1 = bonus token (free target-verified token/round)

            # succession-smoothed suffix score at the chosen handoff (smoothing plot 19):
            # same tree, edge prob (k+1)/(n+2) instead of ML k/n. Selection unchanged.
            budget_k = num_spec - k
            if budget_k > 0:
                suffix.score_mode = "succ"
                _, succ = suffix._spec(ctx_list + block_full[:k], budget_k)
                suffix.score_mode = "raw"
            else:
                succ = 0.0

            rows.append(dict(k=k, ahat=ahat, a=a_real, Ghat=Ghat, G=G_real,
                             Shat=Shat, S=S_real, That=That, T=T_real, succ=float(succ),
                             task=task))

            # per-handoff-position scan (for the handoff plots): at THIS round's tree
            # state, what each candidate handoff k would estimate vs. realize.
            if handoff:
                R = 0                                     # leading head match run
                for j in range(W_):
                    if m + 1 + j < len(gt) and block_full[j] == gt[m + 1 + j]:
                        R += 1
                    else:
                        break
                for kk in range(W_ + 1):
                    Tk = float(tails[kk][1])
                    agg["cnt"][kk] += 1
                    agg["ehead"][kk] += Gk_list[kk]
                    agg["etot_raw"][kk] += 1.0 + Gk_list[kk] + Sk_list[kk] * Tk
                    agg["etot_w"][kk] += 1.0 + Gk_list[kk] + Sk_list[kk] * (hw * Tk)
                    agg["rhead"][kk] += min(kk, R)
                    rtail = 0
                    if kk <= R:                           # head reached the handoff
                        for t, g in zip(tails[kk][0], gt[m + 1 + kk:]):
                            if t == g:
                                rtail += 1
                            else:
                                break
                    agg["rtail"][kk] += rtail

            # advance exactly like the deployed compose replay (tree grows identically)
            accepted_toks = [tree.tokens[i] for i in path]
            nxt = [root] + accepted_toks
            bonus_idx = m + 1 + acc
            if bonus_idx < len(gt):
                nxt.append(gt[bonus_idx])
            suffix.add_response(nxt)
            m += 1 + acc + 1
            if m >= len(gt):
                break
    print(f"collected {len(rows)} rounds over {n_tr} eval traces (num_spec={num_spec})")
    return (rows, agg) if handoff else rows


# --------------------------------------------------------------------------- plots
RED = "#e53e3e"
BLUE = "#2b6cb0"       # DFlash terms
ORANGE = "#ff922b"     # Suffix terms (bright primary orange, softened)
GREY = "#4a5568"       # total (a)


def _int_group(xint, series, cap_pct=None, min_count=20):
    """Group rounds by the (integer) REALIZED value xint; per populated value v
    return v and the mean of each y-series over rounds with realized==v. Axes are
    swapped vs. a reliability plot: x = realized, bar height = mean estimate."""
    import numpy as np
    xint = np.asarray(xint).astype(int)
    if len(xint) == 0:
        return np.array([]), [np.array([]) for _ in series]
    hi = int(np.percentile(xint, cap_pct)) if cap_pct is not None else int(xint.max())
    xs, means = [], [[] for _ in series]
    for v in range(0, hi + 1):
        sel = xint == v
        if sel.sum() >= min_count:
            xs.append(v)
            for i, s in enumerate(series):
                means[i].append(float(np.asarray(s, float)[sel].mean()))
    return np.asarray(xs), [np.asarray(m) for m in means]


def _finish(ax, fig, hi, xlabel, ylabel, title, outpath, fit_w=None, fit_clip=None,
            curve=None, curve_label="fitted calibrator"):
    import numpy as np
    if curve is not None:
        # raw reliability + the fitted calibration FUNCTION as a red curve
        ax.plot([0, hi], [0, hi], color="#9aa5b1", lw=1.6, ls="--", label="y = x (ideal)")
        xx = np.linspace(0, hi, 300)
        yy = np.array([curve(float(v)) for v in xx], float)
        ax.plot(xx, yy, color=RED, lw=2.2, label=curve_label)
    elif fit_w is None:
        ax.plot([0, hi], [0, hi], color=RED, lw=2, label="y = x (target)")
    else:
        # y=x demoted to a grey dashed ideal so the red weighted line reads clearly
        ax.plot([0, hi], [0, hi], color="#9aa5b1", lw=1.6, ls="--", label="y = x (ideal)")
        xx = np.linspace(0, hi, 200)
        yy = fit_w * xx
        if fit_clip is not None:
            yy = np.minimum(yy, fit_clip)
            lbl = rf"weight  $y=\min({fit_clip:g},\,{fit_w:g}\,x)$"
        else:
            lbl = rf"weight  $y={fit_w:g}\,x$"
        ax.plot(xx, yy, color=RED, lw=2.2, label=lbl)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150)
    print("wrote", outpath)


def plot_one(x_real, y_est, color, title, xlabel, ylabel, outpath, cap_pct=None):
    """Single-series swapped bar: x = realized (int), bar = mean estimate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, (ym,) = _int_group(x_real, [y_est], cap_pct=cap_pct)
    hi = (max(xs.max() if len(xs) else 1, ym.max() if len(ym) else 1)) * 1.05
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.bar(xs, ym, width=0.85, color=color, edgecolor="white", linewidth=0.4,
           label="estimate mean per bin")
    _finish(ax, fig, hi, xlabel, ylabel, title, outpath)
    plt.close(fig)


def plot_stacked(a_real, ones, Ghat, ShatThat, title, xlabel, ylabel, outpath, cap_pct=99,
                 head_label=r"DFlash head  $\hat{G}$", tail_label=r"Suffix tail  $\hat{S}\hat{T}$"):
    """Plot 1: x = realized a (int); estimated a_hat stacked into base(1) +
    head (Ghat) + tail (Shat*That), each the per-bin mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, (b, g, t) = _int_group(a_real, [ones, Ghat, ShatThat], cap_pct=cap_pct)
    tot = b + g + t
    hi = (max(xs.max() if len(xs) else 1, tot.max() if len(tot) else 1)) * 1.05
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.bar(xs, b, width=0.85, color=GREY, edgecolor="white", linewidth=0.4,
           label="bonus token = 1")
    ax.bar(xs, g, width=0.85, bottom=b, color=BLUE, edgecolor="white", linewidth=0.4,
           label=head_label)
    ax.bar(xs, t, width=0.85, bottom=b + g, color=ORANGE, edgecolor="white", linewidth=0.4,
           label=tail_label)
    _finish(ax, fig, hi, xlabel, ylabel, title, outpath)
    plt.close(fig)


def plot_est_binned(x_est, y_real, color, title, xlabel, ylabel, outpath,
                    nb=20, cap_pct=99, min_count=50, fit_w=0.125, fit_clip=None,
                    apply_weight=False, xform=None, curve=None,
                    curve_label="fitted calibrator"):
    """Reliability in the estimate->outcome direction: x = ESTIMATE binned, bar
    height = mean REALIZED in that bin. (Twin of plot_head, general continuous x;
    the estimate-binned complement of the realized-binned plot_one.)
    apply_weight: bin the WEIGHTED estimate (fit_w*x). xform: transform x before
    binning (calibrated estimate). curve: overlay this fn as a red line on the raw x."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(x_est, float); y = np.asarray(y_real, float)
    if apply_weight:
        x = fit_w * x
        if fit_clip is not None:
            x = np.minimum(x, fit_clip)
    if xform is not None:
        x = np.array([xform(float(v)) for v in x], float)
    hi_x = float(np.percentile(x, cap_pct)) if cap_pct else float(x.max())
    keep = x <= hi_x
    x, y = x[keep], y[keep]
    edges = np.linspace(0.0, hi_x, nb + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    cx, cy = [], []
    for b in range(nb):
        sel = idx == b
        if sel.sum() >= min_count:
            cx.append(0.5 * (edges[b] + edges[b + 1]))
            cy.append(float(y[sel].mean()))
    cx, cy = np.asarray(cx), np.asarray(cy)
    bw = (edges[1] - edges[0]) * 0.9
    hi = max(hi_x, float(cy.max()) if len(cy) else 0.0) * 1.05
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.bar(cx, cy, width=bw, color=color, edgecolor="white", linewidth=0.4,
           label="realized mean per bin")
    if curve is not None:                       # raw bars + fitted-calibrator curve
        _finish(ax, fig, hi, xlabel, ylabel, title, outpath,
                curve=curve, curve_label=curve_label)
    elif apply_weight or xform is not None:     # calibrated/weighted x -> y=x target
        _finish(ax, fig, hi, xlabel, ylabel, title, outpath)
    else:                                       # raw x + weight line
        _finish(ax, fig, hi, xlabel, ylabel, title, outpath, fit_w=fit_w, fit_clip=fit_clip)
    plt.close(fig)


def plot_head(pairs, color, title, xlabel, ylabel, outpath, nb=20, min_count=50,
              fit_w=1.4, fit_clip=1.0, apply_weight=False, xform=None, curve=None,
              curve_label="fitted calibrator"):
    """Per-token reliability: x = p_DFlash (conf) binned in [0,1], bar height =
    accept rate (mean match) in that bin. y=x = perfectly calibrated head.
    apply_weight: bin the WEIGHTED confidence min(fit_clip, fit_w*conf) and
    reference y=x (the after-correction view) instead of drawing the weight line."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = np.array([p[0] for p in pairs], float)
    match = np.array([p[1] for p in pairs], float)
    if apply_weight:
        conf = fit_w * conf
        if fit_clip is not None:
            conf = np.minimum(conf, fit_clip)
    if xform is not None:
        conf = np.array([xform(float(c)) for c in conf], float)
    edges = np.linspace(0.0, 1.0, nb + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, nb - 1)
    cx, cy = [], []
    for b in range(nb):
        sel = idx == b
        if sel.sum() >= min_count:
            cx.append(0.5 * (edges[b] + edges[b + 1]))
            cy.append(float(match[sel].mean()))
    bw = (edges[1] - edges[0]) * 0.9
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.bar(cx, cy, width=bw, color=color, edgecolor="white", linewidth=0.4,
           label="accept rate per bin")
    if curve is not None:                       # raw bars + fitted-calibrator curve
        _finish(ax, fig, 1.02, xlabel, ylabel, title, outpath,
                curve=curve, curve_label=curve_label)
    elif apply_weight or xform is not None:     # calibrated/weighted x -> y=x target
        _finish(ax, fig, 1.02, xlabel, ylabel, title, outpath)
    else:                                       # raw x + weight line
        _finish(ax, fig, 1.02, xlabel, ylabel, title, outpath, fit_w=fit_w, fit_clip=fit_clip)
    plt.close(fig)


def plot_terms(rows, tag, outdir, base=1, wsub=False):
    """The 5 compose-term est-vs-real plots. base=1 (raw compose, plots 1-5) or
    base=10 (weighted compose, plots 10-14; wsub adds the ',w' subscript)."""
    import numpy as np

    kk = np.array([r["k"] for r in rows])
    a = np.array([r["a"] for r in rows], float)
    Ghat_all = np.array([r["Ghat"] for r in rows], float)
    ShatThat_all = np.array([r["Shat"] * r["That"] for r in rows], float)
    ST_all = np.array([(r["S"] * r["T"]) if r["T"] is not None else 0
                       for r in rows], float)   # realized tail gain (0 if head died)
    ones = np.ones(len(rows))

    hd = kk >= 1                                       # DFlash head used
    Ghat = Ghat_all[hd]
    G = np.array([r["G"] for r in rows], float)[hd]
    Shat = np.array([r["Shat"] for r in rows])[hd]
    S = np.array([r["S"] for r in rows], float)[hd]
    tobs = np.array([r["S"] == 1 for r in rows])       # tail reached
    That = np.array([r["That"] for r in rows])[tobs]
    T = np.array([r["T"] if r["T"] is not None else 0 for r in rows], float)[tobs]

    def hh(X):
        return rf"\hat{{{X}}}_w" if wsub else rf"\hat{{{X}}}"
    A, Gh, Sh, Th = hh("a"), hh("G"), hh("S"), hh("T")
    ST = f"{Sh}{Th}"
    f = "w" if wsub else ""
    tscore = "weighted arctic score" if wsub else "raw arctic score"
    o = lambda n: os.path.join(outdir, f"{tag}_{n}.png")  # noqa: E731

    plot_stacked(a, ones, Ghat_all, ShatThat_all,
                 rf"({base}) ${A}=1+{Gh}+{ST}$  vs  $a$  (total accept length)",
                 r"$a$  realized accept length", rf"${A}$  estimated accept length (stacked)",
                 o(f"{base}_ahat{f}_vs_a"),
                 head_label=rf"DFlash head  ${Gh}$", tail_label=rf"Suffix tail  ${ST}$")
    plot_one(G, Ghat, BLUE,
             rf"({base + 1}) ${Gh}$ vs $G$  (accepted DFlash tokens)",
             r"$G$  realized accepted DFlash", rf"${Gh}$  estimated accepted DFlash",
             o(f"{base + 1}_Ghat{f}_vs_G"))
    plot_one(S, Shat, BLUE,
             rf"({base + 2}) ${Sh}$ vs $S$  (head survival)",
             r"$S$  realized survival (0/1)", rf"${Sh}$  estimated survival prob",
             o(f"{base + 2}_Shat{f}_vs_S"))
    plot_one(T, That, ORANGE,
             rf"({base + 3}) ${Th}$ vs $T$  (suffix tail)",
             r"$T$  realized accepted suffix", rf"${Th}$  {tscore}",
             o(f"{base + 3}_That{f}_vs_T"), cap_pct=99)
    plot_one(ST_all, ShatThat_all, ORANGE,
             rf"({base + 4}) ${ST}$ vs $S\!\cdot\!T$  (tail gain)",
             r"$S\!\cdot\!T$  realized tail gain", rf"${ST}$  estimated tail gain",
             o(f"{base + 4}_Shat{f}That{f}_vs_ST"), cap_pct=99)


def plot_percat_tail(rows, tag, outdir, arm="percat"):
    """The 3 term plots for the deployed arm: ahat vs a (stacked), Ghat vs G,
    ShatThat vs ST. arm='percat' = raw head + per-category tail weight; arm='raw'
    = raw-prob compose (raw head + raw tail)."""
    import numpy as np

    a = np.array([r["a"] for r in rows], float)
    Ghat_all = np.array([r["Ghat"] for r in rows], float)
    ShatThat_all = np.array([r["Shat"] * r["That"] for r in rows], float)
    ST_all = np.array([(r["S"] * r["T"]) if r["T"] is not None else 0 for r in rows], float)
    ones = np.ones(len(rows))
    kk = np.array([r["k"] for r in rows])
    hd = kk >= 1
    Ghat = Ghat_all[hd]
    G = np.array([r["G"] for r in rows], float)[hd]
    o = lambda n: os.path.join(outdir, f"{tag}_{n}.png")  # noqa: E731

    w = "" if arm == "raw" else "_w"                        # tail-score subscript
    ST = rf"\hat{{S}}\hat{{T}}{('_w' if arm != 'raw' else '')}"
    desc = "raw-prob compose" if arm == "raw" else "raw head + per-category tail weight"
    tdesc = "raw" if arm == "raw" else "per-cat"

    plot_stacked(a, ones, Ghat_all, ShatThat_all,
                 rf"$\hat{{a}}=1+\hat{{G}}+{ST}$ vs $a$  ({desc})",
                 r"$a$  realized accept length", r"$\hat{a}$  estimated accept length (stacked)",
                 o("ahat_vs_a"),
                 head_label=r"DFlash head  $\hat{G}$ (raw)",
                 tail_label=rf"Suffix tail  ${ST}$ ({tdesc})")
    plot_one(G, Ghat, BLUE,
             r"$\hat{G}$ vs $G$  (accepted DFlash tokens, raw head)",
             r"$G$  realized accepted DFlash", r"$\hat{G}$  estimated accepted DFlash",
             o("Ghat_vs_G"))
    plot_one(ST_all, ShatThat_all, ORANGE,
             rf"${ST}$ vs $S\!\cdot\!T$  (tail gain, {tdesc})",
             r"$S\!\cdot\!T$  realized tail gain", rf"${ST}$  estimated tail gain",
             o("ShatThat_vs_ST"), cap_pct=99)


def plot_handoff(agg, tag, outdir, version, exp_label):
    """x = handoff position k; bars = realized accept (head gain blue + tail gain
    orange, stacked); lines = expected head gain and expected TOTAL gain (raw or
    tail-weighted). RELATIVE y (bars / max realized, lines / max expected) so the
    argmax positions — what the compose actually selects — are comparable."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cnt = np.array(agg["cnt"], float)
    ks = np.where(cnt > 0)[0]
    mean = lambda key: np.array(agg[key], float)[ks] / cnt[ks]  # noqa: E731
    rh, rt, eh = mean("rhead"), mean("rtail"), mean("ehead")
    et = mean("etot_raw" if version == "raw" else "etot_w")
    rtot = rh + rt
    nr = rtot.max() or 1.0
    ne = et.max() or 1.0
    rh_n, rt_n, eh_n, et_n = rh / nr, rt / nr, eh / ne, et / ne

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.bar(ks, rh_n, width=0.85, color=BLUE, edgecolor="white", linewidth=0.4,
           label="realized head gain")
    ax.bar(ks, rt_n, width=0.85, bottom=rh_n, color=ORANGE, edgecolor="white",
           linewidth=0.4, label="realized tail gain")
    ax.plot(ks, eh_n, color="#12406e", lw=2.0, ls="--", marker="o", ms=3,
            label="expected head gain")
    ax.plot(ks, et_n, color=RED, lw=2.6, marker="o", ms=4,
            label=f"expected total gain  ({exp_label})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("handoff position  k  (DFlash head length)")
    ax.set_ylabel("relative gain  (bars / max realized,  lines / max expected)")
    ax.set_title(f"Handoff-position gains  [{tag}]  —  expectation: {exp_label}", fontsize=11)
    ax.legend(fontsize=8, loc="lower center", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(outdir, f"{tag}_handoff_{version}.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot(rows, tag, outdir):
    import numpy as np

    plot_terms(rows, tag, outdir, base=1, wsub=False)     # plots 1-5 (raw compose)

    tobs = np.array([r["S"] == 1 for r in rows])          # tail reached
    That = np.array([r["That"] for r in rows])[tobs]
    T = np.array([r["T"] if r["T"] is not None else 0 for r in rows], float)[tobs]
    o = lambda n: os.path.join(outdir, f"{tag}_{n}.png")  # noqa: E731
    plot_est_binned(That, T, ORANGE,
                    r"(7) $T$ vs $\hat{T}$  (suffix-tail reliability)",
                    r"$\hat{T}$  raw arctic score", r"$T$  realized accepted suffix (mean)",
                    o("7_T_vs_That"), cap_pct=99)
    plot_est_binned(That, T, ORANGE,
                    r"(9) $T$ vs $\hat{T}_w$  (weighted suffix tail)",
                    r"$\hat{T}_w = 0.125\,\hat{T}$  weighted score",
                    r"$T$  realized accepted suffix (mean)",
                    o("9_T_vs_That_w"), cap_pct=99, fit_w=0.125, apply_weight=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="")
    ap.add_argument("--tag", default="specbench")
    ap.add_argument("--max-rounds", type=int, default=4096)
    ap.add_argument("--outdir", default="readable_outputs/figures/compose_calib")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--head-w", type=float, default=1.4)     # weighted-compose head weight
    ap.add_argument("--head-clip", type=float, default=1.0)
    ap.add_argument("--tail-w", type=float, default=0.125)   # weighted-compose tail weight
    ap.add_argument("--percat-tail", default="",
                    help="json {task->tail_w}: run RAW-head + per-category tail-weight arm "
                         "(three-way test half); caches rows + handoff agg")
    ap.add_argument("--collect-only", action="store_true",
                    help="--percat-tail: cache rows/agg only, no plotting (for parallel runs)")
    ap.add_argument("--arm", default="percat", choices=["percat", "raw"],
                    help="deployed selection arm: 'percat' (raw head + per-category tail "
                         "weight) or 'raw' (raw-prob compose). The handoff 'tail-weight' "
                         "line always uses the per-category weights from --percat-tail.")
    ap.add_argument("--combine", default="",
                    help="comma-separated tags: POOL their cached rows + handoff aggregates "
                         "(average over workloads) and plot the 3 term + 2 handoff graphs "
                         "as tag 'all_<arm>'")
    args = ap.parse_args()

    # cache/output naming per arm
    PAIRS = f"pairs_{args.arm}"
    AGG = "handoff_agg" if args.arm == "percat" else f"handoff_agg_{args.arm}"
    OUTT = "all_percattail" if args.arm == "percat" else f"all_{args.arm}"

    # combine: pool per-workload caches -> average over all workloads
    if args.combine:
        tags = [t for t in args.combine.split(",") if t]
        all_rows, agg = [], None
        for t in tags:
            all_rows += json.load(open(Path(args.outdir) / f"{t}_{PAIRS}.json"))
            a = json.load(open(Path(args.outdir) / f"{t}_{AGG}.json"))
            if agg is None:
                agg = {k: list(map(float, v)) for k, v in a.items()}
            else:
                for k, v in a.items():
                    if len(v) > len(agg[k]):
                        agg[k] += [0.0] * (len(v) - len(agg[k]))
                    for i, x in enumerate(v):
                        agg[k][i] += float(x)
        print(f"combined {len(all_rows)} rounds over {len(tags)} workloads [{args.arm}]: {tags}")
        plot_percat_tail(all_rows, OUTT, args.outdir, arm=args.arm)
        plot_handoff(agg, OUTT, args.outdir, "raw", "raw prob")
        plot_handoff(agg, OUTT, args.outdir, "tailw", "per-category tail weight")
        return

    # dedicated fast path: per-category (deployed) or raw-prob compose arm
    if args.percat_tail:
        tw_by_cat = json.load(open(args.percat_tail))
        sel_map = None if args.arm == "raw" else tw_by_cat   # raw arm: raw tail selection
        print(f"arm={args.arm}  per-category tail weights (handoff line):", tw_by_cat)
        pc = Path(args.outdir) / f"{args.tag}_{PAIRS}.json"
        ac = Path(args.outdir) / f"{args.tag}_{AGG}.json"
        if pc.exists() and ac.exists() and not args.recompute:
            rows_pc = json.load(open(pc))
            agg = json.load(open(ac))
            print(f"loaded {len(rows_pc)} cached rounds from {pc}")
        else:
            rows_pc, agg = collect(args.record, args.max_rounds, head_w=1.0, head_clip=None,
                                   tail_w=1.0, tail_w_by_cat=sel_map, ho_tw_by_cat=tw_by_cat,
                                   three_way=True, group_mode="convlabel", handoff=True)
            os.makedirs(args.outdir, exist_ok=True)
            json.dump(rows_pc, open(pc, "w"))
            json.dump(agg, open(ac, "w"))
            print("cached pairs ->", pc)
        if args.collect_only:
            return
        plot_percat_tail(rows_pc, OUTT, args.outdir, arm=args.arm)
        plot_handoff(agg, OUTT, args.outdir, "raw", "raw prob")
        plot_handoff(agg, OUTT, args.outdir, "tailw", "per-category tail weight")
        return

    cache = Path(args.outdir) / f"{args.tag}_pairs.json"
    if cache.exists() and not args.recompute:
        rows = json.load(open(cache))
        print(f"loaded {len(rows)} cached rounds from {cache}")
    else:
        rows = collect(args.record, args.max_rounds)
        os.makedirs(args.outdir, exist_ok=True)
        json.dump(rows, open(cache, "w"))
        print("cached pairs ->", cache)

    plot(rows, args.tag, args.outdir)

    # plot 6: per-token DFlash head calibration (own cache, no replay needed)
    hcache = Path(args.outdir) / f"{args.tag}_headpairs.json"
    if hcache.exists() and not args.recompute:
        hpairs = json.load(open(hcache))
        print(f"loaded {len(hpairs)} cached head pairs from {hcache}")
    else:
        hpairs = collect_head(args.record)
        os.makedirs(args.outdir, exist_ok=True)
        json.dump(hpairs, open(hcache, "w"))
        print("cached head pairs ->", hcache)
    plot_head(hpairs, BLUE,
              r"(6) $p_{DFlash}$ vs accept rate  (DFlash head)",
              r"$p_{DFlash}$  DFlash confidence", r"accept rate  (realized match)",
              os.path.join(args.outdir, f"{args.tag}_6_pDFlash_vs_acceptrate.png"))
    plot_head(hpairs, BLUE,
              r"(8) $p_{DFlash,w}$ vs accept rate  (weighted DFlash head)",
              r"$p_{DFlash,w} = \min(1,\,1.4\,p_{DFlash})$", r"accept rate  (realized match)",
              os.path.join(args.outdir, f"{args.tag}_8_pDFlashw_vs_acceptrate.png"),
              fit_w=1.4, fit_clip=1.0, apply_weight=True)

    # plots 10-14: the WEIGHTED compose (re-selected with the two-scalar weights)
    wcache = Path(args.outdir) / f"{args.tag}_pairs_w.json"
    if wcache.exists() and not args.recompute:
        rows_w = json.load(open(wcache))
        print(f"loaded {len(rows_w)} cached weighted rounds from {wcache}")
    else:
        rows_w = collect(args.record, args.max_rounds, head_w=args.head_w,
                         head_clip=args.head_clip, tail_w=args.tail_w)
        json.dump(rows_w, open(wcache, "w"))
        print("cached weighted pairs ->", wcache)
    plot_terms(rows_w, args.tag, args.outdir, base=10, wsub=True)

    # ---- plots 15-19: calibration (beta head / isotonic tail) + smoothing ----
    import numpy as np
    from replay_extension import _fit_beta
    from sklearn.isotonic import IsotonicRegression
    o2 = lambda n: os.path.join(args.outdir, f"{args.tag}_{n}.png")  # noqa: E731

    beta = _fit_beta([p[0] for p in hpairs], [p[1] for p in hpairs])   # conf -> P(accept)

    tob = [r for r in rows if r["S"] == 1]                            # tail-observed rounds
    That_arr = np.array([r["That"] for r in tob], float)
    T_arr = np.array([r["T"] for r in tob], float)
    succ_arr = np.array([r.get("succ", 0.0) for r in tob], float)
    _iso = IsotonicRegression(out_of_bounds="clip"); _iso.fit(That_arr, T_arr)
    iso_fn = lambda v: float(_iso.predict([max(0.0, float(v))])[0])   # noqa: E731

    plot_head(hpairs, BLUE,
              r"(15) $\hat{f}_{head}(p_{DFlash})$ vs accept rate  (beta-calibrated head)",
              r"$\hat{f}_{head}(p_{DFlash})$  beta-calibrated prob",
              r"accept rate  (realized match)",
              o2("15_fhead_vs_acceptrate"), xform=beta)
    plot_est_binned(That_arr, T_arr, ORANGE,
                    r"(16) $\hat{f}_{tail}(\hat{T})$ vs accept length  (isotonic-calibrated tail)",
                    r"$\hat{f}_{tail}(\hat{T})$  isotonic-calibrated score",
                    r"$T$  realized accept length (mean)",
                    o2("16_ftail_vs_acceptlen"), cap_pct=99, xform=iso_fn)
    plot_head(hpairs, BLUE,
              r"(17) $p_{DFlash}$ vs accept rate  + fitted calibrator",
              r"$p_{DFlash}$  DFlash confidence", r"accept rate  (realized match)",
              o2("17_pDFlash_beta_curve"), curve=beta,
              curve_label=r"$\hat{f}_{head}$  (beta calibration)")
    plot_est_binned(That_arr, T_arr, ORANGE,
                    r"(18) $\hat{T}$ vs accept length  + fitted calibrator",
                    r"$\hat{T}$  raw arctic score", r"$T$  realized accept length (mean)",
                    o2("18_That_iso_curve"), cap_pct=99, curve=iso_fn,
                    curve_label=r"$\hat{f}_{tail}$  (isotonic)")
    plot_est_binned(succ_arr, T_arr, ORANGE,
                    r"(19) $\hat{T}_{succ}$ vs accept length  (succession smoothing)",
                    r"$\hat{T}_{succ}=\sum_t \frac{N_t+1}{\sum_j N_j+2}$  smoothed score",
                    r"$T$  realized accept length (mean)",
                    o2("19_Thatsucc_vs_acceptlen"), cap_pct=99, fit_w=None)


if __name__ == "__main__":
    main()
