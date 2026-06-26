#!/usr/bin/env python3
"""Consolidated per-technique x per-metric figures for the chain-hybrid
eagle-vs-suffix selection study (14B, pinned eval trajectory = ar+online).

Six metrics, organized by technique AXIS (ref / calibration / feature / mechanism):
  (1) MAT bar
  (2) survival per depth         P(accept_len >= d)
  (3) conditional per depth      survival(d)/survival(d-1)
  (4) selection per depth        eagle-win / suffix-win / agreement (3-panel)
  (5) flip-to-CORRECT vs raw     per depth (offline on the AR oracle log)
  (6) flip-to-WRONG vs raw       per depth

Metrics 1-4 use SERVED logs (timing_/decisions_). Metrics 5-6 are computed OFFLINE
by applying each technique's decision rule to the AR oracle decision log (which has
oracle_hit/gt) on decisive disagreement rows. Online has no frozen map ->
served-only (no flip). tp (target_p offline-calib) is a different trajectory.

Run inside the sglang-bench docker as root (figures dir is root-owned).
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("simulation/results/chain_hybrid_perdepth")
AR, ONLINE, TP, MF = ROOT/"qwen3_14b_ar", ROOT/"qwen3_14b_online", ROOT/"qwen3_14b_tp", ROOT/"qwen3_14b_multifeat"
TRAIN = ROOT/"qwen3_14b_tp_train/decisions_select1_oracle.jsonl"
ORLOG = AR/"decisions_select1_oracle.jsonl"
OUT = ROOT/"figures_consolidated"
STEPS = 16
METHODS = ["histogram", "isotonic", "logistic", "beta"]
CAL = {"histogram": "#1f77b4", "isotonic": "#17becf", "logistic": "#9467bd", "beta": "#8c564b"}


def load_jsonl(p):
    rows = []
    try:
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return rows


def accept_lengths(p):
    v = []
    for r in load_jsonl(p):
        if r.get("phase") and r["phase"] != "decode":
            continue
        a = r.get("accept_lengths")
        if isinstance(a, list):
            v += [int(x) for x in a]
        elif isinstance(a, (int, float)):
            v.append(int(a))
    return np.asarray(v, np.int64)


def survival(acc):
    if acc.size == 0:
        return np.array([]), np.array([])
    d = np.arange(1, int(acc.max())+2)
    return d, np.array([(acc >= k).mean() for k in d])


def conditional(acc):
    d, s = survival(acc)
    if d.size == 0:
        return d, s
    prev = np.concatenate([[1.0], s[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        return d, np.where(prev > 0, s/prev, np.nan)


def selection_by_depth(p):
    # CANONICAL 3-way SELECTION breakdown per depth over CHOICE-AVAILABLE decisions
    # (suffix_token != None: both proposers offered a candidate). Shared definition
    # with plot_o4 / plot_o4_objectives `selection_breakdown_by_depth`:
    #   eagle  = chosen==eagle3 & not agreement  (EAGLE3 uniquely selected)
    #   suffix = chosen==suffix                  (suffix uniquely selected)
    #   tie    = agreement True                  (tokens equal; pick is moot)
    # The three fractions sum to 1.0 at each depth.
    n = defaultdict(int); eg = defaultdict(int); su = defaultdict(int); ag = defaultdict(int)
    for r in load_jsonl(p):
        if r.get("type") != "decision" or r.get("tail") or r.get("suffix_token") is None:
            continue
        d = r["depth"]; n[d] += 1
        if r.get("agreement") is True:
            ag[d] += 1
        elif r.get("chosen") == "suffix":
            su[d] += 1
        else:
            eg[d] += 1
    ds = sorted(n)
    return ds, [eg[d]/n[d] for d in ds], [su[d]/n[d] for d in ds], [ag[d]/n[d] for d in ds]


# ---------- offline decision-rule appliers ----------
def load_pp(path):
    b = json.load(open(path))
    return {g: {int(d): (np.array(m["x"]), np.array(m["y"])) for d, m in dd.items()}
            for g, dd in b["groups"].items()}


def pp_pred(M, g, p, depth):
    m = M[g]
    if depth in m:
        xs, ys = m[depth]
    else:
        le = [k for k in m if k <= depth]
        xs, ys = m[max(le)] if le else m[max(m)]
    return float(np.interp(p, xs, ys))


def load_mf(path):
    return json.load(open(path))["groups"]


def mf_pred(G, g, p, depth, c, n, ml):
    m = G[g]; vals = []
    for f in m["features"]:
        vals.append({"eagle_p": p, "suffix_p": p, "depth": depth, "match_len": ml,
                     "log1p_count": np.log1p(c), "log1p_total": np.log1p(n)}[f])
    x = (np.array(vals, float)-np.array(m["mean"]))/np.array(m["std"])
    z = m["intercept"]+float(np.dot(m["coef"], x))
    return 1/(1+np.exp(-z)) if m["kind"] == "logistic" else min(max(z, 0), 1)


def load_decisive(path, with_prev=False):
    """decisive disagreement rows. Without prev: (depth,ep,sp,c,n,ml,y). With prev:
    appends (prev_sp,prev_ml,prev_suf) before y. y=1 if suffix is the correct pick."""
    byrid = defaultdict(list)
    for r in load_jsonl(path):
        if r.get("type") == "decision" and not r.get("tail"):
            byrid[r.get("rid")].append(r)
    R = []
    for rid, rs in byrid.items():
        rs.sort(key=lambda r: (r.get("decode_step", 0), r.get("depth", 0)))
        psp = pml = psuf = 0.0
        for r in rs:
            oh = r.get("oracle_hit"); et, st = r.get("eagle_token"), r.get("suffix_token")
            ep, sp, c, n, ml = (r.get("eagle_p"), r.get("suffix_p"), r.get("suffix_count"),
                                r.get("suffix_total"), r.get("match_len"))
            if oh in ("eagle", "suffix") and st is not None and et != st and None not in (ep, sp, c, n, ml):
                base = [r["depth"], float(ep), float(sp), float(c), float(n), float(ml)]
                tail = [psp, pml, psuf] if with_prev else []
                R.append(tuple(base + tail + [1 if oh == "suffix" else 0]))
            if sp is not None:
                psp = float(sp)
            if ml is not None:
                pml = float(ml)
            psuf = 1.0 if oh == "suffix" else (0.0 if oh == "eagle" else psuf)
    return R


def picks_rule(rows, kind, **kw):
    out = []
    for r in rows:
        d, ep, sp, c, n, ml = r[0], r[1], r[2], r[3], r[4], r[5]
        if kind == "raw":
            out.append(sp > ep)
        elif kind == "pp":
            out.append(pp_pred(kw["M"], "suffix", sp, d) > pp_pred(kw["M"], "eagle", ep, d))
        elif kind == "mf":
            out.append(mf_pred(kw["G"], "suffix", sp, d, c, n, ml) > mf_pred(kw["G"], "eagle", ep, d, 0, 0, 0))
        elif kind == "signalfree":
            out.append(d >= 2)
    return np.array(out, int)


def flip_curves(rows, picks):
    """per depth: flip-to-correct% and flip-to-wrong% vs raw (raw = sp>ep)."""
    raw = np.array([1 if r[2] > r[1] else 0 for r in rows])
    y = np.array([r[-1] for r in rows]); dep = np.array([r[0] for r in rows])
    fc = (raw != picks) & (picks == y) & (raw != y)
    fw = (raw != picks) & (picks != y) & (raw == y)
    ds = sorted(set(dep.tolist()))
    return (ds, [100*fc[dep == d].sum()/max((dep == d).sum(), 1) for d in ds],
            [100*fw[dep == d].sum()/max((dep == d).sum(), 1) for d in ds])


def served_set():
    S = [(AR, "baseline", "model-only", "#7f7f7f"), (AR, "suffix", "suffix-only", "#d62728"),
         (AR, "select1", "raw", "#1f77b4"), (AR, "select1_oracle", "ORACLE", "#e0b400")]
    for m in METHODS:
        S += [(AR, f"select1_calib_{m}_all-trained", f"calib {m}[all-trained]", CAL[m]),
              (AR, f"select1_calib_{m}_cond-trained", f"calib {m}[cond-trained]", CAL[m])]
    S += [(ONLINE, "multifeat_accept_rate", "multifeat[ar]", "#2ca02c"),
          (ONLINE, "multifeat_target_p", "multifeat[tp]", "#2ca02c")]
    for m in METHODS:
        S.append((ONLINE, f"online_{m}_accept_rate_w1024", f"online {m}", "#ff7f0e"))
    return S


# label format: "<algorithm> [<variant> · <objective>]". raw/ORACLE are uncalibrated
# references (no objective). All calibrated arms here use the accept_rate objective.
LINE = [(AR, "baseline", "model-only", "#7f7f7f", "-"), (AR, "suffix", "suffix-only", "#d62728", "-"),
        (AR, "select1", "raw", "#1f77b4", "-"), (AR, "select1_oracle", "ORACLE", "#e0b400", "-"),
        (AR, "select1_calib_logistic_cond-trained", "logistic [cond-trained · accept_rate]", "#9467bd", "--"),
        (ONLINE, "online_logistic_accept_rate_w1024", "online logistic [accept_rate · w1024]", "#ff7f0e", ":"),
        (ONLINE, "multifeat_accept_rate", "multifeat [accept_rate]", "#2ca02c", "-.")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # (1) MAT bar
    labs, mats, cols = [], [], []
    for d, arm, lab, col in served_set():
        acc = accept_lengths(d/f"timing_{arm}.jsonl")
        if acc.size:
            labs.append(lab); mats.append(float(acc.mean())); cols.append(col)
    fig, ax = plt.subplots(figsize=(max(11, 0.55*len(labs)+2), 5))
    ax.bar(np.arange(len(labs)), mats, color=cols, width=0.74)
    for i, v in enumerate(mats):
        ax.text(i, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels(labs, fontsize=7, rotation=40, ha="right")
    ax.set_ylabel("MAT"); ax.set_ylim(0, max(mats)*1.16); ax.grid(axis="y", alpha=0.3)
    ax.set_title("(1) MAT by technique (14B pinned eval)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT/"m1_mat.png", dpi=150); plt.close(fig)

    # (2)(3) survival / conditional
    accs = {arm: accept_lengths(d/f"timing_{arm}.jsonl") for d, arm, *_ in LINE}
    for tag, fn, ylab, fname in (("2", survival, "survival  P(accept>=d)", "m2_survival.png"),
                                 ("3", conditional, "conditional  P(accept>=d | >=d-1)", "m3_conditional.png")):
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        for d, arm, lab, col, ls in LINE:
            dd, vv = fn(accs[arm])
            if dd.size:
                ax.plot(dd, vv, color=col, ls=ls, lw=1.7, marker="o", ms=3, label=lab)
        ax.axvline(STEPS, color="k", lw=1, ls=":"); ax.set_xlim(0.5, STEPS); ax.set_ylim(0, 1.0)
        ax.set_xlabel("depth d"); ax.set_ylabel(ylab); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2, handlelength=3.5, handletextpad=0.5)
        ax.set_title(f"({tag}) {ylab.split('  ')[0]} per depth", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT/fname, dpi=150); plt.close(fig)

    # (4) selection 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for d, arm, lab, col, ls in LINE:
        dec = d/f"decisions_{arm}.jsonl"
        if not dec.exists():
            continue
        ds, eg, su, ag = selection_by_depth(dec)
        if ds:
            for ax, series in zip(axes, (eg, su, ag)):
                ax.plot(ds, series, color=col, ls=ls, lw=1.6, marker="o", ms=2.5, label=lab)
    for ax, t in zip(axes, ["eagle-win (uniquely selected)", "suffix-win (uniquely selected)", "tie (agreement)"]):
        ax.set_xlim(0.5, STEPS); ax.set_ylim(0, 1.02); ax.set_xlabel("depth d"); ax.set_title(t, fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction (choice-available decisions)")
    axes[0].legend(fontsize=7, handlelength=3.5, handletextpad=0.5)
    fig.suptitle("(4) selection per depth — 3-way over choice-available decisions "
                 "(eagle/suffix/tie sum to 1)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT/"m4_selection.png", dpi=150); plt.close(fig)

    # (5)(6) flip vs raw (offline on AR oracle log) -> precompute (ds,fc,fw) per technique
    rows = load_decisive(ORLOG)
    print(f"decisive rows (AR oracle)={len(rows)}", file=sys.stderr)
    techs = []  # (label, color, ls, ds, fc, fw)
    def add(lab, col, ls, r_, picks):
        ds, fc, fw = flip_curves(r_, picks); techs.append((lab, col, ls, ds, fc, fw))
    add("raw", "#1f77b4", "-", rows, picks_rule(rows, "raw"))
    add("signal-free", "#000000", ":", rows, picks_rule(rows, "signalfree"))
    for m, st in (("logistic", "--"), ("beta", "-.")):
        for v, vls in (("all-trained", st), ("cond-trained", "-")):
            M = load_pp(TP/f"calib_{v}/calib_pp_{m}.json")
            add(f"calib {m}[{v}]", CAL[m], vls, rows, picks_rule(rows, "pp", M=M))
    add("multifeat", "#2ca02c", "-", rows, picks_rule(rows, "mf", G=load_mf(MF/"multifeat_accept_rate.json")))
    add("ORACLE", "#e0b400", "-", rows, np.array([r[-1] for r in rows]))
    # momentum (offline-only) on its own prev-augmented decisive ordering
    try:
        from sklearn.linear_model import LogisticRegression
        Dtr, Dev = load_decisive(TRAIN, True), load_decisive(ORLOG, True)
        def X(D):
            A = np.array(D)
            return np.c_[A[:, 1], A[:, 2], np.log1p(A[:, 3]), np.log1p(A[:, 4]), A[:, 5], A[:, 0], A[:, 6], A[:, 7], A[:, 8]]
        lr = LogisticRegression(max_iter=2000).fit(X(Dtr), np.array([d[-1] for d in Dtr]))
        add("momentum", "#e377c2", "-", Dev, lr.predict(X(Dev)).astype(int))
    except Exception as e:
        print(f"momentum skipped: {e}", file=sys.stderr)

    for k, (which, fname) in enumerate([(1, "m5_flip_correct.png"), (2, "m6_flip_wrong.png")]):
        ylab = "flip-to-CORRECT vs raw (%)" if which == 1 else "flip-to-WRONG vs raw (%)"
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        for lab, col, ls, ds, fc, fw in techs:
            y = (fc if which == 1 else fw)
            ax.plot(ds[:STEPS], y[:STEPS], color=col, ls=ls, lw=1.6, marker="o", ms=3, label=lab)
        ax.set_xlim(0.5, STEPS-0.5); ax.set_xlabel("depth d"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, ncol=2, handlelength=3.5, handletextpad=0.5); ax.set_title(f"({5+k}) {ylab.split(' vs')[0]} per depth (decisive)", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT/fname, dpi=150); plt.close(fig)

    # (7 bonus) NET flip = correct - wrong (the direct "did it help vs raw" view)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for lab, col, ls, ds, fc, fw in techs:
        net = [a-b for a, b in zip(fc, fw)]
        ax.plot(ds[:STEPS], net[:STEPS], color=col, ls=ls, lw=1.6, marker="o", ms=3, label=lab)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlim(0.5, STEPS-0.5); ax.set_xlabel("depth d")
    ax.set_ylabel("NET flip vs raw  (correct - wrong, %)"); ax.grid(alpha=0.3); ax.legend(fontsize=6.5, ncol=2, handlelength=3.5, handletextpad=0.5)
    ax.set_title("(7) NET selection gain vs raw per depth (decisive)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT/"m7_flip_net.png", dpi=150); plt.close(fig)
    print(f"wrote 7 figures -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
