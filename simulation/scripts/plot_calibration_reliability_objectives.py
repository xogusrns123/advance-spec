#!/usr/bin/env python3
"""Calibration reliability diagrams for BOTH objectives, each on its OWN lineage.

Fig 1 (TRAIN, raw prob):   x = raw proposer prob, y = the fit target
    token_gt -> empirical ACCEPT rate  P(token==gt)        (def_train oracle log)
    target_p -> empirical mean q_target                    (tp_train oracle + target_probs)
  Shows what each objective fits to + (peakedness) how close accept-rate and q_target are.

Fig 2 (TEST, calibrated prob): x = calibrated prob = map[group][depth](raw_p), y = target
    token_gt -> ACCEPT rate on def TEST  (def map applied to def-test oracle raw probs)
    target_p -> mean q_target on tp TEST (tp map applied to tp-test oracle raw probs)
  Diagonal y=x = perfectly calibrated. (calibrated prob s should predict outcome s.)

Per-group panels (eagle, suffix); calibration is per-(group,depth) so the map is
applied with each decision's own depth. Calibrated prob uses --method (default isotonic,
the canonical monotone calibrator).

Usage:
  python3 simulation/scripts/plot_calibration_reliability_objectives.py \
    --base simulation/results/o4_perdepth --method isotonic \
    --out-dir simulation/results/comparison_figures
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TG_C, TP_C = "#1f77b4", "#2ca02c"   # token_gt blue, target_p green


def load_maps(d, method):
    return json.load(open(Path(d) / f"calib_pp_{method}.json"))["groups"]


def interp(curve, p):
    x = curve["x"]; n = len(x)
    i = min(n - 1, max(0, int(round(float(p) * (n - 1)))))
    y = curve["y"][i]
    return y if y == y else None   # drop NaN (empty hist bins)


def read_decisions(path):
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("type") == "decision" and not r.get("tail"):
            yield r


def qmap(path):
    """(rid, decode_step, depth) -> (q_eagle, q_suffix)."""
    m = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        m[(r["rid"], r["decode_step"], r["depth"])] = (r.get("q_eagle"), r.get("q_suffix"))
    return m


def collect(dec_path, *, target, qfile=None, maps=None):
    """Return {group: [(x, y)]}. target='accept' uses gt_token; target='q' uses qfile.
    If maps given, x = calibrated prob map[group][depth](raw_p); else x = raw_p."""
    q = qmap(qfile) if qfile else None
    out = {"eagle": [], "suffix": []}
    for r in read_decisions(dec_path):
        dep = str(int(r["depth"]))
        gt = r.get("gt_token")
        key = (r["rid"], r["decode_step"], r["depth"])
        for grp, tok, rawp in (("eagle", r.get("eagle_token"), r.get("eagle_p")),
                               ("suffix", r.get("suffix_token"), r.get("suffix_p"))):
            if rawp is None or tok is None:
                continue
            if target == "accept":
                if gt is None:
                    continue
                y = 1.0 if tok == gt else 0.0
            else:  # q_target
                if q is None or key not in q:
                    continue
                qe, qs = q[key]
                y = qe if grp == "eagle" else qs
                if y is None:
                    continue
            if maps is not None:
                if dep not in maps[grp]:
                    continue
                x = interp(maps[grp][dep], rawp)
                if x is None:
                    continue
            else:
                x = float(rawp)
            out[grp].append((x, float(y)))
    return out


def binned(points, nbins=20, mincount=30):
    if not points:
        return np.array([]), np.array([]), np.array([])
    xs = np.array([p[0] for p in points]); ys = np.array([p[1] for p in points])
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, nbins - 1)
    cx, cy, cn = [], [], []
    for b in range(nbins):
        m = idx == b
        c = int(m.sum())
        if c >= mincount:
            cx.append(xs[m].mean()); cy.append(ys[m].mean()); cn.append(c)
    return np.array(cx), np.array(cy), np.array(cn)


def panel(ax, tg_pts, tp_pts, xlabel, title):
    ax.plot([0, 1], [0, 1], color="#888", lw=1, ls="--", label="perfect (y=x)")
    for pts, c, lab in ((tg_pts, TG_C, "token_gt → accept rate"),
                        (tp_pts, TP_C, "target_p → q_target")):
        bx, by, bn = binned(pts)
        if len(bx):
            sz = 20 + 180 * (bn / bn.max())
            ax.plot(bx, by, color=c, lw=1.3, alpha=0.7, zorder=2)
            ax.scatter(bx, by, s=sz, color=c, alpha=0.85, edgecolor="white",
                       linewidth=0.5, label=lab, zorder=3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel); ax.set_ylabel("empirical target (accept rate / mean q_target)")
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="simulation/results/o4_perdepth")
    ap.add_argument("--method", default="isotonic")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    B = Path(a.base); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- TRAIN (raw prob) ----
    tg_tr = collect(B / "qwen3_14b_def_train" / "decisions_select1_oracle.jsonl", target="accept")
    tp_tr = collect(B / "qwen3_14b_tp_train" / "decisions_select1_oracle.jsonl", target="q",
                    qfile=B / "qwen3_14b_tp_train" / "target_probs.jsonl")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, grp in zip(axs, ("eagle", "suffix")):
        panel(ax, tg_tr[grp], tp_tr[grp], "raw proposer probability",
              f"TRAIN reliability — {grp}  (raw prob → fit target)")
    fig.suptitle("Calibration TRAIN reliability (raw prob, per group) — 14B bfcl web_search\n"
                 "token_gt fits ACCEPT rate (def_train) · target_p fits q_target (tp_train); "
                 "near-coincidence = target peakedness", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "calib_train_raw_reliability.png", dpi=150); plt.close(fig)

    # ---- TEST (calibrated prob) ----
    dmaps = load_maps(B / "qwen3_14b_def", a.method)   # token_gt maps
    tmaps = load_maps(B / "qwen3_14b_tp", a.method)     # target_p maps
    tg_te = collect(B / "qwen3_14b_def" / "decisions_select1_oracle.jsonl",
                    target="accept", maps=dmaps)
    tp_te = collect(B / "qwen3_14b_tp" / "decisions_select1_oracle.jsonl", target="q",
                    qfile=B / "qwen3_14b_tp" / "target_probs_test.jsonl", maps=tmaps)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, grp in zip(axs, ("eagle", "suffix")):
        panel(ax, tg_te[grp], tp_te[grp], f"calibrated probability ({a.method})",
              f"TEST reliability — {grp}  (calibrated prob → outcome)")
    fig.suptitle(f"Calibration TEST reliability (calibrated prob = {a.method} map, per group) — "
                 "14B bfcl web_search\ntoken_gt: cal prob vs accept rate (def test) · "
                 "target_p: cal prob vs q_target (tp test); on-diagonal = well-calibrated",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "calib_test_calibrated_reliability.png", dpi=150); plt.close(fig)

    # numeric summary (ECE-style mean |y - x|, count-weighted)
    def ece(pts, use_x_as_pred):
        bx, by, bn = binned(pts)
        if not len(bx):
            return float("nan"), 0
        pred = bx  # for test, x IS the predicted prob; for train raw, x is raw prob
        return float(np.average(np.abs(by - pred), weights=bn)), int(bn.sum())
    print(f"method={a.method}")
    for name, d in (("TRAIN raw", {"token_gt": tg_tr, "target_p": tp_tr}),
                    ("TEST cal", {"token_gt": tg_te, "target_p": tp_te})):
        for obj, gd in d.items():
            for grp in ("eagle", "suffix"):
                e, n = ece(gd[grp], True)
                print(f"  {name:9s} {obj:9s} {grp:7s} |y-x|(wt)={e:.4f} n={n}")
    print(f"wrote {out}/calib_train_raw_reliability.png + calib_test_calibrated_reliability.png")


if __name__ == "__main__":
    main()
