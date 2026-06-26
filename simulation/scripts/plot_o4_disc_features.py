#!/usr/bin/env python3
"""Per-feature fit + calibrated-reliability plots for the O4 discriminator.

Mirrors the earlier raw-probability calibration figures, but the target here is
the discriminator's LABEL y = 1 iff suffix is the GT pick (on decisive+contested
decisions). For EACH of the 5 signals we treat the feature as a 1-D calibrator
and emit, as SEPARATE images:

  GROUP 1 (fit):     histogram of empirical P(suffix correct) per feature bin
                     (bars) + the fitted 1-D curve (red line).
  GROUP 2 (reliab.): take the fit's calibrated probability and plot
                     calibrated-P vs empirical accept rate (red dashed y=x).

Algorithms: logistic (1-D logistic on the standardized feature) for all 5;
beta (logistic on [ln p, ln(1-p)]) ALSO for the two probability features
suffix_p / eagle_p (beta only changes how a *probability* is encoded; for
match_len / count / total beta == logistic, so those are emitted once).

NOTE these are MARGINAL per-feature fits (each feature alone) — the deployed
discriminator combines all 5 jointly, so a feature's joint weight can differ
from its marginal fit (e.g. count/total share evidence). A joint-model
reliability image is also emitted for reference.

Usage (container, root — figures dir is root-owned):
  python3 simulation/scripts/plot_o4_disc_features.py \
      --oracle-log <dir>/decisions_select1_oracle.jsonl \
      --out-dir   <dir>/figures/disc_features  --label EAGLE3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

RED = "#d62728"
EPS = 1e-4
# (key, display name, is_probability, bar color)
FEATURES = [
    ("suffix_p", "suffix_p  (trie count ratio c/n)", True, "#ff7f0e"),
    ("eagle_p", "eagle_p  (draft top-1 softmax)", True, "#1f77b4"),
    ("match_len", "match_len  (suffix match length)", False, "#2ca02c"),
    ("suffix_count", "suffix_count  (matched count c)", False, "#9467bd"),
    ("suffix_total", "suffix_total  (context total n)", False, "#8c564b"),
]


def fnum(x, d=0.0):
    return d if x is None else float(x)


def load(path):
    cols = {k: [] for k, *_ in FEATURES}
    y = []
    with open(path) as fh:
        for line in fh:
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
            cols["suffix_p"].append(float(r["suffix_p"]))
            cols["eagle_p"].append(float(r["eagle_p"]))
            cols["match_len"].append(fnum(r.get("match_len")))
            cols["suffix_count"].append(fnum(r.get("suffix_count")))
            cols["suffix_total"].append(fnum(r.get("suffix_total")))
            y.append(1 if r["oracle_hit"] == "suffix" else 0)
    return {k: np.asarray(v, float) for k, v in cols.items()}, np.asarray(y, int)


def fit_1d(x, y, algo, is_prob):
    """Return (predict_grid_x, predict_grid_y, predict_fn) for the 1-D fit."""
    if algo == "beta" and is_prob:
        xc = np.clip(x, EPS, 1 - EPS)
        X = np.column_stack([np.log(xc), np.log(1 - xc)])
        clf = LogisticRegression(max_iter=2000).fit(X, y)

        def predict(v):
            vc = np.clip(np.asarray(v, float), EPS, 1 - EPS)
            return clf.predict_proba(
                np.column_stack([np.log(vc), np.log(1 - vc)]))[:, 1]
    else:
        mu, sd = float(x.mean()), float(x.std() or 1.0)
        clf = LogisticRegression(max_iter=2000).fit(((x - mu) / sd)[:, None], y)

        def predict(v):
            return clf.predict_proba(((np.asarray(v, float) - mu) / sd)[:, None])[:, 1]

    lo, hi = (0.0, 1.0) if is_prob else (float(x.min()),
                                         float(np.quantile(x, 0.99)))
    gx = np.linspace(lo, hi, 400)
    return gx, predict(gx), predict


def emit_fit(x, y, feat, name, is_prob, color, algo, out, sub):
    lo, hi = (0.0, 1.0) if is_prob else (float(x.min()),
                                         float(np.quantile(x, 0.995)))
    nb = 20
    edges = np.linspace(lo, hi, nb + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    centers, rates, widths = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() < 25:          # drop noisy sparse bins
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        rates.append(float(y[m].mean()))
        widths.append((edges[b + 1] - edges[b]) * 0.92)
    gx, gy, _ = fit_1d(x, y, algo, is_prob)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.bar(centers, rates, width=widths, color=color, alpha=0.55,
           edgecolor="white", linewidth=0.4, label="empirical P(suffix correct)")
    ax.plot(gx, gy, color=RED, lw=2.4, label=f"{algo} fit")
    ax.axhline(float(y.mean()), color="k", ls=":", lw=0.8,
               label=f"base rate={y.mean():.3f}")
    ax.set_xlabel(name); ax.set_ylabel("P(suffix is the GT pick)")
    ax.set_ylim(0, 1.02); ax.set_xlim(lo, hi)
    ax.set_title(f"{algo} fit — {feat}\n{sub}", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fp = out / f"disc_{algo}_{feat}_fit.png"
    fig.tight_layout(); fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def emit_reliability(x, y, feat, algo, is_prob, out, sub, phat=None,
                     xlabel=None):
    if phat is None:
        _, _, predict = fit_1d(x, y, algo, is_prob)
        phat = predict(x)
    nb = 12
    edges = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(phat, edges) - 1, 0, nb - 1)
    px, py, cnt = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() < 25:
            continue
        px.append(float(phat[m].mean())); py.append(float(y[m].mean()))
        cnt.append(int(m.sum()))
    ece = sum(c * abs(a - b) for a, b, c in zip(px, py, cnt)) / max(sum(cnt), 1)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], color=RED, ls="--", lw=1.3, label="ideal (y=x)")
    sizes = 18 + 240 * np.asarray(cnt) / max(cnt)
    ax.scatter(px, py, s=sizes, color="#1f77b4", alpha=0.7, edgecolor="k",
               linewidth=0.4, zorder=3, label="bin (size∝count)")
    ax.set_xlabel(xlabel or f"calibrated P(suffix correct) from {feat} [{algo}]")
    ax.set_ylabel("empirical accept rate (suffix correct)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{algo} calibrated reliability — {feat}\nECE={ece:.3f}  {sub}",
                 fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_aspect("equal")
    fp = out / f"disc_{algo}_{feat}_reliability.png"
    fig.tight_layout(); fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def joint_reliability(cols, y, algo, out, sub):
    from sklearn.preprocessing import StandardScaler
    parts = []
    for k, _, is_prob, _ in FEATURES:
        x = cols[k]
        if algo == "beta" and is_prob:
            xc = np.clip(x, EPS, 1 - EPS)
            parts += [np.log(xc), np.log(1 - xc)]
        else:
            parts.append(x)
    X = np.column_stack(parts)
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000).fit(Xs, y)
    phat = clf.predict_proba(Xs)[:, 1]
    return emit_reliability(None, y, "JOINT_5feature", algo, False, out, sub,
                            phat=phat,
                            xlabel=f"joint discriminator P(suffix correct) [{algo}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-log", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="EAGLE3")
    args = ap.parse_args()
    cols, y = load(args.oracle_log)
    n = len(y)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sub = f"n={n}, suffix-right={y.mean():.3f} ({args.label})"
    print(f"loaded n={n}  suffix-right={y.mean():.3f}")
    written = []
    for algo in ("logistic", "beta"):
        for key, name, is_prob, color in FEATURES:
            # beta only differs from logistic on probability features
            if algo == "beta" and not is_prob:
                continue
            x = cols[key]
            written.append(emit_fit(x, y, key, name, is_prob, color, algo, out, sub))
            written.append(emit_reliability(x, y, key, algo, is_prob, out, sub))
        written.append(joint_reliability(cols, y, algo, out, sub))
    print(f"wrote {len(written)} images to {out}:")
    for fp in written:
        print("  ", fp.name)


if __name__ == "__main__":
    main()
