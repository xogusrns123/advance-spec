#!/usr/bin/env python3
"""Offline analysis of capture_perpos.jsonl → slides 4, 5, 7, 13b, 18.

Every record is one committed position with the DFlash block's per-depth conf +
gt-match and the warm/cold Suffix probe. From this single dense capture:

  slide 4/5  hazard a_i = P(match_i | match_0..i-1),  survival S_i = P(∀ j<i match_j)
             per task (math/rag/summ/qa or multislot_k*). a_i can be NON-monotone.
  slide 18   conf → a_k calibration: affine refit a ≈ s·conf + b (cf. 0.69/0.29),
             predicted survival Πconf vs measured S_k → Pearson r (target ≈0.99).
  slide 7    DFlash survival S_i vs Suffix survival (warm vs cold) by depth →
             crossover depth where warm suffix overtakes DFlash.
  slide 13b  argmax k* vs first-crossing k* over T, from the measured hazard →
             first-crossing stops at the dip; argmax does not.

Usage (CPU):
  python3 scripts/analyze_perpos.py --glob 'results/perpos/*.jsonl' \
      --fig-dir results/perpos/figures
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import argparse
import glob
import json
import math
from collections import defaultdict


def _load(patterns):
    rows = []
    for pat in patterns:
        for fn in glob.glob(pat):
            for l in open(fn):
                l = l.strip()
                if l:
                    rows.append(json.loads(l))
    return rows


def hazard_survival(rows):
    """Empirical survival S_i and hazard a_i from per-position match vectors.
    S_i = fraction of positions whose first i head tokens ALL match gt."""
    W = rows[0]["W"]
    surv = [0.0] * (W + 1)
    n = len(rows)
    for r in rows:
        m = r["dflash_match"]
        i = 0
        while i < W and m[i] == 1:
            i += 1
        # this position survives to depths 0..i (S_0..S_i get a hit)
        for j in range(i + 1):
            surv[j] += 1
    S = [surv[j] / n if n else 0.0 for j in range(W + 1)]     # S[0]=1
    a = [(S[i] / S[i - 1] if S[i - 1] > 1e-9 else 0.0) for i in range(1, W + 1)]
    return S, a, n


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def calib_fit(rows):
    """affine a ≈ s·conf + b via least squares on (conf[d], match[d]) pairs;
    predicted-survival Πconf vs measured S_k Pearson r."""
    xs, ys = [], []
    for r in rows:
        for c, m in zip(r["dflash_conf"], r["dflash_match"]):
            xs.append(float(c)); ys.append(float(m))
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    s = sxy / sxx if sxx > 0 else 0.0
    b = my - s * mx
    conf_hazard_r = pearson(xs, ys)
    # predicted survival Πconf per position vs realized survival depth
    W = rows[0]["W"]
    pred_S = [0.0] * (W + 1)          # mean predicted survival by depth
    meas_S, _, _ = hazard_survival(rows)
    cnt = 0
    for r in rows:
        p = 1.0
        pred_S[0] += 1.0
        for d in range(W):
            p *= float(r["dflash_conf"][d])
            pred_S[d + 1] += p
        cnt += 1
    pred_S = [v / cnt for v in pred_S]
    surv_r = pearson(pred_S, meas_S)
    return s, b, conf_hazard_r, surv_r, pred_S, meas_S


def suffix_survival(rows, key):
    """P(suffix accept-length >= i) by depth."""
    W = rows[0]["W"]
    n = len(rows)
    out = [0.0] * (W + 1)
    for r in rows:
        ml = int(r[key])
        for i in range(min(ml, W) + 1):
            out[i] += 1
    return [v / n for v in out]


def kstar_curves(a, Ts):
    """slide 13b: argmax vs first-crossing k* as a function of T, from hazard a."""
    W = len(a)
    S = [1.0]
    for aj in a:
        S.append(S[-1] * aj)
    G = [0.0]
    for k in range(1, W + 1):
        G.append(G[-1] + S[k])
    rows = []
    for T in Ts:
        astar = T / (1.0 + T)
        # first-crossing: extend while hazard a_k > a*  (stops at first dip below)
        fc = 0
        for k in range(W):
            if a[k] > astar:
                fc = k + 1
            else:
                break
        # argmax of K(k) = 1 + G_k + S_k·T
        K = [1.0 + G[k] + S[k] * T for k in range(W + 1)]
        am = max(range(W + 1), key=lambda k: K[k])
        rows.append((T, astar, fc, am))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", required=True)
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()

    rows = _load(args.glob)
    if not rows:
        print("no records"); return
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    print(f"loaded {len(rows)} positions across tasks: {sorted(by_task)}")

    # ---- slide 4/5: hazard + survival per task ----
    print("\n== slide 4/5: DFlash hazard a_i / survival S_i (per task) ==")
    task_haz = {}
    for task in sorted(by_task):
        S, a, n = hazard_survival(by_task[task])
        task_haz[task] = (S, a)
        dip = any(a[i] < a[i - 1] - 1e-6 for i in range(1, len(a)))
        print(f"  [{task}] n={n}  a_1..a_5={['%.2f'%x for x in a[:5]]}  "
              f"S_1,S_3,S_5={S[1]:.2f},{S[3]:.2f},{S[5]:.2f}  non-monotone={dip}")

    # ---- slide 18: conf→a_k calibration ----
    print("\n== slide 18: conf → a_k calibration (all tasks) ==")
    s, b, chr_, surv_r, pred_S, meas_S = calib_fit(rows)
    print(f"  affine refit: a ≈ {s:.3f}·conf + {b:.3f}   (Dr.Lee: 0.69·conf+0.29)")
    print(f"  conf↔hazard Pearson r = {chr_:.3f}")
    print(f"  predicted-survival (Πconf) vs measured S_k Pearson r = {surv_r:.3f}  (target ≈0.99)")

    # ---- slide 7: DFlash vs Suffix survival (warm/cold) ----
    print("\n== slide 7: DFlash survival vs Suffix survival (warm/cold) ==")
    S_df, _, _ = hazard_survival(rows)
    S_sw = suffix_survival(rows, "suffix_match_warm")
    S_sc = suffix_survival(rows, "suffix_match_cold")
    cross = next((i for i in range(1, len(S_df)) if S_sw[i] > S_df[i]), None)
    print(f"  depth:      " + "  ".join(f"{i:>4}" for i in range(1, 9)))
    print(f"  DFlash S_i: " + "  ".join(f"{S_df[i]:.2f}" for i in range(1, 9)))
    print(f"  Suffix warm:" + "  ".join(f"{S_sw[i]:.2f}" for i in range(1, 9)))
    print(f"  Suffix cold:" + "  ".join(f"{S_sc[i]:.2f}" for i in range(1, 9)))
    print(f"  warm-suffix overtakes DFlash at depth {cross}")

    # ---- slide 13b: argmax vs first-crossing ----
    print("\n== slide 13b: argmax vs first-crossing k* over T (overall hazard) ==")
    _, a_all, _ = hazard_survival(rows)
    for (T, astar, fc, am) in kstar_curves(a_all, [0.0, 0.5, 1.0, 2.0, 5.0, 9.8]):
        flag = "  <-- differ (dip trap)" if fc != am else ""
        print(f"  T={T:>4}  a*={astar:.2f}  first-crossing k*={fc:>2}  argmax k*={am:>2}{flag}")

    if not args.fig_dir:
        return
    _figs(args.fig_dir, task_haz, pred_S, meas_S, S_df, S_sw, S_sc)


def _figs(fig_dir, task_haz, pred_S, meas_S, S_df, S_sw, S_sc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # slide 4: hazard by task
    fig, ax = plt.subplots(figsize=(6, 4))
    for task, (S, a) in sorted(task_haz.items()):
        ax.plot(range(1, len(a) + 1), a, marker="o", ms=3, label=task)
    ax.set_xlabel("depth i"); ax.set_ylabel("hazard  a_i = P(match_i | survived)")
    ax.set_title("Slide 4 — DFlash hazard a_i by task (can be non-monotone)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/slide4_hazard.png", dpi=140); plt.close(fig)

    # slide 18: predicted vs measured survival
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(meas_S, pred_S, "o", ms=4)
    lo, hi = 0, 1
    ax.plot([lo, hi], [lo, hi], "--", color="#888")
    ax.set_xlabel("measured S_k"); ax.set_ylabel("predicted Πconf")
    ax.set_title("Slide 18 — conf-predicted vs measured survival")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/slide18_calib.png", dpi=140); plt.close(fig)

    # slide 7: survival curves
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(1, len(S_df))
    ax.plot(xs, S_df[1:], marker="o", color="#1f77b4", label="DFlash")
    ax.plot(xs, S_sw[1:], marker="s", color="#ff7f0e", label="Suffix (warm)")
    ax.plot(xs, S_sc[1:], marker="^", color="#888", label="Suffix (cold)")
    ax.set_xlabel("depth i"); ax.set_ylabel("survival S_i")
    ax.set_title("Slide 7 — head (DFlash) vs tail (Suffix) survival")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/slide7_survival.png", dpi=140); plt.close(fig)
    print(f"\nsaved figures -> {fig_dir}/")


if __name__ == "__main__":
    main()
