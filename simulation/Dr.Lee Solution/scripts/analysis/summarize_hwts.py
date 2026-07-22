#!/usr/bin/env python3
r"""Collect the head-weight x tail-smoothing arm matrix (run_headweight_tailsmooth.sh)
plus the existing deployable baselines, into one ranked MAT table + a bar figure.

MAT per (workload, arm) is the `calib: K=` line of the split (TEST-half) replay log.
Mean across the 5 deployable workloads is the headline number. English figure labels
(house rule). Run inside sglang-bench (matplotlib lives there):

  docker exec sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace python3 scripts/analysis/summarize_hwts.py'
"""
from __future__ import annotations
import json
import re
from pathlib import Path

RLOG = Path("readable_outputs/figures/replay_logs")
OUT = Path("results/headweight_tailsmooth")
DSS = ["specbench", "bfcl", "swebench", "spider", "tau2"]

# arm label -> log filename template ({ds} filled). Group tag for the figure.
NEW = {  # from run_headweight_tailsmooth.sh  (mat_{ds}_hwts_{arm}_split.replay.txt)
    "onlbeta_succ_raw":  ("A online", "online-beta + Laplace"),
    "onlbeta_kt_raw":    ("A online", "online-beta + KT"),
    "onllog_succ_raw":   ("A online", "online-logistic + Laplace"),
    "onlbeta_succ_iso":  ("A online", "online-beta + Laplace->iso"),
    "onlbeta_kt_iso":    ("A online", "online-beta + KT->iso"),
    "hs10_succ":         ("B hw+smooth", "head x1.0 + Laplace"),
    "hs12_succ":         ("B hw+smooth", "head x1.2 + Laplace"),
    "hs14_succ":         ("B hw+smooth", "head x1.4 + Laplace"),
    "hs16_succ":         ("B hw+smooth", "head x1.6 + Laplace"),
    "hs18_succ":         ("B hw+smooth", "head x1.8 + Laplace"),
    "hs12_kt":           ("B hw+smooth", "head x1.2 + KT"),
    "hs14_kt":           ("B hw+smooth", "head x1.4 + KT"),
    "hs16_kt":           ("B hw+smooth", "head x1.6 + KT"),
    "hs12_rawtail":      ("C single-w", "head x1.2 only"),
    "hs14_rawtail":      ("C single-w", "head x1.4 only"),
    "hs16_rawtail":      ("C single-w", "head x1.6 only"),
    "rawhead_fix010":    ("C single-w", "tail x0.10 only"),
    "rawhead_fix0125":   ("C single-w", "tail x0.125 only"),
    "rawhead_fix015":    ("C single-w", "tail x0.15 only"),
    "rawhead_fix020":    ("C single-w", "tail x0.20 only"),
    "betahead_succ":     ("D offline", "offline-beta + Laplace"),
    "loghead_succ":      ("D offline", "offline-logistic + Laplace"),
    "betahead_kt":       ("D offline", "offline-beta + KT"),
    "hs14_succscale05":  ("E scaled", "head x1.4 + 0.5*Laplace"),
    "hs14_succscale075": ("E scaled", "head x1.4 + 0.75*Laplace"),
    "hs14_succscale10":  ("E scaled", "head x1.4 + 1.0*Laplace"),
    "hs16_succscale075": ("E scaled", "head x1.6 + 0.75*Laplace"),
    "twoscalar_h14w0125": ("ref", "two-scalar x1.4 / 0.125*raw (repro)"),
}
NEW_TMPL = "mat_{ds}_hwts_{arm}_split.replay.txt"

# existing baselines: (group, pretty, filename template)
BASE = {
    "onl_logistic_isotonic": ("ref", "online-logistic + iso (canonical)",
                              "mat_{ds}_4way_calib_onl_logistic_isotonic_split.replay.txt"),
    "onl_beta_isotonic":     ("ref", "online-beta + iso",
                              "mat_{ds}_4way_calib_onl_beta_isotonic_split.replay.txt"),
    "succ":                  ("ref", "affine head + Laplace",
                              "mat_{ds}_4way_calib_succ_split.replay.txt"),
    "succrawhead":           ("ref", "raw head + Laplace",
                              "mat_{ds}_4way_calib_succrawhead_split.replay.txt"),
}
SINGLE = {  # dflash / suffix / oracle for floor+ceiling context
    "dflash": ("mat_{ds}_4way_singles_split.replay.txt", "dflash"),
    "suffix": ("mat_{ds}_4way_singles_split.replay.txt", "suffix"),
    "oracle": ("mat_{ds}_4way_oracle_split.replay.txt", "oracle"),
}

K_RE = re.compile(r"^\s*(\w+):\s*K=([0-9.]+)")


def read_k(path: Path, prop: str = "calib"):
    if not path.exists():
        return None
    for ln in path.read_text().splitlines():
        m = K_RE.match(ln)
        if m and m.group(1) == prop:
            return float(m.group(2))
    return None


def collect():
    rows = {}  # label -> {group, pretty, per-ds K, mean}
    def add(label, group, pretty, tmpl, prop="calib"):
        ks = {ds: read_k(RLOG / tmpl.format(ds=ds), prop) for ds in DSS}
        have = [v for v in ks.values() if v is not None]
        rows[label] = {"group": group, "pretty": pretty, "ks": ks,
                       "mean": (sum(have) / len(have)) if len(have) == len(DSS) else None,
                       "n": len(have)}
    for arm, (grp, pretty) in NEW.items():
        add(arm, grp, pretty, NEW_TMPL.replace("{arm}", arm))
    for arm, (grp, pretty, tmpl) in BASE.items():
        add(arm, grp, pretty, tmpl)
    for prop, (tmpl, pretty) in SINGLE.items():
        add(prop, "single", pretty, tmpl, prop=prop)
    return rows


def main():
    rows = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "summary.json", "w"), indent=1)

    complete = {k: v for k, v in rows.items() if v["mean"] is not None}
    ranked = sorted(complete.items(), key=lambda kv: -kv[1]["mean"])
    hdr = f"{'arm':32s} {'group':12s} " + " ".join(f"{d:>9s}" for d in DSS) + f"  {'MEAN':>7s}"
    print(hdr); print("-" * len(hdr))
    for label, r in ranked:
        cells = " ".join(f"{r['ks'][d]:9.2f}" if r['ks'][d] is not None else f"{'-':>9s}" for d in DSS)
        print(f"{r['pretty']:32s} {r['group']:12s} {cells}  {r['mean']:7.3f}")
    incomplete = {k: v for k, v in rows.items() if v["mean"] is None}
    if incomplete:
        print("\nINCOMPLETE (missing some workloads):")
        for k, v in incomplete.items():
            print(f"  {k:24s} n={v['n']}/5")

    # ---- figure: mean-MAT bar, colored by group, dflash/suffix/oracle refs ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bars = [(label, r) for label, r in ranked if r["group"] not in ("single",)]
        labels = [r["pretty"] for _, r in bars]
        means = [r["mean"] for _, r in bars]
        gcol = {"A online": "#1f77b4", "B hw+smooth": "#2ca02c", "C single-w": "#ff7f0e",
                "D offline": "#9467bd", "E scaled": "#8c564b", "ref": "#7f7f7f"}
        colors = [gcol.get(r["group"], "#333") for _, r in bars]
        fig, ax = plt.subplots(figsize=(11, 9))
        y = range(len(labels))
        ax.barh(list(y), means, color=colors)
        ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean accepted tokens / round (MAT), test half, 5 workloads")
        ax.set_title("Head-weight x tail-smoothing arms vs deployable baselines")
        for ref, name, c in [("dflash", "DFlash", "#d62728"), ("suffix", "Suffix", "#17becf"),
                             ("oracle", "Oracle", "#000000")]:
            if ref in rows and rows[ref]["mean"] is not None:
                ax.axvline(rows[ref]["mean"], ls="--", lw=1, color=c,
                           label=f"{name} {rows[ref]['mean']:.2f}")
        lo = min(means + [rows["suffix"]["mean"]]) - 0.1 if rows.get("suffix", {}).get("mean") else min(means) - 0.1
        ax.set_xlim(left=max(0, lo))
        for i, v in enumerate(means):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
        leg1 = ax.legend(loc="lower right", fontsize=8, title="reference lines")
        ax.add_artist(leg1)                          # keep line legend when adding group legend
        from matplotlib.patches import Patch
        gl = [Patch(color=c, label=g) for g, c in gcol.items()]
        ax.legend(handles=gl, loc="upper right", fontsize=7, title="group")
        fig.tight_layout()
        fig.savefig(OUT / "hwts_mat_bars.png", dpi=140)
        print(f"\nfigure -> {OUT/'hwts_mat_bars.png'}")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
