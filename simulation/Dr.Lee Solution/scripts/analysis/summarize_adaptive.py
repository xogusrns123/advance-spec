#!/usr/bin/env python3
r"""Collect the adaptive-scalar (feedback-loop) arm sweep (run_adaptive_scalar.sh)
plus the fixed-champion references and the per-workload-best-fixed adaptation
upper bounds, into one ranked MAT table + a bar figure.

MAT per (workload, arm) is the `calib: K=` line of the split (TEST-half) replay
log. English figure labels (house rule). Run inside sglang-bench:

  docker exec sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace python3 scripts/analysis/summarize_adaptive.py'

--headroom-only prints just the E0 adaptation-UB table (no sweep logs needed).
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

RLOG = Path("readable_outputs/figures/replay_logs")
OUT = Path("results/adaptive_scalar")
DSS = ["specbench", "bfcl", "swebench", "spider", "tau2"]

# adaptive arms: label -> (group, pretty)   (mat_{ds}_adw_{arm}_split.replay.txt)
NEW = {
    "deg0075":       ("V gate", "deg grid{0.075} == fixed w0.075"),
    "onlb_deg008":   ("V gate", "deg grid{0.08} beta == onlbeta w0.08"),
    "cftl_w2k":      ("A cFTL", "censored-FTL w2k"),
    "cftl_w8k":      ("A cFTL", "censored-FTL w8k"),
    "cftl_w32k":     ("A cFTL", "censored-FTL w32k"),
    "cftl_ewma2k":   ("A cFTL", "censored-FTL EWMA hl2k"),
    "cftl2d_w8k":    ("A cFTL", "censored-FTL 2D (u,w) w8k"),
    "ratio_g02":     ("B ratio", "ratio g0.2 w8k"),
    "ratio_g03":     ("B ratio", "ratio g0.3 w8k"),
    "succratio_h14": ("B ratio", "label-free succ g0.6 (u1.4 head)"),
    "succratio_raw": ("B ratio", "label-free succ g0.375 (raw head)"),
    "ftlw_inf":      ("C ceiling", "full-info FTL inf-window"),
    "ftlw_w8k":      ("C ceiling", "full-info FTL w8k"),
    "onlb_cftl_w8k": ("D onlb", "online-beta + censored-FTL w8k"),
    "onlb_gb_cftl":  ("D onlb", "online-beta + gbscale + censored-FTL"),
    "ratio_g04":     ("B ratio", "ratio g0.4 w8k (calib-half gamma*)"),
    "sr_g045":       ("B ratio", "label-free succ g0.45 (raw head)"),
    "sgdgain_e3":    ("E sgd-loss", "SGD gain-residual loss (u,w) eta1e-3"),
    "sgdgain_e4":    ("E sgd-loss", "SGD gain-residual loss (u,w) eta3e-4"),
    "sgdmat_e4":     ("E sgd-loss", "SGD total-accept loss (w) eta1e-4"),
    "sgdmat_e3":     ("E sgd-loss", "SGD total-accept loss (w) eta3e-4"),
    "sgdmat_cens_e4": ("E sgd-loss", "SGD total-accept loss censored"),
}
NEW_TMPL = "mat_{ds}_adw_{arm}_split.replay.txt"

# fixed-champion references from EXISTING logs
BASE = {
    "fix0075":  ("ref fixed", "fixed w=0.075 (raw head)",
                 "mat_{ds}_hwts_rawhead_fix0075_split.replay.txt"),
    "twoscalar": ("ref fixed", "two-scalar u1.4/w0.125",
                  "mat_{ds}_4way_calib_h14tw125_split.replay.txt"),
    "onlb_fix008": ("ref fixed", "online-beta + fixed w=0.08",
                    "mat_{ds}_hwts_onlbeta_fixtw08_split.replay.txt"),
    "sota_gb": ("ref fixed", "SOTA onlbeta+gbscale+w0.16",
                "mat_{ds}_hwts_onlbeta_gbdef_b0125_ots016_split.replay.txt"),
}
SINGLE = {
    "dflash": ("mat_{ds}_4way_singles_split.replay.txt", "dflash"),
    "suffix": ("mat_{ds}_4way_singles_split.replay.txt", "suffix"),
    "oracle": ("mat_{ds}_4way_oracle_split.replay.txt", "oracle"),
}

# E0 adaptation upper bounds: per-workload best over the EXISTING fixed sweeps
UB_SWEEPS = {
    "ub_rawhead": ("UB", "per-wl best fixed w (raw head) — adaptation UB",
                   "mat_{ds}_hwts_rawhead_fix{w}_split.replay.txt",
                   ["005", "0075", "010", "0125", "015", "020"]),
    "ub_onlbeta": ("UB", "per-wl best fixed w (online-beta) — adaptation UB",
                   "mat_{ds}_hwts_onlbeta_fixtw{w}_split.replay.txt",
                   ["0075", "0125", "02", "035", "06", "08", "10", "125", "15", "20"]),
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


def ub_rows():
    rows = {}
    for label, (grp, pretty, tmpl, ws) in UB_SWEEPS.items():
        ks, arg = {}, {}
        for ds in DSS:
            best = None, None
            for w in ws:
                k = read_k(RLOG / tmpl.format(ds=ds, w=w))
                if k is not None and (best[0] is None or k > best[0]):
                    best = k, w
            ks[ds], arg[ds] = best
        have = [v for v in ks.values() if v is not None]
        rows[label] = {"group": grp, "pretty": pretty, "ks": ks, "argbest": arg,
                       "mean": (sum(have) / len(have)) if len(have) == len(DSS) else None,
                       "n": len(have)}
    return rows


def collect():
    rows = {}
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
    rows.update(ub_rows())
    return rows


def main():
    if "--headroom-only" in sys.argv:
        rows = ub_rows()
        for label, r in rows.items():
            print(f"{r['pretty']}: mean={r['mean'] and round(r['mean'], 3)}")
            for ds in DSS:
                print(f"  {ds:10s} best={r['ks'][ds]}  at w={r['argbest'][ds]}")
        return

    rows = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "summary.json", "w"), indent=1)

    complete = {k: v for k, v in rows.items() if v["mean"] is not None}
    ranked = sorted(complete.items(), key=lambda kv: -kv[1]["mean"])
    hdr = f"{'arm':40s} {'group':10s} " + " ".join(f"{d:>9s}" for d in DSS) + f"  {'MEAN':>7s}"
    print(hdr); print("-" * len(hdr))
    for label, r in ranked:
        cells = " ".join(f"{r['ks'][d]:9.2f}" if r['ks'][d] is not None else f"{'-':>9s}" for d in DSS)
        print(f"{r['pretty']:40s} {r['group']:10s} {cells}  {r['mean']:7.3f}")
    incomplete = {k: v for k, v in rows.items() if v["mean"] is None}
    if incomplete:
        print("\nINCOMPLETE (missing some workloads):")
        for k, v in incomplete.items():
            print(f"  {k:24s} n={v['n']}/5")

    # ---- figure: mean-MAT bar, colored by group ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bars = [(label, r) for label, r in ranked if r["group"] not in ("single",)]
        labels = [r["pretty"] for _, r in bars]
        means = [r["mean"] for _, r in bars]
        gcol = {"A cFTL": "#1f77b4", "B ratio": "#2ca02c", "C ceiling": "#ff7f0e",
                "D onlb": "#9467bd", "V gate": "#8c564b", "ref fixed": "#7f7f7f",
                "UB": "#d62728", "E sgd-loss": "#17becf"}
        colors = [gcol.get(r["group"], "#333") for _, r in bars]
        fig, ax = plt.subplots(figsize=(11, 8))
        y = range(len(labels))
        ax.barh(list(y), means, color=colors)
        ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean accepted tokens / round (MAT), test half, 5 workloads")
        ax.set_title("Adaptive (feedback-loop) compose scalars vs fixed champions")
        if rows.get("oracle", {}).get("mean"):
            ax.axvline(rows["oracle"]["mean"], ls="--", lw=1, color="#000",
                       label=f"Oracle {rows['oracle']['mean']:.2f}")
            ax.legend(loc="lower right", fontsize=8)
        lo = min(means) - 0.1
        ax.set_xlim(left=max(0, lo))
        for i, v in enumerate(means):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
        from matplotlib.patches import Patch
        gl = [Patch(color=c, label=g) for g, c in gcol.items()]
        ax.legend(handles=gl, loc="upper right", fontsize=7, title="group")
        fig.tight_layout()
        fig.savefig(OUT / "adaptive_mat_bars.png", dpi=140)
        print(f"\nfigure -> {OUT/'adaptive_mat_bars.png'}")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
