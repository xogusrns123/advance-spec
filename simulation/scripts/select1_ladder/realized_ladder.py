"""REALIZED served MAT ladder (the deployment number), replacing the block-anchored
proxy. Pulls accept_length_mean per arm from the served run*.json files (each arm is a
real pinned-trajectory serving run that rebuilds its own draft chain — offline logs
cannot reproduce this). Emits per-cell vertical-bar MAT figures + a printed table with
raw->oracle gap recovery.

WHY realized over block-anchored: block-anchored replays a frozen gt-recording, so it
OVER-counts (e.g. 14B: block-anchored oracle ~5 vs realized 1.78). Realized lets each
arm's picks drive the next proposals -> the true deployment MAT.

Arms (whatever exists in the cell's run*.json): baseline(single dominant), suffix,
select1(raw), select1_calib_{hist,iso,logistic,beta}, select1_mono, select1_bayes,
select1_disc_*, select1_oracle, record(native 3-way), + consensus (separate dir).
"""
import json, glob, os
from collections import OrderedDict, defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "simulation/results/chain_hybrid_perdepth"
OUT = "/tmp/claude-20051/-home-muchwater-advance-spec/24d5967c-83aa-4550-bee6-b36e1dcb2c00/scratchpad"

CELLS = [
    dict(key="14b_2way", title="Qwen3-14B  ·  EAGLE3 + suffix  (2-way)", dir="qwen3_14b_ar"),
    dict(key="27b_2way", title="Qwen3.5-27B  ·  MTP + suffix  (2-way)", dir="qwen35_27b_ar"),
]

# display order + label + colour-group
LADDER = [
    ("suffix",                  "suffix only",            "single"),
    ("baseline",                "single (dominant)",      "single"),
    ("select1",                 "raw  (prob argmax)",     "raw"),
    ("select1_calib_histogram", "calib histogram",        "calib"),
    ("select1_calib_isotonic",  "calib isotonic",         "calib"),
    ("select1_calib_logistic",  "calib logistic",         "calib"),
    ("select1_calib_beta",      "calib beta",             "calib"),
    ("consensus",               "consensus-first",        "rule"),
    ("select1_mono",            "mono (GBM monotone)",    "bayes"),
    ("select1_bayes",           "bayes (GBM)",            "bayes"),
    ("select1_disc_beta",       "disc beta",              "bayes"),
    ("select1_handrule",        "hand-rule (sp>A*ep+B)",  "rule"),
    ("select1_oracle",          "oracle",                 "oracle"),
]
COLORS = {"single":"#9aa7ad","raw":"#c2683a","calib":"#1f7a8c","rule":"#b5892b","bayes":"#6a4c93","oracle":"#2f7a57"}

def collect_mat(cell):
    """arm -> (MAT, n) from the cell's run*.json (first occurrence wins; prefer run.json)."""
    found = OrderedDict()
    files = sorted(glob.glob(f"{ROOT}/{cell['dir']}/run*.json"),
                   key=lambda f: (0 if os.path.basename(f) in ("run.json","run_all-trained.json") else 1, f))
    for f in files:
        try: o = json.load(open(f))
        except Exception: continue
        for arm, v in (o.get("arms") or {}).items():
            if isinstance(v, dict) and v.get("accept_length_mean") is not None and arm not in found:
                found[arm] = (float(v["accept_length_mean"]), v.get("n_samples"))
    # consensus arm from the separate dir (its 'select1' arm == consensus-first selection)
    cdir = cell.get("consensus_dir")
    if cdir:
        for f in sorted(glob.glob(f"{ROOT}/{cdir}/run*.json")):
            try: o = json.load(open(f))
            except Exception: continue
            v = (o.get("arms") or {}).get("select1")
            if isinstance(v, dict) and v.get("accept_length_mean") is not None:
                found["consensus"] = (float(v["accept_length_mean"]), v.get("n_samples")); break
    return found

def make_cell(cell):
    mats = collect_mat(cell)
    bars = [(lbl, mats[arm][0], grp, mats[arm][1]) for arm, lbl, grp in LADDER if arm in mats]
    raw = mats.get("select1", (None,))[0]; orc = mats.get("select1_oracle", (None,))[0]
    print(f"\n### {cell['title']}")
    print(f"   {'arm':22s} {'realized MAT':>12s} {'gap recov':>10s} {'n_tok':>8s}")
    for lbl, v, grp, n in bars:
        rec = ""
        if raw is not None and orc is not None and orc > raw and grp in ("calib","bayes","rule"):
            rec = f"{100*(v-raw)/(orc-raw):+5.0f}%"
        print(f"   {lbl:22s} {v:>12.3f} {rec:>10s} {str(n):>8s}")

    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    xp = np.arange(len(bars)); vals = [b[1] for b in bars]
    ax.bar(xp, vals, color=[COLORS[b[2]] for b in bars], edgecolor="white", width=0.74)
    if raw is not None: ax.axhline(raw, color=COLORS["raw"], ls=":", lw=1, alpha=0.6)
    if orc is not None: ax.axhline(orc, color=COLORS["oracle"], ls=":", lw=1, alpha=0.6)
    for x, b in zip(xp, bars):
        ax.text(x, b[1] + max(vals)*0.01, f"{b[1]:.2f}", va="bottom", ha="center", fontsize=8.5, fontfamily="monospace")
    ax.set_xticks(xp); ax.set_xticklabels([b[0] for b in bars], rotation=30, ha="right", fontsize=8.2)
    ax.set_ylim(0, max(vals)*1.16); ax.set_ylabel("realized MAT (accept length / verify step)")
    ax.set_title(cell["title"], fontsize=13, fontweight="bold", loc="left")
    ax.text(0.0, 1.012, "REALIZED served · held-out (fit 0-29, eval 30-49) · token-pinned (same 89 trajectories) · "
            "all-native proposers · greedy · bfcl_v4 web_search",
            transform=ax.transAxes, fontsize=7.0, color="#5e6e78")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = f"{OUT}/realized_{cell['key']}_mat.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"   -> {out}")

# --- realized selection accuracy (from per-arm pinned decision logs) ---------
# decisive selacc = among positions the arm REACHED where exactly one of
# {eagle, suffix} == gt, fraction where the PICKED token == gt. singles excluded.
# filename CANDIDATES (bare pinned log first; _all-trained fallback). gt_token is logged
# only on PINNED runs, so we pick the candidate that actually has decisive rows -> 14B
# uses bare logs (its _all-trained run was unpinned), 27B uses _all-trained (no bare).
SELACC_LOG = [
    ("select1",                 "raw  (prob argmax)",   "raw",    ["decisions_select1.jsonl"]),
    ("select1_calib_histogram", "calib histogram",      "calib",  ["decisions_select1_calib_histogram.jsonl", "decisions_select1_calib_histogram_all-trained.jsonl"]),
    ("select1_calib_isotonic",  "calib isotonic",       "calib",  ["decisions_select1_calib_isotonic.jsonl", "decisions_select1_calib_isotonic_all-trained.jsonl"]),
    ("select1_calib_logistic",  "calib logistic",       "calib",  ["decisions_select1_calib_logistic.jsonl", "decisions_select1_calib_logistic_all-trained.jsonl"]),
    ("select1_calib_beta",      "calib beta",           "calib",  ["decisions_select1_calib_beta.jsonl", "decisions_select1_calib_beta_all-trained.jsonl"]),
    ("select1_mono",            "mono (GBM monotone)",  "bayes",  ["decisions_select1_mono.jsonl"]),
    ("select1_bayes",           "bayes (GBM)",          "bayes",  ["decisions_select1_bayes.jsonl"]),
    ("select1_disc_beta",       "disc beta",            "bayes",  ["decisions_select1_disc_beta.jsonl"]),
    ("select1_oracle",          "oracle",               "oracle", ["decisions_select1_oracle.jsonl"]),
]

def picked_token(o):
    return o.get("suffix_token") if o.get("chosen") == "suffix" else o.get("eagle_token")

def decisive_selacc(path, d0_only=False):
    hit = tot = 0
    try:
        f = open(path)
    except FileNotFoundError:
        return None
    for line in f:
        o = json.loads(line)
        if o.get("type") != "decision" or o.get("tail"): continue
        if d0_only and o.get("depth") != 0: continue
        gt = o.get("gt_token")
        if gt is None: continue
        et, st = o.get("eagle_token"), o.get("suffix_token")
        avail = [t for t in (et, st) if t is not None]
        hits = [t for t in avail if t == gt]
        if len(avail) >= 2 and 0 < len(hits) < len(avail):   # decisive
            tot += 1
            if picked_token(o) == gt: hit += 1
    return (hit / tot, tot) if tot else None

def make_selacc(cell, d0_only=False):
    d = f"{ROOT}/{cell['dir']}"
    bars = []
    for arm, lbl, grp, cands in SELACC_LOG:
        r = None
        for fn in cands:
            r = decisive_selacc(f"{d}/{fn}", d0_only=d0_only)
            if r is not None: break
        if r is None: continue
        sa, n = r
        bars.append((lbl, sa, grp, n))
    if not bars: return
    scope = "depth-0 only (identical proposals & decisive set across arms — clean A/B)" if d0_only \
            else "decisive positions each arm reached"
    tag = "selacc_d0" if d0_only else "selacc"
    print(f"\n### {cell['title']}  — realized decisive selacc  [{'DEPTH-0' if d0_only else 'ALL DEPTHS'}]")
    for lbl, sa, grp, n in bars:
        print(f"   {lbl:22s} selacc={sa:.4f}  n_decisive={n}")
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    xp = np.arange(len(bars)); vals = [b[1] for b in bars]
    ax.bar(xp, vals, color=[COLORS[b[2]] for b in bars], edgecolor="white", width=0.72)
    raw = next((b[1] for b in bars if b[2] == "raw"), None)
    if raw is not None: ax.axhline(raw, color=COLORS["raw"], ls=":", lw=1, alpha=0.6)
    for x, b in zip(xp, bars):
        ax.text(x, b[1] + max(vals)*0.008, f"{b[1]:.3f}", va="bottom", ha="center", fontsize=8.5, fontfamily="monospace")
    ax.set_xticks(xp); ax.set_xticklabels([b[0] for b in bars], rotation=30, ha="right", fontsize=8.2)
    ax.set_ylim(0, max(vals)*1.12)
    ax.set_ylabel(("depth-0 " if d0_only else "") + "realized decisive selection accuracy")
    ax.set_title(cell["title"], fontsize=13, fontweight="bold", loc="left")
    ax.text(0.0, 1.012, f"REALIZED served · {scope} · held-out · token-pinned · all-native · greedy · "
            "bfcl_v4 web_search", transform=ax.transAxes, fontsize=6.8, color="#5e6e78")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = f"{OUT}/realized_{cell['key']}_{tag}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"   -> {out}")

if __name__ == "__main__":
    for c in CELLS:
        try: make_cell(c)
        except Exception as ex:
            import traceback; print(f"ERR mat {c['key']}: {ex}"); traceback.print_exc()
    for c in CELLS:
        for d0 in (False, True):
            try: make_selacc(c, d0_only=d0)
            except Exception as ex:
                import traceback; print(f"ERR selacc {c['key']} d0={d0}: {ex}"); traceback.print_exc()
