#!/usr/bin/env python3
"""MAT-per-workload extension figures for the method families the user asked
for (2026-07-18). Same skeleton/colors as plot_mat_4way, TEST half of the
disjoint 3-way convlabel split for EVERY bar:

  fixed base : DFlash(single) · Suffix(single) · SD-paper hybrid · Compose(raw)
  { method version bars — inserted here, best per workload/category }
  fixed tail : Oracle(best handoff)

Families (--family):
  weight     head-weight (min(1,u*conf)) / tail-weight (w*rawscore)   [head+tail
             auto-absorbed if within NOISE of tail-weight everywhere]
  calib      ONLINE windowed calibration: head-calib / tail-calib / head+tail;
             each bar picks the best (type, window) over head{logistic,beta,
             linear} x tail{linear,isotonic} x window{1k,4k,8k,cumulative}
  smoothing  classic Laplace (k+1)/(n+2) tail rescore (single bar, 0-param)
  combined   weight-best / calib-best / smoothing-best (each family's per-column
             optimum, side by side)

Every method bar is annotated with its per-column winning hyperparameter.
Two figures per family: per-workload (5 columns) and per-category (all subtask
columns across the 5 workloads, workload separators).

Data:
  base    readable_outputs/figures/replay_logs/mat_{ds}_4way_{singles,calib_raw,
          oracle}_split.replay.txt  + pipeline_deployable/segments fallback sweep
  method  readable_outputs/figures/replay_logs/mat_{ds}_{mfam|hwts}_{arm}_split.replay.txt

  python3 scripts/plot/plot_method_families.py --family weight [--dry]
"""
from __future__ import annotations
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RL = BASE / "readable_outputs" / "figures" / "replay_logs"
OUT = BASE / "readable_outputs" / "figures" / "mat" / "method_families"
FB_DIR = Path("/workspace/simulation/results/pipeline_deployable/segments")
FB_DIR_FALLBACK = Path("/workspace/simulation/results/pipeline_4way/segments")

DS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
DS_NAME = {"specbench": "Spec-Bench\n480 tasks · 6 subtasks",
           "bfcl": "BFCL v4 · bfcl_eval\n753 tasks · 5 categories",
           "swebench": "SWE-bench Verified · mini-swe-agent\n250-step · repos",
           "spider": "Spider2-DBT · spider-agent-dbt\n68 tasks · 68 databases",
           "tau2": "τ²-bench · tau2 sim\n64 tasks · 3 domains"}
NOISE = 0.10   # |A - B| <= NOISE everywhere -> absorb A into simpler B
LOGTAG = {}    # ds -> filename tag override (e.g. swebench -> swelite for --swelite)


def _tag(ds):
    return LOGTAG.get(ds, ds)

# specbench MT-bench per-question categories pooled into one mt_bench column
MT_CATS = {"writing", "roleplay", "reasoning", "math", "coding",
           "extraction", "stem", "humanities"}
SUB_ORDER = {"specbench": ["mt_bench", "translation", "summarization", "qa",
                           "math_reasoning", "rag"],
             "bfcl": ["web_search", "web_search_no_snippet", "memory_kv",
                      "memory_rec_sum", "memory_vector"],
             "tau2": ["airline", "retail", "telecom"]}
# swebench: repos (size-desc, top N); spider: single pooled column
SWE_TOPN = 8

# base bar spec: (key, label, color)
BASE_BARS = [("dflash", "DFlash (single)", "#4C78A8"),
             ("suffix", "Suffix (single)", "#F58518"),
             ("fallback", "SD-paper hybrid", "#9467BD"),
             ("compose_raw", "Compose (raw)", "#cfe8c8")]
ORACLE_BAR = ("oracle", "Oracle (best handoff)", "#E45756")

# ---- latency / cost model (LEGACY measured; fresh GPU0 re-measure = TODO) -----
# results/legacy/latency/method_compare/summary.json — Qwen3.5-27B, num_spec≈32.
# step_ms = draft_ms + verify_ms(+others); vanilla AR per-token = one target fwd.
# BUDGET-AWARE (first-order, one measured point B0=32): draft scales ~linearly with
# the draft-token budget; verify = vanilla forward + a small budget term (27B verify
# is memory-bandwidth bound, so nearly flat in B). A full budget×verify sweep is part
# of the latency TODO.
# LEGACY is the OFFICIAL latency setting (reported results are legacy-based; the
# fresh 2026-07-19 GPU0 measurement was discarded for consistency — it differed by
# only ~1.6% speedup / ~3% throughput and left every ranking unchanged).
_LAT_PATH = BASE.parent / "results" / "legacy" / "latency" / "method_compare" / "summary.json"
_LAT_SRC = "legacy"
_B0 = 32
try:
    _LAT = json.load(open(_LAT_PATH))["Qwen/Qwen3.5-27B"]
    VANILLA_MS = _LAT["Suffix"]["verify_ms"]          # single target forward (1 tok)
except Exception:
    _LAT, VANILLA_MS, _LAT_SRC = {}, 39.42, "fallback"


def step_ms(method, budget):
    """per-round latency of a proposer class at a draft-token budget (ms)."""
    e = _LAT.get(method) or _LAT.get("DFlash")
    if not e:
        return VANILLA_MS
    f = budget / _B0
    draft = e["draft_ms"] * f
    verify = VANILLA_MS + (e["verify_ms"] - VANILLA_MS) * f
    return draft + verify + e.get("others_ms", 0.0)


def _fb_share(ds):
    for d in (FB_DIR, FB_DIR_FALLBACK):
        fp = d / f"fallback_sweep_fresh_{ds}.json"
        if not fp.exists():
            continue
        taus = next(iter(json.load(open(fp)).values()))
        if set(taus) == {"calib", "test"}:
            cal, tst = taus["calib"], taus["test"]
            bt = max(cal, key=lambda t: cal[t]["K"])
            return float(tst[bt].get("suffix_share", 0.0))
        _, v = max(taus.items(), key=lambda kv: kv[1]["K"])
        return float(v.get("suffix_share", 0.0))
    return 0.0


def bar_step_ms(sk, ds, budget):
    """per-round latency of a bar: suffix=tree round, fallback=mixed, else DFlash round."""
    if sk == "suffix":
        return step_ms("Suffix", budget)
    if sk == "fallback":
        s = _fb_share(ds)
        return s * step_ms("Suffix", budget) + (1 - s) * step_ms("DFlash", budget)
    return step_ms("DFlash", budget)      # dflash / compose_raw / method arms / oracle


def transform_metric(mat, sk, ds, metric, budget):
    """MAT -> throughput (tok/s) or speedup (x vs vanilla AR); mat<=0 -> 0."""
    if mat <= 0:
        return 0.0
    st = bar_step_ms(sk, ds, budget)
    if metric == "throughput":
        return (1.0 + mat) * 1000.0 / st          # accepted+1 tokens per round
    if metric == "speedup":
        return (1.0 + mat) * VANILLA_MS / st       # / vanilla per-token latency
    return mat


# ---------------------------------------------------------------- parsing ----
def parse_log(path: Path):
    """-> (overall {prop:(K,rounds)}, by_task {task:{prop:(K,rounds)}})."""
    overall, by_task = {}, {}
    if not path.exists():
        return overall, by_task
    for ln in path.read_text().splitlines():
        m = re.match(r"\s+(\w+): K=([0-9.]+)\s+\(rounds=(\d+)\)", ln)
        if m:
            overall[m.group(1)] = (float(m.group(2)), int(m.group(3)))
            continue
        m = re.match(r"\s+\[(.+?)\]\s+(.*)", ln)
        if m:
            for p, v, r in re.findall(r"(\w+)=([0-9.]+)\((\d+)\)", m.group(2)):
                by_task.setdefault(m.group(1), {})[p] = (float(v), int(r))
    return overall, by_task


def arm_paths(ds, arm):
    for tag in ("mfam", "hwts"):
        p = RL / f"mat_{_tag(ds)}_{tag}_{arm}_split.replay.txt"
        if p.exists():
            yield p


def arm_data(ds, arm):
    """calib arm -> (overall_K, rounds, by_task{cat:(K,rounds)}); None if absent."""
    for p in arm_paths(ds, arm):
        ov, bt = parse_log(p)
        if "calib" in ov:
            bt2 = {c: d["calib"] for c, d in bt.items() if "calib" in d}
            return ov["calib"][0], ov["calib"][1], bt2
    return None


# ------------------------------------------------------------ base bars ----
def load_fallback():
    """SD-paper hybrid (round-level score-threshold fallback) at pooled/calib-best
    tau. Prefer the split sweep (tau* on calib half, K on test half); fall back
    to the non-split pooled-best sweep. -> {ds:(tau, K, rounds, by_task)}."""
    out = {}
    for ds in DS:
        for d, split in ((FB_DIR, True), (FB_DIR_FALLBACK, False)):
            fp = d / f"fallback_sweep_fresh_{_tag(ds)}.json"
            if not fp.exists():
                continue
            taus = next(iter(json.load(open(fp)).values()))
            if set(taus) == {"calib", "test"}:
                cal, tst = taus["calib"], taus["test"]
                bt = max(cal, key=lambda t: cal[t]["K"])
                e = tst[bt]
                out[ds] = (float(bt), e["K"], e.get("rounds", 0),
                           {c: (kn[0], kn[1]) for c, kn in e.get("by_task", {}).items()})
            else:
                bt, e = max(taus.items(), key=lambda kv: kv[1]["K"])
                out[ds] = (float(bt), e["K"], e.get("rounds", 0),
                           {c: (kn[0], kn[1]) for c, kn in e.get("by_task", {}).items()})
            break
    return out


def load_base():
    """-> K[ds][key]=(val,rounds), BT[ds][key][cat]=(val,rounds) for the 5 base
    bars (dflash/suffix/fallback/compose_raw + oracle)."""
    K, BT = {}, {}
    fb = load_fallback()
    fb_tau = {}
    for ds in DS:
        K[ds], BT[ds] = {}, {}
        ov_s, bt_s = parse_log(RL / f"mat_{_tag(ds)}_4way_singles_split.replay.txt")
        for p in ("dflash", "suffix"):
            if p in ov_s:
                K[ds][p] = ov_s[p]
                BT[ds][p] = {c: d[p] for c, d in bt_s.items() if p in d}
        ov_c, bt_c = parse_log(RL / f"mat_{_tag(ds)}_4way_calib_raw_split.replay.txt")
        if "calib" in ov_c:
            K[ds]["compose_raw"] = ov_c["calib"]
            BT[ds]["compose_raw"] = {c: d["calib"] for c, d in bt_c.items() if "calib" in d}
        ov_o, bt_o = parse_log(RL / f"mat_{_tag(ds)}_4way_oracle_split.replay.txt")
        if "oracle" in ov_o:
            K[ds]["oracle"] = ov_o["oracle"]
            BT[ds]["oracle"] = {c: d["oracle"] for c, d in bt_o.items() if "oracle" in d}
        if ds in fb:
            tau, kk, rr, bt = fb[ds]
            K[ds]["fallback"] = (kk, rr)
            BT[ds]["fallback"] = bt
            fb_tau[ds] = tau
    return K, BT, fb_tau


# ---------------------------------------------------- category structure ----
def pool_cats(bt):
    """rounds-weighted-pool MT-bench cats into mt_bench inside a by_task dict."""
    mt = [(c, v) for c, v in bt.items() if c in MT_CATS]
    if not mt:
        return bt
    r = sum(n for _, (_, n) in mt)
    pooled = (sum(k * n for _, (k, n) in mt) / r, r) if r else (0.0, 0)
    out = {c: v for c, v in bt.items() if c not in MT_CATS}
    out["mt_bench"] = pooled
    return out


def categories(ds, base_bt):
    """ordered category keys for a workload's per-category figure."""
    if ds == "spider":
        return ["all"]                      # pooled single column (noisy 68 DBs)
    pooled = pool_cats(base_bt.get("dflash", {}))
    if ds in SUB_ORDER:
        cats = [c for c in SUB_ORDER[ds] if c in pooled]
        cats += [c for c in sorted(pooled) if c not in cats]
        return cats
    # swebench: repos, size-desc by dflash rounds, top N
    order = sorted(pooled, key=lambda c: -pooled[c][1])
    return order[:SWE_TOPN]


def cat_val(bt, cat):
    """(K,rounds) of a by_task dict at a (possibly pooled) category; None if absent."""
    if cat == "all":                        # spider: overall lives under the arm K
        return None
    pooled = pool_cats(bt)
    return pooled.get(cat)


# --------------------------------------------------------- family specs ----
def _rng(lo, hi):
    return None


def weight_family():
    """versions: head-weight / tail-weight / head+tail-weight (absorption checked)."""
    head = [(f, u) for f, u in [
        ("hs10_rawtail", 1.0), ("hs12_rawtail", 1.2), ("hs14_rawtail", 1.4),
        ("hs16_rawtail", 1.6), ("hs18_rawtail", 1.8), ("hs20_rawtail", 2.0),
        ("hs25_rawtail", 2.5), ("hs30_rawtail", 3.0), ("hs40_rawtail", 4.0),
        ("hs50_rawtail", 5.0), ("hs60_rawtail", 6.0), ("hs80_rawtail", 8.0),
        ("hs120_rawtail", 12.0), ("hs200_rawtail", 20.0)]]
    tail = [(f, w) for f, w in [
        ("rawhead_fix001", 0.01), ("rawhead_fix002", 0.02), ("rawhead_fix003", 0.03),
        ("rawhead_fix005", 0.05), ("rawhead_fix0075", 0.075), ("rawhead_fix010", 0.10),
        ("rawhead_fix0125", 0.125), ("rawhead_fix015", 0.15), ("rawhead_fix020", 0.20)]]
    both = [("h14w005", (1.4, 0.05)), ("h14w0075", (1.4, 0.075)), ("h14w010", (1.4, 0.10)),
            ("h16w005", (1.6, 0.05)), ("h16w0075", (1.6, 0.075)), ("h16w010", (1.6, 0.10)),
            ("twoscalar_h14w0125", (1.4, 0.125)), ("h25w003", (2.5, 0.03)),
            ("h25w005", (2.5, 0.05))]
    return {
        "head": ("head-weight", "#5FA85F",
                 [(a, f"u={u:g}") for a, u in head]),
        "tail": ("tail-weight", "#1B5E20",
                 [(a, f"w={w:g}") for a, w in tail]),
        "both": ("head+tail-weight", "#17A398",
                 [(a, f"u={uw[0]:g},w={uw[1]:g}") for a, uw in both]),
    }


_WIN = [("1k", "1K"), ("4k", "4K"), ("8k", "8K"), ("cum", "cumul")]
_HEADN = {"log": "logistic", "beta": "beta", "lin": "linear"}
_TAILN = {"lin": "linear", "iso": "isotonic"}


def calib_family():
    # window {1k,4k,8k,cumulative} is still swept to PICK the best arm, but the
    # winning window is NOT surfaced in the annotation (user: hide the window).
    head = []
    for h in ("log", "beta", "lin"):
        for wk, wl in _WIN:
            head.append((f"onl_{h}_raw_w{wk}", f"{_HEADN[h]}"))
    tail = []
    for t in ("lin", "iso"):
        for wk, wl in _WIN:
            tail.append((f"onl_raw_{t}_w{wk}", f"{_TAILN[t]}"))
    both = []
    for h in ("log", "beta", "lin"):
        for t in ("lin", "iso"):
            for wk, wl in _WIN:
                both.append((f"onl_{h}_{t}_w{wk}", f"{_HEADN[h]}+{_TAILN[t]}"))
    return {
        "head": ("head-calib (online)", "#5FA85F", head),
        "tail": ("tail-calib (online)", "#1B5E20", tail),
        "both": ("head+tail-calib (online)", "#17A398", both),
    }


def smoothing_family():
    # classic Laplace (k+1)/(n+2), raw head, 0-param -> single candidate
    return {"laplace": ("Laplace (k+1)/(n+2)", "#5FA85F",
                        [("rawhead_succ", "λ=1")])}


# genbeta (k+a)/(n+b) arms that exist in the hwts sweep, (a,b) hand-decoded.
_GENBETA = [(0.25, 1), (0.3, 1), (0.375, 1.5), (0.45, 1.5), (0.2, 2), (0.3, 2),
            (0.4, 2), (0.5, 2), (0.6, 2), (0.7, 2), (0.8, 2), (1, 2), (2, 2),
            (4, 2), (0.75, 3), (0.9, 3), (1.2, 4), (0.5, 8), (1, 8), (2, 8),
            (4, 8), (0.5, 32), (1, 32), (2, 32), (4, 32)]
_DFPRIOR = [("s025", 0.25), ("s05", 0.5), ("s1", 1), ("s2", 2), ("s4", 4),
            ("s8", 8), ("s16", 16), ("s32", 32)]


def _gb_arm(a, b):
    at = (f"0{str(a).split('.')[1]}" if a < 1 else str(a).replace(".", "")) \
        if a != int(a) else str(int(a))
    bt = str(b).replace(".", "") if b != int(b) else str(int(b))
    return f"rawhead_genbeta_a{at}_b{bt}"


def smoothing_full_family():
    """existing Laplace bar + the a/b (genbeta) and DFlash-prior (dfprior)
    methodologies, each best-per-column over the EXISTING hwts sweep (no new runs)."""
    ab = [(_gb_arm(a, b), f"a{a:g}/b{b:g}") for a, b in _GENBETA]
    return {
        "laplace": ("Laplace (k+1)/(n+2)", "#5FA85F", [("rawhead_succ", "λ=1")]),
        "ab": ("a/b prior (k+a)/(n+b)", "#1B5E20", ab),
    }


def combined_family():
    w = weight_family()
    c = calib_family()
    s = smoothing_full_family()          # smoothing pool = Laplace + a/b prior
    wc = [(a, "wt " + lab) for _, _, cand in w.values() for a, lab in cand]
    cc = [(a, "cal " + lab) for _, _, cand in c.values() for a, lab in cand]
    sc = [(a, "sm " + lab) for _, _, cand in s.values() for a, lab in cand]
    return {
        "weight": ("Weight (best)", "#5FA85F", wc),
        "calib": ("Calib (best)", "#1B5E20", cc),
        "smooth": ("Smoothing (best)", "#17A398", sc),
    }


FAMILIES = {"weight": weight_family, "calib": calib_family,
            "smoothing": smoothing_family, "smoothing_full": smoothing_full_family,
            "combined": combined_family}


# --------------------------------------------------------- selection ----
def best_overall(ds, candidates):
    """-> (K, rounds, param_label, arm) picking max overall K over candidates."""
    best = None
    for arm, lab in candidates:
        d = arm_data(ds, arm)
        if d is None:
            continue
        if best is None or d[0] > best[0]:
            best = (d[0], d[1], lab, arm)
    return best


def best_cat(ds, cat, candidates):
    """-> (K, rounds, param_label, arm) picking max per-category K."""
    best = None
    for arm, lab in candidates:
        d = arm_data(ds, arm)
        if d is None:
            continue
        if cat == "all":
            v = (d[0], d[1])
        else:
            v = cat_val(d[2], cat)
        if v is None:
            continue
        if best is None or v[0] > best[0]:
            best = (v[0], v[1], lab, arm)
    return best


def _full_cats(ds, BT):
    """ALL category labels of a workload (mt_bench pooled); spider = single 'all'."""
    if ds == "spider":
        return ["all"]
    return list(pool_cats(BT[ds].get("compose_raw", {})).keys())


def catavg_val(ds, sk, K, BT, fam):
    """rounds-weighted mean over the workload's categories of bar sk's per-category
    value (per-category OPTIMUM for method bars). MAT units; transform applied later."""
    num = den = 0.0
    for cat in _full_cats(ds, BT):
        if sk in ("dflash", "suffix", "fallback", "compose_raw", "oracle"):
            vv = K[ds].get(sk) if cat == "all" else cat_val(BT[ds].get(sk, {}), cat)
        else:
            b = best_cat(ds, cat, fam[sk][2])
            vv = (b[0], b[1]) if b else None
        if vv:
            num += vv[0] * vv[1]; den += vv[1]
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--dry", action="store_true",
                    help="print the selected table, no plotting (host-testable)")
    ap.add_argument("--global-fixed", action="store_true",
                    help="ONE weight/config per version, fixed across ALL workloads & "
                         "categories = the candidate with the best MEAN MAT over the 5 "
                         "workloads (the unified-weight baseline). Output *_globalfixed.")
    ap.add_argument("--metric", default="mat", choices=["mat", "throughput", "speedup"],
                    help="bar quantity: mat | throughput (tok/s, legacy 27B latency) | "
                         "speedup (x vs vanilla AR). Output *_throughput / *_speedup.")
    ap.add_argument("--budget", type=int, default=_B0,
                    help="draft-token budget for the latency model (verify/draft scale "
                         "with it); default 32 = the num_spec of the MAT replays.")
    ap.add_argument("--presentation", action="store_true",
                    help="PPT variant -> method_families/presentation/ : per-workload "
                         "only, NO title, bars = DFlash/Suffix/select(=SD hybrid)/"
                         "Compose(raw)/{methods} (no Oracle).")
    ap.add_argument("--catavg", action="store_true",
                    help="per-workload bar = rounds-weighted mean over the workload's "
                         "CATEGORIES of the per-category optimum (pick best per category "
                         "first, then average). No config annotation. Output *_catavg.")
    ap.add_argument("--drop-head", action="store_true",
                    help="drop the 'head' version from the family (e.g. weight: keep only "
                         "tail-weight as the method bar). Output *_nohead.")
    ap.add_argument("--swelite", action="store_true",
                    help="use SWE-bench Lite in the swebench slot (reads swelite-tagged "
                         "logs; other 4 workloads unchanged). Output *_swelite.")
    args = ap.parse_args()

    if args.swelite:
        LOGTAG["swebench"] = "swelite"
        DS_NAME["swebench"] = "SWE-bench Lite · mini-swe-agent\n293 instances · repos"

    fam = FAMILIES[args.family]()
    if args.drop_head and "head" in fam:
        del fam["head"]
    K, BT, fb_tau = load_base()

    # --global-fixed: collapse each version to the single candidate with the best
    # workload-mean MAT, then reuse the normal per-column selection (trivial: 1 arm).
    gf = ""
    if args.global_fixed:
        for ver, (lab, col, cand) in list(fam.items()):
            scored = []
            for arm, albl in cand:
                ks = [arm_data(ds, arm) for ds in DS]
                if all(k is not None for k in ks):
                    scored.append((sum(k[0] for k in ks) / len(DS), arm, albl))
            if scored:
                m, arm, albl = max(scored, key=lambda t: t[0])
                fam[ver] = (lab, col, [(arm, albl)])
                gf += f"  {ver}={albl}(mean {m:.2f})"

    # per-workload selection
    sel = {}   # sel[ds][ver] = (K, rounds, param, arm)
    for ds in DS:
        sel[ds] = {}
        for ver, (lab, col, cand) in fam.items():
            sel[ds][ver] = best_overall(ds, cand)

    # absorption rule (general): a more-complex version (head+tail) is absorbed
    # into the simpler single-side version it matches within NOISE at EVERY column.
    absorbed = {}
    if "both" in fam:
        for simpler in ("tail", "head"):        # prefer collapsing onto tail
            if simpler not in fam:
                continue
            deltas = [(ds, sel[ds]["both"][0] - sel[ds][simpler][0])
                      for ds in DS if sel[ds].get("both") and sel[ds].get(simpler)]
            if deltas and all(abs(dv) <= NOISE for _, dv in deltas):
                absorbed["both"] = (simpler, deltas)
                break

    # ---- report ----
    print(f"=== FAMILY: {args.family} ===  (TEST-half three-way convlabel)")
    hdr = ["workload"] + [k for k in ("dflash", "suffix", "fallback", "compose_raw")] \
        + list(fam) + ["oracle"]
    print("  " + " | ".join(f"{h:>10.10s}" for h in hdr))
    for ds in DS:
        row = [ds]
        for b in ("dflash", "suffix", "fallback", "compose_raw"):
            row.append(f"{K[ds].get(b, (0,0))[0]:.2f}")
        for ver in fam:
            s = sel[ds][ver]
            row.append(f"{s[0]:.2f}[{s[2]}]" if s else "--")
        row.append(f"{K[ds].get('oracle', (0,0))[0]:.2f}")
        print("  " + " | ".join(f"{c:>10.10s}" for c in row))
    if fb_tau:
        print("  fallback τ*:", {d: f"{t:g}" for d, t in fb_tau.items()})
    if absorbed:
        print("  ABSORBED:", {k: f"into {v[0]} (Δ={[round(x,3) for _,x in v[1]]})"
                              for k, v in absorbed.items()})
    missing = [(ds, ver) for ds in DS for ver in fam if sel[ds][ver] is None]
    if missing:
        print("  MISSING (no arm data yet):", missing[:20], "..." if len(missing) > 20 else "")
    if gf:
        print("  GLOBAL-FIXED (best workload-mean):" + gf)

    if args.dry:
        return

    suffix = "_globalfixed" if args.global_fixed else ""
    suffix += "_nohead" if args.drop_head else ""
    suffix += "_catavg" if args.catavg else ""
    suffix += "_swelite" if args.swelite else ""
    _plot(args.family, fam, K, BT, sel, fb_tau, absorbed,
          suffix=suffix, metric=args.metric, budget=args.budget,
          pres=args.presentation, catavg=args.catavg)


def _plot(family, fam, K, BT, sel, fb_tau, absorbed, suffix="", metric="mat", budget=32,
          pres=False, catavg=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = (OUT / "presentation") if pres else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    gfix = "_globalfixed" in suffix
    prefix = {"mat": "MAT", "throughput": "THROUGHPUT", "speedup": "SPEEDUP"}[metric]
    show_vers = [v for v in fam if v not in absorbed]
    # presentation: SD-paper hybrid -> "select", no Oracle bar
    base = [(k, "select" if (pres and k == "fallback") else lab, c)
            for k, lab, c in BASE_BARS]
    series = base + [(v, fam[v][0], fam[v][1]) for v in show_vers] \
        + ([] if pres else [ORACLE_BAR])
    ylabel = {"mat": "mean accept length  (tokens)",
              "throughput": "decode throughput  (tokens / s)",
              "speedup": "decode speedup  (× vs vanilla AR)"}[metric]
    fmt = {"mat": "{:.2f}", "throughput": "{:.0f}", "speedup": "{:.2f}×"}[metric]
    mnote = ("" if metric == "mat" else
             f"\n{_LAT_SRC} Qwen3.5-27B latency; budget={budget} draft-tok · "
             f"vanilla AR = {1000/VANILLA_MS:.1f} tok/s ({VANILLA_MS:.1f} ms/tok); "
             f"DFlash round {step_ms('DFlash', budget):.1f} ms, Suffix {step_ms('Suffix', budget):.1f} ms")
    tv = (lambda base_fn: (lambda key, sk:
          transform_metric(base_fn(key, sk), sk, key if isinstance(key, str) else key[0],
                           metric, budget)))

    # ---------- figure 1: per-workload ----------
    wl_values = (tv(lambda ds, sk: catavg_val(ds, sk, K, BT, fam)) if catavg
                 else tv(lambda ds, sk: _wl_val(ds, sk, K, sel)))
    # catavg has no single config per bar -> no hyperparameter annotation (τ* kept)
    wl_annot = ((lambda ds, sk: (f"τ*={fb_tau.get(ds):g}" if sk == "fallback"
                                 and fb_tau.get(ds) is not None else "")) if catavg
                else (lambda ds, sk: _wl_annot(ds, sk, sel, fb_tau, family)))
    _draw(plt, [(ds, DS_NAME[ds]) for ds in DS], series,
          values=wl_values, annot=wl_annot,
          out=outdir / f"{prefix}_{family}_per_workload{suffix}.png",
          title="" if pres else _title(family, absorbed, gfix=gfix, metric=metric) + mnote,
          per_col=2.9, fs=8, ylabel=ylabel, fmt=fmt)

    if pres:
        if not catavg:                           # tables are per-category (catavg n/a)
            _tables(plt, family, fam, K, BT, fb_tau, absorbed, metric, budget, outdir, prefix, suffix)
        return                                    # presentation: per-workload only
    if catavg:
        return                                    # catavg only changes the per-workload bar

    # ---------- figure 2: per-category ----------
    cols, collabels, seps = [], [], []
    for i, ds in enumerate(DS):
        cats = categories(ds, BT[ds])
        for c in cats:
            cols.append((ds, c))
            collabels.append(_catlabel(ds, c))
        if i < len(DS) - 1:
            seps.append(len(cols) - 0.5)
    _draw(plt, list(zip(cols, collabels)), series,
          values=tv(lambda col, sk: _cat_val(col, sk, K, BT, fam)),
          annot=lambda col, sk: _cat_annot(col, sk, fam, family),
          out=outdir / f"{prefix}_{family}_per_category{suffix}.png",
          title=_title(family, absorbed, gfix=gfix, metric=metric) + " — per category" + mnote,
          per_col=1.15, fs=6.5, seps=seps, ylabel=ylabel, fmt=fmt,
          wl_bands=[(ds, categories(ds, BT[ds])) for ds in DS])


def _title(family, absorbed, gfix=False, metric="mat"):
    names = {"weight": "Weight (per-workload optimal head/tail scalar)",
             "calib": "Online calibration (per-workload optimal calibrator type)",
             "smoothing": "Laplace smoothing (k+1)/(n+2) tail rescore",
             "smoothing_full": "Smoothing — Laplace / a-b prior (each best)",
             "combined": "Combined — weight / calib / smoothing (each best)"}
    if gfix and family == "weight":
        names["weight"] = "Weight (ONE global-fixed head/tail scalar — best workload-mean)"
    metricword = {"mat": "MAT", "throughput": "throughput", "speedup": "speedup"}[metric]
    t = f"Dr.Lee extension — {metricword} per workload  ::  " + names[family]
    if absorbed:
        t += "\n(head+tail-weight absorbed into tail-weight: within ±%.2f everywhere)" % NOISE
    return t


def _wl_val(ds, sk, K, sel):
    if sk in ("dflash", "suffix", "fallback", "compose_raw", "oracle"):
        return K[ds].get(sk, (0.0, 0))[0]
    s = sel[ds].get(sk)
    return s[0] if s else 0.0


def _wl_annot(ds, sk, sel, fb_tau, family):
    if sk == "fallback":
        t = fb_tau.get(ds)
        return f"τ*={t:g}" if t is not None else ""
    if sk in ("dflash", "suffix", "compose_raw", "oracle"):
        return ""
    s = sel[ds].get(sk)
    return s[2] if s else ""


def _cat_val(col, sk, K, BT, fam):
    ds, cat = col
    if sk in ("dflash", "suffix", "fallback", "compose_raw", "oracle"):
        if cat == "all":
            return K[ds].get(sk, (0.0, 0))[0]
        v = cat_val(BT[ds].get(sk, {}), cat)
        return v[0] if v else 0.0
    b = best_cat(ds, cat, fam[sk][2])
    return b[0] if b else 0.0


def _cat_annot(col, sk, fam, family):
    ds, cat = col
    if sk in ("dflash", "suffix", "fallback", "compose_raw", "oracle"):
        return ""
    b = best_cat(ds, cat, fam[sk][2])
    return b[2] if b else ""


def _catlabel(ds, cat):
    return {"mt_bench": "MT-bench", "math_reasoning": "math", "web_search": "web",
            "web_search_no_snippet": "web(no-snip)", "memory_kv": "mem-kv",
            "memory_rec_sum": "mem-recsum", "memory_vector": "mem-vec",
            "summarization": "summ", "all": ds}.get(cat, cat)[:14]


_VER_HEADER = {
    "weight": {"head": "head weight", "tail": "tail weight", "both": "both weight"},
    "calib": {"head": "head calib.", "tail": "tail calib.", "both": "both calib."},
    "smoothing": {"laplace": "Laplace"},
    "smoothing_full": {"laplace": "Laplace", "ab": "a/b prior"},
    "combined": {"weight": "weight", "calib": "calib", "smooth": "smoothing"},
}


def _ver_header(family, v):
    return _VER_HEADER.get(family, {}).get(v, v)


def _tables(plt, family, fam, K, BT, fb_tau, absorbed, metric, budget, outdir, prefix, suffix):
    """per-category rendered as clean table IMAGES (presentation). Two column sets:
    type1 = DFlash/Suffix/select/{methods} (no raw); type2 = Compose(raw)/{methods}.
    Cells also show the chosen config below the value (select -> τ, methods -> their
    winning hyperparameter e.g. tail-weight w, calib type, a/b prior)."""
    show_vers = [v for v in fam if v not in absorbed]
    fmt = {"mat": "{:.2f}", "throughput": "{:.0f}", "speedup": "{:.2f}×"}[metric]
    rows = [(ds, cat) for ds in DS for cat in categories(ds, BT[ds])]

    def cell(ds, cat, sk):
        """-> (value, config-string). config: select -> τ*, method -> winning cfg."""
        if sk in ("dflash", "suffix", "fallback", "compose_raw"):
            mat = _cat_val((ds, cat), sk, K, BT, fam)
            cfg = (f"τ{fb_tau[ds]:g}" if sk == "fallback" and ds in fb_tau else "")
        else:
            b = best_cat(ds, cat, fam[sk][2])
            mat = b[0] if b else 0.0
            cfg = b[2] if b else ""
        return transform_metric(mat, sk, ds, metric, budget), cfg

    base1 = [("dflash", "DFlash"), ("suffix", "Suffix"), ("fallback", "select")]
    base2 = [("compose_raw", "Compose(raw)")]
    meth = [(v, _ver_header(family, v)) for v in show_vers]
    shade = {ds: c for ds, c in zip(DS, ["#eef3f8", "#fef0e6", "#eef7ec",
                                         "#f3eef8", "#fdecec"])}
    for typ, cols in (("type1", base1 + meth), ("type2", base2 + meth)):
        collabels = ["workload", "category"] + [h for _, h in cols]
        text, rcolors = [], []
        for ds, cat in rows:
            r = [ds, _catlabel(ds, cat)]
            for sk, _ in cols:
                v, cfg = cell(ds, cat, sk)
                s = fmt.format(v) if v > 0 else "–"
                r.append(f"{s}\n{cfg}" if (v > 0 and cfg) else s)
            text.append(r)
            rcolors.append(shade[ds])
        fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(collabels), 0.46 * len(rows) + 0.6))
        ax.axis("off")
        tab = ax.table(cellText=text, colLabels=collabels, loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.5)
        tab.scale(1, 2.0)
        for (ri, ci), cellobj in tab.get_celld().items():
            if ri == 0:
                cellobj.set_facecolor("#333"); cellobj.set_text_props(color="w", fontweight="bold")
            else:
                cellobj.set_facecolor(rcolors[ri - 1])
                if ci <= 1:
                    cellobj.set_text_props(fontweight="bold")
            cellobj.set_edgecolor("#ccc")
        out = outdir / f"{prefix}_{family}_per_category_{typ}{suffix}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


def _draw(plt, columns, series, values, annot, out, title, per_col, fs,
          seps=None, wl_bands=None, ylabel="mean accept length  (tokens)", fmt="{:.2f}"):
    n = len(series)
    ncol = len(columns)
    g = 0.82
    bw = g / n
    fig, ax = plt.subplots(figsize=(2.6 + per_col * ncol, 5.8))
    ymax = 0.0
    for pi, (sk, lab, color) in enumerate(series):
        xs = [ci - g / 2 + bw * (pi + 0.5) for ci in range(ncol)]
        ys = [values(_ckey(col), sk) for col in columns]
        ymax = max([ymax] + ys)
        ax.bar(xs, ys, width=bw * 0.9, color=color, label=lab)
        for x, y in zip(xs, ys):
            if y > 0:
                ax.text(x, y + ymax * 0.006, fmt.format(y), ha="center", va="bottom",
                        fontsize=fs - 1.5, color="#555")
    # hyperparameter annotations (above the bar, colored)
    for pi, (sk, lab, color) in enumerate(series):
        xs = [ci - g / 2 + bw * (pi + 0.5) for ci in range(ncol)]
        for x, col in zip(xs, columns):
            a = annot(_ckey(col), sk)
            if a:
                y = values(_ckey(col), sk)
                ax.text(x, y + ymax * 0.045, a, ha="center", va="bottom",
                        fontsize=fs - 1, fontweight="bold", color=color, rotation=90)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([col[1] for col in columns], fontsize=fs, rotation=0)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, ymax * 1.30)
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if seps:
        for xb in seps:
            ax.axvline(xb, color="#BBB", lw=0.8, ls=":")
    if wl_bands:
        # workload band labels along the top; legend lifted ABOVE the axes so it
        # never overlaps the leftmost band label.
        start = 0
        for ds, cats in wl_bands:
            mid = start + (len(cats) - 1) / 2
            ax.text(mid, ymax * 1.27, ds, ha="center", va="top", fontsize=fs + 2,
                    fontweight="bold", color="#333")
            start += len(cats)
        ax.legend(fontsize=8.5, frameon=False, loc="lower left",
                  bbox_to_anchor=(0.0, 1.02), ncol=len(series))
    else:
        ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=3)
    ax.set_title(title, fontsize=11, y=1.06 if wl_bands else 1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _ckey(col):
    """column identity passed to values()/annot(): ds (per-workload) or (ds,cat)."""
    c = col[0]
    return c


def col_key(col):
    return col


if __name__ == "__main__":
    main()
