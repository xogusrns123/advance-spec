"""SERVED Panel-B boundary comparison (fair, pinned-trajectory): selection
accuracy + MAT for the 5 SELECTION policies, every bar from its own real served
+pinned decision log (same eval trajectory) — no offline simulation:

  raw            decisions_select1.jsonl
  best calib(M)  decisions_select1_calib_<best>_(cond-trained|).jsonl
  served mono    decisions_select1_mono.jsonl   (best-monotone GBM, train-fit)
  served Bayes   decisions_select1_bayes.jsonl  (unconstrained GBM, train-fit)
  oracle         decisions_select1_oracle.jsonl

selacc = decisive, alive-conditioned (oracle_hit + chosen); MAT = mean served
per-step accept_len. All tasks. Single proposers are omitted (suffix-only has no
served select-1 arm); their reference lives in mat_compare/mtp_mat_compare.

Colors match Panel B: served-mono = darkorange, served-Bayes = lime."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ladder_style import ladder_bar  # noqa: E402

ROOT = "simulation/results/chain_hybrid_perdepth"
FIGDIR = "simulation/results/calib_why_analysis/figures"
ALIVE = {"eagle", "suffix", "both"}
METHODS = ("histogram", "isotonic", "logistic", "beta")
C_RAW = "#7f7f7f"; C_CALIB = "#e377c2"; C_MONO = "darkorange"; C_BAYES = "lime"; C_ORACLE = "#d62728"
C_SUFFIX = "#8c564b"
CELLS = {
    "eagle3": {"dir": f"{ROOT}/qwen3_14b_ar", "title": "Qwen3-14B EAGLE3", "tag": "eagle3",
               "draft": "EAGLE3", "draft_color": "#1f77b4"},
    "mtp": {"dir": f"{ROOT}/qwen35_27b_ar", "title": "Qwen3.5-27B MTP", "tag": "mtp",
            "draft": "MTP", "draft_color": "#9467bd"},
}


def timing_mat(path):
    """mean per-step accept length from a standalone-decoder timing log."""
    if not os.path.exists(path):
        return None
    A = []
    for line in open(path):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("phase") == "decode":
            A += [float(x) for x in (o.get("accept_lengths") or [])]
    return sum(A) / len(A) if A else None


def _has_oh(p):
    if not os.path.exists(p):
        return False
    for line in open(p):
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            return "oracle_hit" in o
    return False


def calib_arm(d, m):
    cond = f"{d}/decisions_select1_calib_{m}_cond-trained.jsonl"
    return cond if _has_oh(cond) else f"{d}/decisions_select1_calib_{m}.jsonl"


def selacc_mat(path):
    """decisive alive-conditioned selection accuracy + mean per-step accept_len."""
    if not os.path.exists(path):
        return None, None
    chains = defaultdict(list); acc = {}
    for line in open(path):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
        elif o.get("type") == "step":
            acc[(o["rid"], o["decode_step"])] = o.get("accept_len")
    for k in chains: chains[k].sort(key=lambda r: r["depth"])
    n = c = 0
    for rs in chains.values():
        alive = True
        for r in rs:
            if not alive: break
            h = r.get("oracle_hit")
            if h in ("eagle", "suffix"):
                n += 1
                ps = (r.get("chosen") == "suffix")
                c += (ps and h == "suffix") or ((not ps) and h == "eagle")
            if h not in ALIVE: alive = False
    Ls = [v for v in acc.values() if v is not None]
    sa = c / max(n, 1)
    mat = sum(Ls) / max(len(Ls), 1) if Ls else None
    return sa, mat


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["eagle3", "mtp", "both"], default="both")
    args = ap.parse_args()
    items = CELLS.items() if args.cell == "both" else [(args.cell, CELLS[args.cell])]
    for key, c in items:
        d = c["dir"]
        # best calib method by served selacc
        cal = {m: selacc_mat(calib_arm(d, m)) for m in METHODS}
        cal = {m: v for m, v in cal.items() if v[0] is not None}
        best_m = max(cal, key=lambda m: cal[m][0])
        arms = [("raw\n(sp>ep)", f"{d}/decisions_select1.jsonl", C_RAW),
                (f"best calib\n({best_m})", calib_arm(d, best_m), C_CALIB),
                ("served mono\n(best-mono)", f"{d}/decisions_select1_mono.jsonl", C_MONO),
                ("served Bayes", f"{d}/decisions_select1_bayes.jsonl", C_BAYES),
                ("oracle\n(GT)", f"{d}/decisions_select1_oracle.jsonl", C_ORACLE)]
        labels, sels, mats, cols = [], [], [], []
        print(f"=== {c['title']} (best calib={best_m}) SERVED ===")
        for lab, path, col in arms:
            sa, mt = selacc_mat(path)
            if sa is None:
                print(f"  {lab.replace(chr(10),' '):22s} MISSING ({path})"); continue
            labels.append(lab); sels.append(sa); mats.append(mt); cols.append(col)
            print(f"  {lab.replace(chr(10),' '):22s} selacc={sa:.4f}  MAT={mt:.4f}")
        if len(labels) < 5:
            print(f"  [skip plot: only {len(labels)}/5 served arms present]"); continue
        # selection accuracy: only the 5 selection policies (single proposers make
        # no selection -> excluded).
        ladder_bar(sels, labels, "decisive selection accuracy (alive-conditioned)",
                   "Selection accuracy by Panel-B boundary policy (SERVED, pinned)\n"
                   f"({c['title']}, all tasks; real serving)",
                   f"{FIGDIR}/boundary_selacc_{c['tag']}.png", fmt="{:.3f}", colors=cols)
        # MAT: prepend the two single-proposer references (standalone served
        # decoders: draft-only = baseline timing, suffix-only = suffix timing).
        d_mat = timing_mat(f"{d}/timing_baseline.jsonl")
        s_mat = timing_mat(f"{d}/timing_suffix.jsonl")
        m_labels, m_vals, m_cols = list(labels), list(mats), list(cols)
        if s_mat is not None:
            m_labels.insert(0, "suffix\nonly"); m_vals.insert(0, s_mat); m_cols.insert(0, C_SUFFIX)
        if d_mat is not None:
            m_labels.insert(0, f"{c['draft']}\nonly"); m_vals.insert(0, d_mat); m_cols.insert(0, c["draft_color"])
        print(f"  single-proposer (standalone served): {c['draft']}-only={d_mat}  suffix-only={s_mat}")
        ladder_bar(m_vals, m_labels, "MAT (served per-step accept length)",
                   "MAT by Panel-B boundary policy (SERVED)\n"
                   f"({c['title']}, all tasks; real serving; single proposers = standalone decode)",
                   f"{FIGDIR}/boundary_mat_{c['tag']}.png", fmt="{:.3f}", colors=m_cols)


if __name__ == "__main__":
    main()
