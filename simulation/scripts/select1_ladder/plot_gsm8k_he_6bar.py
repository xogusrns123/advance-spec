"""Per-subtask 6-bar REALIZED MAT chart for gsm8k+humaneval select-1. ONE IMAGE PER SUBTASK.
Bars: model-alone, suffix-alone, raw, calib(beta cond-trained), bayes(GBM), oracle.

Per-subtask extraction (all realized, accept_length = draft-tokens accepted per block):
 - raw/calib/bayes/oracle: 'step' accept_len from each arm's decision log, grouped by subtask via
   oracle req-input_ids decode (others order-zipped to oracle).
 - model-alone (baseline): no decision log -> segment timing_baseline.jsonl by prefill markers,
   interleaved dataset => even task=gsm8k, odd=humaneval (validated: overall recovers exactly).
 - suffix-alone: timing_suffix.jsonl has NO prefill markers -> OVERALL only (same value both charts,
   hatched + "(overall)").
"""
import json, argparse, os, statistics
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEC_ARMS = [  # (key, decision-log file, label, color)
    ("raw",    "decisions_select1.jsonl",            "raw\n(argmax)", "#9e9e9e"),
    ("calib",  "decisions_select1_calib_beta.jsonl", "calib\n(beta)", "#e57373"),
    ("bayes",  "decisions_select1_bayes.jsonl",      "bayes\n(GBM)",  "#42a5f5"),
    ("oracle", "decisions_select1_oracle.jsonl",     "oracle",        "#2e7d32"),
]
SUBTASKS = ["gsm8k", "humaneval"]


def timing_persubtask(path):
    """Segment a timing log by prefill markers; even segment=gsm8k, odd=humaneval. Returns
    {subtask: (mat, nblocks)} or None if no prefill markers."""
    segs = []; cur = None; have_prefill = False
    for line in open(path):
        o = json.loads(line); ph = o.get("phase")
        if ph == "prefill":
            have_prefill = True
            if cur is not None: segs.append(cur)
            cur = []
        elif ph == "decode" and cur is not None:
            cur.extend(float(a) for a in (o.get("accept_lengths") or []))
    if cur is not None: segs.append(cur)
    if not have_prefill:
        return None
    segs = [s for s in segs if s]
    out = {}
    for sub, par in (("gsm8k", 0), ("humaneval", 1)):
        pool = [x for i, s in enumerate(segs) for x in s if i % 2 == par]
        out[sub] = (statistics.mean(pool), len(pool)) if pool else (None, 0)
    return out


def timing_overall(path):
    vals = []
    for line in open(path):
        o = json.loads(line)
        if o.get("phase") == "decode":
            vals.extend(float(a) for a in (o.get("accept_lengths") or []))
    return statistics.mean(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--proposer", default="model")
    ap.add_argument("--dataset", default="data/gsm8k_humaneval/dataset_interleaved.jsonl")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--prefix-len", type=int, default=60)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dataset)]
    turn0 = lambda r: (r["turns"][0] if isinstance(r["turns"], list) else r["turns"])
    prefixes = [(turn0(r)[:args.prefix_len], r["subtask"]) for r in rows]

    # --- decision-log arms: rid -> subtask ---
    steps = {k: defaultdict(list) for k, *_ in DEC_ARMS}
    ordered = {k: [] for k, *_ in DEC_ARMS}; seen = {k: set() for k, *_ in DEC_ARMS}
    reqs = {}; present = []
    for k, fn, *_ in DEC_ARMS:
        p = f"{args.dir}/{fn}"
        if not os.path.exists(p):
            continue
        present.append(k)
        for line in open(p):
            o = json.loads(line); t = o.get("type"); rid = o.get("rid")
            if rid is not None and rid not in seen[k] and t in ("req", "step", "decision"):
                seen[k].add(rid); ordered[k].append(rid)
            if t == "req":
                reqs.setdefault(rid, o["input_ids"])
            elif t == "step":
                steps[k][rid].append(float(o.get("accept_len", 0)))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    osub = {}
    for rid in ordered.get("oracle", []):
        ids = reqs.get(rid)
        osub[rid] = None if ids is None else next(
            (sub for pfx, sub in prefixes if pfx and pfx in tok.decode(ids, skip_special_tokens=False)), None)
    oo = ordered.get("oracle", [])
    rid_sub = {"oracle": osub}
    for k in present:
        if k == "oracle":
            continue
        oa = ordered[k]
        rid_sub[k] = {ra: osub.get(ro) for ra, ro in zip(oa, oo)} if len(oa) == len(oo) else {}

    def dec_mat(k, sub):
        acc = [x for rid, lens in steps[k].items() if rid_sub.get(k, {}).get(rid) == sub for x in lens]
        return (sum(acc) / len(acc), len(acc)) if acc else (None, 0)

    # --- single proposers from timing logs ---
    base_ps = timing_persubtask(f"{args.dir}/timing_baseline.jsonl") if \
        os.path.exists(f"{args.dir}/timing_baseline.jsonl") else None
    suf_overall = timing_overall(f"{args.dir}/timing_suffix.jsonl") if \
        os.path.exists(f"{args.dir}/timing_suffix.jsonl") else None

    # bar spec: (label, color, hatch, per-subtask getter)
    def model_alone(sub):
        return base_ps[sub] if base_ps else (None, 0)
    def suffix_alone(sub):
        return (suf_overall, -1)  # overall, same both subtasks

    BARS = [
        (f"model-alone\n({args.proposer})", "#5b8fb0", "", model_alone),
        ("suffix-alone\n(overall)",         "#ff9800", "//", suffix_alone),
        ("raw\n(argmax)",                   "#9e9e9e", "", lambda s: dec_mat("raw", s)),
        ("calib\n(beta)",                   "#e57373", "", lambda s: dec_mat("calib", s)),
        ("bayes\n(GBM)",                    "#42a5f5", "", lambda s: dec_mat("bayes", s)),
        ("oracle",                          "#2e7d32", "", lambda s: dec_mat("oracle", s)),
    ]

    for sub in SUBTASKS:
        labels, vals, colors, hatches = [], [], [], []
        for lab, col, hatch, getter in BARS:
            m, _ = getter(sub)
            labels.append(lab); vals.append(m if m else 0.0); colors.append(col); hatches.append(hatch)
            print(f"  {sub:10} {lab.splitlines()[0]:14} MAT={None if not m else round(m,3)}")
        fig, ax = plt.subplots(figsize=(8.4, 5.3))
        bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="white", width=0.72)
        for b, h in zip(bars, hatches):
            if h: b.set_hatch(h)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.012, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("realized mean accept length (MAT, draft-accepted)")
        ax.set_title(f"{args.model_label} · {sub}  —  select-1 realized MAT\n"
                     f"single proposers ({args.proposer} / suffix) vs raw / calib(beta) / bayes / oracle")
        ax.set_ylim(0, max(vals) * 1.18); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = f"{args.out_prefix}_{sub}.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=140); print("wrote", out)


if __name__ == "__main__":
    main()
