"""Panel-B style DECISION BOUNDARY for the 3-PROPOSER select-1 (DFlash+EAGLE3+suffix
on 8B; MTP+DFlash+suffix on 27B). Panel B is 2-D (one proposer's prob vs the other's)
+ a P(==gt) surface + the raw diagonal + the 0.5 Bayes boundary. With 3 proposers the
joint space is 3-D, so we draw ONE PANEL PER PROPOSER P (3 panels), each in the panel-B
plane (x = p_P, y = max prob of the OTHER two = the best competitor). The surface is the
empirical P(P == gt | p_P, max_other, DECISIVE); the raw select-1 rule "pick P iff
p_P > max_other" is the diagonal y = x; the lime P=0.5 contour is where P actually
becomes the right pick. A gap between the diagonal and the lime contour = the raw
argmax is mis-scaled for that proposer (the calibration story, now 3-way).

Decisive = some-but-not-all available proposers hit gt (the pick matters); alive prefix
= >=1 proposer still covers gt (the 3-way oracle chain has not died). Same conditioning
as analyze_3way_ceiling.py.

  python3 simulation/scripts/analyze_panelB_3way.py --cell 8b  --record-dir <8B ceiling dir> --exclude-loopy
  python3 simulation/scripts/analyze_panelB_3way.py --cell 27b --record-dir <27B ceiling dir> \
       --merge-dflash <dir>/dflash_proposals.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

CELLS = {
    "8b": {"names": ("dflash", "e3", "suffix"),
           "tok": {"dflash": "eagle_token", "e3": "e3_token", "suffix": "suffix_token"},
           "p": {"dflash": "eagle_p", "e3": "e3_p", "suffix": "suffix_p"},
           "model": "Qwen3-8B"},
    "27b": {"names": ("mtp", "dflash", "suffix"),
            "tok": {"mtp": "eagle_token", "dflash": "dflash_token", "suffix": "suffix_token"},
            "p": {"mtp": "eagle_p", "dflash": "dflash_p", "suffix": "suffix_p"},
            "model": "Qwen3.5-27B"},
}


def load_chains(path):
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains:
        chains[k].sort(key=lambda r: r["depth"])
    return chains


def merge_dflash(chains, path):
    idx = {(rid, ds, r["depth"]): r for (rid, ds), rs in chains.items() for r in rs}
    n = 0
    for line in open(path):
        o = json.loads(line)
        r = idx.get((o["rid"], o["decode_step"], o["depth"]))
        if r is not None:
            r["dflash_token"] = o["dflash_token"]; r["dflash_p"] = o.get("dflash_p"); n += 1
    print(f"merged {n} DFlash proposals")


def loopy_rids(record_dir, decisions_file, thresh=0.5, n=4):
    dd = Path(record_dir)
    reqs = {}
    for line in open(dd / decisions_file):
        o = json.loads(line)
        if o.get("type") == "req":
            reqs[o["rid"]] = tuple(o["input_ids"])
    gt = {}
    if (dd / "gt_tokens.jsonl").exists():
        for line in open(dd / "gt_tokens.jsonl"):
            r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
    bad = set()
    for rid, ids in reqs.items():
        out = gt.get(ids)
        if not out or len(out) < n + 1:
            continue
        g = [tuple(out[i:i + n]) for i in range(len(out) - n + 1)]
        if len(set(g)) / max(len(g), 1) < thresh:
            bad.add(rid)
    return bad


def collect(chains, names, tok, pk):
    """Per proposer P: decisive-alive rows -> (p_P, max_other, hit_P)."""
    data = {P: {"x": [], "y": [], "c": []} for P in names}
    for rs in chains.values():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            toks = {P: r.get(tok[P]) for P in names}
            probs = {P: (r.get(pk[P]) if r.get(pk[P]) is not None else 0.0) for P in names}
            avail = [P for P in names if toks[P] is not None]
            hits = [P for P in avail if gt is not None and toks[P] == gt]
            decisive = 0 < len(hits) < len(avail)
            if decisive:
                for P in avail:
                    others = [probs[q] for q in avail if q != P]
                    if not others:
                        continue
                    data[P]["x"].append(probs[P])
                    data[P]["y"].append(max(others))
                    data[P]["c"].append(1 if P in hits else 0)
            if gt is not None and len(hits) == 0:
                alive = False
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["8b", "27b"], required=True)
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--decisions-file", default="decisions_select1_oracle.jsonl")
    ap.add_argument("--merge-dflash", default=None)
    ap.add_argument("--exclude-loopy", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cell = CELLS[args.cell]
    names, tok, pk = cell["names"], cell["tok"], cell["p"]
    chains = load_chains(Path(args.record_dir) / args.decisions_file)
    if args.exclude_loopy:
        bad = loopy_rids(args.record_dir, args.decisions_file)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        print(f"excluded {len(bad)} loopy reqs")
    if args.merge_dflash:
        merge_dflash(chains, args.merge_dflash)
        chains = {k: v for k, v in chains.items()
                  if any(r.get(tok[names[1]]) is not None for r in v)}

    data = collect(chains, names, tok, pk)
    g = np.linspace(0, 1, 220); GX, GY = np.meshgrid(g, g)
    grid = np.c_[GX.ravel(), GY.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    for ax, P in zip(axes, names):
        x = np.array(data[P]["x"]); y = np.array(data[P]["y"]); c = np.array(data[P]["c"])
        n = len(x)
        if n >= 40 and c.min() != c.max():
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                                 learning_rate=0.05).fit(np.c_[x, y], c)
            Z = clf.predict_proba(grid)[:, 1].reshape(GX.shape)
            pc = ax.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.9)
            fig.colorbar(pc, ax=ax).set_label(f"P({P}==gt | decisive)")
            ax.contour(GX, GY, Z, levels=[0.5], colors="lime", linewidths=2.2)
            ax.plot([], [], color="lime", lw=2.2, label="P=0.5 boundary")
        # data density + the raw select-1 rule (pick P iff p_P > best competitor)
        ax.hist2d(x, y, bins=40, range=[[0, 1], [0, 1]], cmin=5, cmap="Greys", alpha=0.35)
        ax.plot([0, 1], [0, 1], "k-", lw=2.0, label=f"raw: pick {P} iff p>max_other")
        win = c.mean() if n else float("nan")
        ax.set_xlabel(f"{P}  prob"); ax.set_ylabel("max prob of other two")
        ax.set_title(f"{P}   (n={n} decisive,  P hits {win:.2f})")
        ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle(f"3-way select-1 decision boundary ({'+'.join(names)}, {cell['model']}, "
                 f"SERVED gt-path)  —  per proposer: P(it==gt) vs the best competitor; "
                 f"gap of lime 0.5-boundary from the black raw diagonal = raw argmax mis-scale",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out or f"{args.record_dir}/figures/panelB_3way_{args.cell}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
