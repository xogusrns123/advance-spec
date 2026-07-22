#!/usr/bin/env python3
"""Regenerate synthetic Picture2 with REAL data — same provenance-stripe format.

One real swebench agent turn (interp_validation curve rid, aligned to its
output_ids in the perpos_swebench_step250 traces), every token colored by
provenance: REPEATED = R (covered by a >=4-gram repeat, rg_cover(s,4), orange)
vs NOVEL = G (the rest, blue). A zoom shows the decoded tokens of one command
line broken into its real R/G segments — the command scaffolding is copied, the
task identifier is where novel tokens appear.

Needs the Qwen3.5 tokenizer → run inside sglang-bench:
  docker exec -i sglang-bench bash -lc 'cd "/workspace/simulation/Dr.Lee Solution" \
      && PYTHONPATH=/workspace python3 scripts/plot/regen_picture2.py'
"""
from __future__ import annotations
import gzip
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from transformers import AutoTokenizer

from _gr import rg_cover

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "results" / "interp_validation"
TRACES = BASE / "results" / "perpos_swebench_alleval" / "swebench_4way.traces.json"
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"

ORANGE, BLUE, NAVY = "#D85A30", "#2A78D6", "#1F2A44"
L_ORANGE, L_BLUE = "#F4D1C5", "#C3D9F3"
# the real matplotlib-20676 turn (conv 35) that emits the SpanSelector search;
# zoom [ZA,ZB) is its bash command line:
#   find /testbed -type f -name "*.py" -path "*/matplotlib/*" | xargs grep -l "class SpanSelector" 2>/dev/null
RID, ZA, ZB = 2183, 49, 84


def provenance(rid):
    cur = {}
    with gzip.open(DATA / "curves_swebench.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l)
            cur[r["rid"]] = r
    tr = json.load(open(TRACES))
    ev = {t["rid"]: t for t in tr["eval_traces"]}
    s, oid = cur[rid]["s"], ev[rid]["output_ids"]
    assert len(s) == len(oid) - 1
    # output token j (>=1) is R iff its copy signal s[j-1] is inside a >=4 repeat;
    # token 0 has no prior context -> G (novel).
    ind = [0] + [1 if c else 0 for c in rg_cover(s)]
    return oid, ind


def runs(ind, a=0, b=None):
    b = len(ind) if b is None else b
    out, i = [], a
    while i < b:
        j = i
        while j < b and ind[j] == ind[i]:
            j += 1
        out.append((i, j, ind[i]))
        i = j
    return out


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    oid, ind = provenance(RID)
    N = len(oid)

    # zoom segments (real R/G runs inside the window)
    segs = []
    for b, e, iw in runs(ind, ZA, ZB):
        txt = tok.decode(oid[b:e])
        segs.append(dict(text=txt, n=e - b, warm=iw))

    fig = plt.figure(figsize=(12.6, 4.0))
    ax_top = fig.add_axes([0.045, 0.62, 0.93, 0.26])
    ax_zoom = fig.add_axes([0.11, 0.06, 0.80, 0.30])

    # ---- top stripe -------------------------------------------------------
    rep = [(b, e - b) for b, e, iw in runs(ind) if iw]
    nov = [(b, e - b) for b, e, iw in runs(ind) if not iw]
    ax_top.broken_barh(rep, (0, 1), facecolors=ORANGE)
    ax_top.broken_barh(nov, (0, 1), facecolors=BLUE)
    for xz in (ZA, ZB):
        ax_top.plot([xz, xz], [0, 1], color="black", lw=1.6)
    ax_top.set_xlim(0, N)
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_xticks(range(0, N + 1, 100))
    ax_top.tick_params(labelsize=11, length=3)
    for sp in ("top", "right", "left"):
        ax_top.spines[sp].set_visible(False)
    ax_top.set_title(f"One agent turn, {N} output tokens — colored by provenance",
                     fontsize=15, color=NAVY, loc="left", pad=10)

    # ---- zoom detail ------------------------------------------------------
    # place every character on an integer grid so the monospace text lines up
    # exactly with the colored provenance boxes (widths = character counts).
    spans = []                                   # (char_start, char_end, seg)
    c = 0
    for s in segs:
        w = len(s["text"])
        spans.append((c, c + w, s))
        c += w
    total = c
    ax_zoom.set_xlim(-1, total + 1)
    ax_zoom.set_ylim(-1.35, 1.0)
    ax_zoom.axis("off")
    subs = "₁₂₃₄₅₆₇₈₉"
    ri = gi = 0
    for k, (x0, x1, s) in enumerate(spans):
        fc, ec = (L_ORANGE, ORANGE) if s["warm"] else (L_BLUE, BLUE)
        ax_zoom.add_patch(Rectangle((x0, 0.12), x1 - x0, 0.76, facecolor=fc,
                                    edgecolor=ec, lw=1.6))
        for i, ch in enumerate(s["text"]):
            ax_zoom.text(x0 + i + 0.5, 0.5, ch, ha="center", va="center",
                         family="monospace", fontsize=12, color="#1a1a1a")
        if s["warm"]:
            ri += 1
            lab = (f"R{subs[ri-1]} (repeated, {s['n']} tok)" if ri == 1
                   else f"R{subs[ri-1]} ({s['n']})")
            col = ORANGE
        else:
            gi += 1
            lab = (f"G{subs[gi-1]} (novel, {s['n']} tok)" if gi == 1
                   else f"G{subs[gi-1]} ({s['n']})")
            col = BLUE
        yl = -0.45 if k % 2 == 0 else -0.95     # stagger to avoid collisions
        ax_zoom.text((x0 + x1) / 2, yl, lab, ha="center", va="center",
                     fontsize=11.5, color=col)

    # ---- dotted connectors stripe->zoom -----------------------------------
    for xz, zx in ((ZA, 0.0), (ZB, total)):
        con = ConnectionPatch(xyA=(xz, 0), coordsA=ax_top.transData,
                              xyB=(zx, 0.85), coordsB=ax_zoom.transData,
                              linestyle=(0, (2, 2)), color="black", lw=1.0)
        fig.add_artist(con)

    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture2.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")
    print(f"  rid={RID} N={N} zoom [{ZA}:{ZB}] -> "
          + " | ".join(f"{'R' if s['warm'] else 'G'}({s['n']}):{s['text']!r}" for s in segs))


if __name__ == "__main__":
    main()
