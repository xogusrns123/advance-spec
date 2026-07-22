#!/usr/bin/env python3
"""Regenerate Picture7 as a TIME graph (schematic, no data).

Two wall-clock timelines over the SAME content — a novel gap G₁, a copy run R,
a second gap G₂ — box width = time:

  Selection (baseline): every region gets its own draft+verify
      model|G₁ · VERIFY · suffix|R · VERIFY · model|G₂ · VERIFY   → 3 verifies
  Handoff (ours): one draft chain crosses the R↔G boundary (head→suffix, no
      verify in between), so that boundary costs 1 fewer verify
      head|G₁ · suffix|R · VERIFY · head|G₂ · VERIFY              → 2 verifies

A verify is one full target-model forward — the dominant cost — so its box is far
WIDER than a draft box. The handoff row therefore finishes earlier by exactly one
verify. Times are illustrative (verify measured at 12–37 ms in this setup).

  python3 scripts/plot/regen_picture7.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent.parent
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"
ORANGE, BLUE, NAVY, GREEN = "#D85A30", "#2A78D6", "#1F2A44", "#2E7D32"

T_DRAFT = 4.0        # ms, illustrative — one draft step (cheap)
T_VERIFY = 18.0      # ms, illustrative — one target forward (dominant; 12–37 ms)
H = 1.0              # row height


def box(ax, x, y, w, color, label, tc="white", fs=10.5, edge="white"):
    ax.add_patch(Rectangle((x, y), w, H, facecolor=color, edgecolor=edge, lw=1.4))
    ax.text(x + w / 2, y + H / 2, label, ha="center", va="center",
            color=tc, fontsize=fs)
    return x + w


def draw_row(ax, y, segs):
    x = 0.0
    for label, color, w, tc in segs:
        x = box(ax, x, y, w, color, label, tc=tc)
    return x                                   # row end time


def main():
    D, V = T_DRAFT, T_VERIFY
    # (label, color, width, text-color)
    sel = [("model\nG₁", BLUE, D, "white"), ("VERIFY", NAVY, V, "white"),
           ("suffix\nR", ORANGE, D, "white"), ("VERIFY", NAVY, V, "white"),
           ("model\nG₂", BLUE, D, "white"), ("VERIFY", NAVY, V, "white")]
    hand = [("head\nG₁", BLUE, D, "white"), ("suffix\nR", ORANGE, D, "white"),
            ("VERIFY", NAVY, V, "white"), ("head\nG₂", BLUE, D, "white"),
            ("VERIFY", NAVY, V, "white")]

    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    y_sel, y_hand = 2.3, 0.5
    sel_end = draw_row(ax, y_sel, sel)
    hand_end = draw_row(ax, y_hand, hand)

    # row titles (above each row, at t=0)
    ax.text(0, y_sel + H + 0.18, "Selection (baseline)  —  3 verifies",
            fontsize=13, fontweight="bold", color=NAVY, ha="left")
    ax.text(0, y_hand + H + 0.18, "Handoff (ours)  —  2 verifies",
            fontsize=13, fontweight="bold", color=NAVY, ha="left")

    # per-row total time
    ax.text(sel_end + 0.6, y_sel + H / 2, f"= {sel_end:.0f} ms",
            va="center", ha="left", fontsize=12.5, color=NAVY, fontweight="bold")
    ax.text(hand_end + 0.6, y_hand + H / 2, f"= {hand_end:.0f} ms",
            va="center", ha="left", fontsize=12.5, color=GREEN, fontweight="bold")

    # time saved bracket between the two row ends
    ax.plot([hand_end, hand_end], [y_hand - 0.3, y_sel + H + 0.45],
            ls=(0, (2, 2)), color="#888", lw=1.2)
    ax.plot([sel_end, sel_end], [y_sel - 0.3, y_sel + H + 0.45],
            ls=(0, (2, 2)), color="#888", lw=1.2)
    ax.add_patch(FancyArrowPatch((hand_end, y_sel + H + 0.32),
                                 (sel_end, y_sel + H + 0.32),
                                 arrowstyle="<->", color=GREEN, lw=1.8,
                                 mutation_scale=14))
    ax.text((hand_end + sel_end) / 2, y_sel + H + 0.5,
            f"saved {sel_end - hand_end:.0f} ms\n(one verify)", ha="center",
            va="bottom", fontsize=11, color=GREEN, fontweight="bold")

    # legend
    handles = [
        Line2D([0], [0], marker="s", ls="", mfc=BLUE, mec="none", ms=13,
               label="model / head draft (cheap)"),
        Line2D([0], [0], marker="s", ls="", mfc=ORANGE, mec="none", ms=13,
               label="suffix draft / copy (cheap)"),
        Line2D([0], [0], marker="s", ls="", mfc=NAVY, mec="none", ms=13,
               label="VERIFY = target forward (dominant cost)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False, fontsize=10.5)

    # time axis
    ax.set_xlim(-0.5, sel_end + 6)
    ax.set_ylim(-0.2, y_sel + H + 1.15)
    ax.set_yticks([])
    ax.set_xlabel("wall-clock time (ms, illustrative)", fontsize=12)
    ax.set_xticks(range(0, int(sel_end) + 1, 12))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture7.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}  selection={sel_end:.0f}ms handoff={hand_end:.0f}ms "
          f"saved={sel_end - hand_end:.0f}ms")


if __name__ == "__main__":
    main()
