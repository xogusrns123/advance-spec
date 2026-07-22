"""Shared bar-ladder style. Non-overlapping color roles:
EAGLE-3=blue, MTP=purple, Suffix=orange, raw=gray, oracle=red(+star),
calib histogram=green, isotonic=cyan, logistic=pink, beta=olive."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# role -> color (no two roles share a color)
ROLE = {
    "EAGLE3":   "#1f77b4",   # blue
    "MTP":      "#9467bd",   # purple
    "suffix":   "#ff7f0e",   # orange
    "raw":      "#7f7f7f",   # gray
    "roundrobin":"#8c564b",  # brown (blind no-score alternation baseline)
    "histogram":"#2ca02c",   # green
    "isotonic": "#17becf",   # cyan
    "logistic": "#e377c2",   # pink
    "beta":     "#bcbd22",   # olive
    "oracle":   "#d62728",   # red (+ star)
}
# 6-bar select-1 ladder (raw, calib x4, oracle)
SELECT1_COLORS = [ROLE[k] for k in ("raw", "histogram", "isotonic", "logistic", "beta", "oracle")]


def ladder_bar(values, xlabels, ylabel, title, out, fmt="{:.3f}", colors=None, star_idx=None):
    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.bar(range(len(values)), values, color=colors or SELECT1_COLORS, width=0.8)
    ymax = max(values)
    for i, v in enumerate(values):
        ax.text(i, v + ymax * 0.012, fmt.format(v), ha="center", va="bottom", fontsize=11)
    _ = star_idx  # star removed (oracle is just the red bar)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, ymax * 1.10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
