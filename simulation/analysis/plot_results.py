"""
Visualize analysis results.

Generates plots for:
1. Case distribution (stacked bar / pie)
2. Per-depth P_accept and P_match curves
3. Agreement vs correctness correlation
4. Top-5 overlap distribution

Usage:
    python -m simulation.analysis.plot_results \
        --complementarity-file results/complementarity/complementarity.json \
        --agreement-file results/agreement/agreement.json \
        --output-dir results/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_case_distribution(data: dict, output_dir: Path) -> None:
    """Stacked bar chart of case distribution."""
    cases = data["cases"]
    rates = cases["rates"]

    labels = ["Case 1\n(Both)", "Case 2a\n(Seq. value)", "Case 2b\n(EAGLE suf.)",
              "Case 3\n(Suffix comp.)", "Case 4\n(Both fail)"]
    values = [rates["case1"], rates["case2a"], rates["case2b"], rates["case3"], rates["case4"]]
    colors = ["#2ecc71", "#3498db", "#95a5a6", "#e67e22", "#e74c3c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = ax1.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Rate")
    ax1.set_title("Case Distribution: EAGLE-3 vs SuffixDecoding")
    ax1.set_ylim(0, max(values) * 1.2 if values else 1.0)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Pie chart: fusion value breakdown
    fusion = cases["fusion_value"]
    seq = cases["sequential_unique_value"]
    eagle_only = rates["case2b"]
    fail = rates["case4"]
    pie_values = [fusion, seq, eagle_only, fail]
    pie_labels = [f"Fusion helps\n({fusion:.1%})",
                  f"Seq. unique\n({seq:.1%})",
                  f"EAGLE suf.\n({eagle_only:.1%})",
                  f"Both fail\n({fail:.1%})"]
    pie_colors = ["#2ecc71", "#3498db", "#95a5a6", "#e74c3c"]
    ax2.pie(pie_values, labels=pie_labels, colors=pie_colors,
            autopct="%1.1f%%", startangle=90)
    ax2.set_title("Value Decomposition")

    plt.tight_layout()
    plt.savefig(output_dir / "case_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved case_distribution.png")


def plot_depth_stats(data: dict, output_dir: Path) -> None:
    """Per-depth P_accept and P_match curves."""
    depth_data = data.get("depth_stats", {})
    if not depth_data:
        print("  No depth stats to plot")
        return

    depths = sorted(int(d) for d in depth_data.keys())
    p_accept = [depth_data[str(d)]["p_accept"] for d in depths]
    p_match = [depth_data[str(d)]["p_match"] for d in depths]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, p_accept, "o-", color="#2ecc71", linewidth=2, markersize=8, label="P_accept (EAGLE-3)")
    ax.plot(depths, p_match, "s-", color="#3498db", linewidth=2, markersize=8, label="P_match (SuffixDecoding)")

    # Expected benefit: P_accept * P_match
    p_benefit = [a * m for a, m in zip(p_accept, p_match)]
    ax.plot(depths, p_benefit, "^--", color="#e67e22", linewidth=1.5, markersize=7,
            label="Expected benefit (P_a * P_m)")

    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Rate")
    ax.set_title("Per-Depth Acceptance and Suffix Match Rates")
    ax.legend()
    ax.set_xticks(depths)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "depth_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved depth_stats.png")


def plot_agreement_correlation(data: dict, output_dir: Path) -> None:
    """Agreement vs correctness correlation."""
    corr = data.get("correlation", {})
    if not corr:
        print("  No correlation data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix style
    matrix = np.array([
        [corr["agree_and_correct"], corr["agree_and_wrong"]],
        [corr["disagree_and_correct"], corr["disagree_and_wrong"]],
    ])
    total = matrix.sum()
    matrix_pct = matrix / max(total, 1) * 100

    im = ax1.imshow(matrix_pct, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Correct", "Wrong"])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Agree", "Disagree"])
    ax1.set_title("Agreement vs Correctness")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{matrix_pct[i, j]:.1f}%\n({int(matrix[i, j])})",
                     ha="center", va="center", fontsize=11)
    plt.colorbar(im, ax=ax1, label="% of total")

    # Conditional probabilities
    labels = ["P(correct|agree)", "P(correct|disagree)"]
    values = [corr["p_correct_given_agree"], corr["p_correct_given_disagree"]]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax2.bar(labels, values, color=colors, edgecolor="white")
    ax2.set_ylabel("Probability")
    ax2.set_title("Correctness Conditioned on Agreement")
    ax2.set_ylim(0, 1.05)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "agreement_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved agreement_correlation.png")


def plot_overlap_distribution(data: dict, output_dir: Path) -> None:
    """Top-5 overlap distribution histogram."""
    overlap_data = data.get("overlap", {}).get("overlap_distribution", {})
    if not overlap_data:
        print("  No overlap data to plot")
        return

    overlaps = sorted(int(k) for k in overlap_data.keys())
    rates = [overlap_data[str(o)]["rate"] for o in overlaps]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(overlaps, rates, color="#3498db", edgecolor="white")
    ax.set_xlabel("Top-5 Overlap Count")
    ax.set_ylabel("Rate")
    ax.set_title("Distribution of EAGLE-3 / SuffixDecoding Top-5 Overlap")
    ax.set_xticks(overlaps)
    mean_overlap = data["overlap"].get("mean_overlap", 0)
    ax.axvline(mean_overlap, color="#e74c3c", linestyle="--", label=f"Mean={mean_overlap:.2f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "overlap_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved overlap_distribution.png")


def main():
    parser = argparse.ArgumentParser(description="Plot analysis results")
    parser.add_argument("--complementarity-file", default=None)
    parser.add_argument("--agreement-file", default=None)
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")

    if args.complementarity_file and Path(args.complementarity_file).exists():
        with open(args.complementarity_file) as f:
            comp_data = json.load(f)
        plot_case_distribution(comp_data, output_dir)
        plot_depth_stats(comp_data, output_dir)

    if args.agreement_file and Path(args.agreement_file).exists():
        with open(args.agreement_file) as f:
            agree_data = json.load(f)
        plot_agreement_correlation(agree_data, output_dir)
        plot_overlap_distribution(agree_data, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
