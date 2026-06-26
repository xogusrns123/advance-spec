"""Plot per-method decoding latency from measure_methods_latency summary.json.

For each model, two bar-chart PNGs:
  <slug>_verify_step_latency.png : stacked bar (verify / draft / others) per method
  <slug>_draft_step_latency.png  : single bar (per-draft-token latency) per method,
                                   LOG y-scale so a near-zero draft (e.g. SUFFIX's
                                   model-free trie lookup) stays visible next to the
                                   multi-ms model drafts.
Methods are ordered EAGLE-3 -> MTP -> small-model -> DFlash -> Suffix (present ones).
Every bar/segment is annotated with its value; the unit auto-switches between ms
and us so small magnitudes are never hidden. The tiny draft segment in the stacked
plot gets a leader-line callout when it is too thin to hold an in-place label.

Run INSIDE the sglang-bench container as root (figures dir is root-owned):
    docker exec -u root sglang-bench bash -lc \
      ". /opt/venv/bin/activate && cd /workspace && \
       python3 simulation/scripts/plot_method_latency.py \
         --summary simulation/results/latency/method_compare/summary.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_ORDER = ["EAGLE-3", "MTP", "small-model", "DFlash", "Suffix"]
COLORS = {"verify": "#4C72B0", "draft": "#DD8452", "others": "#8C8C8C"}
# distinct per-method colors for the single-bar draft-step plot
BAR_COLORS = ["#55A868", "#C44E52", "#8172B3", "#CCB974", "#DD8452"]


def slug(model: str) -> str:
    return (model.split("/")[-1].lower()
            .replace(".", "").replace("-", "_"))


def methods_in_order(by_method: dict) -> list[str]:
    return [m for m in METHOD_ORDER if m in by_method]


def fmt_latency(ms: float) -> str:
    """Auto ms/us so tiny values (e.g. the suffix trie draft) read clearly."""
    if ms <= 0:
        return "0"
    if ms < 1.0:
        return f"{ms * 1e3:.1f} µs"   # microseconds
    return f"{ms:.2f} ms"


def plot_verify_step(model: str, by_method: dict, fig_dir: Path) -> Path:
    methods = methods_in_order(by_method)
    verify = [by_method[m]["verify_ms"] for m in methods]
    draft = [by_method[m]["draft_ms"] for m in methods]
    # Clamp tiny negative "others" (median-sum artifact: per-component medians
    # don't sum exactly to the step median) to 0 for the stacked bar.
    others = [max(0.0, by_method[m]["others_ms"]) for m in methods]
    totals = [verify[i] + draft[i] + others[i] for i in range(len(methods))]
    maxtotal = max(totals) if totals else 1.0

    fig, ax = plt.subplots(figsize=(1.7 * len(methods) + 2.5, 5.4))
    x = range(len(methods))
    ax.bar(x, verify, color=COLORS["verify"], label="verify")
    ax.bar(x, draft, bottom=verify, color=COLORS["draft"], label="draft")
    bottom2 = [v + d for v, d in zip(verify, draft)]
    ax.bar(x, others, bottom=bottom2, color=COLORS["others"], label="others")

    headroom = maxtotal * 0.28
    for i in x:
        total = totals[i]
        # total label above the bar
        ax.text(i, total + maxtotal * 0.012, fmt_latency(total),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        # verify / others in-segment labels (white) when tall enough to read
        for seg, base in ((verify[i], 0.0), (others[i], bottom2[i])):
            if seg > 0.06 * maxtotal:
                ax.text(i, base + seg / 2, fmt_latency(seg), ha="center",
                        va="center", fontsize=8, color="white")
        # DRAFT value: ALWAYS shown. If the segment is tall enough, label it in
        # place; otherwise (suffix-style sliver) draw a leader-line callout so the
        # number is never hidden by the verify-dominated linear scale.
        if draft[i] > 0.06 * maxtotal:
            ax.text(i, verify[i] + draft[i] / 2, fmt_latency(draft[i]),
                    ha="center", va="center", fontsize=8, color="white")
        else:
            ax.annotate(
                f"draft {fmt_latency(draft[i])}",
                xy=(i, verify[i] + draft[i]),
                xytext=(i, total + headroom * 0.55),
                ha="center", va="bottom", fontsize=8,
                color=COLORS["draft"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=COLORS["draft"], lw=0.9),
            )

    ax.set_ylim(0, maxtotal + headroom * 1.15)
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel("latency per decode step (ms)")
    ax.set_title(f"{model}  —  verify-step scope (verify + draft + others)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / f"{slug(model)}_verify_step_latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_draft_step(model: str, by_method: dict, fig_dir: Path) -> Path:
    methods = methods_in_order(by_method)
    vals = [by_method[m]["per_draft_token_ms"] for m in methods]
    colors = BAR_COLORS[:len(methods)]

    # Log y-scale: drafts span orders of magnitude (suffix ~us, small-model ~ms),
    # so a linear axis would flatten the tiny ones to invisibility. Floor non-
    # positive values to a small positive so they still render a visible bar.
    pos = [v for v in vals if v > 0]
    floor = (min(pos) * 0.25) if pos else 1e-3
    plot_vals = [v if v > 0 else floor for v in vals]
    top = max(plot_vals)

    fig, ax = plt.subplots(figsize=(1.7 * len(methods) + 2.5, 5.4))
    x = range(len(methods))
    ax.bar(x, plot_vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.5, top * 4.0)
    for i, (v, pv) in enumerate(zip(vals, plot_vals)):
        ax.text(i, pv * 1.08, fmt_latency(v), ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel("latency per draft token  (log scale)")
    ax.set_title(f"{model}  —  draft-step scope (one draft token)")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / f"{slug(model)}_draft_step_latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary",
                    default="simulation/results/latency/method_compare/summary.json")
    ap.add_argument("--fig-dir", default=None,
                    help="default: <summary dir>/figures")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    fig_dir = Path(args.fig_dir) if args.fig_dir else summary_path.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    made = []
    for model, by_method in summary.items():
        if not by_method:
            continue
        made.append(plot_verify_step(model, by_method, fig_dir))
        made.append(plot_draft_step(model, by_method, fig_dir))
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
