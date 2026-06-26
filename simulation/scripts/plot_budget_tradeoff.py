"""Dual-axis Node-Budget tradeoff figure, one per (method, model).

Reproduces the reference image.png style: X = node budget (log2), left axis
(blue) = Per-Token Time (ms/tok), right axis (orange) = Acceptance Length (tau).

Consumes the per-(model, method) JSON emitted by
simulation/scripts/experiments/measure_budget_sweep.py:
    { "model": "...", "method": "...",
      "results": [ {"budget":16,"accept_length_mean":..,"per_token_ms":..}, ... ] }

Run INSIDE the sglang-bench container as root (figures dir is root-owned):
    docker exec -u root sglang-bench bash -lc \
      ". /opt/venv/bin/activate && cd /workspace && \
       python3 simulation/scripts/plot_budget_tradeoff.py \
         --glob 'simulation/results/budget_tradeoff/qwen3_8b/*.json'"
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

BLUE = "#1f77b4"   # Per-Token Time
ORANGE = "#ff7f0e"  # Acceptance Length (tau)


def slug(model: str) -> str:
    """Same convention as plot_method_latency.slug."""
    return (model.split("/")[-1].lower().replace(".", "").replace("-", "_"))


def extract_series(doc: dict):
    """-> (budgets, ms_per_tok, tau) keeping only points that have BOTH metrics
    on the ms axis; tau may be None for a point (then that tau marker is skipped)."""
    rows = [r for r in doc.get("results", []) if "error" not in r and "budget" in r]
    # X position = effective budget (the config actually run; differs from the
    # requested budget when a method clamps, e.g. DFlash block size).
    for r in rows:
        r["_x"] = int(r.get("effective_budget", r["budget"]))
    rows.sort(key=lambda r: r["_x"])
    budgets, ms, tau = [], [], []
    seen_x = set()
    for r in rows:
        ptm = r.get("per_token_ms")
        if ptm is None or r["_x"] in seen_x:
            continue
        seen_x.add(r["_x"])
        budgets.append(r["_x"])
        ms.append(float(ptm))
        t = r.get("accept_length_mean")
        tau.append(float(t) if t is not None else None)
    return budgets, ms, tau


def plot_one(doc: dict, fig_dir: Path) -> Path | None:
    model = doc.get("model", "model")
    method = doc.get("method", "method")
    budgets, ms, tau = extract_series(doc)
    if not budgets:
        print(f"  skip {model}/{method}: no plottable points")
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax2 = ax.twinx()

    l1, = ax.plot(budgets, ms, "o-", color=BLUE, lw=2, ms=6,
                  label=f"{method} ms/tok")
    # tau may have None gaps; plot only the present ones.
    tb = [b for b, t in zip(budgets, tau) if t is not None]
    tv = [t for t in tau if t is not None]
    l2 = None
    if tb:
        l2, = ax2.plot(tb, tv, "s-", color=ORANGE, lw=2, ms=6,
                       label=f"{method} Acc. Length (τ)")

    ax.set_xscale("log", base=2)
    # Markers sit at every budget, but only label powers of 2 (16,32,...,1024)
    # so the dense 1.5x points (24,48,...) don't collide at the high end.
    tick_b = [b for b in budgets if b > 0 and (b & (b - 1)) == 0] or budgets
    ax.xaxis.set_major_locator(FixedLocator(tick_b))
    ax.xaxis.set_minor_locator(NullLocator())
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel("Node Budget")
    ax.set_ylabel("Per-Token Time (ms/tok)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2.set_ylabel("Acceptance Length (τ)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax.grid(alpha=0.3)
    # Title above, legend in a clean band just under it — never over the data.
    ax.set_title(f"{model} — {method}", pad=30)

    handles = [h for h in (l1, l2) if h is not None]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9, fontsize=8)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"budget_tradeoff_{slug(model)}_{method.replace('/', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", default=[],
                    help="per-(model,method) sweep JSON; repeatable")
    ap.add_argument("--glob", default=None,
                    help="glob of sweep JSONs (e.g. '.../qwen3_8b/*.json')")
    ap.add_argument("--fig-dir", default=None,
                    help="default: <each input's dir>/figures")
    args = ap.parse_args()

    paths = list(args.input)
    if args.glob:
        paths += sorted(globlib.glob(args.glob))
    if not paths:
        raise SystemExit("no inputs (use --input and/or --glob)")

    made = []
    for p in paths:
        path = Path(p)
        doc = json.loads(path.read_text())
        fig_dir = Path(args.fig_dir) if args.fig_dir else path.parent / "figures"
        out = plot_one(doc, fig_dir)
        if out:
            made.append(out)
    for m in made:
        print("wrote", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
