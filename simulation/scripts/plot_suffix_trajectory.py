"""Microscopic suffix-decoding accept-length trajectory for a SINGLE request.

Model-free: replays one recorded GT trajectory (gt_tokens.jsonl line = --rid)
through a SuffixDecodingCache, and at each speculative step records the accept
length (drafted tokens that match GT), the suffix context match_len, and the
suffix score. Advances by acc+1 tokens (the +1 is the always-correct next
token, i.e. the target's bonus token in real spec decoding), which reproduces
suffix decoding's OWN step structure -- so no model forward / no serving.

Answers the question: does one match yield a long accept, or do many short
accepts accumulate? Produces a 3-panel PNG:
  A) accept-length spike train  (x = output token position, y = accept length)
  B) accept-length distribution (# steps  and  token-contribution per value)
  C) cumulative contribution (Pareto/Lorenz) with Gini -- how concentrated the
     accepted tokens are in a few long spikes.

Run inside the sglang-bench container (has arctic_inference + matplotlib):
  docker exec sglang-bench python3 /workspace/simulation/scripts/plot_suffix_trajectory.py \
    --gt /workspace/simulation/results/chain_hybrid_perdepth/qwen3_14b_ar/gt_tokens.jsonl \
    --rid 4 --label "Qwen3-14B  bfcl_v4 web_search  (normal)" \
    --out /workspace/simulation/results/suffix_trajectory/qwen3_14b_normal.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# unified proposer color convention (shared with plot_proposer_trajectories.py)
SUFFIX_ORANGE = "#e8820e"
SUFFIX_DARK = "#9c4221"
REF_GRAY = "#4a5568"
REF_GRAY_LT = "#718096"


def replay_one(gt_path: str, rid: int, max_spec_tokens: int, max_spec_factor: float):
    """Replay a single request; return per-step arrays."""
    from arctic_inference.suffix_decoding import SuffixDecodingCache
    from simulation.evaluation.tree_knapsack import greedy_tree_walk

    rows = [json.loads(l) for l in open(gt_path) if l.strip()]
    if rid < 0 or rid >= len(rows):
        raise SystemExit(f"rid {rid} out of range (n={len(rows)})")
    r = rows[rid]
    prompt = list(r.get("input_ids") or [])
    gt = list(r.get("output_ids") or [])
    if not gt:
        raise SystemExit(f"rid {rid} has empty output_ids")

    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
    cache.start_request(0, np.asarray(prompt, dtype=np.int32))
    ctx = list(prompt)
    pos = 0
    steps = []  # (pos, acc, match_len, score, proposed_depth)
    while pos < len(gt):
        try:
            draft = cache.speculate(
                0, np.asarray(ctx, dtype=np.int32),
                max_spec_tokens=max_spec_tokens, max_spec_factor=max_spec_factor,
                min_token_prob=0.0, use_tree_spec=True)
            tok = list(draft.token_ids)
            par = list(draft.parents)
            acc = greedy_tree_walk(tok, par, gt[pos:]) if tok else 0
            match_len = int(getattr(draft, "match_len", 0))
            score = float(getattr(draft, "score", 0.0))
            # longest root->leaf path length = the deepest the draft could reach
            if tok:
                depth = [0] * len(tok)
                for i in range(len(tok)):
                    p = par[i]
                    depth[i] = 1 if (p is None or p < 0 or p == i) else depth[p] + 1
                proposed_depth = max(depth)
            else:
                proposed_depth = 0
        except Exception:
            acc, match_len, score, proposed_depth = 0, 0, 0.0, 0
        steps.append((pos, int(acc), match_len, score, int(proposed_depth)))
        commit = min(acc + 1, len(gt) - pos)
        seg = gt[pos:pos + commit]
        cache.add_active_response(0, [int(t) for t in seg])
        ctx.extend(seg)
        pos += commit
    cache.stop_request(0)

    arr = np.array(steps, dtype=float)
    return {
        "pos": arr[:, 0], "acc": arr[:, 1], "match_len": arr[:, 2],
        "score": arr[:, 3], "proposed": arr[:, 4],
        "out_len": len(gt), "prompt_len": len(prompt),
    }


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--rid", type=int, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-spec-tokens", type=int, default=256)
    ap.add_argument("--max-spec-factor", type=float, default=4.0)
    args = ap.parse_args()

    d = replay_one(args.gt, args.rid, args.max_spec_tokens, args.max_spec_factor)
    acc = d["acc"]
    n_steps = len(acc)
    accepted_tokens = acc.sum()          # speculative tokens saved
    total_tokens = accepted_tokens + n_steps  # +1 baseline token per step
    mat = acc.mean()                     # mean accept length (spec tokens / step)
    toks_per_step = mat + 1.0            # tokens committed / model forward

    # --- console summary -------------------------------------------------
    print(f"[{args.label}]")
    print(f"  out_len={int(d['out_len'])}  steps={n_steps}  "
          f"prompt_len={int(d['prompt_len'])}")
    print(f"  mean accept length={mat:.3f}  tokens/step={toks_per_step:.3f}")
    print(f"  accept==0 steps: {(acc == 0).mean()*100:.1f}%   "
          f"accept>=5: {(acc >= 5).mean()*100:.1f}%   max={int(acc.max())}")
    order = np.argsort(acc)[::-1]
    csum = np.cumsum(acc[order])
    for frac in (0.05, 0.10, 0.20):
        k = max(1, int(np.ceil(frac * n_steps)))
        share = csum[k - 1] / accepted_tokens if accepted_tokens else 0
        print(f"  top {frac*100:.0f}% steps -> {share*100:.1f}% of accepted tokens")
    print(f"  Gini(accept length)={gini(acc):.3f}")

    # --- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0],
                          hspace=0.32, wspace=0.22)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # A) spike train: x = output token position, y = accept length
    ax0.vlines(d["pos"], 0, acc, color=SUFFIX_ORANGE, linewidth=0.9, alpha=0.9)
    ax0.axhline(mat, color=REF_GRAY, lw=1.4, ls="--",
                label=f"mean accept length = {mat:.2f}")
    # annotate the tallest spikes with their context match_len; greedily pick
    # well-separated positions so tied/adjacent spikes don't overprint.
    gap = max(1.0, 0.06 * d["out_len"])
    chosen: list[int] = []
    for idx in order:
        if acc[idx] < 3 or len(chosen) >= 3:
            break
        if all(abs(d["pos"][idx] - d["pos"][j]) > gap for j in chosen):
            chosen.append(idx)
    for rank, idx in enumerate(chosen):
        ax0.annotate(f"acc={int(acc[idx])}\nmatch_len={int(d['match_len'][idx])}",
                     (d["pos"][idx], acc[idx]),
                     textcoords="offset points", xytext=(0, 8 + 4 * rank),
                     fontsize=7.5, ha="center", color=SUFFIX_DARK)
    ax0.set_xlabel("output token position (trajectory)")
    ax0.set_ylabel("accept length (spec tokens accepted this step)")
    ax0.set_title(f"A. Accept-length spike train  —  {args.label}", fontsize=11)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.margins(x=0.005)
    ax0.set_ylim(bottom=0)

    # B) distribution: per accept-length value -> #steps and token-contribution
    maxv = int(acc.max())
    vals = np.arange(0, maxv + 1)
    step_counts = np.array([(acc == v).sum() for v in vals], float)
    tok_contrib = vals * step_counts  # accepted tokens contributed by that value
    w = 0.42
    axb2 = ax1.twinx()
    b1 = ax1.bar(vals - w / 2, step_counts, width=w, color=REF_GRAY_LT,
                 label="# steps")
    b2 = axb2.bar(vals + w / 2, tok_contrib, width=w, color=SUFFIX_ORANGE,
                  label="accepted tokens")
    ax1.set_xlabel("accept length value")
    ax1.set_ylabel("# steps", color=REF_GRAY_LT)
    axb2.set_ylabel("accepted tokens contributed", color=SUFFIX_ORANGE)
    ax1.set_title("B. Distribution: frequency vs token contribution", fontsize=10.5)
    ax1.legend(handles=[b1, b2], loc="upper right", fontsize=8.5)
    ax1.set_xlim(-0.5, maxv + 0.5)

    # C) cumulative contribution (Lorenz/Pareto)
    xfrac = np.arange(1, n_steps + 1) / n_steps
    yfrac = csum / accepted_tokens if accepted_tokens else np.zeros(n_steps)
    ax2.plot(xfrac * 100, yfrac * 100, color=SUFFIX_ORANGE, lw=2)
    ax2.plot([0, 100], [0, 100], color="#a0aec0", ls=":", lw=1)  # equality line
    for frac, c in ((0.10, REF_GRAY), (0.20, "#2c7a7b")):
        k = max(1, int(np.ceil(frac * n_steps)))
        share = (csum[k - 1] / accepted_tokens * 100) if accepted_tokens else 0
        ax2.axvline(frac * 100, color=c, ls="--", lw=1)
        ax2.annotate(f"top {int(frac*100)}% steps\n= {share:.0f}% tokens",
                     (frac * 100, share), fontsize=8, color=c,
                     xytext=(6, -18), textcoords="offset points")
    ax2.set_xlabel("% of steps (sorted by accept length, desc)")
    ax2.set_ylabel("% of accepted tokens (cumulative)")
    ax2.set_title(f"C. Concentration (Pareto)  —  Gini = {gini(acc):.2f}",
                  fontsize=10.5)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)

    fig.suptitle(
        f"Suffix decoding accept-length characterization   "
        f"(steps={n_steps}, out_len={int(d['out_len'])}, "
        f"mean accept={mat:.2f}, tokens/step={toks_per_step:.2f})",
        fontsize=12, y=0.995)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
