"""Model-free suffix-only accept-length simulation over a recorded GT trajectory.

Suffix decoding proposes from a trie (no model forward), so suffix-only MAT needs
no real serving: replay the recorded GT tokens through a shared
SuffixDecodingCache, and at each step speculate() + greedy-walk vs GT, advancing
by the suffix's OWN accept (step += acc + 1) — its own step structure. Mirrors
simulation/evaluation/run_side_suffix_trajectory.py's _live_suffix_walk, but
walks the raw GT token sequence (gt_tokens.jsonl: {input_ids, output_ids}) instead
of Stage-1 artifacts, so it runs on the SAME trajectory as the serving arms.

Writes timing_suffix.jsonl (accept_lengths schema plot_o4 reads) into --out-dir
and records the suffix MAT in run.json (marked simulated).

Usage:
  python3 simulation/scripts/sim_suffix_only.py \
    --gt simulation/results/o4_perdepth/qwen3_14b_ocalib/gt_tokens.jsonl \
    --out-dir simulation/results/o4_perdepth/qwen3_14b_ocalib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.evaluation.tree_knapsack import greedy_tree_walk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="gt_tokens.jsonl {input_ids,output_ids}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-spec-tokens", type=int, default=256)
    ap.add_argument("--max-spec-factor", type=float, default=4.0)
    args = ap.parse_args()

    from arctic_inference.suffix_decoding import SuffixDecodingCache
    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)

    accs = []
    rid = 0
    n_req = 0
    for line in open(args.gt):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prompt = list(r.get("input_ids") or [])
        gt = list(r.get("output_ids") or [])
        if not gt:
            continue
        n_req += 1
        cache.start_request(rid, np.asarray(prompt, dtype=np.int32))
        ctx = list(prompt)
        pos = 0
        while pos < len(gt):
            try:
                draft = cache.speculate(
                    rid, np.asarray(ctx, dtype=np.int32),
                    max_spec_tokens=args.max_spec_tokens,
                    max_spec_factor=args.max_spec_factor,
                    min_token_prob=0.0, use_tree_spec=True)
                acc = (greedy_tree_walk(list(draft.token_ids), list(draft.parents),
                                        gt[pos:]) if draft.token_ids else 0)
            except Exception:
                acc = 0
            accs.append(int(acc))
            commit = min(acc + 1, len(gt) - pos)
            seg = gt[pos:pos + commit]
            for t in seg:
                cache.add_active_response(rid, [int(t)])
            ctx.extend(seg)
            pos += commit
        cache.stop_request(rid)
        rid += 1

    a = np.asarray(accs, float)
    print(f"requests={n_req}  suffix-only steps={len(a)}  MAT={a.mean():.3f}")
    for d in (1, 2, 3, 5, 8):
        print(f"  survival P(accept>=%d) = %.3f" % (d, (a >= d).mean()))

    out = Path(args.out_dir)
    with open(out / "timing_suffix.jsonl", "w") as f:
        for v in accs:
            f.write(json.dumps({"phase": "decode", "accept_lengths": [int(v)]}) + "\n")
    rj = out / "run.json"
    s = json.load(open(rj))
    s["arms"]["suffix"] = {"accept_length_mean": float(a.mean()),
                           "n_samples": int(len(a)), "simulated": "suffix_trie"}
    json.dump(s, open(rj, "w"), indent=2)
    print(f"wrote timing_suffix.jsonl + updated {rj}")


if __name__ == "__main__":
    main()
