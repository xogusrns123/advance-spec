"""Verify step — offline greedy GT-walk (user-selected reproduction of the
missing ``tree_verify_hybrid``).

Dr. Lee's spec calls ``TVH.tree_verify(target, ctx_root, tree, cfg)`` expecting a
LIVE masked forward through the 27B hybrid (GatedDeltaNet+attention), returning
the accepted node path + the bonus token. That source was not shipped, and a
correct tree-mask over the linear-attention layers is research-grade on this
arch. Since greedy speculative decoding is *lossless* — the accepted set is
exactly the longest draft-tree path matching the target's greedy continuation —
we verify by walking the tree against the target-greedy trace. This yields the
SAME accept length K as a live verify (composition.md/controller.md rely on the
same "spec-decode == AR greedy" equivalence), needs only plain causal target
forwards (robust on Blackwell 27B), and keeps ``run_partialwarm_tree.py``
unmodified.

The signature is preserved exactly: ``target`` is the ``TargetWrapper`` from
``dflash_candidates.load`` (it can produce the greedy continuation from any
cached-prefix ctx_root); ``cfg`` (the {Hk,Hv,Dk,Dv,Keep} GatedDeltaNet dims a
live verify would need) is accepted and ignored.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

sys.path.insert(0, "/workspace")
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402


def _tree_max_depth(tree) -> int:
    d = 0
    for i in range(len(tree.tokens)):
        d = max(d, tree.depth_of(i))
    return d


def tree_verify(target, ctx_root, tree, cfg) -> Tuple[List[int], int, None]:
    """Greedy tree-walk verify.

    Returns (accepted_node_indices, bonus_token, None):
      - accepted_node_indices: nodes of ``tree`` on the longest root→node path
        whose tokens match the target-greedy continuation from ctx_root
        (== accept length K = len(this list)).
      - bonus_token: the target's next greedy token AFTER the accepted prefix
        (the free token every verify round commits). At EOS the continuation
        ends, so the bonus is the trace's final (EOS) token.
    """
    need = max(1, _tree_max_depth(tree) + 1)
    gt_cont = target.greedy_continuation(ctx_root, need)   # list[int], target-greedy
    if not gt_cont:
        # ctx_root already at EOS/end — nothing to accept, no bonus token.
        return [], int(target.eos_token_id), None
    path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt_cont)
    if len(path) < len(gt_cont):
        bonus = int(gt_cont[len(path)])
    else:
        # Tree matched the entire fetched continuation; fetch one more token.
        more = target.greedy_continuation(ctx_root, len(path) + 1)
        bonus = int(more[len(path)]) if len(more) > len(path) else int(target.eos_token_id)
    return list(path), bonus, None
