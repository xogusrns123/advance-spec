"""Reslice captured EAGLE3 full pool to (s', k') sub-configurations.

Stage 1 capture (with ``SGLANG_CAPTURE_FULL_POOL=1``) stores the full
score pool produced by SGLang's EAGLE3 draft. Each per-step record's
``eagle3_pool_full`` field contains:

    {
      "draft_tokens": [pool_size]                # token id per pool position
      "parent_list":  [(S-1)*K + 1]              # alive-parent pool positions per step
      "path_probs":   [pool_size + 1]            # cumulative path prob (root at idx 0)
      "pool_size":    K + (S-1)*K²               # total candidates produced
    }

Pool layout (for original config (S, K)):

  * step 0 candidates at positions ``[0, K)`` (= K children of root)
  * step i ≥ 1 candidates at positions ``[K + (i-1)*K², K + i*K²)``
    (= K² children, K alive parents × K children per parent)

``parent_list`` layout (= ``cat(parents_list[:-1])`` from SGLang's
``select_top_k_tokens``):

  * index 0:                       root sentinel (-1)
  * indices 1..K:                  pool positions of step-0's K alive going INTO step 1
                                   (trivially 0..K-1 since step 0 keeps all K)
  * indices K+1+i*K..K+1+(i+1)*K:  K alive going INTO step i+2  (for i in [0, S-2))

Total length: 1 + K + (S-2)*K = (S-1)*K + 1.

Reslicing algorithm for (s', k') with 1 ≤ s' ≤ S, 1 ≤ k' ≤ K:

  step 0:
    keep top-k' of root's K children by path_prob — these are alive going INTO step 1
  step i ≥ 1 (for i in 1..s'-1):
    for each alive parent at depth i:
      gather its K children from pool
      keep top-k' of them by individual score (= path_prob[child] / path_prob[parent])
    among the resulting k'·k' = k'² children, top-k' by path_prob become alive going
    INTO step i+1

The resliced tree size is ``1 + k' + (s' - 1) * k'^2`` (root + s' layers).
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def _build_pool_parents(
    parent_list: List[int], pool_size: int, S: int, K: int,
) -> List[int]:
    """Compute each pool position's parent in pool indexing.

    Returns ``pool_parents`` where ``pool_parents[p]`` is the parent's
    pool position (or -1 if the parent is the verified root).
    """
    expected_pool = K + (S - 1) * K * K
    if pool_size != expected_pool:
        raise ValueError(
            f"pool_size={pool_size} doesn't match K + (S-1)*K² = {expected_pool} "
            f"for (S={S}, K={K})")

    pool_parents = [-1] * pool_size
    for p in range(pool_size):
        if p < K:
            pool_parents[p] = -1
            continue
        step = (p - K) // (K * K) + 1            # 1..S-1
        local_j = (p - K) - (step - 1) * K * K   # 0..K²-1
        alive_slot = local_j // K                # 0..K-1
        if step == 1:
            # parents_list[0] = parent_list[0..K] = [-1, alive_0_pos*K]
            pool_parents[p] = parent_list[1 + alive_slot]
        else:
            # parents_list[step-1] starts at parent_list[K + 1 + (step-2)*K]
            pool_parents[p] = parent_list[
                K + 1 + (step - 2) * K + alive_slot]
    return pool_parents


def _orig_alive_step(parent_list: List[int], step_idx: int, K: int) -> List[int]:
    """The K alive parents going INTO step (step_idx + 1) in the original capture.

    For step_idx == 0: parent_list[1..K+1] (= original step-0 alive, trivially [0..K-1]).
    For step_idx ≥ 1: parent_list[K+1 + (step_idx-1)*K .. K+1 + step_idx*K].
    """
    if step_idx == 0:
        return parent_list[1:K + 1]
    base = K + 1 + (step_idx - 1) * K
    return parent_list[base:base + K]


def reslice_eagle3_pool(
    draft_tokens: List[int],
    parent_list: List[int],
    path_probs: List[float],
    pool_size: int,
    S: int, K: int,
    s_prime: int, k_prime: int,
) -> Tuple[List[int], List[int], List[float]]:
    """Reslice the captured full pool to a (s', k') sub-tree.

    Returns ``(sub_token_ids, sub_parents, sub_path_probs)``:
      * ``sub_token_ids[i]`` — token id of the i-th kept *child* (root NOT included)
      * ``sub_parents[i]`` — parent index in the resliced list, or -1 if child of root
      * ``sub_path_probs[i]`` — cumulative path probability of the i-th kept node

    Format mirrors Stage 1's per-step ``eagle3_tree`` so the simulator's
    ``greedy_tree_walk`` works without modification.

    Tree shape (children only): layer 1 = k', layers 2..s' = k'^2 each.
    Total kept = k' + (s' - 1) * k'^2.

    When (s', k') == (S, K) the result reproduces the original full pool tree.
    """
    if not (1 <= s_prime <= S):
        raise ValueError(f"s_prime must be in [1, {S}], got {s_prime}")
    if not (1 <= k_prime <= K):
        raise ValueError(f"k_prime must be in [1, {K}], got {k_prime}")
    if len(draft_tokens) != pool_size:
        raise ValueError(
            f"len(draft_tokens)={len(draft_tokens)} != pool_size={pool_size}")
    if len(path_probs) != pool_size + 1:
        raise ValueError(
            f"len(path_probs)={len(path_probs)} != pool_size+1={pool_size + 1}")
    expected_parent_len = (S - 1) * K + 1
    if len(parent_list) != expected_parent_len:
        raise ValueError(
            f"len(parent_list)={len(parent_list)} != (S-1)*K+1={expected_parent_len}")

    pool_parents = _build_pool_parents(parent_list, pool_size, S, K)

    # Step 0: top-k' of root's K children by path_prob
    step0_candidates = list(range(K))
    step0_sorted = sorted(
        step0_candidates, key=lambda p: -path_probs[p + 1])
    kept_step_0 = step0_sorted[:k_prime]

    # alive_pool_pos[i] = k' parent positions going INTO step i+1
    # kept_per_step[i] = nodes to add to the tree at depth i+1
    alive_pool_pos: List[List[int]] = [list(kept_step_0)]
    kept_per_step: List[List[int]] = [list(kept_step_0)]

    for i in range(1, s_prime):
        orig_alive = _orig_alive_step(parent_list, i - 1, K)
        slot_of: Dict[int, int] = {pp: s for s, pp in enumerate(orig_alive)}

        candidates_step_i: List[int] = []
        for parent_pp in alive_pool_pos[i - 1]:
            s_orig = slot_of.get(parent_pp)
            if s_orig is None:
                # Resliced alive set must always be a subset of original
                # alive (since we only keep top-k' of original K) — but guard
                # against unexpected data.
                continue
            block_start = K + (i - 1) * K * K + s_orig * K
            children = list(range(block_start, block_start + K))
            parent_pp_prob = path_probs[parent_pp + 1]
            if parent_pp_prob <= 0:
                # Degenerate parent prob — fall back to path-prob ranking
                children_sorted = sorted(
                    children, key=lambda c: -path_probs[c + 1])
            else:
                children_sorted = sorted(
                    children,
                    key=lambda c: -(path_probs[c + 1] / parent_pp_prob))
            candidates_step_i.extend(children_sorted[:k_prime])

        kept_per_step.append(list(candidates_step_i))

        # Top-k' alive going INTO step i+1 by cumulative path prob
        next_alive = sorted(
            candidates_step_i, key=lambda c: -path_probs[c + 1])[:k_prime]
        alive_pool_pos.append(next_alive)

    # Collect all kept pool positions across layers, then sort GLOBALLY by
    # path_prob descending. Sorting by path_prob preserves ancestor closure
    # for free: path_prob[child] = path_prob[parent] · step_score ≤
    # path_prob[parent], so a parent always sorts before any of its
    # descendants. After this sort, ``tids[:B]`` truncation in the simulator
    # picks the *top-B by score*, which matches SGLang's
    # ``organize_draft_results(num_draft_token=B+1)`` (top-(B-1) by score
    # then sorted by index — same set, just different intra-set order; the
    # simulator only cares about the set + parent edges).
    kept_flat: List[int] = []
    for layer in kept_per_step:
        kept_flat.extend(layer)
    kept_sorted = sorted(kept_flat, key=lambda p: -path_probs[p + 1])

    # Build resliced tree — same format as Stage 1's eagle3_tree:
    # token_ids/parents/path_probs are indexed by *kept child* (root NOT
    # included). parents[i] = -1 means "child of root"; otherwise parents[i]
    # is the index of the parent node within the resliced list.
    # Order = global path_prob descending (matches SGLang's organize semantics).
    pool_to_sub: Dict[int, int] = {-1: -1}
    sub_token_ids: List[int] = []
    sub_parents: List[int] = []
    sub_path_probs: List[float] = []

    for pool_pos in kept_sorted:
        sub_idx = len(sub_token_ids)
        pool_to_sub[pool_pos] = sub_idx
        sub_token_ids.append(int(draft_tokens[pool_pos]))
        parent_pool = pool_parents[pool_pos]
        sub_parents.append(pool_to_sub.get(parent_pool, -1))
        sub_path_probs.append(float(path_probs[pool_pos + 1]))

    return sub_token_ids, sub_parents, sub_path_probs
