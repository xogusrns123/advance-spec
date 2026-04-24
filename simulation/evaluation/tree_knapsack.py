"""Greedy tree walk utility — counts ground-truth-matching tokens in a tree."""

from __future__ import annotations


def greedy_tree_walk(
    token_ids: list[int],
    parents: list[int],
    ground_truth: list[int],
) -> int:
    """Count accepted tokens by greedy tree walk matching ground truth.

    Walks from virtual root, at each level picking the child matching
    the next ground truth token.  Returns consecutive match count.
    """
    accepted = 0
    node = -1  # virtual root
    for gt_token in ground_truth:
        matched = False
        for i in range(len(parents)):
            if parents[i] == node and token_ids[i] == gt_token:
                accepted += 1
                node = i
                matched = True
                break
        if not matched:
            break
    return accepted
