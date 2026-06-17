#!/usr/bin/env python3
"""Synthetic check for route-b tail append: verify_tree_greedy on a chain
extended beyond S.

Builds a linear chain of ndt' = (S+1) + t tokens with the exact tensor
shapes chain_hybrid_patch._tail_append produces (retrive_index = arange,
next_token = [1..ndt'-1, -1], sibling = -1, accept_index rows =
spec_steps+1 = ndt'), then drives target_predict so the target accepts
exactly j draft tokens, for j sweeping across the head/tail boundary.
Asserts accept_length == j every time — proving the verify kernel walks an
extended chain correctly and the bumped accept_index sizing suffices.

Run inside the sglang-bench container (CUDA kernel):
    python3 simulation/scripts/experiments/test_tail_verify_kernel.py
"""
import torch

from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func

S = 16          # head draft steps
T = 32          # tail tokens appended
NDT = S + 1 + T  # extended draft_token count (root + S head + T tail)
VOCAB = 1000


def run_case(j: int) -> int:
    """Target accepts exactly j draft tokens (chain positions 1..j)."""
    bs = 1
    device = "cuda"
    candidates = torch.arange(100, 100 + NDT, dtype=torch.int64,
                              device=device).view(bs, NDT)
    retrive_index = torch.arange(NDT, dtype=torch.long,
                                 device=device).view(bs, NDT)
    nxt = torch.full((bs, NDT), -1, dtype=torch.long, device=device)
    nxt[0, : NDT - 1] = torch.arange(1, NDT, dtype=torch.long, device=device)
    sib = torch.full((bs, NDT), -1, dtype=torch.long, device=device)

    # target_predict[i] = target's next token after chain position i.
    # Accept chain token i+1 iff target_predict[i] == candidates[i+1].
    target_predict = torch.full((bs, NDT), VOCAB - 1, dtype=torch.int64,
                                device=device)
    for i in range(j):
        target_predict[0, i] = candidates[0, i + 1]

    spec_steps = NDT - 1  # patched value: S + t
    predict = torch.empty((bs * NDT + 1,), dtype=torch.int32, device=device)
    accept_index = torch.full((bs, spec_steps + 1), -1, dtype=torch.int32,
                              device=device)
    accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

    verify_tree_greedy_func(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=accept_length,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=nxt,
        retrive_next_sibling=sib,
        target_predict=target_predict,
        topk=1,
    )
    got = int(accept_length[0].item())
    # accept_index must hold the walked chain prefix: 0..got, rest -1.
    idx = accept_index[0].tolist()
    expect_idx = list(range(got + 1)) + [-1] * (spec_steps - got)
    assert idx == expect_idx, f"j={j}: accept_index {idx[:got+3]}..."
    return got


def main() -> None:
    # Sweep: none, partial head, full head (the old S cap), into the tail,
    # full tail.
    for j in [0, 1, S - 1, S, S + 1, S + T // 2, S + T]:
        got = run_case(j)
        status = "head" if j <= S else "TAIL"
        assert got == j, f"j={j}: accept_length={got}"
        print(f"  j={j:3d} ({status}) accept_length={got:3d} OK")
    print(f"verify_tree_greedy extended-chain test PASSED "
          f"(S={S}, T={T}, ndt'={NDT})")


if __name__ == "__main__":
    main()
