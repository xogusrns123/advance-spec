"""
Oracle Vanilla Patch for SGLang EAGLE3/MTP Worker.

Patches verify_tree_greedy_func to force accept_length=0, ensuring
1-token/step advance with correct batch state updates.
Works for both EAGLEWorker (EAGLE3) and MultiLayerEagleWorker (MTP)
since they share the same EagleVerifyInput format.

Two modes:
- Vanilla (SGLANG_ORACLE_VANILLA=1): generate trajectory + log drafts
- Replay (SGLANG_ORACLE_REPLAY=path): follow pre-recorded trajectory + log drafts

Output format (per line):
    {"eagle3": [[flat_draft_ids...]], "tokens": [[next_token]],
     "req_id": "...", "proposer": "eagle3"|"mtp",
     "eagle3_tree": {"token_ids": [...], "parents": [...]}}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleVerifyInput, EagleVerifyOutput
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

logger = logging.getLogger(__name__)

ORACLE_LOG_PATH = Path("/tmp/sglang_oracle_vanilla.jsonl")
ORACLE_REPLAY_PATH = os.environ.get("SGLANG_ORACLE_REPLAY", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_oracle_enabled() -> bool:
    return os.environ.get("SGLANG_ORACLE_VANILLA", "0") == "1"

def is_replay_mode() -> bool:
    return bool(ORACLE_REPLAY_PATH)

def clear_oracle_log() -> None:
    try:
        open(ORACLE_LOG_PATH, "w").close()
    except OSError:
        pass

def read_oracle_log() -> list[dict]:
    entries = []
    try:
        with open(ORACLE_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        pass
    return entries

def _log_entry(entry: dict) -> None:
    try:
        with open(ORACLE_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.warning(f"Oracle log write failed: {e}")

def _load_trajectory(path: str) -> dict[str, list[int]]:
    with open(path) as f:
        return json.load(f)


class TrajectoryState:
    def __init__(self, trajectories: dict[str, list[int]]):
        self.trajectories = trajectories
        self.positions: dict[str, int] = {}

    def get_next_token(self, req_id: str) -> int | None:
        if req_id not in self.trajectories:
            return None
        pos = self.positions.get(req_id, 0)
        traj = self.trajectories[req_id]
        if pos >= len(traj):
            return None
        token = traj[pos]
        self.positions[req_id] = pos + 1
        return token


# ---------------------------------------------------------------------------
# Tree structure extraction
# ---------------------------------------------------------------------------

def _extract_eagle3_tree(
    draft_cpu: list,
    retrive_next_token: "torch.Tensor",
    retrive_next_sibling: "torch.Tensor",
    req_idx: int,
    num_draft: int,
) -> dict | None:
    """Extract (token_ids, parents) from EAGLE3 draft tree.

    Uses retrive_next_token (first child) and retrive_next_sibling
    (next sibling) to walk the tree structure via BFS.

    Position 0 in each request's candidates is the verified root token.
    Positions 1..num_draft-1 are draft nodes.

    Returns {"token_ids": [...], "parents": [...]} where parents use
    -1 for children of root, and indices are 0-based into the returned
    token_ids (which excludes the root).
    """
    try:
        req_start = req_idx * num_draft
        req_end = req_start + num_draft

        if req_end > len(draft_cpu):
            return None

        candidates = draft_cpu[req_start:req_end]
        rnt = retrive_next_token[req_idx].tolist()   # first child
        rns = retrive_next_sibling[req_idx].tolist()  # next sibling

        # Global offset for this request in the batch
        req_offset = req_idx * num_draft

        # BFS from root (position 0) to collect tree structure
        # pos_to_idx: maps candidates position → output index
        token_ids = []
        parents = []
        candidates_positions = []  # candidates pos for each output node
        pos_to_idx = {}

        # Queue: (candidates_position, parent_output_index)
        # Start with root's children
        queue = []
        first_child = rnt[0]
        if first_child >= 0:
            # Convert global index to request-local
            local_first = first_child - req_offset
            queue.append((local_first, -1))

        while queue:
            pos, parent_idx = queue.pop(0)

            if pos < 0 or pos >= num_draft:
                continue
            tid = candidates[pos]
            if tid == 0:
                continue

            idx = len(token_ids)
            pos_to_idx[pos] = idx
            token_ids.append(tid)
            parents.append(parent_idx)
            candidates_positions.append(pos)

            # First child of this node
            child = rnt[pos]
            if child >= 0:
                local_child = child - req_offset
                queue.append((local_child, idx))

            # Next sibling (shares same parent)
            sib = rns[pos]
            if sib >= 0:
                local_sib = sib - req_offset
                queue.append((local_sib, parent_idx))

        if not token_ids:
            return None

        # BFS guarantees parent[i] < i (enables safe truncation)
        for i, p in enumerate(parents):
            if p != -1 and p >= i:
                return None  # safety: skip malformed tree

        return {
            "token_ids": token_ids,
            "parents": parents,
            "candidates_positions": candidates_positions,
        }
    except Exception:
        return None


def _extract_tree_p_t(
    verify_logits: "torch.Tensor",
    draft_cpu: list,
    eagle3_tree: dict,
    req_idx: int,
    num_draft: int,
) -> list | None:
    """Extract per-node p_t from verification logits.

    verify_logits shape: [bs * num_draft, vocab_size]
    Each position's logits predict the NEXT token after that position.
    So p_t(node) = softmax(logits[parent_candidates_pos])[node_token_id].

    eagle3_tree contains candidates_positions mapping each tree node
    to its candidates position (0-based within this request).
    """
    import torch
    import torch.nn.functional as F

    try:
        token_ids = eagle3_tree["token_ids"]
        parents = eagle3_tree["parents"]
        cand_pos = eagle3_tree.get("candidates_positions")
        n = len(token_ids)

        if n == 0 or cand_pos is None:
            return None

        req_offset = req_idx * num_draft
        p_t = []

        for i in range(n):
            tid = token_ids[i]

            if parents[i] == -1:
                # Root child: parent is candidates position 0
                parent_global = req_offset + 0
            else:
                # Parent is tree node parents[i], whose candidates pos is cand_pos[parents[i]]
                parent_global = req_offset + cand_pos[parents[i]]

            if parent_global >= len(verify_logits):
                p_t.append(0.0)
                continue

            probs = F.softmax(verify_logits[parent_global].float(), dim=-1)
            p_t.append(probs[tid].item())

        return p_t
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main patch
# ---------------------------------------------------------------------------

def _detect_proposer_type(eagle_worker) -> str:
    """Detect whether the worker is EAGLE3 or MTP."""
    cls_name = type(eagle_worker).__name__
    if "MultiLayer" in cls_name:
        return "mtp"
    return "eagle3"


def patch_eagle_worker_full(eagle_worker: "EAGLEWorker") -> None:
    """Apply oracle patch.  Works for both EAGLEWorker and MultiLayerEagleWorker.

    Strategy: Patch verify_tree_greedy_func (the core acceptance function)
    to always return accept_length=0. This way verify() naturally processes
    only 1 token, and ALL batch state updates (output_ids, kv_committed_len,
    hidden_states, accept_length) are correctly set for 1-token advance.

    No rollback needed — the acceptance decision is intercepted at the source.
    """
    eagle_worker._oracle_proposer_type = _detect_proposer_type(eagle_worker)
    replay_state = None
    if ORACLE_REPLAY_PATH:
        trajectory = _load_trajectory(ORACLE_REPLAY_PATH)
        replay_state = TrajectoryState(trajectory)
        logger.info(f"Oracle REPLAY: {len(trajectory)} trajectories")

    _patch_verify_greedy_func()
    _patch_draft_stash(eagle_worker)
    _patch_verify_logits(eagle_worker)
    _patch_forward_log(eagle_worker, replay_state)

    mode = "REPLAY" if replay_state else "VANILLA"
    logger.info(f"Oracle {mode} patch applied to EAGLEWorker")


def _patch_verify_greedy_func() -> None:
    """Patch verify_tree_greedy_func to force accept_length=0.

    This function is called inside EagleVerifyInput.verify() to determine
    which draft tokens to accept. By forcing accept_length=0, verify()
    only accepts the bonus token (target model's argmax at root position).
    """
    import sglang.srt.speculative.eagle_info as eagle_info

    if getattr(eagle_info, "_oracle_patched_greedy", False):
        return  # Already patched

    original_func = eagle_info.verify_tree_greedy_func

    def patched_verify_tree_greedy_func(
        predicts, accept_index, accept_token_num,
        candidates, retrive_index, retrive_next_token,
        retrive_next_sibling, target_predict, topk,
    ):
        # Run original to get correct target_predict (bonus token)
        predicts, accept_index, accept_token_num = original_func(
            predicts, accept_index, accept_token_num,
            candidates, retrive_index, retrive_next_token,
            retrive_next_sibling, target_predict, topk,
        )

        # Force accept_length=0: only keep the first accepted token (bonus)
        # accept_index[i] = [first_idx, -1, -1, ...] → only bonus token
        bs = accept_index.shape[0]
        first_col = accept_index[:, 0].clone()
        accept_index.fill_(-1)
        accept_index[:, 0] = first_col
        accept_token_num.fill_(0)

        return predicts, accept_index, accept_token_num

    eagle_info.verify_tree_greedy_func = patched_verify_tree_greedy_func
    eagle_info._oracle_patched_greedy = True
    logger.info("Patched verify_tree_greedy_func for oracle vanilla (accept_length=0)")


def _patch_draft_stash(eagle_worker: "EAGLEWorker") -> None:
    """Patch draft() to stash full draft tree (token_ids + tree structure)."""
    original_draft = eagle_worker.draft

    def patched_draft(batch: "ScheduleBatch") -> "EagleVerifyInput":
        spec_info = original_draft(batch)

        try:
            dt = spec_info.draft_token
            if dt is not None and dt.numel() > 0:
                eagle_worker._oracle_stashed_draft = dt.cpu().clone()
                eagle_worker._oracle_stashed_num_draft = spec_info.draft_token_num
                eagle_worker._oracle_stashed_topk = spec_info.topk
                # Stash tree navigation indices for full tree extraction
                eagle_worker._oracle_stashed_retrive_next_token = (
                    spec_info.retrive_next_token.cpu().clone()
                    if getattr(spec_info, "retrive_next_token", None) is not None
                    else None
                )
                eagle_worker._oracle_stashed_retrive_next_sibling = (
                    spec_info.retrive_next_sibling.cpu().clone()
                    if getattr(spec_info, "retrive_next_sibling", None) is not None
                    else None
                )
            else:
                eagle_worker._oracle_stashed_draft = None
                eagle_worker._oracle_stashed_retrive_next_token = None
                eagle_worker._oracle_stashed_retrive_next_sibling = None
        except Exception:
            eagle_worker._oracle_stashed_draft = None
            eagle_worker._oracle_stashed_retrive_next_token = None
            eagle_worker._oracle_stashed_retrive_next_sibling = None

        return spec_info

    eagle_worker.draft = patched_draft


def _patch_verify_logits(eagle_worker: "EAGLEWorker") -> None:
    """Patch verify() to stash full target-model logits before acceptance filtering."""
    original_verify = eagle_worker.verify

    def patched_verify(batch, spec_info):
        result = original_verify(batch, spec_info)
        # result is (logits_output, verify_output, model_worker_batch, can_run_cuda_graph)
        # At this point logits_output.next_token_logits is already filtered to accepted_indices.
        # But we need the FULL logits before filtering.
        # Unfortunately verify() modifies logits_output in-place.
        # So we need to capture them inside verify() before the filtering.
        # Alternative: we stash from the target_worker forward result directly.
        return result

    # Instead of patching verify(), patch target_worker.forward_batch_generation
    # to capture logits before verify filters them.
    original_target_forward = eagle_worker.target_worker.forward_batch_generation

    def patched_target_forward(model_worker_batch, is_verify=False):
        result = original_target_forward(model_worker_batch, is_verify=is_verify)
        if is_verify and result.logits_output is not None:
            try:
                logits = result.logits_output.next_token_logits
                if logits is not None and logits.numel() > 0:
                    eagle_worker._oracle_stashed_verify_logits = logits.cpu().clone()
                else:
                    eagle_worker._oracle_stashed_verify_logits = None
            except Exception:
                eagle_worker._oracle_stashed_verify_logits = None
        return result

    eagle_worker.target_worker.forward_batch_generation = patched_target_forward


def _patch_forward_log(
    eagle_worker: "EAGLEWorker",
    replay_state: TrajectoryState | None,
) -> None:
    """Patch forward_batch_generation() to log draft + accepted token."""
    original_forward = eagle_worker.forward_batch_generation

    def patched_forward(batch: "ScheduleBatch") -> "GenerationBatchResult":
        import torch

        result = original_forward(batch)

        # Only process decode steps
        is_decode = not (
            batch.forward_mode.is_extend()
            or getattr(batch, "is_extend_in_batch", False)
        )
        if not is_decode:
            return result

        # Only log on TP rank 0 to avoid duplicate entries
        tp_rank = getattr(eagle_worker, "tp_rank", 0)
        if tp_rank != 0:
            return result

        try:
            stashed_draft = getattr(eagle_worker, "_oracle_stashed_draft", None)
            if stashed_draft is None:
                return result

            accept_lengths = getattr(result, "accept_length_per_req_cpu", None)
            verified_ids = result.next_token_ids
            if verified_ids is None or verified_ids.numel() == 0:
                return result

            verified_cpu = verified_ids.cpu().tolist()
            draft_cpu = stashed_draft.tolist()
            num_draft = getattr(eagle_worker, "_oracle_stashed_num_draft", 16)

            # Determine actual batch size (not inflated by speculative topk)
            num_reqs = len(accept_lengths) if accept_lengths else len(verified_cpu)

            # Log per-request (use num_reqs, not len(batch.reqs))
            v_offset = 0
            for i in range(num_reqs):
                accept_len = accept_lengths[i] if accept_lengths else 0
                n_tokens = accept_len + 1
                req_accepted = verified_cpu[v_offset:v_offset + n_tokens]
                v_offset += n_tokens

                vanilla_token = req_accepted[0] if req_accepted else 0

                req = batch.reqs[i] if i < len(batch.reqs) else None
                req_id = getattr(req, "rid", str(i)) if req else str(i)

                # Replay: override token
                if replay_state is not None:
                    forced = replay_state.get_next_token(req_id)
                    if forced is not None:
                        vanilla_token = forced
                        if req and req.output_ids:
                            req.output_ids[-1] = forced

                # draft_token layout: [root_verified, draft_0, draft_1, ...]
                # root_verified is the previous step's token, skip it
                d_start = i * num_draft + 1  # +1 to skip root
                d_end = (i + 1) * num_draft
                req_draft = draft_cpu[d_start:d_end] if d_end <= len(draft_cpu) else []

                # Extract full tree structure
                eagle3_tree = None
                rnt = getattr(eagle_worker, "_oracle_stashed_retrive_next_token", None)
                rns = getattr(eagle_worker, "_oracle_stashed_retrive_next_sibling", None)
                if rnt is not None and rns is not None:
                    eagle3_tree = _extract_eagle3_tree(
                        draft_cpu, rnt, rns, i, num_draft)

                # Extract per-node p_t from verification logits
                tree_p_t = None
                verify_logits = getattr(eagle_worker, "_oracle_stashed_verify_logits", None)
                if verify_logits is not None and eagle3_tree is not None:
                    tree_p_t = _extract_tree_p_t(
                        verify_logits, draft_cpu, eagle3_tree, i, num_draft)

                proposer = getattr(eagle_worker, "_oracle_proposer_type", "eagle3")
                entry = {
                    "eagle3": [req_draft],
                    "tokens": [[vanilla_token]],
                    "req_id": req_id,
                    "proposer": proposer,
                }
                if eagle3_tree is not None:
                    # Don't store internal candidates_positions in log
                    entry["eagle3_tree"] = {
                        "token_ids": eagle3_tree["token_ids"],
                        "parents": eagle3_tree["parents"],
                    }
                if tree_p_t is not None:
                    entry["eagle3_tree_p_t"] = tree_p_t
                _log_entry(entry)

            eagle_worker._oracle_stashed_draft = None
            eagle_worker._oracle_stashed_verify_logits = None

        except Exception as e:
            logger.warning(f"Oracle logging failed: {e}")

        return result

    eagle_worker.forward_batch_generation = patched_forward
