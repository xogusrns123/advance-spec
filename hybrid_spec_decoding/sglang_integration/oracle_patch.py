"""
Oracle Vanilla Patch for SGLang EAGLE3/MTP Worker.

Patches verify_tree_greedy_func to force accept_length=0, ensuring
1-token/step advance with correct batch state updates.

Two modes:
- Vanilla (SGLANG_ORACLE_VANILLA=1): generate trajectory + log drafts
- Replay (SGLANG_ORACLE_REPLAY=path): follow pre-recorded trajectory + log drafts

Output format (per line):
    {"eagle3": [[flat_draft_ids...]], "tokens": [[next_token]], "req_id": "..."}
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
# Main patch
# ---------------------------------------------------------------------------

def patch_eagle_worker_full(eagle_worker: "EAGLEWorker") -> None:
    """Apply oracle patch.

    Strategy: Patch verify_tree_greedy_func (the core acceptance function)
    to always return accept_length=0. This way verify() naturally processes
    only 1 token, and ALL batch state updates (output_ids, kv_committed_len,
    hidden_states, accept_length) are correctly set for 1-token advance.

    No rollback needed — the acceptance decision is intercepted at the source.
    """
    replay_state = None
    if ORACLE_REPLAY_PATH:
        trajectory = _load_trajectory(ORACLE_REPLAY_PATH)
        replay_state = TrajectoryState(trajectory)
        logger.info(f"Oracle REPLAY: {len(trajectory)} trajectories")

    _patch_verify_greedy_func()
    _patch_draft_stash(eagle_worker)
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
    """Patch draft() to stash full draft tree."""
    original_draft = eagle_worker.draft

    def patched_draft(batch: "ScheduleBatch") -> "EagleVerifyInput":
        spec_info = original_draft(batch)

        try:
            dt = spec_info.draft_token
            if dt is not None and dt.numel() > 0:
                eagle_worker._oracle_stashed_draft = dt.cpu().clone()
                eagle_worker._oracle_stashed_num_draft = spec_info.draft_token_num
                eagle_worker._oracle_stashed_topk = spec_info.topk
            else:
                eagle_worker._oracle_stashed_draft = None
        except Exception:
            eagle_worker._oracle_stashed_draft = None

        return spec_info

    eagle_worker.draft = patched_draft


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
                # TODO: store full tree structure for accurate simulation
                d_start = i * num_draft + 1  # +1 to skip root
                d_end = (i + 1) * num_draft
                req_draft = draft_cpu[d_start:d_end] if d_end <= len(draft_cpu) else []

                # TODO: add "eagle3_tree": {"token_ids": [...], "parents": [...]}
                _log_entry({
                    "eagle3": [req_draft],
                    "tokens": [[vanilla_token]],
                    "req_id": req_id,
                })

            eagle_worker._oracle_stashed_draft = None

        except Exception as e:
            logger.warning(f"Oracle logging failed: {e}")

        return result

    eagle_worker.forward_batch_generation = patched_forward
