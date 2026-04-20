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
ORACLE_TIMING_PATH = Path("/tmp/sglang_oracle_timing.jsonl")
ORACLE_REPLAY_PATH = os.environ.get("SGLANG_ORACLE_REPLAY", "")
ORACLE_DRAFT_BUDGET = os.environ.get("SGLANG_DRAFT_BUDGET", "")  # override draft token count
ORACLE_VERIFY_TRIES_PATH = os.environ.get("SGLANG_ORACLE_VERIFY_TRIES", "")
ORACLE_VERIFY_RID_MAP_PATH = os.environ.get("SGLANG_ORACLE_VERIFY_RID_MAP", "")


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

def get_oracle_log_position() -> int:
    """Return the current byte offset (end) of the oracle log file.

    Call this before an LLM request to mark where new entries will start.
    """
    try:
        return ORACLE_LOG_PATH.stat().st_size
    except (OSError, FileNotFoundError):
        return 0

def read_oracle_log(start_position: int | None = None) -> list[dict]:
    """Read oracle log entries appended after start_position.

    When start_position is given (from get_oracle_log_position()), only
    entries written after that byte offset are returned.  This enables
    concurrent requests to share the same log file safely — each caller
    reads only its own entries without clearing others'.

    When start_position is None (legacy), all entries are returned.
    """
    entries = []
    try:
        with open(ORACLE_LOG_PATH, "rb" if start_position is not None else "r") as f:
            if start_position is not None:
                f.seek(start_position)
            for line in f:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
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

def _log_timing(entry: dict) -> None:
    try:
        with open(ORACLE_TIMING_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.warning(f"Oracle timing write failed: {e}")

def _load_trajectory(path: str) -> dict[str, list[int]]:
    with open(path) as f:
        return json.load(f)


class TrajectoryState:
    def __init__(self, trajectories: dict[str, list[int]]):
        self.trajectories = trajectories
        self.positions: dict[str, int] = {}
        self._rid_map: dict[str, str] = {}  # new_rid → original_rid
        # Deterministic FIFO queue of unmatched trajectory keys (sorted)
        self._unmatched_queue: list[str] = sorted(trajectories.keys())

    def get_next_token(self, req_id: str) -> int | None:
        orig_rid = self._rid_map.get(req_id)

        # Unknown rid: assign next unmatched trajectory
        if orig_rid is None:
            if self._unmatched_queue:
                orig_rid = self._unmatched_queue.pop(0)
                self._rid_map[req_id] = orig_rid
            else:
                return None

        traj = self.trajectories.get(orig_rid)
        if traj is None:
            return None
        pos = self.positions.get(orig_rid, 0)
        if pos >= len(traj):
            return None
        token = traj[pos]
        self.positions[orig_rid] = pos + 1
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

    # Override draft budget if requested (for latency measurement)
    if ORACLE_DRAFT_BUDGET:
        budget = int(ORACLE_DRAFT_BUDGET)
        original = eagle_worker.speculative_num_draft_tokens
        eagle_worker.speculative_num_draft_tokens = budget
        eagle_worker.server_args.speculative_num_draft_tokens = budget
        logger.info(f"Oracle DRAFT BUDGET override: {original} → {budget}")

    # Verify-tries mode: inject pre-built union tries into draft()
    trie_feeder = None
    if ORACLE_VERIFY_TRIES_PATH:
        from .oracle_verify_patch import UnionTrieFeeder
        trie_feeder = UnionTrieFeeder(ORACLE_VERIFY_TRIES_PATH)

    # LATENCY-ONLY mode: pure timing instrumentation. No force-accept, no tree
    # extraction, no p_t computation — so the server runs REAL speculative
    # decoding (target actually skips accepted draft tokens) and we just measure
    # each step's sub-costs + the real accept_length.
    if os.environ.get("SGLANG_LATENCY_ONLY", "0") == "1":
        _setup_latency_only(eagle_worker)
        return

    _patch_verify_greedy_func()
    _patch_draft_stash(eagle_worker, trie_feeder)
    _patch_verify_logits(eagle_worker, stash_verify_logits=True)

    # Optional: replace eagle_worker.verify + EagleVerifyInput.verify with
    # instrumented versions that time each sub-section (see verify_detail_patch.py).
    if os.environ.get("SGLANG_VERIFY_DETAILED", "0") == "1":
        from .verify_detail_patch import install_detailed_verify_patch
        install_detailed_verify_patch(eagle_worker)

    _patch_forward_log(eagle_worker, replay_state, trie_feeder)

    mode = "VERIFY_TRIES" if trie_feeder else ("REPLAY" if replay_state else "VANILLA")
    logger.info(f"Oracle {mode} patch applied to EAGLEWorker")


def _setup_latency_only(eagle_worker) -> None:
    """Pure timing instrumentation for real speculative decoding.

    Wraps draft / target_forward / verify / forward_batch_generation with
    perf_counter timers and writes per-step timing + accept_length to the
    oracle timing log. Does NOT force accept_length=0 (target actually commits
    accepted draft tokens) and does NOT extract draft tree or p_t.
    """
    import time as _time

    # Draft timer
    original_draft = eagle_worker.draft

    def timed_draft(batch):
        t0 = _time.perf_counter()
        result = original_draft(batch)
        eagle_worker._oracle_last_draft_ms = (_time.perf_counter() - t0) * 1000
        return result

    eagle_worker.draft = timed_draft

    # verify + target_forward timers. stash_verify_logits=False: skip the
    # .cpu().clone() so target_forward_ms reflects only CPU dispatch (the true
    # GPU wall time flows into verify_overhead, and the sum equals real target).
    _patch_verify_logits(eagle_worker, stash_verify_logits=False)

    # Optional detailed verify instrumentation
    if os.environ.get("SGLANG_VERIFY_DETAILED", "0") == "1":
        from .verify_detail_patch import install_detailed_verify_patch
        install_detailed_verify_patch(eagle_worker)

    # Forward-batch timer + JSONL log (timing + accept_length only)
    original_forward = eagle_worker.forward_batch_generation

    def timed_forward(batch):
        t0 = _time.perf_counter()
        result = original_forward(batch)
        step_total_ms = (_time.perf_counter() - t0) * 1000
        eagle_worker._oracle_last_step_total_ms = step_total_ms

        is_decode = not (
            batch.forward_mode.is_extend()
            or getattr(batch, "is_extend_in_batch", False)
        )
        if is_decode:
            draft_ms = getattr(eagle_worker, "_oracle_last_draft_ms", None)
            fwd_ms = getattr(eagle_worker, "_oracle_last_target_forward_ms", None)
            verify_total_ms = getattr(eagle_worker, "_oracle_last_verify_total_ms", None)
            accept_lens = getattr(eagle_worker, "_oracle_last_accept_lengths", []) or []

            entry = {
                "step_total_ms": round(step_total_ms, 3),
                "eagle3_draft_ms": round(draft_ms, 3) if draft_ms is not None else None,
                "target_forward_ms": round(fwd_ms, 3) if fwd_ms is not None else None,
                "verify_total_ms": round(verify_total_ms, 3) if verify_total_ms is not None else None,
                # Real accepted draft tokens per request this step (does not include bonus)
                "accept_lengths": accept_lens,
                # Committed tokens this step = accept_length + 1 per request (bonus)
                "committed_tokens": [n + 1 for n in accept_lens] if accept_lens else [],
            }
            if verify_total_ms is not None and fwd_ms is not None:
                entry["verify_overhead_ms"] = round(verify_total_ms - fwd_ms, 3)
            if (step_total_ms is not None and draft_ms is not None
                    and verify_total_ms is not None):
                entry["post_verify_ms"] = round(
                    step_total_ms - draft_ms - verify_total_ms, 3)

            detail = getattr(eagle_worker, "_oracle_last_verify_detail", None)
            if detail:
                for k, v in detail.items():
                    entry[f"vd_{k}"] = v
                eagle_worker._oracle_last_verify_detail = None

            _log_timing(entry)

            # Reset per-step stashes
            eagle_worker._oracle_last_draft_ms = None
            eagle_worker._oracle_last_target_forward_ms = None
            eagle_worker._oracle_last_verify_total_ms = None
            eagle_worker._oracle_last_step_total_ms = None
            eagle_worker._oracle_last_accept_lengths = []

        return result

    eagle_worker.forward_batch_generation = timed_forward

    logger.info(
        "Oracle LATENCY-ONLY patch applied "
        "(real speculative decoding, timing instrumentation only)"
    )


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
        import time as _time
        _t0 = _time.perf_counter()
        # Run original to get correct target_predict (bonus token)
        predicts, accept_index, accept_token_num = original_func(
            predicts, accept_index, accept_token_num,
            candidates, retrive_index, retrive_next_token,
            retrive_next_sibling, target_predict, topk,
        )
        _t1 = _time.perf_counter()

        # Store timing in module-level var (eagle_worker not in scope)
        eagle_info._oracle_last_greedy_ms = (_t1 - _t0) * 1000

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


def _patch_draft_stash(eagle_worker: "EAGLEWorker", trie_feeder=None) -> None:
    """Patch draft() to stash full draft tree (token_ids + tree structure).

    When trie_feeder is provided (verify-tries mode), the draft tree is
    replaced with the pre-built union trie after running the original draft
    (which is still needed for hidden state updates).
    """
    original_draft = eagle_worker.draft

    def patched_draft(batch: "ScheduleBatch") -> "EagleVerifyInput":
        import time as _time
        _t_draft_start = _time.perf_counter()
        spec_info = original_draft(batch)
        _t_draft_end = _time.perf_counter()
        eagle_worker._oracle_last_draft_ms = (_t_draft_end - _t_draft_start) * 1000

        # Verify-tries mode: replace draft tree with union trie
        if trie_feeder is not None:
            try:
                # Debug: log tensor shapes once
                if not getattr(eagle_worker, "_oracle_shapes_logged", False):
                    logger.info(f"EagleVerifyInput shapes: "
                                f"draft_token={spec_info.draft_token.shape}, "
                                f"custom_mask={spec_info.custom_mask.shape}, "
                                f"positions={spec_info.positions.shape}, "
                                f"retrive_index={spec_info.retrive_index.shape}, "
                                f"retrive_next_token={spec_info.retrive_next_token.shape}, "
                                f"retrive_next_sibling={spec_info.retrive_next_sibling.shape}, "
                                f"draft_token_num={spec_info.draft_token_num}, "
                                f"topk={spec_info.topk}, "
                                f"spec_steps={spec_info.spec_steps}")
                    eagle_worker._oracle_shapes_logged = True
                _inject_union_trie(spec_info, batch, trie_feeder, eagle_worker)
            except Exception as e:
                logger.warning(f"Union trie injection failed: {e}", exc_info=True)

        try:
            dt = spec_info.draft_token
            if dt is not None and dt.numel() > 0:
                eagle_worker._oracle_stashed_draft = dt.cpu().clone()
                eagle_worker._oracle_stashed_num_draft = spec_info.draft_token_num
                eagle_worker._oracle_stashed_topk = spec_info.topk
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


def _inject_union_trie(spec_info, batch, trie_feeder, eagle_worker):
    """Replace draft tree in spec_info with pre-built union trie.

    Pops the next record from the sequential feeder and overwrites
    draft tokens + tree navigation in the EagleVerifyInput.
    """
    import torch

    num_draft = spec_info.draft_token_num
    bs = len(batch.reqs)

    for i in range(bs):
        req = batch.reqs[i]
        req_id = getattr(req, "rid", str(i))
        rec = trie_feeder.get_next_trie()
        if rec is None:
            continue

        trie = rec.get("union_trie", {})
        token_ids = trie.get("token_ids", [])
        parents = trie.get("parents", [])

        # Stash for p_t logging (single request at a time with workers=1)
        eagle_worker._oracle_last_trie_rec = rec

        # Truncate to fit draft_token_num - 1
        max_nodes = num_draft - 1
        if len(token_ids) > max_nodes:
            token_ids = token_ids[:max_nodes]
            parents = parents[:max_nodes]

        n = len(token_ids)
        if n == 0:
            continue

        # Overwrite draft tokens for this request
        # Layout: [root_verified, draft_0, draft_1, ...] per request
        req_start = i * num_draft
        for j in range(n):
            spec_info.draft_token[req_start + 1 + j] = token_ids[j]

        # Rebuild tree navigation (retrive_next_token, retrive_next_sibling)
        # from flat (token_ids, parents) to child/sibling linked list
        children = {}  # parent_idx → list of child indices (in trie order)
        for j in range(n):
            p = parents[j]
            children.setdefault(p, []).append(j)

        rnt = spec_info.retrive_next_token
        rns = spec_info.retrive_next_sibling

        # Clear this request's navigation
        rnt[i].fill_(-1)
        rns[i].fill_(-1)

        req_offset = i * num_draft

        # Set first child of root (position 0)
        root_children = children.get(-1, [])
        if root_children:
            rnt[i, 0] = req_offset + 1 + root_children[0]

        # Set child/sibling for each node
        for j in range(n):
            pos = j + 1  # +1 because position 0 is root
            node_children = children.get(j, [])
            if node_children:
                rnt[i, pos] = req_offset + 1 + node_children[0]
            # Sibling: next child of same parent
            p = parents[j]
            siblings = children.get(p, [])
            my_idx = siblings.index(j)
            if my_idx + 1 < len(siblings):
                rns[i, pos] = req_offset + 1 + siblings[my_idx + 1]


def _patch_verify_logits(eagle_worker: "EAGLEWorker",
                         stash_verify_logits: bool = True) -> None:
    """Patch verify() to measure total verify time + stash accept lengths.

    `stash_verify_logits` controls whether the target's verify logits are
    copied to CPU for later p_t extraction. In latency-only mode this should
    be False to avoid the .cpu().clone() sync overhead contaminating the
    target forward timing.
    """
    original_verify = eagle_worker.verify

    def patched_verify(batch, spec_info):
        import time as _time
        _t0 = _time.perf_counter()
        result = original_verify(batch, spec_info)
        _t1 = _time.perf_counter()
        eagle_worker._oracle_last_verify_total_ms = (_t1 - _t0) * 1000
        # Stash real accept_length per request (list of ints; does NOT include bonus)
        try:
            _, _res, _, _ = result
            eagle_worker._oracle_last_accept_lengths = list(
                getattr(_res, "accept_length_per_req_cpu", []) or [])
        except Exception:
            eagle_worker._oracle_last_accept_lengths = []
        return result

    eagle_worker.verify = patched_verify

    # Patch target_worker.forward_batch_generation for target forward timing + (optionally) logit stashing
    original_target_forward = eagle_worker.target_worker.forward_batch_generation

    def patched_target_forward(model_worker_batch, is_verify=False):
        import time as _time
        _t_fwd_start = _time.perf_counter()
        result = original_target_forward(model_worker_batch, is_verify=is_verify)
        _t_fwd_end = _time.perf_counter()
        if is_verify:
            eagle_worker._oracle_last_target_forward_ms = (
                _t_fwd_end - _t_fwd_start) * 1000
        if stash_verify_logits and is_verify and result.logits_output is not None:
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


VERIFY_P_T_LOG_PATH = Path("/tmp/sglang_oracle_verify_p_t.jsonl")


def _patch_forward_log(
    eagle_worker: "EAGLEWorker",
    replay_state: TrajectoryState | None,
    trie_feeder=None,
) -> None:
    """Patch forward_batch_generation() to log draft + accepted token."""
    original_forward = eagle_worker.forward_batch_generation

    def patched_forward(batch: "ScheduleBatch") -> "GenerationBatchResult":
        import time as _time
        import torch

        _t_step_start = _time.perf_counter()
        result = original_forward(batch)
        _t_step_end = _time.perf_counter()
        eagle_worker._oracle_last_step_total_ms = (_t_step_end - _t_step_start) * 1000

        # Only process decode steps
        is_decode = not (
            batch.forward_mode.is_extend()
            or getattr(batch, "is_extend_in_batch", False)
        )
        if not is_decode:
            return result

        tp_rank = getattr(eagle_worker, "tp_rank", 0)

        # Replay: override tokens on ALL TP ranks to keep them in sync
        if replay_state is not None and is_decode:
            accept_lengths_r = getattr(result, "accept_length_per_req_cpu", None)
            if accept_lengths_r:
                num_reqs_r = len(accept_lengths_r)
                for i in range(num_reqs_r):
                    req = batch.reqs[i] if i < len(batch.reqs) else None
                    req_id = getattr(req, "rid", str(i)) if req else str(i)
                    forced = replay_state.get_next_token(req_id)
                    if forced is not None and req and req.output_ids:
                        req.output_ids[-1] = forced

        # Only log on TP rank 0
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

            num_reqs = len(accept_lengths) if accept_lengths else len(verified_cpu)

            v_offset = 0
            for i in range(num_reqs):
                accept_len = accept_lengths[i] if accept_lengths else 0
                n_tokens = accept_len + 1
                req_accepted = verified_cpu[v_offset:v_offset + n_tokens]
                v_offset += n_tokens

                vanilla_token = req_accepted[0] if req_accepted else 0

                req = batch.reqs[i] if i < len(batch.reqs) else None
                req_id = getattr(req, "rid", str(i)) if req else str(i)

                # Update vanilla_token for logging (already overridden above)
                if replay_state is not None and req and req.output_ids:
                    vanilla_token = req.output_ids[-1]

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

                # Verify-tries mode: log p_t for union trie nodes
                if trie_feeder is not None and verify_logits is not None:
                    trie_rec = getattr(eagle_worker, "_oracle_last_trie_rec", None)
                    if trie_rec is not None:
                        _log_verify_trie_p_t(
                            trie_rec, verify_logits, i, num_draft)

            eagle_worker._oracle_stashed_draft = None
            eagle_worker._oracle_stashed_verify_logits = None

            # Log per-step decomposed timing
            import sglang.srt.speculative.eagle_info as _eagle_info
            draft_ms = getattr(eagle_worker, "_oracle_last_draft_ms", None)
            fwd_ms = getattr(eagle_worker, "_oracle_last_target_forward_ms", None)
            verify_total_ms = getattr(eagle_worker, "_oracle_last_verify_total_ms", None)
            step_total_ms = getattr(eagle_worker, "_oracle_last_step_total_ms", None)
            greedy_ms = getattr(_eagle_info, "_oracle_last_greedy_ms", None)

            if draft_ms is not None or fwd_ms is not None:
                # Decompose verify: pre-forward + forward + greedy + post-greedy
                verify_pre_ms = None
                verify_post_ms = None
                post_verify_ms = None
                if verify_total_ms is not None and fwd_ms is not None:
                    # verify_total = verify_pre + target_forward + greedy + verify_post
                    greedy_t = greedy_ms or 0
                    verify_other = verify_total_ms - fwd_ms - greedy_t
                    # We can't easily split pre/post without more hooks,
                    # but we log verify_other = pre + post
                    verify_pre_ms = None  # unknown split
                    verify_post_ms = None
                if step_total_ms is not None and draft_ms is not None and verify_total_ms is not None:
                    post_verify_ms = step_total_ms - draft_ms - verify_total_ms

                entry = {
                    "eagle3_draft_ms": draft_ms,
                    "target_forward_ms": fwd_ms,
                    "verify_total_ms": verify_total_ms,
                    "verify_greedy_ms": greedy_ms,
                    "verify_overhead_ms": round(verify_total_ms - fwd_ms - (greedy_ms or 0), 3) if verify_total_ms and fwd_ms else None,
                    "step_total_ms": step_total_ms,
                    "post_verify_ms": round(post_verify_ms, 3) if post_verify_ms is not None else None,
                    "num_draft": num_draft,
                    "num_reqs": num_reqs,
                }
                # Flatten the verify-detail dict into the log row if present
                detail = getattr(eagle_worker, "_oracle_last_verify_detail", None)
                if detail:
                    for k, v in detail.items():
                        entry[f"vd_{k}"] = v
                    eagle_worker._oracle_last_verify_detail = None
                _log_timing(entry)
                eagle_worker._oracle_last_draft_ms = None
                eagle_worker._oracle_last_target_forward_ms = None
                eagle_worker._oracle_last_verify_total_ms = None
                eagle_worker._oracle_last_step_total_ms = None
                _eagle_info._oracle_last_greedy_ms = None

        except Exception as e:
            logger.warning(f"Oracle logging failed: {e}")

        return result

    eagle_worker.forward_batch_generation = patched_forward


def _log_verify_trie_p_t(rec, verify_logits, req_idx, num_draft):
    """Log p_t for union trie nodes from verification logits."""
    import torch.nn.functional as F

    trie = rec.get("union_trie", {})
    token_ids = trie.get("token_ids", [])
    parents = trie.get("parents", [])
    n = min(len(token_ids), num_draft - 1)
    if n == 0:
        return

    req_offset = req_idx * num_draft
    p_t = []
    for j in range(n):
        tid = token_ids[j]
        if parents[j] == -1:
            parent_pos = req_offset + 0  # root position
        else:
            parent_pos = req_offset + 1 + parents[j]

        if parent_pos >= len(verify_logits):
            p_t.append(0.0)
            continue

        probs = F.softmax(verify_logits[parent_pos].float(), dim=-1)
        p_t.append(probs[tid].item())

    entry = {
        "request_id": rec.get("request_id", ""),
        "call_idx": rec.get("call_idx", 0),
        "step_idx": rec.get("step_idx", 0),
        "p_t": p_t,
    }
    try:
        with open(VERIFY_P_T_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
