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


def _extract_tree_path_draft_p_t(
    path_probs: "torch.Tensor",
    eagle3_tree: dict,
    req_idx: int,
) -> list | None:
    """Per-tree-node cumulative path probability from the DRAFT model.

    path_probs shape: (bs, num_draft). Each cell is the path probability
    (product of per-edge conditional probs along root→node) that EAGLE3's
    internal ranking produced. path_probs[req, 0] = 1.0 (root); for
    positions 1..num_draft-1 it's the cumulative prob of the selected
    tree node at that candidates position.

    eagle3_tree["candidates_positions"][bfs_idx] tells us which candidates
    position each BFS-ordered node occupies; we just look it up.
    """
    try:
        cand_pos = eagle3_tree.get("candidates_positions")
        if cand_pos is None:
            return None
        row = path_probs[req_idx]
        n = len(eagle3_tree["token_ids"])
        out = []
        for i in range(n):
            cp = cand_pos[i]
            if cp < 0 or cp >= row.shape[0]:
                out.append(0.0)
            else:
                out.append(float(row[cp].item()))
        return out
    except Exception:
        return None


def _install_draft_p_t_tracer() -> None:
    """Monkey-patch organize_draft_results to stash the per-position path
    probability tensor on the eagle_worker module.

    SGLang's organize_draft_results selects the top-N candidate tree
    positions by cumulative path probability. We compute those same
    probabilities alongside, via torch.gather on the same flat score_list
    concatenation, and stash them so patched_draft can use them per-request.
    """
    import sglang.srt.speculative.eagle_worker as ew_module
    if getattr(ew_module, "_oracle_organize_traced", False):
        return
    original = ew_module.organize_draft_results

    def traced(score_list, token_list, parents_list, num_draft_token):
        import torch
        try:
            flat_scores = torch.cat(score_list, dim=1).flatten(1)  # (bs, total)
        except Exception as e:
            if not getattr(ew_module, "_oracle_tracer_logged_err1", False):
                logger.warning(f"draft-p_t tracer: cat/flatten failed: {e}")
                ew_module._oracle_tracer_logged_err1 = True
            flat_scores = None
        parent_list, top_scores_index, draft_tokens = original(
            score_list, token_list, parents_list, num_draft_token)
        # Compute per-candidate-position path prob:
        # draft_tokens[req, i] corresponds to top_scores_index[req, i-1] for i >= 1
        # (tree position 0 is the verified root, prob = 1.0).
        try:
            if flat_scores is not None:
                path_probs = torch.gather(
                    flat_scores, 1, top_scores_index)  # (bs, num_draft-1)
                bs = path_probs.shape[0]
                root_col = torch.ones(
                    bs, 1, device=path_probs.device, dtype=path_probs.dtype)
                path_probs_full = torch.cat([root_col, path_probs], dim=1)
                ew_module._oracle_last_path_probs = \
                    path_probs_full.detach().cpu().clone()
                # Log the shape distribution across calls (each unique bs only once)
                seen_bs = getattr(ew_module, "_oracle_tracer_seen_bs", set())
                if bs not in seen_bs:
                    seen_bs.add(bs)
                    ew_module._oracle_tracer_seen_bs = seen_bs
                    logger.info(
                        f"draft-p_t tracer: saw bs={bs}, "
                        f"path_probs shape={tuple(path_probs_full.shape)}, "
                        f"row0[:5]={path_probs_full[0, :5].tolist()}")
            else:
                ew_module._oracle_last_path_probs = None
        except Exception as e:
            if not getattr(ew_module, "_oracle_tracer_logged_err2", False):
                logger.warning(f"draft-p_t tracer: gather failed: {e}")
                ew_module._oracle_tracer_logged_err2 = True
            ew_module._oracle_last_path_probs = None
        return parent_list, top_scores_index, draft_tokens

    ew_module.organize_draft_results = traced
    ew_module._oracle_organize_traced = True
    logger.info("Installed EAGLE3 draft-p_t tracer on organize_draft_results")


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

    # LATENCY-ONLY mode: pure timing instrumentation. No force-accept, no
    # tree/p_t extraction — server runs REAL speculative decoding so target
    # actually commits accepted draft tokens. Used by measure_eagle3_cost.py.
    if os.environ.get("SGLANG_LATENCY_ONLY", "0") == "1":
        _setup_latency_only(eagle_worker)
        return

    _patch_verify_greedy_func()
    _patch_draft_stash(eagle_worker, trie_feeder)
    _patch_verify_logits(eagle_worker)
    _patch_forward_log(eagle_worker, replay_state, trie_feeder)

    mode = "VERIFY_TRIES" if trie_feeder else ("REPLAY" if replay_state else "VANILLA")
    logger.info(f"Oracle {mode} patch applied to EAGLEWorker")


def _setup_latency_only(eagle_worker) -> None:
    """Pure timing instrumentation for real speculative decoding.

    Wraps draft / target_forward / verify / forward_batch_generation with
    perf_counter timers and writes per-step timing + real accept_length to
    the oracle timing log. Does NOT force accept_length=0 (target actually
    commits accepted draft tokens) and does NOT extract draft tree or p_t.
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
    # .cpu().clone() so target_forward_ms reflects only CPU dispatch (the
    # true GPU wall time flows into verify_overhead, and the sum equals the
    # real end-to-end target cost).
    _patch_verify_logits(eagle_worker, stash_verify_logits=False)

    # Optional detailed verify instrumentation (requires verify_detail_patch
    # module which is not shipped with this branch; silently skip if absent).
    if os.environ.get("SGLANG_VERIFY_DETAILED", "0") == "1":
        try:
            from .verify_detail_patch import install_detailed_verify_patch
            install_detailed_verify_patch(eagle_worker)
        except ImportError:
            logger.warning(
                "SGLANG_VERIFY_DETAILED=1 set but verify_detail_patch "
                "module not available; continuing without detail patch.")

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
        phase = "decode" if is_decode else "prefill"

        draft_ms = getattr(eagle_worker, "_oracle_last_draft_ms", None)
        fwd_ms = getattr(eagle_worker, "_oracle_last_target_forward_ms", None)
        verify_total_ms = getattr(eagle_worker, "_oracle_last_verify_total_ms", None)
        accept_lens = getattr(eagle_worker, "_oracle_last_accept_lengths", []) or []

        entry = {
            "phase": phase,
            "step_total_ms": round(step_total_ms, 3),
        }
        try:
            entry["num_tokens"] = int(batch.input_ids.numel())
        except Exception:
            pass
        if draft_ms is not None:
            entry["eagle3_draft_ms"] = round(draft_ms, 3)
        if fwd_ms is not None:
            entry["target_forward_ms"] = round(fwd_ms, 3)
        if verify_total_ms is not None:
            entry["verify_total_ms"] = round(verify_total_ms, 3)
        if verify_total_ms is not None and fwd_ms is not None:
            entry["verify_overhead_ms"] = round(verify_total_ms - fwd_ms, 3)
        if (step_total_ms is not None and draft_ms is not None
                and verify_total_ms is not None):
            entry["post_verify_ms"] = round(
                step_total_ms - draft_ms - verify_total_ms, 3)

        if is_decode:
            entry["accept_lengths"] = list(accept_lens)
            entry["committed_tokens"] = (
                [int(n) + 1 for n in accept_lens] if accept_lens else [])

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
        "(real speculative decoding, timing instrumentation only)")


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


def _patch_draft_stash(eagle_worker: "EAGLEWorker", trie_feeder=None) -> None:
    """Patch draft() to stash full draft tree (token_ids + tree structure).

    When trie_feeder is provided (verify-tries mode), the draft tree is
    replaced with the pre-built union trie after running the original draft
    (which is still needed for hidden state updates).
    """
    _install_draft_p_t_tracer()
    original_draft = eagle_worker.draft

    def patched_draft(batch: "ScheduleBatch") -> "EagleVerifyInput":
        import time as _time
        _t_draft_start = _time.perf_counter()
        spec_info = original_draft(batch)
        _t_draft_end = _time.perf_counter()
        eagle_worker._oracle_last_draft_ms = (_t_draft_end - _t_draft_start) * 1000

        # Pick up the draft-side path probs from organize_draft_results tracer
        try:
            import sglang.srt.speculative.eagle_worker as ew_module
            pp = getattr(ew_module, "_oracle_last_path_probs", None)
            eagle_worker._oracle_stashed_path_probs = pp
            ew_module._oracle_last_path_probs = None  # consume
            pp_shape = None if pp is None else tuple(pp.shape)
            batch_bs = len(batch.reqs) if hasattr(batch, "reqs") else None
            dt_numel = (spec_info.draft_token.numel()
                        if getattr(spec_info, "draft_token", None) is not None
                        else None)
            num_draft = getattr(spec_info, "draft_token_num", None)
            inferred_bs = (dt_numel // num_draft
                           if dt_numel and num_draft else None)
            seen_sigs = getattr(
                eagle_worker, "_oracle_draft_stash_seen", set())
            sig = (batch_bs, inferred_bs, pp_shape)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                eagle_worker._oracle_draft_stash_seen = seen_sigs
                logger.info(
                    f"patched_draft: batch.reqs={batch_bs}, "
                    f"inferred_bs={inferred_bs}, path_probs_shape={pp_shape}, "
                    f"forward_mode={getattr(batch, 'forward_mode', None)}")
        except Exception as e:
            eagle_worker._oracle_stashed_path_probs = None
            logger.warning(f"patched_draft: path-probs stash failed: {e}")

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
    """Patch verify() / target_worker.forward_batch_generation to:

    * measure total verify time (``_oracle_last_verify_total_ms``)
    * stash real accept_length per request (``_oracle_last_accept_lengths``)
    * optionally stash the target's verify logits for p_t extraction
      (``stash_verify_logits=True``; oracle-vanilla mode needs this; in
      LATENCY_ONLY mode we pass False to avoid the .cpu().clone() sync
      overhead contaminating the target forward timing).
    """
    original_verify = eagle_worker.verify

    def patched_verify(batch, spec_info):
        import time as _time
        _t0 = _time.perf_counter()
        result = original_verify(batch, spec_info)
        _t1 = _time.perf_counter()
        eagle_worker._oracle_last_verify_total_ms = (_t1 - _t0) * 1000
        # Stash real accept_length per request (list of ints; excludes bonus).
        try:
            _, _res, _, _ = result
            eagle_worker._oracle_last_accept_lengths = list(
                getattr(_res, "accept_length_per_req_cpu", []) or [])
        except Exception:
            eagle_worker._oracle_last_accept_lengths = []
        return result

    eagle_worker.verify = patched_verify

    # Patch target_worker.forward_batch_generation for target forward timing
    # and (optionally) logit stashing before verify filters them.
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
        import torch

        result = original_forward(batch)

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

                # Extract per-node draft-side cumulative path prob from the
                # organize_draft_results tracer. This is available PRE-verify
                # so filters using it are deployment-realistic.
                tree_path_draft_p_t = None
                path_probs = getattr(eagle_worker, "_oracle_stashed_path_probs", None)
                if path_probs is not None and eagle3_tree is not None:
                    tree_path_draft_p_t = _extract_tree_path_draft_p_t(
                        path_probs, eagle3_tree, i)

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
                if tree_path_draft_p_t is not None:
                    entry["eagle3_tree_path_draft_p_t"] = tree_path_draft_p_t
                _log_entry(entry)

                # Verify-tries mode: log p_t for union trie nodes
                if trie_feeder is not None and verify_logits is not None:
                    trie_rec = getattr(eagle_worker, "_oracle_last_trie_rec", None)
                    if trie_rec is not None:
                        _log_verify_trie_p_t(
                            trie_rec, verify_logits, i, num_draft)

            eagle_worker._oracle_stashed_draft = None
            eagle_worker._oracle_stashed_verify_logits = None
            eagle_worker._oracle_stashed_path_probs = None

            # Log per-step timing (eagle3_draft + target_forward)
            draft_ms = getattr(eagle_worker, "_oracle_last_draft_ms", None)
            fwd_ms = getattr(eagle_worker, "_oracle_last_target_forward_ms", None)
            if draft_ms is not None or fwd_ms is not None:
                _log_timing({
                    "eagle3_draft_ms": draft_ms,
                    "target_forward_ms": fwd_ms,
                    "num_draft": num_draft,
                    "num_reqs": num_reqs,
                })
                eagle_worker._oracle_last_draft_ms = None
                eagle_worker._oracle_last_target_forward_ms = None

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
