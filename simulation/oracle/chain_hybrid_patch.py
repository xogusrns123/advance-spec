"""Chain-Hybrid (EAGLE3 + Suffix) per-depth online construction patch.

At each EAGLE3 chain draft step (speculative_eagle_topk=1), the suffix
decoding cache proposes one candidate token alongside eagle3's top-1.
The candidate with the higher per-token probability wins:

    choose suffix iff suffix_p > eagle_p   (and the tokens differ)

where eagle_p is the draft model's full-vocab softmax prob (conditional on
the chain so far) and suffix_p is the trie edge count ratio of the suffix
draft's first token. The chosen token is written into the draft loop's
``topk_index`` / ``topk_p`` before ``select_top_k_tokens`` consumes them,
so the next draft forward continues the chain from the winner and the
verify input is assembled with zero downstream changes.

Trie semantics follow ArcticInference's official usage: tokens committed by
verify are fed to ``add_active_response`` immediately every step
(incremental), not batched at request finish.

Activation (all required, validated at patch time):
  * SGLANG_CHAIN_HYBRID=1  (read by oracle_patch.patch_eagle_worker_full)
  * SGLANG_LATENCY_ONLY=1  (real speculative decoding; vanilla force-accept
    would nullify the experiment)
  * --speculative-eagle-topk 1  (chain mode)
  * --disable-cuda-graph  (the per-step GPU->CPU sync requires the eager
    draft path; with graphs enabled draft_forward is bypassed entirely and
    the patch would be a silent no-op)

Decision log (JSONL, SGLANG_CHAIN_HYBRID_LOG, tp_rank 0 only):
  {"type": "decision", "rid": ..., "decode_step": N, "depth": i,
   "eagle_token": ..., "eagle_p": ..., "suffix_token": ..., "suffix_p": ...,
   "match_len": ..., "suffix_score": ..., "chosen": "eagle3"|"suffix",
   "agreement": bool|null}
  {"type": "step", "rid": ..., "decode_step": N, "accept_len": a}

Join rule for offline analysis: the decision at depth d (0-based) was
accepted by verify iff the same (rid, decode_step)'s accept_len >= d + 1.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

logger = logging.getLogger(__name__)

# Stale-request GC: every GC_INTERVAL decode batches, stop_request() any rid
# not seen for GC_MAX_IDLE batches (covers aborted/retracted requests that
# never reach req.finished()).
GC_INTERVAL = 512
GC_MAX_IDLE = 2048


class _ChainHybridState:
    """Per-process state shared by the three hooks (one EAGLEWorker/process)."""

    def __init__(self, eagle_worker, suffix_cache, log_path: str):
        self.worker = eagle_worker
        self.cache = suffix_cache
        self.log_path = log_path
        self.active: set = set()        # rids with start_request() done
        self.last_out_len: dict = {}    # rid -> len(output_ids) already fed to trie
        self.decode_step: dict = {}     # rid -> decode step counter (1-based)
        self.last_seen: dict = {}       # rid -> batch_counter at last sighting
        self.batch_counter = 0
        # Per-draft-call transient state (set by the draft wrapper, consumed
        # by the select_top_k_tokens wrapper, cleared when draft returns).
        self.stash: list | None = None      # [(rid, context_tail)] in batch.reqs order
        self.chains: list | None = None     # per-row chain tokens chosen so far
        self.pending: list = []             # records awaiting flush
        self._warned: set = set()

    def warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(f"chain-hybrid [{key}]: {msg}")

    def flush(self) -> None:
        if not self.pending:
            return
        records, self.pending = self.pending, []
        if getattr(self.worker, "tp_rank", 0) != 0:
            return
        try:
            with open(self.log_path, "a") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
        except OSError as e:
            self.warn_once("log-write", str(e))


_STATE: _ChainHybridState | None = None


# ---------------------------------------------------------------------------
# Per-depth decision
# ---------------------------------------------------------------------------

def _decide_and_inject(st: _ChainHybridState, depth: int, topk_p, topk_index):
    """Run the suffix-vs-eagle3 decision for every batch row at this depth.

    Returns (topk_p, topk_index) — clones with injected tokens when at least
    one row chose suffix, the originals otherwise. Cloning matters at depth 0
    where the tensors are spec_info.topk_p / a fresh gather of topk_index;
    in-place writes could leak into state SGLang still owns.
    """
    p_cpu = topk_p.detach().cpu()
    idx_cpu = topk_index.detach().cpu()
    new_p = None
    new_idx = None

    for r, (rid, ctx_tail) in enumerate(st.stash):
        eagle_tok = int(idx_cpu[r, 0])
        eagle_p = float(p_cpu[r, 0])
        chain = st.chains[r]
        rec = {
            "type": "decision",
            "rid": rid,
            "decode_step": st.decode_step.get(rid, 0),
            "depth": depth,
            "eagle_token": eagle_tok,
            "eagle_p": round(eagle_p, 6),
            "suffix_token": None,
            "suffix_p": None,
            "match_len": None,
            "suffix_score": None,
            "chosen": "eagle3",
            "agreement": None,
        }

        suffix_tok = None
        suffix_p = None
        if rid in st.active:
            try:
                ctx = (list(ctx_tail) + chain)[-st.cache.max_tree_depth:]
                if chain:
                    # The trie learned only the committed context; insert the
                    # speculative chain so matching can extend through it,
                    # then roll back bit-exactly (requires enable_undo=True).
                    with st.cache.temporary_extension(rid, chain):
                        draft = st.cache.speculate(
                            rid, ctx, max_spec_tokens=1, use_tree_spec=False)
                else:
                    draft = st.cache.speculate(
                        rid, ctx, max_spec_tokens=1, use_tree_spec=False)
                if not draft.is_empty:
                    suffix_tok = int(draft.token_ids[0])
                    suffix_p = float(draft.probs[0]) if draft.probs else 0.0
                    rec["suffix_token"] = suffix_tok
                    rec["suffix_p"] = round(suffix_p, 6)
                    rec["match_len"] = int(draft.match_len)
                    rec["suffix_score"] = round(float(draft.score), 4)
            except Exception as e:
                st.warn_once("speculate", str(e))

        chosen_tok = eagle_tok
        if suffix_tok is not None:
            rec["agreement"] = suffix_tok == eagle_tok
            if suffix_tok != eagle_tok and suffix_p is not None \
                    and suffix_p > eagle_p:
                if new_p is None:
                    new_p = topk_p.clone()
                    new_idx = topk_index.clone()
                new_idx[r, 0] = suffix_tok
                # Clamp keeps cumulative chain scores monotone non-increasing
                # (probs from the trie are <= 1 by construction; defensive).
                new_p[r, 0] = min(max(suffix_p, 0.0), 1.0)
                chosen_tok = suffix_tok
                rec["chosen"] = "suffix"

        chain.append(chosen_tok)
        st.pending.append(rec)

    return (new_p if new_p is not None else topk_p,
            new_idx if new_idx is not None else topk_index)


def _install_select_wrapper() -> None:
    """Rebind eagle_worker module's select_top_k_tokens with the decision hook.

    draft_forward resolves select_top_k_tokens as a module global at call
    time and invokes it at the top of every draft step with exactly the
    state we need (depth i, topk_p, topk_index) — and crucially AFTER both
    hot_token_id remap sites, so injected suffix tokens (full-vocab ids)
    are never remapped. Same rebind pattern as oracle_patch's
    organize_draft_results tracer.
    """
    import sglang.srt.speculative.eagle_worker as ew_module

    if getattr(ew_module, "_chain_hybrid_select_patched", False):
        return
    original = ew_module.select_top_k_tokens

    def chain_hybrid_select(i, topk_p, topk_index, hidden_states, scores, topk):
        st = _STATE
        if (st is None or st.stash is None or topk != 1
                or topk_p is None or topk_p.dim() != 2):
            return original(i, topk_p, topk_index, hidden_states, scores, topk)
        if topk_p.shape[0] != len(st.stash):
            st.warn_once(
                "shape-mismatch",
                f"topk_p rows={topk_p.shape[0]} != stash={len(st.stash)}; "
                f"passing through (decisions skipped this step)")
            return original(i, topk_p, topk_index, hidden_states, scores, topk)
        try:
            topk_p, topk_index = _decide_and_inject(st, i, topk_p, topk_index)
        except Exception as e:
            st.warn_once("decide", str(e))
        return original(i, topk_p, topk_index, hidden_states, scores, topk)

    ew_module.select_top_k_tokens = chain_hybrid_select
    ew_module._chain_hybrid_select_patched = True
    logger.info("chain-hybrid: select_top_k_tokens decision hook installed")


# ---------------------------------------------------------------------------
# Draft wrapper: per-request context capture + lazy lifecycle
# ---------------------------------------------------------------------------

def _patch_draft(eagle_worker) -> None:
    original_draft = eagle_worker.draft

    def chain_draft(batch: "ScheduleBatch"):
        st = _STATE
        try:
            if batch.forward_mode.is_idle():
                st.stash = None
                st.chains = None
            else:
                st.batch_counter += 1
                stash = []
                chains = []
                for req in batch.reqs:
                    rid = req.rid
                    if rid not in st.active:
                        try:
                            st.cache.start_request(
                                rid, list(req.origin_input_ids))
                            st.active.add(rid)
                            st.last_out_len[rid] = 0
                            st.decode_step[rid] = 0
                        except Exception as e:
                            st.warn_once("start_request", str(e))
                    st.decode_step[rid] = st.decode_step.get(rid, 0) + 1
                    st.last_seen[rid] = st.batch_counter
                    ctx = (list(req.origin_input_ids)
                           + list(req.output_ids))[-st.cache.max_tree_depth:]
                    stash.append((rid, ctx))
                    chains.append([])
                if len(stash) > 1:
                    st.warn_once(
                        "bs>1",
                        f"batch size {len(stash)} > 1: decision path is "
                        f"written batch-safe but only validated at bs=1")
                st.stash = stash
                st.chains = chains
        except Exception as e:
            st.stash = None
            st.chains = None
            st.warn_once("draft-stash", str(e))
        try:
            return original_draft(batch)
        finally:
            st.stash = None
            st.chains = None
            st.flush()

    eagle_worker.draft = chain_draft


# ---------------------------------------------------------------------------
# Forward wrapper: incremental trie updates + per-step accept records
# ---------------------------------------------------------------------------

def _patch_forward(eagle_worker) -> None:
    original_forward = eagle_worker.forward_batch_generation

    def chain_forward(batch: "ScheduleBatch"):
        result = original_forward(batch)
        st = _STATE
        try:
            is_decode = not (
                batch.forward_mode.is_extend()
                or getattr(batch, "is_extend_in_batch", False)
            )
            if not is_decode:
                return result

            accept_lens = list(
                getattr(result, "accept_length_per_req_cpu", []) or [])

            for i, req in enumerate(batch.reqs):
                rid = req.rid
                if rid not in st.active:
                    continue

                # Incremental trie update (ArcticInference official
                # semantics): feed exactly the tokens verify committed this
                # step, tracked via output_ids length delta.
                out = req.output_ids or []
                prev = st.last_out_len.get(rid, 0)
                if len(out) > prev:
                    try:
                        st.cache.add_active_response(rid, list(out[prev:]))
                    except Exception as e:
                        st.warn_once("add_active_response", str(e))
                    st.last_out_len[rid] = len(out)

                if i < len(accept_lens):
                    st.pending.append({
                        "type": "step",
                        "rid": rid,
                        "decode_step": st.decode_step.get(rid, 0),
                        "accept_len": int(accept_lens[i]),
                    })

                if req.finished():
                    try:
                        st.cache.stop_request(rid)
                    except Exception as e:
                        st.warn_once("stop_request", str(e))
                    st.active.discard(rid)
                    st.last_out_len.pop(rid, None)
                    st.decode_step.pop(rid, None)
                    st.last_seen.pop(rid, None)

            if st.batch_counter % GC_INTERVAL == 0:
                _gc_stale(st)
            st.flush()
        except Exception as e:
            st.warn_once("forward-hook", str(e))
        return result

    eagle_worker.forward_batch_generation = chain_forward


def _gc_stale(st: _ChainHybridState) -> None:
    cutoff = st.batch_counter - GC_MAX_IDLE
    stale = [rid for rid, seen in st.last_seen.items() if seen < cutoff]
    for rid in stale:
        try:
            st.cache.stop_request(rid)
        except Exception:
            pass
        st.active.discard(rid)
        st.last_out_len.pop(rid, None)
        st.decode_step.pop(rid, None)
        st.last_seen.pop(rid, None)
    if stale:
        logger.info(f"chain-hybrid: GC'd {len(stale)} stale request(s)")


# ---------------------------------------------------------------------------
# Entry point (called from oracle_patch.patch_eagle_worker_full)
# ---------------------------------------------------------------------------

def patch_chain_hybrid(eagle_worker: "EAGLEWorker") -> None:
    global _STATE

    if os.environ.get("SGLANG_LATENCY_ONLY", "0") != "1":
        raise RuntimeError(
            "SGLANG_CHAIN_HYBRID=1 requires SGLANG_LATENCY_ONLY=1: oracle "
            "vanilla mode force-accepts 0 tokens per step, which would "
            "nullify the chain-hybrid experiment.")
    if getattr(eagle_worker, "topk", None) != 1:
        raise RuntimeError(
            f"SGLANG_CHAIN_HYBRID=1 requires chain mode "
            f"(--speculative-eagle-topk 1), got topk={eagle_worker.topk}.")
    if not getattr(eagle_worker.server_args, "disable_cuda_graph", False):
        raise RuntimeError(
            "SGLANG_CHAIN_HYBRID=1 requires --disable-cuda-graph: the CUDA "
            "graph draft path bypasses draft_forward, so the decision hook "
            "would silently never run.")

    # Lazy import: needs arctic_inference (container-only) and the repo root
    # on sys.path (server is launched with cwd=/workspace, same mechanism
    # that resolves simulation.oracle imports).
    from hybrid_spec_decoding.suffix_decoding.suffix_tree import (
        SuffixDecodingCache,
    )

    # ArcticInference official defaults (max_spec_factor=1.0,
    # min_token_prob=0.1); enable_undo for temporary_extension;
    # max_spec_tokens=1 — only the first suffix token is ever used.
    suffix_cache = SuffixDecodingCache(
        max_tree_depth=64,
        max_cached_requests=100000,
        max_spec_tokens=1,
        max_spec_factor=1.0,
        max_spec_offset=0.0,
        min_token_prob=0.1,
        use_tree_spec=False,
        enable_undo=True,
    )

    log_path = os.environ.get(
        "SGLANG_CHAIN_HYBRID_LOG", "/tmp/sglang_chain_hybrid_decisions.jsonl")
    _STATE = _ChainHybridState(eagle_worker, suffix_cache, log_path)

    _install_select_wrapper()
    _patch_draft(eagle_worker)
    _patch_forward(eagle_worker)

    logger.info(
        f"Chain-hybrid patch applied: per-depth eagle3+suffix selection "
        f"(suffix_p > eagle_p), steps={eagle_worker.speculative_num_steps}, "
        f"decision log -> {log_path}")
