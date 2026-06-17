"""Chain-Hybrid (EAGLE3 + Suffix) per-depth online construction patch.

At each EAGLE3 chain draft step (speculative_eagle_topk=1), the suffix
decoding cache proposes one candidate token alongside eagle3's top-1.
The candidate with the higher per-token probability wins:

    choose suffix iff S(suffix) > eagle_p   (and the tokens differ)

where eagle_p is the draft model's full-vocab softmax prob (conditional on
the chain so far) and S(suffix) is the suffix draft's first-token score.
By default S(suffix) = suffix_p, the trie edge count ratio (the original
select-1). When SGLANG_CHAIN_HYBRID_CALIB names a frozen isotonic map,
S(suffix) is the calibrated P(accept) the map assigns to that edge prob
(optionally Jeffreys count-shrunk first) — putting it on the same scale as
eagle_p instead of comparing a raw count ratio against a softmax prob. The
chosen token is written into the draft loop's ``topk_index`` / ``topk_p``
before ``select_top_k_tokens`` consumes them, so the next draft forward
continues the chain from the winner and the verify input is assembled with
zero downstream changes.

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
   "eagle_token": ..., "eagle_p": ..., "eagle_p_cal": p|null,
   "suffix_token": ..., "suffix_p": ..., "suffix_count": c|null,
   "suffix_total": n|null, "suffix_p_cal": p|null, "match_len": ...,
   "suffix_score": ..., "chosen": "eagle3"|"suffix", "agreement": bool|null}
  {"type": "step", "rid": ..., "decode_step": N, "accept_len": a}

suffix_count (c) and suffix_total (n) are the raw trie counts for the first
suffix edge (suffix_p = c/n); fit_chain_hybrid_calib.py uses them to build
both the raw and Jeffreys-shrunk calibration maps. suffix_p_cal is the
calibrated score actually compared against eagle_p (null when uncalibrated).

Suffix tail append (SGLANG_CHAIN_HYBRID_TAIL=<T_max>, 0/unset = off):
after the S-step head chain is fully drafted, the suffix cache is queried
once more with (context + full chain) and the returned continuation run
(up to T_max tokens, bounded by SGLANG_CHAIN_HYBRID_TAIL_FACTOR x match_len
and SGLANG_CHAIN_HYBRID_TAIL_MIN_PROB) is appended to the linear chain in
the EagleVerifyInput — verify scores S+1+t tokens in ONE target forward,
with zero extra eagle forwards. Requires bs=1 (--max-running-requests 1)
and the triton target attention backend. Two sglang-internal constants are
handled per step: the triton backend's num_draft_tokens (overridden around
verify) and the draft-extend kernel's next_power_of_2(num_steps+1)
constexpr (speculative_num_steps temporarily bumped by t during
forward_draft_extend_after_decode; silent position truncation otherwise).
Tail decision records carry "tail": true and depth = S+k; their suffix_p is
the CUMULATIVE path probability (not a single-edge count ratio), so the
calibration fitter must exclude them. The join rule is unchanged: tail
token at depth d accepted iff accept_len >= d+1 (accept_len may now exceed
S). Only validated for greedy sampling (temperature 0).
SGLANG_CHAIN_HYBRID_TAIL_CHECK=1 additionally verifies, every step, that
the hand reconstruction of positions/retrive_*/custom_mask at the original
size bit-matches the build kernel's output before extending (debug).

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
    """Per-process state shared by the hooks (one EAGLEWorker/process)."""

    def __init__(self, eagle_worker, suffix_cache, log_path: str,
                 tail_max: int = 0, tail_factor: float = 4.0,
                 tail_min_prob: float = 0.1, tail_check: bool = False,
                 mode: str = "select1", score_threshold: float = 5.0,
                 fb_factor: float = 1.0, fb_min_prob: float = 0.1):
        self.worker = eagle_worker
        self.cache = suffix_cache
        self.log_path = log_path
        # Decision mode: "select1" = per-depth competition (default);
        # "score_fallback" = Arctic-style per-STEP hybrid — one suffix run is
        # drafted from the committed context; if its score >= threshold the
        # run occupies the chain (eagle fills depths past the run end), else
        # the step is pure eagle3/MTP. Mirrors the simulator's hybrid_e3:t
        # (paper-faithful defaults F=1.0, T=0.1; score = sum of path probs).
        # "record" = pure pass-through baseline that dumps each finished
        # request's (input_ids, output_ids) to SGLANG_CHAIN_HYBRID_GT_OUT —
        # the greedy ground-truth source for the oracle arm.
        # "oracle" = per-depth selection ORACLE: ground-truth token gt[L+d]
        # (from a record-arm dump, request matched by exact input_ids) decides
        # the chain — eagle if it matches, else suffix (injected) if it
        # matches, else the chain is dead at d. The arm's accept_mean IS the
        # selection-policy ceiling. Requires the agent to --replay the record
        # arm's conversation so prompts match byte-exactly.
        self.mode = mode
        self.score_threshold = score_threshold
        self.fb_factor = fb_factor
        self.fb_min_prob = fb_min_prob
        self.fallback_runs: list | None = None    # per-row token runs (or None)
        self.fallback_meta: list | None = None    # per-row (score, probs, match_len)
        # record/oracle mode state
        self.gt_out_path: str | None = None       # record: dump path
        self.gt_map: dict | None = None           # oracle: input_ids -> output_ids
        self.gt: dict = {}                        # rid -> gt output list (or None)
        self.gt_offtrack: dict = {}               # rid -> bool (FP divergence)
        self.gt_pos: list | None = None           # per-row L at stash time
        self.gt_stats = {"matched": 0, "unmatched": 0, "offtrack": 0}
        # Suffix tail-append config (0 = disabled = original behavior).
        self.tail_max = tail_max
        self.tail_factor = tail_factor
        self.tail_min_prob = tail_min_prob
        self.tail_check = tail_check
        self.last_tail_len = 0   # tokens appended this step; consumed by the
                                 # draft-extend wrapper, reset at each draft
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
# Suffix-probability calibration (optional)
# ---------------------------------------------------------------------------
#
# A faithful serving-side port of run_tree_oracle_sim._FrozenIsoCalibrator:
# loads the same JSON map ({"meta": {"shrink": ...}, "groups": {"suffix":
# {"x": [...], "y": [...]}}}) produced by fit_chain_hybrid_calib.py /
# fit_iso_calibration.py and does the identical step-function lookup
# raw_edge_prob -> P(accept). When the map was fitted with Jeffreys
# count-shrink (meta.shrink truthy), the caller must shrink the suffix edge
# prob the same way BEFORE lookup — mirrored here via ``wants_shrunk``.

class _ServingIsoCalibrator:
    """Frozen isotonic map suffix_p -> P(accept), loaded once at patch time."""

    def __init__(self, blob: dict):
        import numpy as np
        self._maps = {
            grp: (np.asarray(m["x"], dtype=np.float64),
                  np.asarray(m["y"], dtype=np.float64))
            for grp, m in blob["groups"].items()
        }
        self.wants_shrunk = bool(blob.get("meta", {}).get("shrink"))

    @classmethod
    def load(cls, path: str) -> "_ServingIsoCalibrator":
        with open(path) as f:
            return cls(json.load(f))

    def predict(self, group: str, p: float, fallback: float) -> float:
        import numpy as np
        m = self._maps.get(group)
        if m is None:
            return fallback
        xs, ys = m
        i = int(np.searchsorted(xs, p, side="right")) - 1
        if i < 0:
            i = 0
        v = float(ys[i])
        return v if v > 1e-6 else 1e-6


_CALIB: _ServingIsoCalibrator | None = None


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
            "eagle_p_cal": None,
            "suffix_token": None,
            "suffix_p": None,
            "suffix_count": None,
            "suffix_total": None,
            "suffix_p_cal": None,
            "match_len": None,
            "suffix_score": None,
            "chosen": "eagle3",
            "agreement": None,
        }

        suffix_tok = None
        suffix_p = None
        suffix_count = None
        suffix_total = None
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
                    # Raw trie counts for the first edge: c = matched count,
                    # n = c / (c/n) recovered from the count-ratio prob. These
                    # let the fitter compute both the raw (c/n) and the
                    # Jeffreys-shrunk ((c+.5)/(n+1)) edge prob offline, and let
                    # the Jeffreys calibration variant shrink at eval here.
                    if draft.counts and suffix_p > 0.0:
                        suffix_count = int(draft.counts[0])
                        suffix_total = int(round(suffix_count / suffix_p))
                        rec["suffix_count"] = suffix_count
                        rec["suffix_total"] = suffix_total
            except Exception as e:
                st.warn_once("speculate", str(e))

        # ORACLE mode: the ground-truth token gt[L+depth] decides — eagle if
        # it matches, else suffix (injected) if it matches, else the chain is
        # dead here (proposal-limited selection ceiling).
        if st.mode == "oracle":
            gt_list = st.gt.get(rid)
            L = (st.gt_pos[r] if st.gt_pos is not None
                 and r < len(st.gt_pos) else None)
            gt_tok = None
            if (gt_list is not None and L is not None
                    and L + depth < len(gt_list)):
                gt_tok = int(gt_list[L + depth])
            rec["gt_token"] = gt_tok
            rec["agreement"] = (suffix_tok == eagle_tok
                                if suffix_tok is not None else None)
            chosen_tok = eagle_tok
            hit = "nogt" if gt_tok is None else "none"
            if gt_tok is not None:
                e_hit = eagle_tok == gt_tok
                s_hit = suffix_tok == gt_tok if suffix_tok is not None else False
                if e_hit:
                    hit = "both" if s_hit else "eagle"
                elif s_hit:
                    hit = "suffix"
                    if new_p is None:
                        new_p = topk_p.clone()
                        new_idx = topk_index.clone()
                    new_idx[r, 0] = suffix_tok
                    new_p[r, 0] = 1.0
                    chosen_tok = suffix_tok
                    rec["chosen"] = "suffix"
            rec["oracle_hit"] = hit
            chain.append(chosen_tok)
            st.pending.append(rec)
            continue

        # The two values the decision compares. Raw select-1 compares the raw
        # suffix count-ratio against eagle's raw softmax prob. The calibrated
        # variants map BOTH through their frozen isotonic curves so the
        # comparison is accept-prob vs accept-prob (calibrating only suffix
        # would pit a calibrated accept-prob against an inflated softmax and
        # over-suppress suffix). suffix may be Jeffreys-shrunk first; eagle_p
        # has no counts, so its group is never shrunk.
        suffix_cmp = suffix_p
        eagle_cmp = eagle_p
        if _CALIB is not None:
            eagle_cmp = _CALIB.predict("eagle", eagle_p, eagle_p)
            rec["eagle_p_cal"] = round(eagle_cmp, 6)
            if suffix_p is not None:
                p_in = suffix_p
                if _CALIB.wants_shrunk:
                    if suffix_count is not None and suffix_total:
                        p_in = (suffix_count + 0.5) / (suffix_total + 1)
                    else:
                        st.warn_once(
                            "no-counts",
                            "calib map wants Jeffreys-shrunk probs but the "
                            "suffix draft exposed no counts; using raw suffix_p")
                suffix_cmp = _CALIB.predict("suffix", p_in, suffix_p)
                rec["suffix_p_cal"] = round(suffix_cmp, 6)

        chosen_tok = eagle_tok
        if suffix_tok is not None:
            rec["agreement"] = suffix_tok == eagle_tok
            if suffix_tok != eagle_tok and suffix_cmp is not None \
                    and suffix_cmp > eagle_cmp:
                if new_p is None:
                    new_p = topk_p.clone()
                    new_idx = topk_index.clone()
                new_idx[r, 0] = suffix_tok
                # Clamp keeps cumulative chain scores monotone non-increasing
                # (probs are <= 1 by construction; defensive).
                new_p[r, 0] = min(max(suffix_cmp, 0.0), 1.0)
                chosen_tok = suffix_tok
                rec["chosen"] = "suffix"

        chain.append(chosen_tok)
        st.pending.append(rec)

    return (new_p if new_p is not None else topk_p,
            new_idx if new_idx is not None else topk_index)


def _inject_fallback_run(st: _ChainHybridState, depth: int, topk_p, topk_index):
    """score_fallback mode: occupy depth with the step's suffix run token.

    The run was drafted ONCE per step (in the draft wrapper) from the
    committed context; rows whose run is None (score < threshold, or no
    match) keep eagle's candidate untouched — and depths past the run end
    are eagle continuation tokens drafted conditioned on the suffix prefix.
    """
    runs = st.fallback_runs
    new_p = None
    new_idx = None
    idx_cpu = None

    for r, (rid, _ctx) in enumerate(st.stash):
        run = runs[r] if runs and r < len(runs) else None
        score, probs, match_len = (
            st.fallback_meta[r] if st.fallback_meta and r < len(st.fallback_meta)
            else (None, [], None))
        rec = {
            "type": "decision",
            "policy": "score_fallback",
            "rid": rid,
            "decode_step": st.decode_step.get(rid, 0),
            "depth": depth,
            "eagle_token": None,
            "eagle_p": None,
            "eagle_p_cal": None,
            "suffix_token": None,
            "suffix_p": None,
            "suffix_count": None,
            "suffix_total": None,
            "suffix_p_cal": None,
            "match_len": match_len,
            "suffix_score": round(float(score), 4) if score is not None else None,
            "chosen": "eagle3",
            "agreement": None,
        }
        if run is not None and depth < len(run):
            if new_p is None:
                new_p = topk_p.clone()
                new_idx = topk_index.clone()
            tok = int(run[depth])
            p = float(probs[depth]) if depth < len(probs) else 0.0
            new_idx[r, 0] = tok
            new_p[r, 0] = min(max(p, 0.0), 1.0)
            rec["suffix_token"] = tok
            rec["suffix_p"] = round(p, 6)
            rec["chosen"] = "suffix"
            st.chains[r].append(tok)
        else:
            # eagle's candidate stays; record its token for the log.
            if idx_cpu is None:
                idx_cpu = topk_index.detach().cpu()
            eagle_tok = int(idx_cpu[r, 0])
            rec["eagle_token"] = eagle_tok
            st.chains[r].append(eagle_tok)
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
            if st.mode == "record":
                pass  # pure pass-through baseline (GT capture only)
            elif st.mode == "score_fallback":
                topk_p, topk_index = _inject_fallback_run(
                    st, i, topk_p, topk_index)
            else:  # select1 (raw/calibrated) and oracle
                topk_p, topk_index = _decide_and_inject(
                    st, i, topk_p, topk_index)
        except Exception as e:
            st.warn_once("decide", str(e))
        return original(i, topk_p, topk_index, hidden_states, scores, topk)

    ew_module.select_top_k_tokens = chain_hybrid_select
    ew_module._chain_hybrid_select_patched = True
    logger.info("chain-hybrid: select_top_k_tokens decision hook installed")


# ---------------------------------------------------------------------------
# Suffix tail append (route b): extend the chain EagleVerifyInput in place
# ---------------------------------------------------------------------------

def _rebuild_chain_tensors(ndt: int, seq_len: int, device, dtypes: dict):
    """Hand-build the linear-chain verify tensors at size ndt (bs=1).

    Mirrors build_tree_kernel_efficient's FULL_MASK output for topk=1:
      positions  = seq_len + arange(ndt)
      retrive_index = arange(ndt)                row vector (1, ndt)
      retrive_next_token = [1..ndt-1, -1]
      retrive_next_sibling = all -1
      custom_mask rows: row i = [ones(seq_len), causal_{j<=i}(ndt)], flattened
    """
    import torch
    positions = torch.arange(
        seq_len, seq_len + ndt, dtype=dtypes["positions"], device=device)
    retrive_index = torch.arange(
        ndt, dtype=dtypes["retrive"], device=device).view(1, -1)
    nxt = torch.full((1, ndt), -1, dtype=dtypes["retrive"], device=device)
    nxt[0, : ndt - 1] = torch.arange(
        1, ndt, dtype=dtypes["retrive"], device=device)
    sib = torch.full((1, ndt), -1, dtype=dtypes["retrive"], device=device)
    mask = torch.cat(
        [
            torch.ones(ndt, seq_len, dtype=dtypes["mask"], device=device),
            torch.tril(torch.ones(ndt, ndt, dtype=dtypes["mask"],
                                  device=device)),
        ],
        dim=1,
    ).flatten()
    return positions, retrive_index, nxt, sib, mask


def _tail_append(st: _ChainHybridState, spec_info) -> None:
    """Append a suffix continuation run to the chain EagleVerifyInput.

    Runs after original_draft returns and before verify consumes spec_info.
    Mutates spec_info in place; on any guard failure it simply returns,
    leaving the original (correct) verify input untouched.
    """
    import torch

    if st.tail_max <= 0 or st.stash is None:
        return
    if len(st.stash) != 1:
        st.warn_once("tail-bs>1", f"batch size {len(st.stash)} > 1: tail "
                                  f"append only supports bs=1; skipping")
        return
    if getattr(spec_info, "topk", None) != 1:
        return
    rid, ctx_tail = st.stash[0]
    if rid not in st.active:
        return

    ndt_old = int(spec_info.draft_token_num)
    if spec_info.draft_token.numel() != ndt_old:
        st.warn_once("tail-shape", "draft_token numel != draft_token_num; "
                                   "skipping tail")
        return
    # Derive prefix length from the mask size (bs=1: ndt*(seq_len+ndt)).
    mask_numel = spec_info.custom_mask.numel()
    if mask_numel % ndt_old != 0:
        st.warn_once("tail-mask", "unexpected custom_mask size; skipping")
        return
    seq_len = mask_numel // ndt_old - ndt_old
    if seq_len <= 0 or seq_len != int(spec_info.seq_lens_sum):
        st.warn_once(
            "tail-seqlen",
            f"mask-derived seq_len {seq_len} != seq_lens_sum "
            f"{int(spec_info.seq_lens_sum)}; skipping")
        return

    # KV headroom: verify allocates ndt_new slots but the scheduler only
    # reserved server_ndt; skip the tail when the pool is nearly full.
    alloc = getattr(st.worker, "token_to_kv_pool_allocator", None)
    if alloc is not None:
        try:
            if alloc.available_size() < ndt_old + st.tail_max + 64:
                st.warn_once("tail-headroom", "KV pool nearly full; "
                                              "skipping tail this step")
                return
        except Exception:
            pass

    dev = spec_info.draft_token.device
    dtypes = {
        "positions": spec_info.positions.dtype,
        "retrive": spec_info.retrive_index.dtype,
        "mask": spec_info.custom_mask.dtype,
    }

    if st.tail_check:
        # Debug gate: our reconstruction at the ORIGINAL size must bit-match
        # the build kernel's output, else the layout assumption is wrong and
        # extending would corrupt verify — skip and warn.
        pos0, ri0, nx0, sb0, mk0 = _rebuild_chain_tensors(
            ndt_old, seq_len, dev, dtypes)
        checks = [
            ("positions", torch.equal(pos0, spec_info.positions)),
            ("retrive_index", torch.equal(ri0, spec_info.retrive_index)),
            ("retrive_next_token",
             torch.equal(nx0, spec_info.retrive_next_token)),
            ("retrive_next_sibling",
             torch.equal(sb0, spec_info.retrive_next_sibling)),
            ("custom_mask", torch.equal(mk0, spec_info.custom_mask)),
        ]
        bad = [name for name, ok in checks if not ok]
        if bad:
            st.warn_once("tail-check",
                         f"chain reconstruction mismatch in {bad}; "
                         f"tail disabled this step")
            return

    # The chain verify will walk — taken from the verify input itself
    # (bit-exact, robust to any dropped per-depth decision).
    chain = spec_info.draft_token[1:].tolist()
    if len(chain) != ndt_old - 1:
        return

    # Suffix continuation run from (context + chain).
    try:
        ctx = (list(ctx_tail) + chain)[-st.cache.max_tree_depth:]
        with st.cache.temporary_extension(rid, chain):
            draft = st.cache.speculate(
                rid, ctx,
                max_spec_tokens=st.tail_max,
                max_spec_factor=st.tail_factor,
                min_token_prob=st.tail_min_prob,
                use_tree_spec=False,
            )
    except Exception as e:
        st.warn_once("tail-speculate", str(e))
        return
    if draft.is_empty:
        return

    tail_tokens = [int(x) for x in draft.token_ids[: st.tail_max]]
    t = len(tail_tokens)
    ndt_new = ndt_old + t

    # Tensor surgery: extend the linear chain to ndt_new.
    positions, ri, nx, sb, mask = _rebuild_chain_tensors(
        ndt_new, seq_len, dev, dtypes)
    spec_info.draft_token = torch.cat(
        [spec_info.draft_token,
         torch.tensor(tail_tokens, dtype=spec_info.draft_token.dtype,
                      device=dev)])
    spec_info.positions = positions
    spec_info.retrive_index = ri
    spec_info.retrive_next_token = nx
    spec_info.retrive_next_sibling = sb
    spec_info.custom_mask = mask
    spec_info.spec_steps = int(spec_info.spec_steps) + t  # sizes accept_index
    spec_info.draft_token_num = ndt_new
    st.last_tail_len = t

    # Decision records for the tail (join rule unchanged).
    decode_step = st.decode_step.get(rid, 0)
    probs = list(draft.probs or [])
    counts = list(draft.counts or [])
    for k, tok in enumerate(tail_tokens):
        st.pending.append({
            "type": "decision",
            "tail": True,
            "rid": rid,
            "decode_step": decode_step,
            "depth": (ndt_old - 1) + k,
            "eagle_token": None,
            "eagle_p": None,
            "eagle_p_cal": None,
            "suffix_token": tok,
            "suffix_p": round(float(probs[k]), 6) if k < len(probs) else None,
            "suffix_count": int(counts[k]) if k < len(counts) else None,
            "suffix_total": None,
            "suffix_p_cal": None,
            "match_len": int(draft.match_len),
            "suffix_score": round(float(draft.score), 4),
            "chosen": "suffix",
            "agreement": None,
        })


# ---------------------------------------------------------------------------
# Draft wrapper: per-request context capture + lazy lifecycle
# ---------------------------------------------------------------------------

def _patch_draft(eagle_worker) -> None:
    original_draft = eagle_worker.draft

    def chain_draft(batch: "ScheduleBatch"):
        st = _STATE
        st.last_tail_len = 0
        try:
            if batch.forward_mode.is_idle():
                st.stash = None
                st.chains = None
                st.fallback_runs = None
                st.fallback_meta = None
            else:
                st.batch_counter += 1
                stash = []
                chains = []
                fb_runs = []
                fb_meta = []
                gt_pos = []
                for req in batch.reqs:
                    rid = req.rid
                    if st.mode == "record":
                        # pass-through baseline: no trie, no decisions —
                        # the forward hook dumps GT at request finish.
                        stash.append((rid, ()))
                        chains.append([])
                        continue
                    if rid not in st.active:
                        try:
                            st.cache.start_request(
                                rid, list(req.origin_input_ids))
                            st.active.add(rid)
                            st.last_out_len[rid] = 0
                            st.decode_step[rid] = 0
                        except Exception as e:
                            st.warn_once("start_request", str(e))
                        if st.mode == "oracle":
                            g = (st.gt_map or {}).get(
                                tuple(req.origin_input_ids))
                            st.gt[rid] = list(g) if g is not None else None
                            st.gt_offtrack[rid] = False
                            st.gt_stats[
                                "matched" if g is not None else "unmatched"
                            ] += 1
                            if g is None:
                                st.warn_once(
                                    "gt-unmatched",
                                    "request prompt not found in GT dump; "
                                    "running pure eagle for unmatched "
                                    "requests (count in gt_stats)")
                    st.decode_step[rid] = st.decode_step.get(rid, 0) + 1
                    st.last_seen[rid] = st.batch_counter
                    ctx = (list(req.origin_input_ids)
                           + list(req.output_ids))[-st.cache.max_tree_depth:]
                    stash.append((rid, ctx))
                    chains.append([])
                    gt_pos.append(len(req.output_ids or []))
                    # score_fallback: ONE suffix run per step from the
                    # committed context (sim hybrid_e3 semantics: suffix iff
                    # draft.score >= threshold, else pure eagle this step).
                    if st.mode == "score_fallback":
                        run = None
                        meta = (None, [], None)
                        if rid in st.active:
                            try:
                                d = st.cache.speculate(
                                    rid, ctx,
                                    max_spec_tokens=(
                                        st.worker.speculative_num_steps),
                                    max_spec_factor=st.fb_factor,
                                    min_token_prob=st.fb_min_prob,
                                    use_tree_spec=False)
                                if not d.is_empty:
                                    meta = (float(d.score),
                                            [float(p) for p in d.probs],
                                            int(d.match_len))
                                    if d.score >= st.score_threshold:
                                        run = [int(t) for t in d.token_ids]
                            except Exception as e:
                                st.warn_once("fallback-speculate", str(e))
                        fb_runs.append(run)
                        fb_meta.append(meta)
                if len(stash) > 1:
                    st.warn_once(
                        "bs>1",
                        f"batch size {len(stash)} > 1: decision path is "
                        f"written batch-safe but only validated at bs=1")
                st.stash = stash
                st.chains = chains
                st.fallback_runs = fb_runs if st.mode == "score_fallback" else None
                st.fallback_meta = fb_meta if st.mode == "score_fallback" else None
                st.gt_pos = gt_pos if st.mode == "oracle" else None
        except Exception as e:
            st.stash = None
            st.chains = None
            st.fallback_runs = None
            st.fallback_meta = None
            st.warn_once("draft-stash", str(e))
        try:
            result = original_draft(batch)
            if st.tail_max > 0 and st.stash is not None \
                    and st.mode != "record":
                try:
                    _tail_append(st, result)
                except Exception as e:
                    st.warn_once("tail", str(e))
            return result
        finally:
            st.stash = None
            st.chains = None
            st.fallback_runs = None
            st.fallback_meta = None
            st.flush()

    eagle_worker.draft = chain_draft


def _dump_gt(st: _ChainHybridState, req) -> None:
    """record mode: append the finished request's (input_ids, output_ids) to
    the GT dump — the oracle arm matches requests by exact input_ids."""
    if getattr(st.worker, "tp_rank", 0) != 0 or not st.gt_out_path:
        return
    try:
        with open(st.gt_out_path, "a") as f:
            f.write(json.dumps({
                "input_ids": [int(x) for x in req.origin_input_ids],
                "output_ids": [int(x) for x in (req.output_ids or [])],
            }) + "\n")
    except OSError as e:
        st.warn_once("gt-dump", str(e))


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
                if st.mode == "record":
                    if req.finished():
                        _dump_gt(st, req)
                    continue
                if rid not in st.active:
                    continue

                # Incremental trie update (ArcticInference official
                # semantics): feed exactly the tokens verify committed this
                # step, tracked via output_ids length delta.
                out = req.output_ids or []
                prev = st.last_out_len.get(rid, 0)
                if len(out) > prev:
                    # Oracle GT alignment: greedy FP flips would otherwise
                    # push the realized output off the recorded path and
                    # invalidate gt indexing for the rest of the turn. Force
                    # divergent committed tokens back to GT (same mechanism
                    # as oracle_patch's REPLAY verify-override: the token
                    # enters the next forward via its embedding, so the
                    # context stays self-consistent). The next-step chain
                    # seed (spec_info.verified_id) is mirrored below.
                    if st.mode == "oracle":
                        g = st.gt.get(rid)
                        if g is not None:
                            forced = False
                            for k in range(prev, min(len(out), len(g))):
                                if int(out[k]) != int(g[k]):
                                    req.output_ids[k] = int(g[k])
                                    st.gt_stats["forced"] = (
                                        st.gt_stats.get("forced", 0) + 1)
                                    forced = True
                            if len(out) > len(g):
                                st.gt_stats["past_gt_end"] = (
                                    st.gt_stats.get("past_gt_end", 0)
                                    + len(out) - len(g))
                            if forced:
                                try:
                                    di = getattr(batch, "spec_info", None)
                                    vid = getattr(di, "verified_id", None)
                                    if vid is not None \
                                            and vid.numel() == len(batch.reqs):
                                        vid[i] = int(req.output_ids[-1])
                                except Exception as e:
                                    st.warn_once("gt-force-vid", str(e))
                            out = req.output_ids
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
                    st.gt.pop(rid, None)
                    st.gt_offtrack.pop(rid, None)

            if st.batch_counter % GC_INTERVAL == 0:
                _gc_stale(st)
                if st.mode == "oracle":
                    logger.info(f"chain-hybrid oracle gt_stats: {st.gt_stats}")
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
# Tail support wrappers: per-step sglang-internal constant fixes
# ---------------------------------------------------------------------------

def _resolve_target_attn_backend(eagle_worker):
    """The target attention backend that owns ``num_draft_tokens``. For
    Mamba-hybrid models the model-runner backend is a HybridLinearAttnBackend
    wrapper; the full-attention sub-backend carries the constant."""
    backend = eagle_worker.target_worker.model_runner.attn_backend
    if not hasattr(backend, "num_draft_tokens") \
            and hasattr(backend, "full_attn_backend"):
        backend = backend.full_attn_backend
    return backend


def _patch_verify_for_tail(eagle_worker) -> None:
    """Wrap eagle_worker.verify to override the triton target attention
    backend's worker-level ``num_draft_tokens`` constant for steps whose
    verify input was tail-extended (it sizes qo_indptr / mask_indptr /
    max_extend_len in TARGET_VERIFY and never reads spec_info)."""
    original_verify = eagle_worker.verify
    backend = _resolve_target_attn_backend(eagle_worker)
    server_ndt = eagle_worker.server_args.speculative_num_draft_tokens

    def tail_verify(batch, spec_info):
        ndt = getattr(spec_info, "draft_token_num", server_ndt)
        override = ndt != server_ndt
        if override:
            backend.num_draft_tokens = ndt
        try:
            return original_verify(batch, spec_info)
        finally:
            if override:
                backend.num_draft_tokens = server_ndt

    eagle_worker.verify = tail_verify


def _patch_draft_extend_for_tail(eagle_worker) -> None:
    """Wrap forward_draft_extend_after_decode to bump speculative_num_steps
    by the tail length for the duration of the call. The draft-extend triton
    kernel's position/verified_id stores are masked by a
    next_power_of_2(num_steps+1) constexpr — accept lengths beyond it are
    SILENTLY truncated, so the bump is required for correctness whenever
    accept_len can exceed S (t >= next_pow2(S+1) - S - 1)."""
    original_fdead = eagle_worker.forward_draft_extend_after_decode

    def tail_fdead(batch):
        st = _STATE
        bump = st.last_tail_len if st is not None else 0
        if st is not None:
            st.last_tail_len = 0
        if bump <= 0:
            return original_fdead(batch)
        saved = eagle_worker.speculative_num_steps
        eagle_worker.speculative_num_steps = saved + bump
        try:
            return original_fdead(batch)
        finally:
            eagle_worker.speculative_num_steps = saved

    eagle_worker.forward_draft_extend_after_decode = tail_fdead


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

    # Suffix tail append (route b) config — validated up front so a
    # misconfigured run fails at boot instead of silently mis-masking.
    tail_max = int(os.environ.get("SGLANG_CHAIN_HYBRID_TAIL", "0"))
    tail_factor = float(os.environ.get("SGLANG_CHAIN_HYBRID_TAIL_FACTOR", "4.0"))
    tail_min_prob = float(
        os.environ.get("SGLANG_CHAIN_HYBRID_TAIL_MIN_PROB", "0.1"))
    tail_check = os.environ.get("SGLANG_CHAIN_HYBRID_TAIL_CHECK", "0") == "1"
    if tail_max > 0:
        if getattr(eagle_worker.server_args, "max_running_requests", None) != 1:
            raise RuntimeError(
                "SGLANG_CHAIN_HYBRID_TAIL requires --max-running-requests 1 "
                "(tail tensor surgery only supports bs=1).")
        backend = _resolve_target_attn_backend(eagle_worker)
        if not hasattr(backend, "num_draft_tokens"):
            raise RuntimeError(
                f"SGLANG_CHAIN_HYBRID_TAIL requires a target attention "
                f"backend exposing num_draft_tokens (triton); got "
                f"{type(backend).__name__}.")
        if hasattr(backend, "linear_attn_backend") or hasattr(
                eagle_worker.target_worker.model_runner.attn_backend,
                "linear_attn_backend"):
            # Mamba-hybrid archs (Qwen3.5/qwen3-next): the speculative mamba
            # intermediate caches (intermediate_ssm / intermediate_conv_window
            # in memory_pool.py) are statically allocated with the SERVER's
            # speculative_num_draft_tokens — a tail-extended verify would
            # write past them. Unsupported until that pool can be oversized.
            raise RuntimeError(
                "SGLANG_CHAIN_HYBRID_TAIL is unsupported on Mamba-hybrid "
                "models: speculative mamba intermediate caches are statically "
                "sized to the server num_draft_tokens. Run with tail "
                "disabled (SGLANG_CHAIN_HYBRID_TAIL=0).")

    # Decision mode (default per-depth select-1). "score_fallback" mirrors
    # the simulator's hybrid_e3:t baseline; "record" dumps GT trajectories;
    # "oracle" is the per-depth selection ceiling driven by a GT dump.
    mode = os.environ.get("SGLANG_CHAIN_HYBRID_MODE", "select1")
    if mode not in ("select1", "score_fallback", "record", "oracle"):
        raise RuntimeError(f"unknown SGLANG_CHAIN_HYBRID_MODE={mode!r}")
    score_threshold = float(
        os.environ.get("SGLANG_CHAIN_HYBRID_SCORE_THRESHOLD", "5.0"))
    fb_factor = float(os.environ.get("SGLANG_CHAIN_HYBRID_FB_FACTOR", "1.0"))
    fb_min_prob = float(
        os.environ.get("SGLANG_CHAIN_HYBRID_FB_MIN_PROB", "0.1"))

    _STATE = _ChainHybridState(
        eagle_worker, suffix_cache, log_path,
        tail_max=tail_max, tail_factor=tail_factor,
        tail_min_prob=tail_min_prob, tail_check=tail_check,
        mode=mode, score_threshold=score_threshold,
        fb_factor=fb_factor, fb_min_prob=fb_min_prob)

    # Optional suffix-prob calibration. SGLANG_CHAIN_HYBRID_CALIB points at a
    # frozen isotonic map; absent -> raw count-ratio comparison (the original
    # select-1). The map's meta.shrink decides whether suffix probs are
    # Jeffreys-shrunk before lookup (must match how the map was fitted).
    global _CALIB
    calib_path = os.environ.get("SGLANG_CHAIN_HYBRID_CALIB")
    if calib_path:
        _CALIB = _ServingIsoCalibrator.load(calib_path)
        calib_desc = (f"calibrated (map={calib_path}, "
                      f"shrink={'jeffreys' if _CALIB.wants_shrunk else 'none'})")
    else:
        _CALIB = None
        calib_desc = "raw suffix_p (uncalibrated)"

    _install_select_wrapper()
    _patch_draft(eagle_worker)
    _patch_forward(eagle_worker)
    if tail_max > 0:
        _patch_verify_for_tail(eagle_worker)
        _patch_draft_extend_for_tail(eagle_worker)
        tail_desc = (f"tail=on (T_max={tail_max}, factor={tail_factor}, "
                     f"min_p={tail_min_prob}, check={tail_check})")
    else:
        tail_desc = "tail=off"

    if mode == "record":
        _STATE.gt_out_path = os.environ.get(
            "SGLANG_CHAIN_HYBRID_GT_OUT", "/tmp/sglang_chain_hybrid_gt.jsonl")
        mode_desc = f"mode=record (GT dump -> {_STATE.gt_out_path})"
    elif mode == "oracle":
        gt_path = os.environ.get("SGLANG_CHAIN_HYBRID_GT")
        if not gt_path:
            raise RuntimeError(
                "SGLANG_CHAIN_HYBRID_MODE=oracle requires "
                "SGLANG_CHAIN_HYBRID_GT=<gt_tokens.jsonl from a record arm>")
        gt_map: dict = {}
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gt_map[tuple(rec["input_ids"])] = rec["output_ids"]
        _STATE.gt_map = gt_map
        mode_desc = (f"mode=oracle (per-depth selection ceiling, "
                     f"{len(gt_map)} GT trajectories from {gt_path})")
    elif mode == "score_fallback":
        mode_desc = (f"mode=score_fallback (suffix iff score>="
                     f"{score_threshold}, F={fb_factor}, T={fb_min_prob})")
    else:
        mode_desc = f"mode=select1, decision={calib_desc} > eagle_p"
    logger.info(
        f"Chain-hybrid patch applied: {mode_desc}, {tail_desc}, "
        f"steps={eagle_worker.speculative_num_steps}, "
        f"decision log -> {log_path}")
