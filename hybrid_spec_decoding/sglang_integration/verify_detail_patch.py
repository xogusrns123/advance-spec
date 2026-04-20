"""Detailed verify-path profiling patch.

Replaces `eagle_worker.verify()` and `EagleVerifyInput.verify()` with
instrumented versions that time each sub-section. A `torch.cuda.synchronize()`
call precedes each time-reading point so elapsed values include actual GPU
completion, not just host-side dispatch.

Activation: set env var `SGLANG_VERIFY_DETAILED=1` when launching the server.
Timings are stashed on `eagle_worker._oracle_last_verify_detail` as a dict and
picked up by oracle_patch.py's forward_log emitter.

WARNING: The function bodies below are copied from SGLang's source (eagle_worker.py
and eagle_info.py); if SGLang is upgraded, cross-check with the installed version.
Tested against sglang == 0.5.9.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _sync_now() -> float:
    """Return perf_counter() after flushing GPU queue."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# Ordered list of timing keys we emit. Keep in sync with the section markers
# inside the instrumented functions below.
TIMING_KEYS_OUTER = [
    "outer_pre_prepare",        # entry until prepare_for_verify called
    "outer_prepare_for_verify",
    "outer_build_mwb",          # batch.get_model_worker_batch
    "outer_grammar_pre",        # grammar .cpu() copies (0 if no grammar)
    # target forward itself is timed separately (target_forward_ms)
    "outer_grammar_post",       # generate_token_bitmask (0 if no grammar)
    "outer_nan_detect",
    "outer_spec_verify",        # total time of spec_info.verify; drills down below
    "outer_target_forward_wall",  # sync-bracketed wall time of target_worker.forward_batch_generation
    "outer_accept_gather",      # next_token_logits/hidden_states gather
    "outer_mamba_update",       # 0 for non-mamba models
    "outer_logprob",            # 0 unless return_logprob set
    "outer_final",              # mode reset
]

TIMING_KEYS_INNER = [
    "inner_init_tensors",
    "inner_logit_ops",
    "inner_argmax",             # greedy path: argmax + reshape
    # inner_tree_greedy is captured as verify_greedy_ms (oracle no-op)
    "inner_tree_sampling",      # non-greedy path; 0 if greedy
    "inner_simulate",           # SIMULATE_ACC_LEN (0 unless enabled)
    "inner_tolist",             # accept_index/predict .tolist()
    "inner_pyloop",             # for-loop over reqs
    "inner_accept_sync",        # accept_length.cpu() + accept_index extraction + evict_mask
    "inner_kv_free",            # KV cache free/move
    "inner_construct_out",      # seq_lens/out_cache_loc + draft_input + return
]


def _make_instrumented_verify_inner(
    EagleVerifyOutput,
    EagleDraftInput,
    CaptureHiddenMode,
    verify_tree_greedy_func,
    tree_speculative_sampling_target_only,
    top_k_renorm_prob,
    top_p_renorm_prob,
    apply_custom_logit_processor,
    align_evict_mask_to_page_size,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
    assign_req_to_token_pool_func,
    create_accept_length_filter,
    filter_finished_cache_loc_kernel,
    generate_simulated_accept_index,
    next_power_of_2,
    get_global_server_args,
    SIMULATE_ACC_LEN,
    TREE_SPEC_KERNEL_AVAILABLE,
    record_sink,
):
    """Build the instrumented EagleVerifyInput.verify method.

    `record_sink(name, delta_ms)` is called after each section to push the timing
    into the caller's dict. `record_sink` is injected by the outer wrapper.
    """

    def verify_instrumented(
        self,
        batch,
        logits_output,
        token_to_kv_pool_allocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,
    ):
        if batch.forward_mode.is_idle():
            return EagleVerifyOutput(
                draft_input=EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                ),
                logits_output=logits_output,
                verified_id=torch.empty(0, dtype=torch.long, device=batch.device),
                accept_length_per_req_cpu=[],
                accepted_indices=torch.full(
                    (0, self.spec_steps + 1),
                    -1,
                    dtype=torch.int32,
                    device=batch.device,
                ),
            )

        # ---------- inner_init_tensors ----------
        t = _sync_now()
        bs = self.retrive_index.shape[0]
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.empty(predict_shape, dtype=torch.int32, device=batch.device)
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=batch.device)

        if bs != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            sampling_info.filter_batch(self.retrive_index.tolist(), self.retrive_index)
        record_sink("inner_init_tensors", (_sync_now() - t) * 1000)

        # ---------- inner_logit_ops ----------
        t = _sync_now()
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        if (
            sampling_info.penalizer_orchestrator.is_required
            or sampling_info.logit_bias is not None
        ):
            linear_penalty = torch.zeros(
                (bs, logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device=batch.device,
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
            )

        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )
        record_sink("inner_logit_ops", (_sync_now() - t) * 1000)

        # ---------- inner_argmax / inner_tree_sampling ----------
        is_all_greedy = sampling_info.is_all_greedy
        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            logger.warning(
                "Tree speculative sampling kernel unavailable (likely AMD/HIP build). "
                "Falling back to greedy verification."
            )

        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
            t = _sync_now()
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            record_sink("inner_argmax", (_sync_now() - t) * 1000)
            record_sink("inner_tree_sampling", 0.0)

            # verify_tree_greedy_func is timed elsewhere as verify_greedy_ms
            predict, accept_index, accept_length = verify_tree_greedy_func(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            record_sink("inner_argmax", 0.0)
            t = _sync_now()
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )
            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )
            if not torch.all(sampling_info.top_ps == 1.0):
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    ),
                )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device=batch.device
            )
            coins = torch.rand_like(
                candidates, dtype=torch.float32, device=batch.device
            )
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=batch.device
            )
            tree_speculative_sampling_target_only(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )
            record_sink("inner_tree_sampling", (_sync_now() - t) * 1000)

        # ---------- inner_simulate ----------
        t = _sync_now()
        if SIMULATE_ACC_LEN > 0.0:
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,
                accept_length=accept_length,
                bs=bs,
                spec_steps=self.spec_steps,
            )
        record_sink("inner_simulate", (_sync_now() - t) * 1000)

        # ---------- inner_tolist ----------
        t = _sync_now()
        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False
        record_sink("inner_tolist", (_sync_now() - t) * 1000)

        # ---------- inner_pyloop ----------
        t = _sync_now()
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            num_accepted = 0
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                num_accepted += 1
                id = predict_cpu[idx]
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n" f"{accept_index=}\n" f"{predict=}\n"
                            )
                            raise e
            req.kv_committed_len += num_accepted
            req.kv_allocated_len = req.kv_committed_len
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1
            accepted_draft_tokens = sum(1 for idx in accept_index_row if idx != -1) - 1
            req.spec_accepted_tokens += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)
        record_sink("inner_pyloop", (_sync_now() - t) * 1000)

        # ---------- inner_accept_sync ----------
        t = _sync_now()
        if has_finished:
            accept_length = (accept_index != -1).sum(dim=1) - 1

        accept_index = accept_index[accept_index != -1]
        verified_id = predict[accept_index]
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        accept_length_cpu = accept_length.cpu()
        accept_length_list = accept_length_cpu.tolist()
        record_sink("inner_accept_sync", (_sync_now() - t) * 1000)

        # ---------- inner_kv_free ----------
        t = _sync_now()
        if page_size == 1:
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            src_cache_loc = None
            tgt_cache_loc = None
        else:
            if self.topk == 1:
                align_evict_mask_to_page_size[len(batch.seq_lens),](
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
                src_cache_loc = None
                tgt_cache_loc = None
            else:
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    accept_length,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=torch.int64,
                    device=to_free_num_slots.device,
                )
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    accept_length,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )
                token_to_kv_pool_allocator.free(to_free_slots)
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )
        record_sink("inner_kv_free", (_sync_now() - t) * 1000)

        # ---------- inner_construct_out ----------
        t = _sync_now()
        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc,
                    bs,
                )
            else:
                batch.out_cache_loc = tgt_cache_loc
            batch.seq_lens.add_(accept_length + 1)
            batch.seq_lens_cpu.add_(accept_length_cpu + 1)

            draft_input = EagleDraftInput(
                hidden_states=batch.spec_info.hidden_states[accept_index],
                verified_id=verified_id,
                accept_length=accept_length,
                accept_length_cpu=accept_length_list,
                seq_lens_for_draft_extend=batch.seq_lens,
                seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu,
                req_pool_indices_for_draft_extend=batch.req_pool_indices,
            )
            result = EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=draft_input.accept_length_cpu,
                accepted_indices=accept_index,
            )
        else:
            if page_size == 1 or self.topk == 1:
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc[accept_index],
                    bs,
                )
                batch.seq_lens.add_(accept_length + 1)
                batch.seq_lens_cpu.add_(accept_length_cpu + 1)

            if len(unfinished_accept_index) > 0:
                unfinished_accept_index = torch.cat(unfinished_accept_index)
                unfinished_index_device = torch.tensor(
                    unfinished_index, dtype=torch.int64, device=predict.device
                )
                draft_input_accept_length_cpu = [
                    accept_length_list[i] for i in unfinished_index
                ]
                if page_size == 1 or self.topk == 1:
                    batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index]
                else:
                    batch.out_cache_loc = torch.empty(
                        len(unfinished_index) + sum(draft_input_accept_length_cpu),
                        dtype=torch.int64,
                        device=predict.device,
                    )
                    accept_length_filter = create_accept_length_filter(
                        accept_length,
                        unfinished_index_device,
                        batch.seq_lens,
                    )
                    batch.seq_lens_cpu.add_(accept_length_cpu + 1)
                    filter_finished_cache_loc_kernel[(bs,)](
                        batch.out_cache_loc,
                        tgt_cache_loc,
                        accept_length,
                        accept_length_filter,
                        next_power_of_2(bs),
                        next_power_of_2(self.draft_token_num),
                    )

                draft_input = EagleDraftInput(
                    hidden_states=batch.spec_info.hidden_states[
                        unfinished_accept_index
                    ],
                    verified_id=predict[unfinished_accept_index],
                    accept_length_cpu=draft_input_accept_length_cpu,
                    accept_length=accept_length[unfinished_index_device],
                    seq_lens_for_draft_extend=batch.seq_lens[unfinished_index_device],
                    seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu[unfinished_index],
                    req_pool_indices_for_draft_extend=batch.req_pool_indices[
                        unfinished_index_device
                    ],
                )
            else:
                draft_input = EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )

            result = EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_list,
                accepted_indices=accept_index,
            )
        record_sink("inner_construct_out", (_sync_now() - t) * 1000)
        return result

    return verify_instrumented


def install_detailed_verify_patch(eagle_worker) -> None:
    """Replace eagle_worker.verify + EagleVerifyInput.verify with instrumented versions.

    Call this AFTER oracle_patch's _patch_verify_logits because we overwrite
    eagle_worker.verify here and want our wrapping to include oracle total timing.
    """
    # Pull all dependencies from the live sglang installation so we don't
    # accidentally bind stale imports.
    from sglang.srt.speculative import eagle_info as _ei
    from sglang.srt.speculative import eagle_worker as _ew
    from sglang.srt.speculative import eagle_utils as _eu
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode as _CaptureHiddenMode,
        ForwardMode as _ForwardMode,
    )

    # Build closure-captured inner replacement.
    def _outer_record_sink(name, val):
        d = getattr(eagle_worker, "_oracle_last_verify_detail", None)
        if d is None:
            d = {}
            eagle_worker._oracle_last_verify_detail = d
        d[name] = round(val, 3)

    verify_instrumented_inner = _make_instrumented_verify_inner(
        EagleVerifyOutput=_ei.EagleVerifyOutput,
        EagleDraftInput=_ei.EagleDraftInput,
        CaptureHiddenMode=_CaptureHiddenMode,
        verify_tree_greedy_func=_ei.verify_tree_greedy_func,
        tree_speculative_sampling_target_only=getattr(
            _ei, "tree_speculative_sampling_target_only", None),
        top_k_renorm_prob=getattr(_ei, "top_k_renorm_prob", None),
        top_p_renorm_prob=getattr(_ei, "top_p_renorm_prob", None),
        apply_custom_logit_processor=getattr(_ei, "apply_custom_logit_processor", None),
        align_evict_mask_to_page_size=getattr(_ei, "align_evict_mask_to_page_size", None),
        get_src_tgt_cache_loc=getattr(_ei, "get_src_tgt_cache_loc", None),
        get_target_cache_loc=getattr(_ei, "get_target_cache_loc", None),
        assign_req_to_token_pool_func=getattr(_ei, "assign_req_to_token_pool_func", None),
        create_accept_length_filter=getattr(_ei, "create_accept_length_filter", None),
        filter_finished_cache_loc_kernel=getattr(_ei, "filter_finished_cache_loc_kernel", None),
        generate_simulated_accept_index=getattr(_ei, "generate_simulated_accept_index", None),
        next_power_of_2=getattr(_ei, "next_power_of_2", None),
        get_global_server_args=getattr(_ei, "get_global_server_args", None),
        SIMULATE_ACC_LEN=getattr(_ei, "SIMULATE_ACC_LEN", 0.0),
        TREE_SPEC_KERNEL_AVAILABLE=getattr(_ei, "TREE_SPEC_KERNEL_AVAILABLE", True),
        record_sink=_outer_record_sink,
    )

    # Patch the class so any new EagleVerifyInput instance uses it.
    _ei.EagleVerifyInput.verify = verify_instrumented_inner

    # Now replace eagle_worker.verify with an instrumented outer wrapper.
    # We shadow any previously installed wrapper (oracle_patch's) but still
    # stash verify_total_ms + verify_logits, same contract as oracle_patch.
    # target_worker.forward_batch_generation remains wrapped by oracle_patch,
    # so target_forward_ms still flows through unchanged.

    def verify_instrumented_outer(batch, spec_info):
        details = {}
        eagle_worker._oracle_last_verify_detail = details
        t_v_start = _sync_now()

        # outer_pre_prepare
        t = _sync_now()
        seq_lens_pre_verify = batch.seq_lens.clone()
        details["outer_pre_prepare"] = round((_sync_now() - t) * 1000, 3)

        # outer_prepare_for_verify
        t = _sync_now()
        spec_info.prepare_for_verify(batch, eagle_worker.page_size)
        spec_info.num_tokens_per_req = eagle_worker.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            _ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else _ForwardMode.IDLE
        )
        batch.spec_info = spec_info
        details["outer_prepare_for_verify"] = round((_sync_now() - t) * 1000, 3)

        # outer_build_mwb
        t = _sync_now()
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode
        details["outer_build_mwb"] = round((_sync_now() - t) * 1000, 3)

        # outer_grammar_pre
        t = _sync_now()
        retrieve_next_token_cpu = None
        retrieve_next_sibling_cpu = None
        draft_tokens_cpu = None
        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()
        details["outer_grammar_pre"] = round((_sync_now() - t) * 1000, 3)

        # outer_target_forward_wall — sync-bracketed wall time covering the
        # ENTIRE target_worker.forward_batch_generation call (includes async
        # GPU completion). oracle_patch's target_forward_ms only measures CPU
        # dispatch, so the real target-side GPU work would otherwise leak into
        # the next section's sync boundary.
        t = _sync_now()
        batch_result = eagle_worker.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        details["outer_target_forward_wall"] = round((_sync_now() - t) * 1000, 3)

        # outer_grammar_post
        t = _sync_now()
        vocab_mask = None
        if batch.has_grammar:
            from sglang.srt.speculative.spec_utils import generate_token_bitmask
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )
            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                batch.sampling_info.vocab_mask = None
        details["outer_grammar_post"] = round((_sync_now() - t) * 1000, 3)

        # outer_nan_detect
        t = _sync_now()
        if eagle_worker.enable_nan_detection:
            from sglang.srt.speculative.spec_utils import detect_nan
            detect_nan(logits_output)
        details["outer_nan_detect"] = round((_sync_now() - t) * 1000, 3)

        # outer_spec_verify (calls the instrumented inner verify)
        t = _sync_now()
        spec_info.hidden_states = logits_output.hidden_states
        res = spec_info.verify(
            batch,
            logits_output,
            eagle_worker.token_to_kv_pool_allocator,
            eagle_worker.page_size,
            vocab_mask,
        )
        details["outer_spec_verify"] = round((_sync_now() - t) * 1000, 3)

        # outer_accept_gather
        t = _sync_now()
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]
        details["outer_accept_gather"] = round((_sync_now() - t) * 1000, 3)

        # outer_mamba_update
        t = _sync_now()
        if (
            eagle_worker.target_worker.model_runner.hybrid_gdn_config is not None
            or eagle_worker.target_worker.model_runner.mamba2_config is not None
            or eagle_worker.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            eagle_worker._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )
        details["outer_mamba_update"] = round((_sync_now() - t) * 1000, 3)

        # outer_logprob
        t = _sync_now()
        if batch.return_logprob:
            from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
            add_output_logprobs_for_spec_v1(batch, res, logits_output)
        details["outer_logprob"] = round((_sync_now() - t) * 1000, 3)

        # outer_final
        t = _sync_now()
        batch.forward_mode = (
            _ForwardMode.DECODE if not batch.forward_mode.is_idle() else _ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input
        details["outer_final"] = round((_sync_now() - t) * 1000, 3)

        # Match oracle_patch's contract for verify_total.
        eagle_worker._oracle_last_verify_total_ms = (_sync_now() - t_v_start) * 1000
        # Stash verify logits for oracle_patch consumers that expect it
        try:
            _lg = logits_output.next_token_logits
            eagle_worker._oracle_stashed_verify_logits = (
                _lg.detach().cpu().clone() if _lg is not None and _lg.numel() > 0 else None
            )
        except Exception:
            eagle_worker._oracle_stashed_verify_logits = None

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    eagle_worker.verify = verify_instrumented_outer
    logger.info(
        "Detailed verify patch installed (SGLANG_VERIFY_DETAILED). "
        "Timings stashed at eagle_worker._oracle_last_verify_detail."
    )
