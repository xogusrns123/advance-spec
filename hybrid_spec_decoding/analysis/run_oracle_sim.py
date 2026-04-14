"""Run offline suffix decoding simulation and save results as JSON.

Uses SuffixDecodingCache to simulate suffix/hybrid/oracle methods on
oracle_vanilla trajectory. Saves per-method, per-request results that
the notebook can load without needing arctic_inference.

Requires arctic_inference (run inside the container).

Ported from agentic-bench/analysis/bfcl/run_oracle_sim.py for use with
SGLang + GLM-4.7-Flash in the advance-spec project.

Usage:
  # GLM-4.7-Flash with SGLang-collected oracle data
  python3 -m hybrid_spec_decoding.analysis.run_oracle_sim \
    --agent-results results/glm4_flash/oracle_vanilla/agent_results.json \
    --output results/glm4_flash/oracle_vanilla/oracle_sim.json \
    --model zai-org/GLM-4.7-Flash \
    --print-summary

  # With BFCL dataset for prompt reconstruction
  python3 -m hybrid_spec_decoding.analysis.run_oracle_sim \
    --agent-results results/glm4_flash/oracle_vanilla/agent_results.json \
    --output results/glm4_flash/oracle_vanilla/oracle_sim.json \
    --model zai-org/GLM-4.7-Flash \
    --dataset data/bfcl_multi_turn/dataset.jsonl \
    --print-summary

  # Override step latencies (measured on RTX 4090)
  python3 -m hybrid_spec_decoding.analysis.run_oracle_sim \
    --agent-results results/glm4_flash/oracle_vanilla/agent_results.json \
    --output results/glm4_flash/oracle_vanilla/oracle_sim.json \
    --step-latencies eagle3=0.015,suffix=0.012,vanilla=0.011 \
    --print-summary
"""

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Step latencies (defaults from Qwen3-8B, overridable via --step-latencies)
# TODO: Recalibrate for GLM-4.7-Flash on RTX 4090 x4 using calibrate_latency.py
EAGLE3_STEP_S = 0.013311
SUFFIX_STEP_S = 0.010279
VANILLA_STEP_S = 0.010023
DRAFT_MODEL_STEP_S = 0.043227

# Default full draft depths (overridable via --eagle3-depths / --dm-depths max)
EAGLE3_FULL_DEPTH = 5
DM_FULL_DEPTH = 4


def _interpolate_step_latency(full_step_s, k, full_depth):
    """Linearly interpolate step latency for reduced draft depth k.

    step_latency(k) = VANILLA_STEP_S + (full_step_s - VANILLA_STEP_S) * k / full_depth
    """
    draft_only = full_step_s - VANILLA_STEP_S
    return VANILLA_STEP_S + draft_only * k / full_depth


class MethodSpec:
    """Fully describes one experimental case."""
    __slots__ = ("name", "method", "e3_k", "dm_k", "threshold")

    def __init__(self, name, method, e3_k=None, dm_k=None, threshold=0.0):
        self.name = name
        self.method = method
        self.e3_k = e3_k
        self.dm_k = dm_k
        self.threshold = threshold

    def __repr__(self):
        return f"MethodSpec({self.name!r})"


def build_method_specs(eagle3_depths, dm_depths, thresholds, has_dm):
    """Generate all combinatorial MethodSpec cases.

    88 cases total (with 5 eagle3 depths, 4 dm depths, 5 thresholds):
      eagle3(5) + suffix(1) + draft_model(4) + hybrid_E3S(25) + hybrid_DMS(20)
      + E3⊕S(5) + DM⊕S(4) + E3⊕DM(20) + S⊕DM(4)
    """
    specs = []

    # Eagle3 standalone: sweep e3_k
    for k in eagle3_depths:
        specs.append(MethodSpec(f"eagle3_e3k{k}", "eagle3", e3_k=k))

    # Suffix: no depth parameter
    specs.append(MethodSpec("suffix", "suffix"))

    # Draft model standalone: sweep dm_k
    if has_dm:
        for k in dm_depths:
            specs.append(MethodSpec(f"draft_model_dmk{k}", "draft_model", dm_k=k))

    # Hybrid E3+suffix: e3_k × threshold
    for k in eagle3_depths:
        for t in thresholds:
            specs.append(MethodSpec(
                f"hybrid_e3k{k}_t{t}", "hybrid", e3_k=k, threshold=t))

    # Hybrid DM+suffix: dm_k × threshold
    if has_dm:
        for k in dm_depths:
            for t in thresholds:
                specs.append(MethodSpec(
                    f"hybrid_dm_dmk{k}_t{t}", "hybrid_dm", dm_k=k, threshold=t))

    # E3 ⊕ Suffix: e3_k
    for k in eagle3_depths:
        specs.append(MethodSpec(
            f"eagle3_ext_suffix_e3k{k}", "eagle3_ext_suffix", e3_k=k))

    # DM ⊕ Suffix: dm_k
    if has_dm:
        for k in dm_depths:
            specs.append(MethodSpec(
                f"dm_ext_suffix_dmk{k}", "dm_ext_suffix", dm_k=k))

    # E3 ⊕ DM: e3_k × dm_k
    if has_dm:
        for ek in eagle3_depths:
            for dk in dm_depths:
                specs.append(MethodSpec(
                    f"eagle3_ext_dm_e3k{ek}_dmk{dk}", "eagle3_ext_dm",
                    e3_k=ek, dm_k=dk))

    # Suffix ⊕ DM: dm_k
    if has_dm:
        for k in dm_depths:
            specs.append(MethodSpec(
                f"suffix_ext_dm_dmk{k}", "suffix_ext_dm", dm_k=k))

    # --- Hybrid + Extension ---
    # hybrid_ext_e3s (E3+S): suffix decides, extension on exhaust
    if has_dm:
        for k in eagle3_depths:
            for t in thresholds:
                specs.append(MethodSpec(
                    f"hybrid_ext_e3s_e3k{k}_t{t}", "hybrid_ext_e3s",
                    e3_k=k, threshold=t))

        # hybrid_ext_dms (DM+S): suffix decides, extension on exhaust
        for k in dm_depths:
            for t in thresholds:
                specs.append(MethodSpec(
                    f"hybrid_ext_dms_dmk{k}_t{t}", "hybrid_ext_dms",
                    dm_k=k, threshold=t))

        # hybrid_ext_e3dm (E3+DM with suffix selection)
        for ek in eagle3_depths:
            for dk in dm_depths:
                for t in thresholds:
                    specs.append(MethodSpec(
                        f"hybrid_ext_e3dm_e3k{ek}_dmk{dk}_t{t}",
                        "hybrid_ext_e3dm",
                        e3_k=ek, dm_k=dk, threshold=t))

        # hybrid_ext_sdm (S+DM): same logic as hybrid_ext_dms
        for k in dm_depths:
            for t in thresholds:
                specs.append(MethodSpec(
                    f"hybrid_ext_sdm_dmk{k}_t{t}", "hybrid_ext_sdm",
                    dm_k=k, threshold=t))

    # --- Tree-based Extension ---
    if has_dm:
        # E3 ⊗ DM: e3_k × dm_k
        for ek in eagle3_depths:
            for dk in dm_depths:
                specs.append(MethodSpec(
                    f"eagle3_tree_dm_e3k{ek}_dmk{dk}", "eagle3_tree_dm",
                    e3_k=ek, dm_k=dk))

        # DM ⊗ E3: dm_k × e3_k
        for dk in dm_depths:
            for ek in eagle3_depths:
                specs.append(MethodSpec(
                    f"dm_tree_eagle3_dmk{dk}_e3k{ek}", "dm_tree_eagle3",
                    dm_k=dk, e3_k=ek))

    # E3 ⊗ Suffix: e3_k
    for k in eagle3_depths:
        specs.append(MethodSpec(
            f"eagle3_tree_suffix_e3k{k}", "eagle3_tree_suffix", e3_k=k))

    if has_dm:
        # DM ⊗ Suffix: dm_k
        for k in dm_depths:
            specs.append(MethodSpec(
                f"dm_tree_suffix_dmk{k}", "dm_tree_suffix", dm_k=k))

        # Suffix ⊗ DM: dm_k
        for k in dm_depths:
            specs.append(MethodSpec(
                f"suffix_tree_dm_dmk{k}", "suffix_tree_dm", dm_k=k))

    # Suffix ⊗ E3: e3_k
    for k in eagle3_depths:
        specs.append(MethodSpec(
            f"suffix_tree_eagle3_e3k{k}", "suffix_tree_eagle3", e3_k=k))

    return specs


def prefix_match(draft, future):
    n = 0
    for a, b in zip(draft, future):
        if a != b:
            break
        n += 1
    return n


def count_accepted_tree(draft, future_tokens):
    """Count accepted tokens by walking the draft tree greedily.

    Returns (accepted, exhausted) where exhausted=True means the walk
    reached a leaf node (tree ran out of candidates), not a mismatch.
    """
    accepted = 0
    node = -1
    exhausted = True  # assume exhausted unless mismatch breaks
    for token_id in future_tokens:
        children = [i for i, p in enumerate(draft.parents) if p == node]
        if not children:
            # Leaf node reached — tree exhausted
            break
        matched = False
        for c in children:
            if draft.token_ids[c] == token_id:
                accepted += 1
                node = c
                matched = True
                break
        if not matched:
            exhausted = False
            break
    return accepted, exhausted



def _compute_suffix_extension_acc(ext_pos, tokens, N,
                                  suffix_cache, req_id, prompt, decoded,
                                  pos, max_spec_tokens, max_spec_factor,
                                  min_token_prob):
    """Compute suffix extension acceptance at ext_pos.

    Called only when method A's draft was fully accepted (exhausted).
    Suffix extension is exact (no hidden state dependency, CPU-only).
    """
    if ext_pos >= N:
        return 0
    ext_future = tokens[ext_pos + 1:]
    ext_response = decoded + list(tokens[pos:ext_pos + 1])
    if len(prompt) > 0:
        ext_context = np.concatenate(
            [prompt, np.array(ext_response, dtype=np.int32)])
    else:
        ext_context = np.array(ext_response, dtype=np.int32)
    ext_draft = suffix_cache.speculate(
        req_id, ext_context,
        max_spec_tokens=max_spec_tokens,
        max_spec_factor=max_spec_factor,
        min_token_prob=min_token_prob)
    ext_acc, _ = count_accepted_tree(ext_draft, ext_future)
    return ext_acc


# ---------------------------------------------------------------------------
# Tree-based extension helpers
# ---------------------------------------------------------------------------

class DraftTree:
    """Lightweight draft tree compatible with count_accepted_tree()."""
    __slots__ = ("parents", "token_ids", "score")

    def __init__(self, parents, token_ids, score=0.0):
        self.parents = parents
        self.token_ids = token_ids
        self.score = score


def _flat_to_tree(draft_tokens):
    """Convert flat draft list to (parents, token_ids) chain."""
    parents = [i - 1 for i in range(len(draft_tokens))]
    return parents, list(draft_tokens)


def _attach_subtree(parents, token_ids, attach_to, sub_parents, sub_token_ids):
    """Attach a sub-tree to an existing tree at node *attach_to*.

    Re-indexes sub-tree nodes.  Sub-tree root children (parent == -1)
    become children of *attach_to*.
    """
    base = len(parents)
    for i in range(len(sub_parents)):
        p = sub_parents[i]
        parents.append(attach_to if p == -1 else base + p)
        token_ids.append(sub_token_ids[i])


def build_combined_tree(primary_parents, primary_token_ids,
                        get_secondary_at, max_primary_len):
    """Build combined tree: primary structure + secondary branches at root
    and each primary node.

    *get_secondary_at(offset)* returns ``(sub_parents, sub_token_ids)``
    for the secondary draft at position ``pos + offset``.

    * offset == 0 → secondary at root (pos itself)
    * offset == depth + 1 → secondary at primary node at *depth*
    """
    parents = list(primary_parents)
    token_ids = list(primary_token_ids)

    # Compute depth of each primary node (assumes topological order)
    n_primary = len(primary_parents)
    depths = [0] * n_primary
    for i in range(n_primary):
        if primary_parents[i] == -1:
            depths[i] = 0
        else:
            depths[i] = depths[primary_parents[i]] + 1

    # Secondary at root (offset 0)
    sub_p, sub_t = get_secondary_at(0)
    _attach_subtree(parents, token_ids, -1, sub_p, sub_t)

    # Secondary at each primary node
    for node_idx in range(n_primary):
        offset = depths[node_idx] + 1
        if offset > max_primary_len:
            continue
        sub_p, sub_t = get_secondary_at(offset)
        _attach_subtree(parents, token_ids, node_idx, sub_p, sub_t)

    return DraftTree(parents, token_ids)


def load_exclude_ids(path):
    ids = set()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.add(line)
    except FileNotFoundError:
        pass
    return ids


def _extract_entries(entries):
    """Extract tokens and eagle3 drafts from oracle_vanilla_entries."""
    call_tokens = []
    call_eagle3s = []
    for e in entries:
        toks = (e["tokens"][0]
                if e.get("tokens") and e["tokens"] else [])
        call_tokens.extend(toks)
        call_eagle3s.append(
            e["eagle3"][0]
            if e.get("eagle3") and e["eagle3"] else [])
    return call_tokens, call_eagle3s


def _langchain_to_openai_messages(messages):
    """Convert LangChain message format (type/content) to OpenAI format (role/content)."""
    TYPE_MAP = {"system": "system", "human": "user", "ai": "assistant",
                "tool": "tool"}
    result = []
    for m in messages:
        msg_type = m.get("type", m.get("role", ""))
        role = TYPE_MAP.get(msg_type, msg_type)
        msg = {"role": role, "content": m.get("content", "")}
        # Preserve tool_calls for assistant messages
        if role == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["args"])
                              if isinstance(tc["args"], dict) else tc["args"]}}
                for tc in m["tool_calls"]
            ]
        # Preserve tool_call_id for tool result messages
        if role == "tool" and m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        result.append(msg)
    return result


def _reconstruct_swebench_prompts(q, tokenizer):
    """Reconstruct prompt token IDs for SWE-bench (messages in turns[])."""
    turns = q.get("turns", [])
    llm_steps = [s for s in q["agent_metrics"]["steps"]
                 if s.get("type") == "llm"]
    prompt_ids_list = []
    for step_idx in range(len(llm_steps)):
        if step_idx < len(turns) and isinstance(turns[step_idx], dict):
            messages = turns[step_idx].get("messages", [])
            if messages:
                # Convert LangChain format to OpenAI format if needed
                if messages[0].get("type") and not messages[0].get("role"):
                    messages = _langchain_to_openai_messages(messages)
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                prompt_ids_list.append(tokenizer.encode(text))
                continue
        prompt_ids_list.append([])
    return prompt_ids_list


def _reconstruct_bfcl_prompts(bfcl_entry, resp_entry, tokenizer):
    """Reconstruct prompt token IDs for BFCL (needs dataset + responses)."""
    try:
        from bfcl_eval.model_handler.utils import (
            system_prompt_pre_processing_chat_model,
        )
        from bfcl_eval.constants.default_prompts import (
            DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
        )
    except ImportError:
        return None

    conversation_turns = bfcl_entry["question"]
    functions = list(bfcl_entry.get("function", []))
    holdout_function = bfcl_entry.get("missed_function", {})
    entry_id = bfcl_entry["bfcl_id"]

    all_turn_messages = copy.deepcopy(conversation_turns)
    all_turn_messages[0] = system_prompt_pre_processing_chat_model(
        all_turn_messages[0], functions, entry_id)

    messages = []
    prompt_ids_list = []

    for turn in resp_entry["conversation"]:
        turn_idx = turn["turn_idx"]
        if str(turn_idx) in holdout_function:
            holdout_docs = holdout_function[str(turn_idx)]
            turn_messages = [{
                "role": "user",
                "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                    functions=holdout_docs),
            }]
        else:
            turn_messages = (all_turn_messages[turn_idx]
                             if turn_idx < len(all_turn_messages) else [])
        messages.extend(turn_messages)

        for step_data in turn["steps"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt_ids_list.append(tokenizer.encode(text))
            messages.append({"role": "assistant",
                             "content": step_data["model_response"]})
            for exec_result in step_data.get("exec_results", []):
                messages.append({"role": "tool", "content": str(exec_result)})

    return prompt_ids_list


def _reconstruct_specbench_prompts(dataset_entry, result_question, tokenizer):
    """Reconstruct prompt token IDs for SpecBench (needs dataset)."""
    user_turns = dataset_entry["turns"]
    result_turns = result_question["turns"]
    messages = []
    prompt_ids_list = []

    for turn_idx, (user_msg, result_turn) in enumerate(
            zip(user_turns, result_turns)):
        messages.append({"role": "user", "content": user_msg})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_ids_list.append(tokenizer.encode(text))
        messages.append({"role": "assistant",
                         "content": result_turn.get("response", "")})

    return prompt_ids_list


def extract_requests(data, exclude_ids, dm_by_id=None,
                     tokenizer=None, bfcl_dataset=None, resp_by_id=None,
                     specbench_dataset=None):
    """Extract per-request tokens and eagle3 drafts (per LLM call).

    Per-question format detection:
    - agent_metrics present → BFCL/SWE-bench path
    - turns with dicts → SpecBench path
    Supports mixed-format questions (cross-workload merge).
    """
    questions = data.get("questions", [])
    requests = []

    for q in questions:
        if "agent_metrics" in q:
            reqs = _extract_bfcl([q], exclude_ids, dm_by_id,
                                 tokenizer, bfcl_dataset, resp_by_id)
        else:
            reqs = _extract_online(
                {"questions": [q], "per_request": []}, exclude_ids, dm_by_id,
                tokenizer, specbench_dataset)
        requests.extend(reqs)

    return requests


def _extract_bfcl(questions, exclude_ids, dm_by_id=None,
                  tokenizer=None, bfcl_dataset=None, resp_by_id=None):
    """Extract from agent format (BFCL / SWE-bench)."""
    requests = []
    for q in questions:
        bfcl_id = q.get("bfcl_id") or q.get("instance_id") or str(q.get("question_id", ""))
        if bfcl_id in exclude_ids:
            continue

        per_call_tokens = []
        per_call_eagle3s = []

        for s in q["agent_metrics"]["steps"]:
            entries = s.get("spec_decode", {}).get(
                "oracle_vanilla_entries", [])
            if not entries:
                continue
            ct, ce = _extract_entries(entries)
            per_call_tokens.append(ct)
            per_call_eagle3s.append(ce)

        # Reconstruct prompt token IDs for suffix cache local tree
        per_call_prompt_ids = None
        if tokenizer:
            turns = q.get("turns", [])
            has_messages = (turns and isinstance(turns[0], dict)
                           and "messages" in turns[0])
            if has_messages:
                # SWE-bench: messages embedded in agent_results.json
                per_call_prompt_ids = _reconstruct_swebench_prompts(
                    q, tokenizer)
            elif bfcl_dataset and resp_by_id and bfcl_id in bfcl_dataset:
                # BFCL: needs external dataset + responses file
                resp = resp_by_id.get(bfcl_id)
                if resp:
                    per_call_prompt_ids = _reconstruct_bfcl_prompts(
                        bfcl_dataset[bfcl_id], resp, tokenizer)

        requests.append({
            "bfcl_id": bfcl_id,
            "category": q.get("category", ""),
            "per_call_tokens": per_call_tokens,
            "per_call_eagle3s": per_call_eagle3s,
            "per_call_prompt_ids": per_call_prompt_ids,
            "n_tokens": sum(len(ct) for ct in per_call_tokens),
            "draft_model_drafts": dm_by_id.get(bfcl_id) if dm_by_id else None,
        })
    return requests


def _extract_online(data, exclude_ids, dm_by_id=None,
                    tokenizer=None, specbench_dataset=None):
    """Extract from online.py format (SpecBench, etc.)."""
    questions = data.get("questions", [])
    per_request = data.get("per_request", [])
    requests = []

    # Build question_id → category mapping
    q_map = {}
    for q in questions:
        qid = str(q.get("question_id", ""))
        q_map[qid] = q.get("category", "")

    # Try questions[].turns[] first, fallback to per_request[]
    if questions and "turns" in questions[0] and isinstance(questions[0]["turns"], list):
        # questions[].turns[] has per-turn spec_decode
        for qi, q in enumerate(questions):
            qid = str(q.get("question_id", qi))
            if qid in exclude_ids:
                continue

            per_call_tokens = []
            per_call_eagle3s = []

            for turn in q["turns"]:
                if isinstance(turn, dict):
                    entries = turn.get("spec_decode", {}).get(
                        "oracle_vanilla_entries", [])
                    if not entries:
                        continue
                    ct, ce = _extract_entries(entries)
                    per_call_tokens.append(ct)
                    per_call_eagle3s.append(ce)

            if not per_call_tokens:
                continue

            # Reconstruct prompt token IDs for SpecBench
            per_call_prompt_ids = None
            if tokenizer and specbench_dataset:
                ds_entry = specbench_dataset.get(q.get("question_id"))
                if ds_entry:
                    per_call_prompt_ids = _reconstruct_specbench_prompts(
                        ds_entry, q, tokenizer)

            requests.append({
                "bfcl_id": qid,
                "category": q.get("category", ""),
                "per_call_tokens": per_call_tokens,
                "per_call_eagle3s": per_call_eagle3s,
                "per_call_prompt_ids": per_call_prompt_ids,
                "n_tokens": sum(len(ct) for ct in per_call_tokens),
                "draft_model_drafts": dm_by_id.get(qid) if dm_by_id else None,
            })
    else:
        # Fallback: per_request[] (flat, one entry per LLM call)
        for ri, r in enumerate(per_request):
            entries = r.get("spec_decode", {}).get(
                "oracle_vanilla_entries", [])
            if not entries:
                continue

            ct, ce = _extract_entries(entries)
            qid = str(ri)
            category = q_map.get(qid, "") if ri < len(questions) else ""

            requests.append({
                "bfcl_id": qid,
                "category": category,
                "per_call_tokens": [ct],
                "per_call_eagle3s": [ce],
                "per_call_prompt_ids": None,
                "n_tokens": len(ct),
                "draft_model_drafts": dm_by_id.get(qid) if dm_by_id else None,
            })

    return requests


def simulate_request(req, method, suffix_cache, threshold=0.0,
                     max_spec_tokens=64, max_spec_factor=1.0,
                     min_token_prob=0.1,
                     eagle3_max_depth=None, dm_max_depth=None):
    """Simulate one method on one request using suffix cache.

    Returns dict with steps, e_wins, s_wins, d_wins, ties,
    per_step_e_acc, per_step_s_acc, per_step_acc.
    """
    # Effective step latencies (interpolated if depth is reduced)
    eff_eagle3_step = (_interpolate_step_latency(EAGLE3_STEP_S, eagle3_max_depth, EAGLE3_FULL_DEPTH)
                       if eagle3_max_depth is not None else EAGLE3_STEP_S)
    eff_dm_step = (_interpolate_step_latency(DRAFT_MODEL_STEP_S, dm_max_depth, DM_FULL_DEPTH)
                   if dm_max_depth is not None else DRAFT_MODEL_STEP_S)

    steps = 0
    e_wins = s_wins = d_wins = ties = 0
    total_step_time = 0.0
    per_step_e_acc = []
    per_step_s_acc = []
    per_step_dm_acc = []
    per_step_acc = []
    per_step_e_draft_len = []
    per_step_s_exhausted = []
    per_step_dm_draft_len = []
    req_id = 0
    dm_drafts = req.get("draft_model_drafts")
    global_pos = 0

    prompt_ids_list = req.get("per_call_prompt_ids")

    for call_idx in range(len(req["per_call_tokens"])):
        tokens = req["per_call_tokens"][call_idx]
        eagle3s = req["per_call_eagle3s"][call_idx]
        N = len(tokens)
        if N == 0:
            continue

        # Pass prompt tokens to suffix cache local tree when available
        if prompt_ids_list and call_idx < len(prompt_ids_list):
            prompt = np.array(prompt_ids_list[call_idx], dtype=np.int32)
        else:
            prompt = np.array([], dtype=np.int32)
        suffix_cache.start_request(req_id, prompt)
        decoded = []
        pos = 0

        while pos < N:
            future = tokens[pos + 1:]

            # Eagle3
            e_draft = eagle3s[pos] if pos < len(eagle3s) else []
            if eagle3_max_depth is not None:
                e_draft = e_draft[:eagle3_max_depth]
            e_acc = prefix_match(e_draft, future)

            # Suffix: offline simulation
            # Context must include prompt + response (see simulator.py:58)
            response_so_far = decoded + [tokens[pos]]
            if len(prompt) > 0:
                context = np.concatenate(
                    [prompt, np.array(response_so_far, dtype=np.int32)])
            else:
                context = np.array(response_so_far, dtype=np.int32)
            draft = suffix_cache.speculate(
                req_id, context,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=max_spec_factor,
                min_token_prob=min_token_prob)
            s_acc, s_exhausted = count_accepted_tree(draft, future)

            # Draft model
            dm_d = (dm_drafts[global_pos + pos]
                    if dm_drafts and (global_pos + pos) < len(dm_drafts)
                    else [])
            if dm_max_depth is not None:
                dm_d = dm_d[:dm_max_depth]
            dm_acc = prefix_match(dm_d, future)

            # Extension "all accepted" (draft exhausted) checks
            e_exhausted = (e_acc == len(e_draft) and len(e_draft) > 0)
            dm_exhausted = (dm_acc == len(dm_d) and len(dm_d) > 0)

            # Method selection
            suffix_chosen = False  # track hybrid's actual choice
            chosen_step_time = VANILLA_STEP_S
            if method == "eagle3":
                acc = e_acc
                chosen_step_time = eff_eagle3_step
            elif method == "suffix":
                acc = s_acc
                chosen_step_time = SUFFIX_STEP_S
            elif method == "draft_model":
                acc = dm_acc
                chosen_step_time = eff_dm_step
            elif method == "hybrid":
                suffix_chosen = (draft.score >= threshold
                                 and bool(draft.token_ids))
                acc = s_acc if suffix_chosen else e_acc
                chosen_step_time = SUFFIX_STEP_S if suffix_chosen else (
                    SUFFIX_STEP_S + eff_eagle3_step - VANILLA_STEP_S)
            elif method == "hybrid_dm":
                suffix_chosen = (draft.score >= threshold
                                 and bool(draft.token_ids))
                acc = s_acc if suffix_chosen else dm_acc
                chosen_step_time = SUFFIX_STEP_S if suffix_chosen else (
                    SUFFIX_STEP_S + eff_dm_step - VANILLA_STEP_S)
            # --- DEPRECATED: MAT oracles (kept for reference) ---
            # elif method == "oracle_mat_se":
            #     acc = max(e_acc, s_acc)
            #     chosen_step_time = (eff_eagle3_step if e_acc > s_acc
            #                         else SUFFIX_STEP_S if s_acc > e_acc
            #                         else min(eff_eagle3_step, SUFFIX_STEP_S))
            # elif method == "oracle_mat_sd":
            #     acc = max(s_acc, dm_acc)
            #     chosen_step_time = (SUFFIX_STEP_S if s_acc > dm_acc
            #                         else eff_dm_step if dm_acc > s_acc
            #                         else min(SUFFIX_STEP_S, eff_dm_step))
            # elif method == "oracle_mat_base":
            #     acc = (max(e_acc, s_acc, dm_acc) if dm_drafts
            #            else max(e_acc, s_acc))
            #     ...
            # elif method == "oracle_mat_ext":
            #     ... (see git history)
            # --- Greedy latency oracles ---
            elif method in ("oracle_latency_base", "oracle_latency_ext",
                            "oracle_latency_ext_tree"):
                # Greedy: pick max acceptance at each position.
                # Tie-break by cheapest step time.
                candidates = [(e_acc, eff_eagle3_step),
                              (s_acc, SUFFIX_STEP_S)]
                if dm_drafts:
                    candidates.append((dm_acc, eff_dm_step))

                S_DRAFT_ONLY = SUFFIX_STEP_S - VANILLA_STEP_S
                DM_DRAFT_ONLY = eff_dm_step - VANILLA_STEP_S
                E3_DRAFT_ONLY = eff_eagle3_step - VANILLA_STEP_S

                # Linear extension candidates (ext and ext_tree)
                if method in ("oracle_latency_ext",
                              "oracle_latency_ext_tree"):
                    if e_exhausted:
                        ext_pos_e = pos + e_acc + 1
                        ext_s_acc = _compute_suffix_extension_acc(
                            ext_pos_e, tokens, N, suffix_cache, req_id,
                            prompt, decoded, pos,
                            max_spec_tokens, max_spec_factor,
                            min_token_prob)
                        candidates.append((e_acc + ext_s_acc,
                                           eff_eagle3_step + S_DRAFT_ONLY))
                        if dm_drafts:
                            ext_dm_d = (
                                dm_drafts[global_pos + ext_pos_e]
                                if (global_pos + ext_pos_e) < len(dm_drafts)
                                else [])
                            if dm_max_depth is not None:
                                ext_dm_d = ext_dm_d[:dm_max_depth]
                            ext_d_acc = prefix_match(
                                ext_dm_d,
                                tokens[ext_pos_e + 1:]
                                if ext_pos_e < N else [])
                            candidates.append((e_acc + ext_d_acc,
                                               eff_eagle3_step + DM_DRAFT_ONLY))
                    if dm_drafts and dm_exhausted:
                        ext_pos_d = pos + dm_acc + 1
                        ext_s_acc2 = _compute_suffix_extension_acc(
                            ext_pos_d, tokens, N, suffix_cache, req_id,
                            prompt, decoded, pos,
                            max_spec_tokens, max_spec_factor,
                            min_token_prob)
                        candidates.append((dm_acc + ext_s_acc2,
                                           eff_dm_step + S_DRAFT_ONLY))
                    if s_exhausted and s_acc > 0 and dm_drafts:
                        ext_pos_s = pos + s_acc + 1
                        ext_dm_d2 = (
                            dm_drafts[global_pos + ext_pos_s]
                            if (global_pos + ext_pos_s) < len(dm_drafts)
                            else [])
                        if dm_max_depth is not None:
                            ext_dm_d2 = ext_dm_d2[:dm_max_depth]
                        ext_d_acc2 = prefix_match(
                            ext_dm_d2,
                            tokens[ext_pos_s + 1:]
                            if ext_pos_s < N else [])
                        candidates.append((s_acc + ext_d_acc2,
                                           SUFFIX_STEP_S + DM_DRAFT_ONLY))

                # Tree extension candidates (ext_tree only)
                # Includes 4 cheap tree types (no extra suffix speculate).
                # eagle3_tree_suffix and dm_tree_suffix excluded (too expensive).
                if method == "oracle_latency_ext_tree":
                    # E3 ⊗ DM
                    if dm_drafts:
                        e_p, e_t = _flat_to_tree(e_draft)
                        def _orc_dm(off, _gp=global_pos, _p=pos):
                            t = _gp + _p + off
                            d = (dm_drafts[t]
                                 if t < len(dm_drafts) else [])
                            if dm_max_depth is not None:
                                d = d[:dm_max_depth]
                            return _flat_to_tree(d)
                        comb = build_combined_tree(
                            e_p, e_t, _orc_dm, len(e_draft))
                        tree_acc, _ = count_accepted_tree(comb, future)
                        candidates.append((tree_acc,
                                           eff_eagle3_step + DM_DRAFT_ONLY))
                    # DM ⊗ E3
                    if dm_drafts:
                        dm_p, dm_t = _flat_to_tree(dm_d)
                        def _orc_e3(off, _p=pos):
                            t = _p + off
                            d = (eagle3s[t]
                                 if t < len(eagle3s) else [])
                            if eagle3_max_depth is not None:
                                d = d[:eagle3_max_depth]
                            return _flat_to_tree(d)
                        comb = build_combined_tree(
                            dm_p, dm_t, _orc_e3, len(dm_d))
                        tree_acc, _ = count_accepted_tree(comb, future)
                        candidates.append((tree_acc,
                                           eff_dm_step + E3_DRAFT_ONLY))
                    # Suffix ⊗ DM
                    if dm_drafts:
                        def _orc_dm2(off, _gp=global_pos, _p=pos):
                            t = _gp + _p + off
                            d = (dm_drafts[t]
                                 if t < len(dm_drafts) else [])
                            if dm_max_depth is not None:
                                d = d[:dm_max_depth]
                            return _flat_to_tree(d)
                        comb = build_combined_tree(
                            list(draft.parents), list(draft.token_ids),
                            _orc_dm2, max_spec_tokens)
                        tree_acc, _ = count_accepted_tree(comb, future)
                        candidates.append((tree_acc,
                                           SUFFIX_STEP_S + DM_DRAFT_ONLY))
                    # Suffix ⊗ E3
                    def _orc_e3b(off, _p=pos):
                        t = _p + off
                        d = (eagle3s[t]
                             if t < len(eagle3s) else [])
                        if eagle3_max_depth is not None:
                            d = d[:eagle3_max_depth]
                        return _flat_to_tree(d)
                    comb = build_combined_tree(
                        list(draft.parents), list(draft.token_ids),
                        _orc_e3b, max_spec_tokens)
                    tree_acc, _ = count_accepted_tree(comb, future)
                    candidates.append((tree_acc,
                                       SUFFIX_STEP_S + E3_DRAFT_ONLY))

                acc = max(c[0] for c in candidates)
                best_candidates = [c for c in candidates if c[0] == acc]
                chosen_step_time = min(c[1] for c in best_candidates)
            # --- Extension by suffix: A drafts first, if all accepted → suffix extends ---
            elif method == "eagle3_ext_suffix":
                ext_acc = 0
                if e_exhausted:
                    ext_pos = pos + e_acc + 1
                    ext_acc = _compute_suffix_extension_acc(
                        ext_pos, tokens, N, suffix_cache, req_id,
                        prompt, decoded, pos,
                        max_spec_tokens, max_spec_factor, min_token_prob)
                acc = e_acc + ext_acc
                chosen_step_time = eff_eagle3_step + (SUFFIX_STEP_S - VANILLA_STEP_S)
            elif method == "dm_ext_suffix":
                ext_acc = 0
                if dm_exhausted:
                    ext_pos = pos + dm_acc + 1
                    ext_acc = _compute_suffix_extension_acc(
                        ext_pos, tokens, N, suffix_cache, req_id,
                        prompt, decoded, pos,
                        max_spec_tokens, max_spec_factor, min_token_prob)
                acc = dm_acc + ext_acc
                chosen_step_time = eff_dm_step + (SUFFIX_STEP_S - VANILLA_STEP_S)
            # --- Extension by DM: all-accepted condition ---
            elif method == "eagle3_ext_dm":
                ext_acc = 0
                if e_exhausted:
                    ext_pos = pos + e_acc + 1
                    ext_dm_d = (dm_drafts[global_pos + ext_pos]
                                if dm_drafts and (global_pos + ext_pos) < len(dm_drafts)
                                else [])
                    if dm_max_depth is not None:
                        ext_dm_d = ext_dm_d[:dm_max_depth]
                    ext_acc = prefix_match(ext_dm_d, tokens[ext_pos + 1:] if ext_pos < N else [])
                acc = e_acc + ext_acc
                chosen_step_time = eff_eagle3_step + (eff_dm_step - VANILLA_STEP_S)
            elif method == "suffix_ext_dm":
                ext_acc = 0
                if s_exhausted and s_acc > 0:
                    ext_pos = pos + s_acc + 1
                    ext_dm_d = (dm_drafts[global_pos + ext_pos]
                                if dm_drafts and (global_pos + ext_pos) < len(dm_drafts)
                                else [])
                    if dm_max_depth is not None:
                        ext_dm_d = ext_dm_d[:dm_max_depth]
                    ext_acc = prefix_match(ext_dm_d, tokens[ext_pos + 1:] if ext_pos < N else [])
                acc = s_acc + ext_acc
                chosen_step_time = SUFFIX_STEP_S + (eff_dm_step - VANILLA_STEP_S)
            # --- Hybrid + Extension ---
            elif method == "hybrid_ext_e3s":
                # E3+S: suffix decides primary; exhaust → extend with other
                suffix_chosen = (draft.score >= threshold
                                 and bool(draft.token_ids))
                S_DRAFT_ONLY = SUFFIX_STEP_S - VANILLA_STEP_S
                E3_DRAFT_ONLY = eff_eagle3_step - VANILLA_STEP_S
                if suffix_chosen:
                    ext_acc = 0
                    if s_exhausted and s_acc > 0:
                        ext_pos = pos + s_acc + 1
                        ext_e = (eagle3s[ext_pos]
                                 if ext_pos < len(eagle3s) else [])
                        if eagle3_max_depth is not None:
                            ext_e = ext_e[:eagle3_max_depth]
                        ext_acc = prefix_match(
                            ext_e,
                            tokens[ext_pos + 1:] if ext_pos < N else [])
                    acc = s_acc + ext_acc
                    chosen_step_time = SUFFIX_STEP_S + E3_DRAFT_ONLY
                else:
                    ext_acc = 0
                    if e_exhausted:
                        ext_pos = pos + e_acc + 1
                        ext_acc = _compute_suffix_extension_acc(
                            ext_pos, tokens, N, suffix_cache, req_id,
                            prompt, decoded, pos,
                            max_spec_tokens, max_spec_factor,
                            min_token_prob)
                    acc = e_acc + ext_acc
                    chosen_step_time = (SUFFIX_STEP_S + E3_DRAFT_ONLY
                                        + S_DRAFT_ONLY)
            elif method == "hybrid_ext_dms":
                # DM+S: suffix decides primary; exhaust → extend with other
                suffix_chosen = (draft.score >= threshold
                                 and bool(draft.token_ids))
                S_DRAFT_ONLY = SUFFIX_STEP_S - VANILLA_STEP_S
                DM_DRAFT_ONLY = eff_dm_step - VANILLA_STEP_S
                if suffix_chosen:
                    ext_acc = 0
                    if s_exhausted and s_acc > 0:
                        ext_pos = pos + s_acc + 1
                        ext_dm_d2 = (
                            dm_drafts[global_pos + ext_pos]
                            if dm_drafts
                            and (global_pos + ext_pos) < len(dm_drafts)
                            else [])
                        if dm_max_depth is not None:
                            ext_dm_d2 = ext_dm_d2[:dm_max_depth]
                        ext_acc = prefix_match(
                            ext_dm_d2,
                            tokens[ext_pos + 1:] if ext_pos < N else [])
                    acc = s_acc + ext_acc
                    chosen_step_time = SUFFIX_STEP_S + DM_DRAFT_ONLY
                else:
                    ext_acc = 0
                    if dm_exhausted:
                        ext_pos = pos + dm_acc + 1
                        ext_acc = _compute_suffix_extension_acc(
                            ext_pos, tokens, N, suffix_cache, req_id,
                            prompt, decoded, pos,
                            max_spec_tokens, max_spec_factor,
                            min_token_prob)
                    acc = dm_acc + ext_acc
                    chosen_step_time = (SUFFIX_STEP_S + DM_DRAFT_ONLY
                                        + S_DRAFT_ONLY)
            elif method == "hybrid_ext_e3dm":
                # E3+DM: suffix chosen → S standalone; else → E3⊕DM
                suffix_chosen = (draft.score >= threshold
                                 and bool(draft.token_ids))
                DM_DRAFT_ONLY = eff_dm_step - VANILLA_STEP_S
                if suffix_chosen:
                    acc = s_acc
                    chosen_step_time = SUFFIX_STEP_S
                else:
                    ext_acc = 0
                    if e_exhausted:
                        ext_pos = pos + e_acc + 1
                        ext_dm_d2 = (
                            dm_drafts[global_pos + ext_pos]
                            if dm_drafts
                            and (global_pos + ext_pos) < len(dm_drafts)
                            else [])
                        if dm_max_depth is not None:
                            ext_dm_d2 = ext_dm_d2[:dm_max_depth]
                        ext_acc = prefix_match(
                            ext_dm_d2,
                            tokens[ext_pos + 1:] if ext_pos < N else [])
                    acc = e_acc + ext_acc
                    chosen_step_time = (SUFFIX_STEP_S
                                        + (eff_eagle3_step - VANILLA_STEP_S)
                                        + DM_DRAFT_ONLY)
            elif method == "hybrid_ext_sdm":
                # S+DM: same logic as hybrid_ext_dms
                suffix_chosen = (draft.score >= threshold
                                 and bool(draft.token_ids))
                S_DRAFT_ONLY = SUFFIX_STEP_S - VANILLA_STEP_S
                DM_DRAFT_ONLY = eff_dm_step - VANILLA_STEP_S
                if suffix_chosen:
                    ext_acc = 0
                    if s_exhausted and s_acc > 0:
                        ext_pos = pos + s_acc + 1
                        ext_dm_d2 = (
                            dm_drafts[global_pos + ext_pos]
                            if dm_drafts
                            and (global_pos + ext_pos) < len(dm_drafts)
                            else [])
                        if dm_max_depth is not None:
                            ext_dm_d2 = ext_dm_d2[:dm_max_depth]
                        ext_acc = prefix_match(
                            ext_dm_d2,
                            tokens[ext_pos + 1:] if ext_pos < N else [])
                    acc = s_acc + ext_acc
                    chosen_step_time = SUFFIX_STEP_S + DM_DRAFT_ONLY
                else:
                    ext_acc = 0
                    if dm_exhausted:
                        ext_pos = pos + dm_acc + 1
                        ext_acc = _compute_suffix_extension_acc(
                            ext_pos, tokens, N, suffix_cache, req_id,
                            prompt, decoded, pos,
                            max_spec_tokens, max_spec_factor,
                            min_token_prob)
                    acc = dm_acc + ext_acc
                    chosen_step_time = (SUFFIX_STEP_S + DM_DRAFT_ONLY
                                        + S_DRAFT_ONLY)
            # --- Tree-based Extension ---
            elif method == "eagle3_tree_dm":
                e_p, e_t = _flat_to_tree(e_draft)
                def _get_dm(off, _gp=global_pos, _p=pos):
                    t = _gp + _p + off
                    d = (dm_drafts[t]
                         if dm_drafts and t < len(dm_drafts) else [])
                    if dm_max_depth is not None:
                        d = d[:dm_max_depth]
                    return _flat_to_tree(d)
                combined = build_combined_tree(
                    e_p, e_t, _get_dm, len(e_draft))
                acc, _ = count_accepted_tree(combined, future)
                chosen_step_time = (eff_eagle3_step
                                    + (eff_dm_step - VANILLA_STEP_S))
            elif method == "dm_tree_eagle3":
                dm_p, dm_t = _flat_to_tree(dm_d)
                def _get_e3(off, _p=pos):
                    t = _p + off
                    d = eagle3s[t] if t < len(eagle3s) else []
                    if eagle3_max_depth is not None:
                        d = d[:eagle3_max_depth]
                    return _flat_to_tree(d)
                combined = build_combined_tree(
                    dm_p, dm_t, _get_e3, len(dm_d))
                acc, _ = count_accepted_tree(combined, future)
                chosen_step_time = (eff_dm_step
                                    + (eff_eagle3_step - VANILLA_STEP_S))
            elif method == "eagle3_tree_suffix":
                e_p, e_t = _flat_to_tree(e_draft)
                def _get_sfx(off, _p=pos, _dec=list(decoded),
                             _tok=tokens, _N=N, _pr=prompt):
                    tp = _p + off
                    if tp >= _N:
                        return [], []
                    ext_resp = _dec + list(_tok[_p:tp + 1])
                    if len(_pr) > 0:
                        ctx = np.concatenate(
                            [_pr, np.array(ext_resp, dtype=np.int32)])
                    else:
                        ctx = np.array(ext_resp, dtype=np.int32)
                    sd = suffix_cache.speculate(
                        req_id, ctx,
                        max_spec_tokens=max_spec_tokens,
                        max_spec_factor=max_spec_factor,
                        min_token_prob=min_token_prob)
                    return list(sd.parents), list(sd.token_ids)
                combined = build_combined_tree(
                    e_p, e_t, _get_sfx, len(e_draft))
                acc, _ = count_accepted_tree(combined, future)
                chosen_step_time = (eff_eagle3_step
                                    + (SUFFIX_STEP_S - VANILLA_STEP_S))
            elif method == "dm_tree_suffix":
                dm_p, dm_t = _flat_to_tree(dm_d)
                def _get_sfx2(off, _p=pos, _dec=list(decoded),
                              _tok=tokens, _N=N, _pr=prompt):
                    tp = _p + off
                    if tp >= _N:
                        return [], []
                    ext_resp = _dec + list(_tok[_p:tp + 1])
                    if len(_pr) > 0:
                        ctx = np.concatenate(
                            [_pr, np.array(ext_resp, dtype=np.int32)])
                    else:
                        ctx = np.array(ext_resp, dtype=np.int32)
                    sd = suffix_cache.speculate(
                        req_id, ctx,
                        max_spec_tokens=max_spec_tokens,
                        max_spec_factor=max_spec_factor,
                        min_token_prob=min_token_prob)
                    return list(sd.parents), list(sd.token_ids)
                combined = build_combined_tree(
                    dm_p, dm_t, _get_sfx2, len(dm_d))
                acc, _ = count_accepted_tree(combined, future)
                chosen_step_time = (eff_dm_step
                                    + (SUFFIX_STEP_S - VANILLA_STEP_S))
            elif method == "suffix_tree_dm":
                def _get_dm2(off, _gp=global_pos, _p=pos):
                    t = _gp + _p + off
                    d = (dm_drafts[t]
                         if dm_drafts and t < len(dm_drafts) else [])
                    if dm_max_depth is not None:
                        d = d[:dm_max_depth]
                    return _flat_to_tree(d)
                combined = build_combined_tree(
                    list(draft.parents), list(draft.token_ids),
                    _get_dm2, max_spec_tokens)
                acc, _ = count_accepted_tree(combined, future)
                chosen_step_time = (SUFFIX_STEP_S
                                    + (eff_dm_step - VANILLA_STEP_S))
            elif method == "suffix_tree_eagle3":
                def _get_e3b(off, _p=pos):
                    t = _p + off
                    d = eagle3s[t] if t < len(eagle3s) else []
                    if eagle3_max_depth is not None:
                        d = d[:eagle3_max_depth]
                    return _flat_to_tree(d)
                combined = build_combined_tree(
                    list(draft.parents), list(draft.token_ids),
                    _get_e3b, max_spec_tokens)
                acc, _ = count_accepted_tree(combined, future)
                chosen_step_time = (SUFFIX_STEP_S
                                    + (eff_eagle3_step - VANILLA_STEP_S))
            elif method == "vanilla":
                acc = 0
                chosen_step_time = VANILLA_STEP_S
            else:
                acc = e_acc
                chosen_step_time = eff_eagle3_step

            per_step_e_acc.append(e_acc)
            per_step_s_acc.append(s_acc)
            per_step_dm_acc.append(dm_acc)
            per_step_acc.append(acc)
            per_step_e_draft_len.append(len(e_draft))
            per_step_s_exhausted.append(s_exhausted)
            per_step_dm_draft_len.append(len(dm_d))

            # Winner tracking
            if method in ("hybrid", "hybrid_dm",
                          "hybrid_ext_e3s", "hybrid_ext_dms",
                          "hybrid_ext_e3dm", "hybrid_ext_sdm"):
                if suffix_chosen:
                    s_wins += 1
                else:
                    e_wins += 1
            # oracle_mat_se / oracle_mat_sd removed (deprecated)
            else:
                best = (max(e_acc, s_acc, dm_acc) if dm_drafts
                        else max(e_acc, s_acc))
                winners = []
                if e_acc == best:
                    winners.append("e")
                if s_acc == best:
                    winners.append("s")
                if dm_acc == best and dm_drafts:
                    winners.append("d")
                if len(winners) > 1:
                    ties += 1
                elif "e" in winners:
                    e_wins += 1
                elif "s" in winners:
                    s_wins += 1
                elif "d" in winners:
                    d_wins += 1
                else:
                    ties += 1

            # Consume
            end = min(pos + acc + 1, N)
            consumed = tokens[pos:end]
            suffix_cache.add_active_response(req_id, consumed)
            decoded.extend(consumed)
            pos = end
            steps += 1
            total_step_time += chosen_step_time

        suffix_cache.stop_request(req_id)
        global_pos += N
        req_id += 1

    return {
        "steps": steps,
        "e_wins": e_wins,
        "s_wins": s_wins,
        "d_wins": d_wins,
        "ties": ties,
        "total_step_time": total_step_time,
        "per_step_e_acc": per_step_e_acc,
        "per_step_s_acc": per_step_s_acc,
        "per_step_dm_acc": per_step_dm_acc,
        "per_step_acc": per_step_acc,
        "per_step_e_draft_len": per_step_e_draft_len,
        "per_step_s_exhausted": per_step_s_exhausted,
        "per_step_dm_draft_len": per_step_dm_draft_len,
    }


def dp_oracle_from_vanilla(vanilla_result, has_dm, exclude_eagle3=False,
                           include_extensions=True, e3_k=None, dm_k=None):
    """Compute global-optimal oracle via backward DP on vanilla walk data.

    Vanilla walk (acc=0 every step) visits every position, so:
      per_step_e_acc[i]  = eagle3 acceptance at position i
      per_step_s_acc[i]  = suffix acceptance at position i
      per_step_dm_acc[i] = draft model acceptance at position i

    DP: dp[pos] = min total latency from position pos to end.
    At each position, choose the method that minimizes
      step_time(m) + dp[pos + acc_m + 1].

    When include_extensions=False, only base methods (e, s, d) are considered.
    When e3_k/dm_k are set, truncate acceptances and interpolate step costs.

    Returns dict compatible with simulate_request output.
    """
    raw_e_accs = vanilla_result["per_step_e_acc"]
    s_accs = vanilla_result["per_step_s_acc"]
    raw_dm_accs = vanilla_result.get("per_step_dm_acc", [])
    raw_e_draft_lens = vanilla_result.get("per_step_e_draft_len", [])
    s_exhausteds = vanilla_result.get("per_step_s_exhausted", [])
    raw_dm_draft_lens = vanilla_result.get("per_step_dm_draft_len", [])
    N = len(raw_e_accs)
    if N == 0:
        return {"steps": 0, "e_wins": 0, "s_wins": 0, "d_wins": 0,
                "ties": 0, "total_step_time": 0.0,
                "per_step_e_acc": [], "per_step_s_acc": [],
                "per_step_dm_acc": [], "per_step_acc": []}

    # Truncate acceptances and draft lens for reduced k
    if e3_k is not None:
        e_accs = [min(a, e3_k) for a in raw_e_accs]
        e_draft_lens = [min(d, e3_k) for d in raw_e_draft_lens] if raw_e_draft_lens else []
    else:
        e_accs = raw_e_accs
        e_draft_lens = raw_e_draft_lens

    if dm_k is not None:
        dm_accs = [min(a, dm_k) for a in raw_dm_accs] if raw_dm_accs else []
        dm_draft_lens = [min(d, dm_k) for d in raw_dm_draft_lens] if raw_dm_draft_lens else []
    else:
        dm_accs = raw_dm_accs
        dm_draft_lens = raw_dm_draft_lens

    # Interpolated step costs
    eff_e3 = (_interpolate_step_latency(EAGLE3_STEP_S, e3_k, EAGLE3_FULL_DEPTH)
              if e3_k is not None else EAGLE3_STEP_S)
    eff_dm = (_interpolate_step_latency(DRAFT_MODEL_STEP_S, dm_k, DM_FULL_DEPTH)
              if dm_k is not None else DRAFT_MODEL_STEP_S)

    SUFFIX_DRAFT_ONLY = SUFFIX_STEP_S - VANILLA_STEP_S
    DM_DRAFT_ONLY = eff_dm - VANILLA_STEP_S
    E3_DRAFT_ONLY = eff_e3 - VANILLA_STEP_S

    # Backward DP
    INF = float("inf")
    dp = [INF] * (N + 1)
    dp[N] = 0.0
    choice = [None] * N  # (method_name, acc) at each position

    for pos in range(N - 1, -1, -1):
        candidates = [(s_accs[pos], SUFFIX_STEP_S, "s")]
        if not exclude_eagle3:
            candidates.append((e_accs[pos], eff_e3, "e"))
        if has_dm and dm_accs:
            candidates.append((dm_accs[pos], eff_dm, "d"))

        # Extension candidates (only when include_extensions=True)
        if include_extensions:
            # Extension by suffix (exact: suffix is CPU-only, no hidden state)
            if e_draft_lens and not exclude_eagle3:
                if (e_accs[pos] == e_draft_lens[pos]
                        and e_draft_lens[pos] > 0):
                    ext_pos = min(pos + e_accs[pos] + 1, N)
                    ext_s = s_accs[ext_pos] if ext_pos < N else 0
                    candidates.append(
                        (e_accs[pos] + ext_s,
                         eff_e3 + SUFFIX_DRAFT_ONLY, "e+s"))
            if dm_draft_lens and has_dm and dm_accs:
                if (dm_accs[pos] == dm_draft_lens[pos]
                        and dm_draft_lens[pos] > 0):
                    ext_pos = min(pos + dm_accs[pos] + 1, N)
                    ext_s = s_accs[ext_pos] if ext_pos < N else 0
                    candidates.append(
                        (dm_accs[pos] + ext_s,
                         eff_dm + SUFFIX_DRAFT_ONLY, "d+s"))

            # Extension by DM (all-accepted condition)
            if has_dm and dm_accs:
                # eagle3 → dm extension
                if (e_draft_lens and not exclude_eagle3
                        and e_accs[pos] == e_draft_lens[pos]
                        and e_draft_lens[pos] > 0):
                    ext_pos = min(pos + e_accs[pos] + 1, N)
                    ext_d = dm_accs[ext_pos] if ext_pos < N else 0
                    candidates.append(
                        (e_accs[pos] + ext_d,
                         eff_e3 + DM_DRAFT_ONLY, "e+d"))
                # suffix → dm extension
                if (s_exhausteds
                        and s_exhausteds[pos] and s_accs[pos] > 0):
                    ext_pos = min(pos + s_accs[pos] + 1, N)
                    ext_d = dm_accs[ext_pos] if ext_pos < N else 0
                    candidates.append(
                        (s_accs[pos] + ext_d,
                         SUFFIX_STEP_S + DM_DRAFT_ONLY, "s+d"))

        for acc, step_t, name in candidates:
            nxt = min(pos + acc + 1, N)
            cost = step_t + dp[nxt]
            if cost < dp[pos]:
                dp[pos] = cost
                choice[pos] = (name, acc)

    # Step time lookup (base + extension methods)
    EXT_STEP_TIMES = {
        "e": eff_e3, "s": SUFFIX_STEP_S, "d": eff_dm,
        "e+s": eff_e3 + SUFFIX_DRAFT_ONLY,
        "d+s": eff_dm + SUFFIX_DRAFT_ONLY,
        "e+d": eff_e3 + DM_DRAFT_ONLY,
        "s+d": SUFFIX_STEP_S + DM_DRAFT_ONLY,
    }

    # Forward pass: walk along optimal path
    steps = 0
    e_wins = s_wins = d_wins = 0
    total_step_time = 0.0
    per_step_e = []
    per_step_s = []
    per_step_dm = []
    per_step_a = []
    pos = 0
    while pos < N:
        name, acc = choice[pos]
        per_step_e.append(e_accs[pos])
        per_step_s.append(s_accs[pos])
        per_step_dm.append(dm_accs[pos] if dm_accs else 0)
        per_step_a.append(acc)

        total_step_time += EXT_STEP_TIMES.get(name, SUFFIX_STEP_S)
        if "e" in name:
            e_wins += 1
        if "s" in name:
            s_wins += 1
        if "d" in name:
            d_wins += 1
        steps += 1
        pos = min(pos + acc + 1, N)

    return {
        "steps": steps,
        "e_wins": e_wins,
        "s_wins": s_wins,
        "d_wins": d_wins,
        "ties": 0,
        "total_step_time": total_step_time,
        "per_step_e_acc": per_step_e,
        "per_step_s_acc": per_step_s,
        "per_step_dm_acc": per_step_dm,
        "per_step_acc": per_step_a,
    }


def dp_oracle_best_k(vanilla_result, has_dm, eagle3_depths, dm_depths,
                     include_extensions=True):
    """Find the (e3_k, dm_k) that minimizes total latency via DP.

    Runs dp_oracle_from_vanilla for each k combination and returns the best.
    Also returns the chosen (e3_k, dm_k) in the result dict.
    """
    best_result = None
    best_time = float("inf")
    best_ks = (None, None)

    for ek in eagle3_depths:
        dk_list = dm_depths if has_dm else [None]
        for dk in dk_list:
            r = dp_oracle_from_vanilla(
                vanilla_result, has_dm,
                include_extensions=include_extensions,
                e3_k=ek, dm_k=dk)
            t = r.get("total_step_time", float("inf"))
            if t < best_time:
                best_time = t
                best_result = r
                best_ks = (ek, dk)

    if best_result is not None:
        best_result["best_e3_k"] = best_ks[0]
        best_result["best_dm_k"] = best_ks[1]
    return best_result


def _compute_tpot(spec, result, N, has_dm):
    """Compute TPOT (ms) for a single MethodSpec + result."""
    st = result["steps"]
    ew, sw = result["e_wins"], result["s_wins"]
    dw, ti = result.get("d_wins", 0), result["ties"]

    eff_e3 = (_interpolate_step_latency(EAGLE3_STEP_S, spec.e3_k, EAGLE3_FULL_DEPTH)
              if spec.e3_k is not None else EAGLE3_STEP_S)
    eff_dm = (_interpolate_step_latency(DRAFT_MODEL_STEP_S, spec.dm_k, DM_FULL_DEPTH)
              if spec.dm_k is not None else DRAFT_MODEL_STEP_S)

    E3_DRAFT = eff_e3 - VANILLA_STEP_S
    DM_DRAFT = eff_dm - VANILLA_STEP_S
    S_DRAFT = SUFFIX_STEP_S - VANILLA_STEP_S

    # Uniform-cost methods: every step pays the same latency
    UNIFORM_COSTS = {
        "eagle3":            eff_e3,
        "suffix":            SUFFIX_STEP_S,
        "draft_model":       eff_dm,
        "eagle3_ext_suffix": eff_e3 + S_DRAFT,
        "dm_ext_suffix":     eff_dm + S_DRAFT,
        "eagle3_ext_dm":     eff_e3 + DM_DRAFT,
        "suffix_ext_dm":     SUFFIX_STEP_S + DM_DRAFT,
        # Tree-based extension (same cost model as linear extension)
        "eagle3_tree_dm":      eff_e3 + DM_DRAFT,
        "dm_tree_eagle3":      eff_dm + E3_DRAFT,
        "eagle3_tree_suffix":  eff_e3 + S_DRAFT,
        "dm_tree_suffix":      eff_dm + S_DRAFT,
        "suffix_tree_dm":      SUFFIX_STEP_S + DM_DRAFT,
        "suffix_tree_eagle3":  SUFFIX_STEP_S + E3_DRAFT,
    }

    m = spec.method
    if m in UNIFORM_COSTS:
        return st * UNIFORM_COSTS[m] / max(N, 1) * 1000

    # Hybrid+Extension: use tracked total_step_time
    if m.startswith("hybrid_ext_"):
        return result.get("total_step_time", 0) / max(N, 1) * 1000

    if m == "hybrid":
        fallback = SUFFIX_STEP_S + E3_DRAFT
        return (sw * SUFFIX_STEP_S + ew * fallback) / max(N, 1) * 1000

    if m == "hybrid_dm":
        fallback = SUFFIX_STEP_S + DM_DRAFT
        return (sw * SUFFIX_STEP_S + ew * fallback) / max(N, 1) * 1000

    # All greedy oracles use total_step_time tracked during simulation
    if m.startswith("oracle_"):
        return result.get("total_step_time", 0) / max(N, 1) * 1000

    # Fallback
    return st * eff_e3 / max(N, 1) * 1000


def _make_label(spec, has_dm):
    """Generate a human-readable label from a MethodSpec."""
    METHOD_LABELS = {
        "eagle3": "Eagle3",
        "suffix": "Suffix",
        "draft_model": "DM",
        "hybrid": "Hybrid(S-E)",
        "hybrid_dm": "Hybrid(S-DM)",
        "eagle3_ext_suffix": "E3⊕Suffix",
        "dm_ext_suffix": "DM⊕Suffix",
        "eagle3_ext_dm": "E3⊕DM",
        "suffix_ext_dm": "Suffix⊕DM",
        "oracle_latency_base": "Oracle(base)",
        "oracle_latency_ext": "Oracle(ext)",
        "oracle_latency_ext_tree": "Oracle(ext+tree)",
        # Hybrid + Extension
        "hybrid_ext_e3s":      "HybridExt(E3+S)",
        "hybrid_ext_dms":      "HybridExt(DM+S)",
        "hybrid_ext_e3dm":     "HybridExt(E3+DM|S)",
        "hybrid_ext_sdm":      "HybridExt(S+DM)",
        # Tree-based Extension
        "eagle3_tree_dm":      "E3⊗DM",
        "dm_tree_eagle3":      "DM⊗E3",
        "eagle3_tree_suffix":  "E3⊗Suffix",
        "dm_tree_suffix":      "DM⊗Suffix",
        "suffix_tree_dm":      "Suffix⊗DM",
        "suffix_tree_eagle3":  "Suffix⊗E3",
    }
    base = METHOD_LABELS.get(spec.method, spec.method)
    parts = [base]
    if spec.e3_k is not None:
        parts.append(f"e3k={spec.e3_k}")
    if spec.dm_k is not None:
        parts.append(f"dmk={spec.dm_k}")
    if spec.threshold > 0:
        parts.append(f"t={spec.threshold}")
    return " ".join(parts)


def print_summary(method_results, spec_map, N, has_dm, n_train=0,
                  n_with_prompt=0, n_requests=0):
    """Print text summary to stdout."""
    print("=" * 60)
    print(f"Vanilla-Trajectory Oracle Analysis")
    print("  Suffix: offline SuffixDecodingCache simulation per method")
    if n_with_prompt > 0:
        print(f"  Prompt tokens: {n_with_prompt}/{n_requests} requests "
              f"(suffix local tree)")
    else:
        print("  Prompt tokens: NONE (suffix local tree empty)")
    if n_train > 0:
        print(f"  Cache warmup: {n_train} train requests")
    print("=" * 60)

    print(f"\nTotal tokens: {N:,}")
    print()
    print("--- All Methods on Same Vanilla Trajectory ---")
    print(f"  {'Method':<30s}  {'Steps':>8s}  {'MAT':>6s}  {'TPOT':>10s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*6}  {'-'*10}")
    print(f"  {'Vanilla (no spec)':<30s}  {N:>8,}  {'0.000':>6s}  {'—':>10s}")

    methods = list(method_results.keys())
    for method_name in methods:
        spec = spec_map[method_name]
        r = method_results[method_name]
        st = r["steps"]
        mat = (N - st) / max(st, 1)
        tpot = _compute_tpot(spec, r, N, has_dm)
        label = _make_label(spec, has_dm)
        print(f"  {label:<30s}  {st:>8,}  {mat:>6.3f}  {tpot:>8.2f}ms")

    # Oracle improvement (use best latency oracle as the reference)
    o_key = None
    for k in ("oracle_latency_ext_tree", "oracle_latency_ext",
              "oracle_latency_base"):
        if k in method_results:
            o_key = k
            break
    if o_key is None or N == 0:
        return

    o_r = method_results[o_key]
    o_mat = (N - o_r["steps"]) / max(o_r["steps"], 1)

    print()
    print("--- Oracle Improvement ---")

    # Compare vs max-k standalone methods
    for method_type, ref_label in [("eagle3", "Eagle3"), ("suffix", "Suffix"),
                                   ("draft_model", "Draft Model")]:
        # Find the max-k variant for this method type
        ref_names = [n for n, s in spec_map.items()
                     if s.method == method_type and s.threshold == 0.0]
        if not ref_names:
            continue
        ref_name = max(ref_names, key=lambda n: (spec_map[n].e3_k or 0) + (spec_map[n].dm_k or 0))
        ref_st = method_results[ref_name]["steps"]
        ref_mat = (N - ref_st) / max(ref_st, 1)
        print(f"  vs {ref_label:<14s} MAT +{((o_mat - ref_mat) / max(ref_mat, 1e-9)) * 100:.1f}%")

    # Best hybrid (S-E)
    hybrid_se = [n for n, s in spec_map.items() if s.method == "hybrid"]
    if hybrid_se:
        best_h = min(hybrid_se, key=lambda n: method_results[n]["steps"])
        best_h_st = method_results[best_h]["steps"]
        best_h_mat = (N - best_h_st) / max(best_h_st, 1)
        best_spec = spec_map[best_h]
        print(f"  vs Best Hybrid(S-E):  MAT +{((o_mat - best_h_mat) / max(best_h_mat, 1e-9)) * 100:.1f}%"
              f"  (best=e3k={best_spec.e3_k},t={best_spec.threshold}, MAT={best_h_mat:.3f})")

    # Best hybrid (S-DM)
    hybrid_sd = [n for n, s in spec_map.items() if s.method == "hybrid_dm"]
    if has_dm and hybrid_sd:
        best_hd = min(hybrid_sd, key=lambda n: method_results[n]["steps"])
        best_hd_st = method_results[best_hd]["steps"]
        best_hd_mat = (N - best_hd_st) / max(best_hd_st, 1)
        best_spec = spec_map[best_hd]
        print(f"  vs Best Hybrid(S-DM): MAT +{((o_mat - best_hd_mat) / max(best_hd_mat, 1e-9)) * 100:.1f}%"
              f"  (best=dmk={best_spec.dm_k},t={best_spec.threshold}, MAT={best_hd_mat:.3f})")

    print()
    print(f"--- Oracle Method Selection ({o_key}) ---")
    ew, sw = o_r["e_wins"], o_r["s_wins"]
    dw, ti = o_r.get("d_wins", 0), o_r["ties"]
    o_tpot = o_r.get("total_step_time", 0) / max(N, 1) * 1000
    print(f"  MAT={o_mat:.3f}  TPOT={o_tpot:.2f}ms  steps={o_r['steps']:,}")
    print(f"  Eagle3 better:  {ew:>8,} ({ew / max(o_r['steps'], 1):.1%})")
    print(f"  Suffix better:  {sw:>8,} ({sw / max(o_r['steps'], 1):.1%})")
    if dw > 0:
        print(f"  DM better:      {dw:>8,} ({dw / max(o_r['steps'], 1):.1%})")
    print(f"  Tie:            {ti:>8,} ({ti / max(o_r['steps'], 1):.1%})")

    # Compare all oracle variants
    for oname in ["oracle_latency_base", "oracle_latency_ext",
                   "oracle_latency_ext_tree"]:
        if oname in method_results and oname != o_key:
            or2 = method_results[oname]
            or2_mat = (N - or2["steps"]) / max(or2["steps"], 1)
            or2_tpot = or2.get("total_step_time", 0) / max(N, 1) * 1000
            print(f"  {oname}: MAT={or2_mat:.3f} TPOT={or2_tpot:.2f}ms")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-results", required=True)
    parser.add_argument("--output", default=None,
                        help="Output JSON path for simulation results")
    parser.add_argument("--exclude", default=None,
                        help="Path to exclude_repetitive.txt")
    parser.add_argument("--thresholds", default="0.5,1.0,2.0,3.0,5.0,10.0",
                        help="Comma-separated hybrid thresholds")
    parser.add_argument("--draft-model-drafts", default=None,
                        help="Path to draft_model_drafts.json")
    parser.add_argument("--print-summary", action="store_true",
                        help="Print text summary to stdout")
    parser.add_argument("--train-ratio", type=float, default=0.0,
                        help="Fraction of requests for cache warmup (stratified by category, default: 0)")
    parser.add_argument("--model", default=None,
                        help="Target model name for tokenizer (enables prompt reconstruction for suffix cache)")
    parser.add_argument("--responses", default=None,
                        help="Path to agent_results_responses.json (BFCL prompt reconstruction)")
    parser.add_argument("--dataset", default=None,
                        help="Path to dataset.jsonl (BFCL/SpecBench prompt reconstruction)")
    parser.add_argument("--step-latencies", default=None,
                        help="Step latencies: eagle3,suffix,vanilla,draft_model (seconds). "
                             "Example: 0.013311,0.010279,0.010023,0.043227")
    parser.add_argument("--eagle3-depths", default=None,
                        help="Comma-separated eagle3 draft depths for extension sweep. "
                             "Example: 1,2,3,4,5")
    parser.add_argument("--dm-depths", default=None,
                        help="Comma-separated draft model depths for extension sweep. "
                             "Example: 1,2,3,4")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated method names to run (default: all). "
                             "Example: oracle_mat_ext,oracle_latency_base")
    parser.add_argument("--merge", default=None,
                        help="Path to existing output JSON to merge new results into. "
                             "New methods are added; existing methods are preserved.")
    args = parser.parse_args()

    if not args.output and not args.print_summary:
        parser.error("At least one of --output or --print-summary is required")

    # Override step latencies if provided
    global EAGLE3_STEP_S, SUFFIX_STEP_S, VANILLA_STEP_S, DRAFT_MODEL_STEP_S
    global EAGLE3_FULL_DEPTH, DM_FULL_DEPTH
    if args.step_latencies:
        parts = [float(x) for x in args.step_latencies.split(",")]
        EAGLE3_STEP_S, SUFFIX_STEP_S, VANILLA_STEP_S = parts[0], parts[1], parts[2]
        DRAFT_MODEL_STEP_S = parts[3] if len(parts) > 3 else DRAFT_MODEL_STEP_S
        print(f"Step latencies: E={EAGLE3_STEP_S*1000:.2f}ms S={SUFFIX_STEP_S*1000:.2f}ms "
              f"V={VANILLA_STEP_S*1000:.2f}ms D={DRAFT_MODEL_STEP_S*1000:.2f}ms",
              file=sys.stderr)

    from arctic_inference.suffix_decoding import SuffixDecodingCache

    exclude_ids = load_exclude_ids(args.exclude) if args.exclude else set()
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"Loading: {args.agent_results}", file=sys.stderr)
    with open(args.agent_results) as f:
        data = json.load(f)

    # Load draft model drafts
    dm_by_id = {}
    has_dm = False
    if args.draft_model_drafts:
        with open(args.draft_model_drafts) as f:
            dm_data = json.load(f)
        dm_by_id = {r["bfcl_id"]: r["drafts"] for r in dm_data["requests"]}
        has_dm = True
        print(f"Draft model drafts: {len(dm_by_id)} requests", file=sys.stderr)

    # Load tokenizer for prompt reconstruction (suffix cache local tree)
    tokenizer = None
    if args.model:
        from transformers import AutoTokenizer
        model_name = args.model
        print(f"Loading tokenizer: {model_name}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load BFCL dataset + responses for prompt reconstruction
    bfcl_dataset = None
    resp_by_id = None
    if args.dataset and args.responses:
        # BFCL mode: need both dataset and responses
        try:
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(PROJECT_ROOT))
            sys.path.insert(0, str(PROJECT_ROOT / "bench"))
            from bench.bfcl_agent import preprocess_bfcl_requests
            entries = []
            with open(args.dataset) as f:
                for line in f:
                    entries.append(json.loads(line))
            preprocess_bfcl_requests(entries)
            bfcl_dataset = {e["bfcl_id"]: e for e in entries}
            with open(args.responses) as f:
                resp_data = json.load(f)
            resp_by_id = {r["bfcl_id"]: r for r in resp_data}
            print(f"BFCL dataset: {len(bfcl_dataset)}, "
                  f"responses: {len(resp_by_id)}", file=sys.stderr)
        except Exception as e:
            print(f"WARN: BFCL prompt reconstruction failed: {e}",
                  file=sys.stderr)

    # Load SpecBench dataset for prompt reconstruction
    specbench_dataset = None
    if args.dataset and not args.responses:
        # SpecBench mode: dataset only (no responses file)
        try:
            specbench_dataset = {}
            with open(args.dataset) as f:
                for line in f:
                    entry = json.loads(line)
                    specbench_dataset[entry["question_id"]] = entry
            print(f"SpecBench dataset: {len(specbench_dataset)}",
                  file=sys.stderr)
        except Exception as e:
            print(f"WARN: SpecBench dataset load failed: {e}",
                  file=sys.stderr)

    n_prompts_msg = ""
    if tokenizer:
        n_prompts_msg = " (with prompt reconstruction)"

    all_requests = extract_requests(data, exclude_ids, dm_by_id,
                                    tokenizer, bfcl_dataset, resp_by_id,
                                    specbench_dataset)
    n_excluded = len(data["questions"]) - len(all_requests)

    if tokenizer:
        n_with_prompts = sum(1 for r in all_requests
                             if r.get("per_call_prompt_ids"))
        print(f"Prompt reconstruction: {n_with_prompts}/{len(all_requests)} "
              f"requests", file=sys.stderr)

    # Stratified train/test split by category
    train_ratio = args.train_ratio
    if train_ratio > 0:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for req in all_requests:
            by_cat[req["category"]].append(req)
        train_requests, test_requests = [], []
        for cat in sorted(by_cat):
            reqs = by_cat[cat]
            n_train = int(len(reqs) * train_ratio)
            train_requests.extend(reqs[:n_train])
            test_requests.extend(reqs[n_train:])
        requests = test_requests
        print(f"Train/test split: {len(train_requests)} train (warmup), "
              f"{len(test_requests)} test", file=sys.stderr)
    else:
        train_requests = []
        requests = all_requests

    N = sum(r["n_tokens"] for r in requests)
    print(f"Requests: {len(requests)} test (excluded {n_excluded}), "
          f"Tokens: {N:,}", file=sys.stderr)

    # Parse depth sweep parameters
    eagle3_depths = ([int(x) for x in args.eagle3_depths.split(",")]
                     if args.eagle3_depths else [])
    dm_depths = ([int(x) for x in args.dm_depths.split(",")]
                 if args.dm_depths else [])
    # Set full depth from max of sweep depths (for latency interpolation)
    if eagle3_depths:
        EAGLE3_FULL_DEPTH = max(eagle3_depths)
    if dm_depths:
        DM_FULL_DEPTH = max(dm_depths)

    # Build all experiment specs (88 combinatorial cases + oracle methods)
    experiment_specs = build_method_specs(eagle3_depths, dm_depths, thresholds, has_dm)

    # Oracle methods (not part of combinatorial sweep)
    # DEPRECATED: oracle_mat_* and oracle_latency_se/sd (see git history)
    oracle_specs = []
    oracle_specs.append(MethodSpec("oracle_latency_base", "oracle_latency_base"))
    oracle_specs.append(MethodSpec("oracle_latency_ext", "oracle_latency_ext"))
    oracle_specs.append(MethodSpec("oracle_latency_ext_tree", "oracle_latency_ext_tree"))

    all_specs = experiment_specs + oracle_specs

    # Filter to specific methods if requested
    if args.methods:
        method_filter = set(args.methods.split(","))
        all_specs = [s for s in all_specs if s.name in method_filter]
        if not all_specs:
            parser.error(f"No matching methods for --methods={args.methods}")

    spec_map = {s.name: s for s in all_specs}
    methods = [s.name for s in all_specs]

    print(f"Methods: {len(methods)}", file=sys.stderr)

    n_with_prompt = sum(1 for r in requests if r.get("per_call_prompt_ids"))
    output = {
        "metadata": {
            "agent_results": args.agent_results,
            "n_requests": len(requests),
            "n_tokens": N,
            "n_excluded": n_excluded,
            "n_train": len(train_requests),
            "train_ratio": train_ratio,
            "thresholds": thresholds,
            "eagle3_depths": eagle3_depths,
            "dm_depths": dm_depths,
            "methods": methods,
            "has_draft_model": has_dm,
            "has_prompt_tokens": n_with_prompt > 0,
            "n_with_prompt": n_with_prompt,
            "model": args.model,
        },
        "methods": {},
    }

    method_results = {}  # for print_summary

    for spec in all_specs:
        t0 = time.time()

        # Warmup: fill global tree with train set (metrics discarded)
        cache = SuffixDecodingCache(
            max_tree_depth=64, max_cached_requests=100000)
        for req in train_requests:
            simulate_request(req, spec.method, cache, threshold=spec.threshold,
                             eagle3_max_depth=spec.e3_k,
                             dm_max_depth=spec.dm_k)

        # Test: measure on warmed-up cache
        total = {"steps": 0, "e_wins": 0, "s_wins": 0, "d_wins": 0, "ties": 0,
                 "total_step_time": 0.0}
        per_req = []
        all_per_step_e_acc = []
        all_per_step_s_acc = []
        all_per_step_acc = []

        for ri, req in enumerate(requests):
            r = simulate_request(req, spec.method, cache,
                                 threshold=spec.threshold,
                                 eagle3_max_depth=spec.e3_k,
                                 dm_max_depth=spec.dm_k)

            # Greedy latency oracles run inline (no post-hoc DP needed)

            for k in total:
                total[k] += r[k]
            per_req.append({
                "bfcl_id": req["bfcl_id"],
                "category": req["category"],
                "n_tokens": req["n_tokens"],
                "steps": r["steps"],
                "e_wins": r["e_wins"],
                "s_wins": r["s_wins"],
                "d_wins": r["d_wins"],
                "ties": r["ties"],
            })
            all_per_step_e_acc.extend(r["per_step_e_acc"])
            all_per_step_s_acc.extend(r["per_step_s_acc"])
            all_per_step_acc.extend(r["per_step_acc"])

        # DEPRECATED: DP-based latency oracle k-sweep (see git history)
        # Greedy oracles now run inline in simulate_request.

        elapsed = time.time() - t0
        mat = (N - total["steps"]) / max(total["steps"], 1)
        warmup_str = f", warmup={len(train_requests)}" if train_requests else ""
        best_k_str = ""
        if total.get("best_e3_k") is not None:
            best_k_str = f", best_k=e3k{total['best_e3_k']}"
            if total.get("best_dm_k") is not None:
                best_k_str += f"_dmk{total['best_dm_k']}"
        print(f"  {spec.name}: steps={total['steps']:,}, MAT={mat:.3f} "
              f"({elapsed:.1f}s{warmup_str}{best_k_str})", file=sys.stderr)

        method_results[spec.name] = total
        method_output = {
            "global": total,
            "per_request": per_req,
            "per_step_e_acc": all_per_step_e_acc,
            "per_step_s_acc": all_per_step_s_acc,
            "per_step_acc": all_per_step_acc,
        }

        # DEPRECATED: vanilla walk data for DP oracle (see git history)

        output["methods"][spec.name] = method_output

    if args.output:
        # Merge into existing file if --merge is specified
        if args.merge:
            print(f"Merging into: {args.merge}", file=sys.stderr)
            with open(args.merge) as f:
                existing = json.load(f)

            # Rename old oracle keys for consistency
            RENAME_MAP = {
                "oracle_mat": "oracle_mat_base",
                "oracle_latency": "oracle_latency_ext",
            }
            for old_name, new_name in RENAME_MAP.items():
                if old_name in existing["methods"] and new_name not in existing["methods"]:
                    existing["methods"][new_name] = existing["methods"].pop(old_name)
                    print(f"  Renamed: {old_name} → {new_name}", file=sys.stderr)
            # Rename in metadata methods list too
            existing_methods = existing["metadata"].get("methods", [])
            existing_methods = [RENAME_MAP.get(m, m) for m in existing_methods]
            existing["metadata"]["methods"] = existing_methods

            # Add new methods, preserve existing ones
            for method_name, method_data in output["methods"].items():
                existing["methods"][method_name] = method_data
            # Update metadata methods list
            existing_methods_set = set(existing["metadata"]["methods"])
            for m in methods:
                existing_methods_set.add(m)
            existing["metadata"]["methods"] = list(existing_methods_set)
            output = existing

        print(f"Writing: {args.output}", file=sys.stderr)
        with open(args.output, "w") as f:
            json.dump(output, f)

    if args.print_summary:
        print_summary(method_results, spec_map, N, has_dm,
                      n_train=len(train_requests),
                      n_with_prompt=n_with_prompt,
                      n_requests=len(requests))

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
