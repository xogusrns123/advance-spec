"""Shared helpers for reading Stage 1 agent_results JSON and reconstructing
per-call token/tree data + prompt ids for the downstream pipeline stages.

Used by:
  * simulation/pipeline/collect_draft_model.py  (Stage 2)
  * simulation/pipeline/collect_union_trie.py   (Stage 3 record assembly)

Not a public API — callers inside ``simulation.pipeline`` only.
"""

from __future__ import annotations

import copy
import json
from typing import List, Optional


def _flat_to_tree(draft_tokens):
    """Convert flat draft list to (parents, token_ids) chain."""
    parents = [i - 1 for i in range(len(draft_tokens))]
    return parents, list(draft_tokens)


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
    """Extract tokens, eagle3 drafts, eagle3 trees, p_t, mtp trees from
    ``oracle_vanilla_entries``."""
    # Filter interleaved entries from concurrent requests (workers>1):
    # keep only entries matching the most common req_id.
    if entries and len(set(e.get("req_id", "") for e in entries)) > 1:
        from collections import Counter
        rid_counts = Counter(e.get("req_id", "") for e in entries)
        primary_rid = rid_counts.most_common(1)[0][0]
        entries = [e for e in entries if e.get("req_id") == primary_rid]

    call_tokens = []
    call_eagle3s = []
    call_eagle3_trees = []
    call_eagle3_tree_p_ts = []
    call_eagle3_tree_path_draft_p_ts = []
    call_mtp_trees = []
    for e in entries:
        toks = (e["tokens"][0]
                if e.get("tokens") and e["tokens"] else [])
        call_tokens.extend(toks)
        call_eagle3s.append(
            e["eagle3"][0]
            if e.get("eagle3") and e["eagle3"] else [])
        call_eagle3_trees.append(e.get("eagle3_tree"))
        call_eagle3_tree_p_ts.append(e.get("eagle3_tree_p_t"))
        call_eagle3_tree_path_draft_p_ts.append(
            e.get("eagle3_tree_path_draft_p_t"))
        call_mtp_trees.append(e.get("mtp_tree"))
    return (call_tokens, call_eagle3s, call_eagle3_trees,
            call_eagle3_tree_p_ts, call_eagle3_tree_path_draft_p_ts,
            call_mtp_trees)


def _langchain_to_openai_messages(messages):
    """Convert LangChain message format (type/content) to OpenAI format
    (role/content)."""
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

    for user_msg, result_turn in zip(user_turns, result_turns):
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
      * agent_metrics present → BFCL/SWE-bench path
      * turns with dicts → SpecBench path
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
        bfcl_id = (q.get("bfcl_id") or q.get("instance_id")
                   or str(q.get("question_id", "")))
        if bfcl_id in exclude_ids:
            continue

        per_call_tokens = []
        per_call_eagle3s = []
        per_call_eagle3_trees = []
        per_call_eagle3_tree_p_ts = []
        per_call_eagle3_tree_path_draft_p_ts = []
        per_call_mtp_trees = []

        for s in q["agent_metrics"]["steps"]:
            entries = s.get("spec_decode", {}).get(
                "oracle_vanilla_entries", [])
            if not entries:
                continue
            ct, ce, c_trees, c_p_ts, c_draft_p_ts, c_mtp_trees = \
                _extract_entries(entries)
            per_call_tokens.append(ct)
            per_call_eagle3s.append(ce)
            per_call_eagle3_trees.append(c_trees)
            per_call_eagle3_tree_p_ts.append(c_p_ts)
            per_call_eagle3_tree_path_draft_p_ts.append(c_draft_p_ts)
            per_call_mtp_trees.append(c_mtp_trees)

        # Reconstruct prompt token IDs for suffix cache local tree
        per_call_prompt_ids = None
        if tokenizer:
            steps = q.get("agent_metrics", {}).get("steps", [])
            steps_with_msgs = [s for s in steps if s.get("messages")]
            if steps_with_msgs:
                per_call_prompt_ids = []
                for s in steps_with_msgs:
                    result = tokenizer.apply_chat_template(
                        s["messages"], add_generation_prompt=True)
                    if isinstance(result, list):
                        prompt_ids = result
                    else:
                        prompt_ids = result["input_ids"]
                    per_call_prompt_ids.append(prompt_ids)
            else:
                turns = q.get("turns", [])
                has_messages = (turns and isinstance(turns[0], dict)
                                and "messages" in turns[0])
                if has_messages:
                    per_call_prompt_ids = _reconstruct_swebench_prompts(
                        q, tokenizer)
                elif bfcl_dataset and resp_by_id and bfcl_id in bfcl_dataset:
                    resp = resp_by_id.get(bfcl_id)
                    if resp:
                        per_call_prompt_ids = _reconstruct_bfcl_prompts(
                            bfcl_dataset[bfcl_id], resp, tokenizer)

        has_trees = any(t is not None for trees in per_call_eagle3_trees
                        for t in trees)
        has_p_ts = any(p is not None for pts in per_call_eagle3_tree_p_ts
                       for p in pts)
        has_draft_p_ts = any(
            p is not None
            for pts in per_call_eagle3_tree_path_draft_p_ts for p in pts)

        req_data = {
            "bfcl_id": bfcl_id,
            "category": q.get("category", ""),
            "per_call_tokens": per_call_tokens,
            "per_call_eagle3s": per_call_eagle3s,
            "per_call_prompt_ids": per_call_prompt_ids,
            "n_tokens": sum(len(ct) for ct in per_call_tokens),
            "draft_model_drafts": dm_by_id.get(bfcl_id) if dm_by_id else None,
        }
        if has_trees:
            req_data["per_call_eagle3_trees"] = per_call_eagle3_trees
        if has_p_ts:
            req_data["per_call_eagle3_tree_p_ts"] = per_call_eagle3_tree_p_ts
        if has_draft_p_ts:
            req_data["per_call_eagle3_tree_path_draft_p_ts"] = \
                per_call_eagle3_tree_path_draft_p_ts
        has_mtp_trees = any(t is not None for trees in per_call_mtp_trees
                            for t in trees)
        if has_mtp_trees:
            req_data["per_call_mtp_trees"] = per_call_mtp_trees
        requests.append(req_data)
    return requests


def _extract_online(data, exclude_ids, dm_by_id=None,
                    tokenizer=None, specbench_dataset=None):
    """Extract from online.py format (SpecBench, etc.)."""
    questions = data.get("questions", [])
    per_request = data.get("per_request", [])
    requests = []

    q_map = {}
    for q in questions:
        qid = str(q.get("question_id", ""))
        q_map[qid] = q.get("category", "")

    if (questions and "turns" in questions[0]
            and isinstance(questions[0]["turns"], list)):
        for qi, q in enumerate(questions):
            qid = str(q.get("question_id", qi))
            if qid in exclude_ids:
                continue

            per_call_tokens = []
            per_call_eagle3s = []
            per_call_eagle3_trees = []
            per_call_eagle3_tree_p_ts = []
            per_call_eagle3_tree_path_draft_p_ts = []
            per_call_mtp_trees = []

            for turn in q["turns"]:
                if isinstance(turn, dict):
                    entries = turn.get("spec_decode", {}).get(
                        "oracle_vanilla_entries", [])
                    if not entries:
                        continue
                    ct, ce, c_trees, c_p_ts, c_draft_p_ts, c_mtp_trees = \
                        _extract_entries(entries)
                    per_call_tokens.append(ct)
                    per_call_eagle3s.append(ce)
                    per_call_eagle3_trees.append(c_trees)
                    per_call_eagle3_tree_p_ts.append(c_p_ts)
                    per_call_eagle3_tree_path_draft_p_ts.append(c_draft_p_ts)
                    per_call_mtp_trees.append(c_mtp_trees)

            if not per_call_tokens:
                continue

            per_call_prompt_ids = None
            if tokenizer and specbench_dataset:
                ds_entry = specbench_dataset.get(q.get("question_id"))
                if ds_entry:
                    per_call_prompt_ids = _reconstruct_specbench_prompts(
                        ds_entry, q, tokenizer)

            has_trees = any(t is not None for trees in per_call_eagle3_trees
                            for t in trees)
            req_data = {
                "bfcl_id": qid,
                "category": q.get("category", ""),
                "per_call_tokens": per_call_tokens,
                "per_call_eagle3s": per_call_eagle3s,
                "per_call_prompt_ids": per_call_prompt_ids,
                "n_tokens": sum(len(ct) for ct in per_call_tokens),
                "draft_model_drafts": dm_by_id.get(qid) if dm_by_id else None,
            }
            if has_trees:
                req_data["per_call_eagle3_trees"] = per_call_eagle3_trees
            has_p_ts = any(p is not None
                           for pts in per_call_eagle3_tree_p_ts for p in pts)
            if has_p_ts:
                req_data["per_call_eagle3_tree_p_ts"] = \
                    per_call_eagle3_tree_p_ts
            has_draft_p_ts = any(
                p is not None
                for pts in per_call_eagle3_tree_path_draft_p_ts for p in pts)
            if has_draft_p_ts:
                req_data["per_call_eagle3_tree_path_draft_p_ts"] = \
                    per_call_eagle3_tree_path_draft_p_ts
            has_mtp_trees = any(t is not None for trees in per_call_mtp_trees
                                for t in trees)
            if has_mtp_trees:
                req_data["per_call_mtp_trees"] = per_call_mtp_trees
            requests.append(req_data)
    else:
        # Fallback: per_request[] (flat, one entry per LLM call)
        for ri, r in enumerate(per_request):
            entries = r.get("spec_decode", {}).get(
                "oracle_vanilla_entries", [])
            if not entries:
                continue

            ct, ce, c_trees, c_p_ts, c_draft_p_ts, c_mtp_trees = \
                _extract_entries(entries)
            qid = str(ri)
            category = q_map.get(qid, "") if ri < len(questions) else ""

            has_trees = any(t is not None for t in c_trees)
            has_p_ts = any(p is not None for p in c_p_ts)
            has_draft_p_ts = any(p is not None for p in c_draft_p_ts)
            req_data = {
                "bfcl_id": qid,
                "category": category,
                "per_call_tokens": [ct],
                "per_call_eagle3s": [ce],
                "per_call_prompt_ids": None,
                "n_tokens": len(ct),
                "draft_model_drafts": dm_by_id.get(qid) if dm_by_id else None,
            }
            if has_trees:
                req_data["per_call_eagle3_trees"] = [c_trees]
            if has_p_ts:
                req_data["per_call_eagle3_tree_p_ts"] = [c_p_ts]
            if has_draft_p_ts:
                req_data["per_call_eagle3_tree_path_draft_p_ts"] = [c_draft_p_ts]
            has_mtp_trees = any(t is not None for t in c_mtp_trees)
            if has_mtp_trees:
                req_data["per_call_mtp_trees"] = [c_mtp_trees]
            requests.append(req_data)

    return requests
