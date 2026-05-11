"""Replay-only DM/draft capture (bypasses agent loop).

For each (question, turn) in a source EAGLE3 capture, submits the captured
input prompt directly to SGLang via /v1/completions (pre-tokenized prompt).
SGLang's ORACLE_REPLAY forces output to the expected token sequence, while
oracle hooks capture the new draft proposer's predictions per step.

This script does NOT execute tools, parse responses, or run agent loops:
in replay mode none of that is needed. The trajectory file is built with
zero-padded submission-order keys (`q000123_t007`) so that
TrajectoryState's sorted-key concat order matches the submission order
exactly — eliminating the global-queue misalignment that broke prior
multi-turn captures.

Output format mirrors agent_results_eagle3.json so downstream
assemble_records_from_artifacts can consume it as a dm-capture source.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
import uuid
from pathlib import Path

import ijson
import openai
from openai import OpenAI


# ---------------------------------------------------------------------------
# langchain → OpenAI message conversion
# ---------------------------------------------------------------------------

_LC_TYPE_MAP = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
}


def _coerce_tool_calls(raw):
    """Tool calls in EAGLE3 capture sometimes appear as a Python repr string
    (e.g., `"[{'name': 'bash', 'args': {...}}]"`). Try literal_eval first.
    Returns a list of {name, args} dicts or None on failure."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            return v if isinstance(v, list) else None
        except (ValueError, SyntaxError):
            return None
    return None


def _lc_to_openai(messages):
    """Convert langchain {type, content, tool_calls?, tool_call_id?} list
    to OpenAI messages format."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        mtype = m.get("type") or m.get("role")
        # Pass through if already in OpenAI form.
        if mtype in ("user", "assistant", "system", "tool"):
            role = mtype
        elif mtype in _LC_TYPE_MAP:
            role = _LC_TYPE_MAP[mtype]
        else:
            continue
        msg = {"role": role, "content": m.get("content") or ""}
        tcs = _coerce_tool_calls(m.get("tool_calls"))
        if tcs:
            openai_tcs = []
            for i, tc in enumerate(tcs):
                if not isinstance(tc, dict):
                    continue
                # langchain has {name, args, id?}; OpenAI wants
                # {id, type:'function', function:{name, arguments(json str)}}
                tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                args = tc.get("args", tc.get("arguments", {}))
                if isinstance(args, dict):
                    args_str = json.dumps(args)
                elif isinstance(args, str):
                    args_str = args
                else:
                    args_str = json.dumps(args) if args is not None else "{}"
                openai_tcs.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": args_str,
                    },
                })
            if openai_tcs:
                msg["tool_calls"] = openai_tcs
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


# ---------------------------------------------------------------------------
# EAGLE3 capture streaming
# ---------------------------------------------------------------------------

def _extract_expected_tokens(turn):
    """Raw verified tokens from oracle_vanilla_entries.

    These ARE what 14B's target naturally output during the eagle3 capture
    (one token per step in vanilla mode). Replay forces target to repeat
    this exact sequence, which keeps dm capture's prefix byte-identical to
    eagle3's prefix — needed so the simulator's gt (eagle3 verified) and
    the chain DM's prefix-conditional predictions stay aligned.

    BPE round-trip via tokenize(response_text) does NOT preserve the raw
    sequence — the captured response field is sometimes reconstructed
    (e.g., '<think>' may be appended at the start of response but not
    present in raw verified tokens), and re-tokenizing the response text
    can produce different token IDs at boundaries where Qwen's BPE merges
    differ between encode→decode→encode. Using raw verified tokens here
    side-steps both issues."""
    sd = turn.get("spec_decode") or {}
    evs = sd.get("oracle_vanilla_entries") or []
    out = []
    for e in evs:
        toks = e.get("tokens") or []
        if toks and toks[0]:
            out.append(int(toks[0][0]))
    return out


def _has_oracle_data(item):
    if not isinstance(item, dict):
        return False
    sd = item.get("spec_decode") or {}
    return bool(sd.get("oracle_vanilla_entries"))


def _drop_trailing_assistant(msgs):
    if not msgs:
        return msgs
    last = msgs[-1]
    if isinstance(last, dict) and (
            last.get("type") == "ai" or last.get("role") == "assistant"):
        return msgs[:-1]
    return msgs


def _build_specbench_messages(dataset_q, eagle_q, t_idx):
    """Reconstruct messages for specbench turn t_idx.

    specbench doesn't store messages in EAGLE3 capture — rebuild from the
    dataset's user-turn list and prior turns' captured responses."""
    user_turns = (dataset_q or {}).get("turns") or []
    eagle_turns = (eagle_q or {}).get("turns") or []
    if t_idx >= len(user_turns):
        return []
    msgs = []
    for prev in range(t_idx):
        if prev < len(user_turns):
            msgs.append({"role": "user", "content": user_turns[prev]})
        if prev < len(eagle_turns):
            prev_resp = eagle_turns[prev].get("response") or ""
            msgs.append({"role": "assistant", "content": prev_resp})
    msgs.append({"role": "user", "content": user_turns[t_idx]})
    return msgs


def stream_turns(eagle3_path, tokenizer, dataset=None, max_questions=None):
    """Yield {q_idx, t_idx, q_id, q_meta, input_messages, response,
    expected_tokens} per (question, turn).

    expected_tokens are derived by tokenizing the captured response text
    (NOT from oracle_vanilla_entries' tokens, which are offset by SGLang's
    spec-decode warmup steps and don't match the response text)."""
    if dataset is not None:
        dataset_by_id = {}
        for d in dataset:
            qid = d.get("question_id") or d.get("instance_id") or d.get("bfcl_id")
            if qid is not None:
                dataset_by_id[qid] = d
    else:
        dataset_by_id = None

    with open(eagle3_path, "rb") as f:
        for q_idx, q in enumerate(ijson.items(f, "questions.item")):
            if max_questions is not None and q_idx >= max_questions:
                break
            q_id = (q.get("question_id") or q.get("instance_id")
                    or q.get("bfcl_id") or str(q_idx))
            q_meta = {k: v for k, v in q.items()
                      if k not in ("turns", "agent_metrics", "spec_decode")}
            qturns = q.get("turns") or []
            steps = (q.get("agent_metrics") or {}).get("steps") or []

            # Choose the source that actually carries spec_decode entries.
            # - specbench: q.turns has spec_decode (steps absent)
            # - bfcl_v4: q.agent_metrics.steps has spec_decode (turns absent)
            # - swebench: BOTH exist; only steps has spec_decode
            def _has_oracle(item):
                if not isinstance(item, dict):
                    return False
                sd = item.get("spec_decode") or {}
                return bool(sd.get("oracle_vanilla_entries"))

            if qturns and any(_has_oracle_data(t) for t in qturns):
                source_units = qturns
                source_kind = "turns"
            elif steps and any(_has_oracle_data(s) for s in steps):
                source_units = steps
                source_kind = "steps"
            else:
                continue

            ds_q = (dataset_by_id or {}).get(q_id)
            for t_idx, u in enumerate(source_units):
                if not isinstance(u, dict):
                    continue
                msgs_raw = u.get("messages") or []
                if not msgs_raw and ds_q is not None and source_kind == "turns":
                    msgs_raw = _build_specbench_messages(ds_q, q, t_idx)
                input_msgs = _drop_trailing_assistant(msgs_raw)
                response_text = u.get("response") or u.get("content") or ""
                if not input_msgs:
                    continue
                # Use raw verified tokens (NOT tokenize(response)) so the dm
                # capture's prefix matches eagle3's byte-for-byte. See
                # _extract_expected_tokens docstring for why.
                expected = _extract_expected_tokens(u)
                if not expected:
                    continue
                yield {
                    "q_idx": q_idx,
                    "t_idx": t_idx,
                    "q_id": q_id,
                    "q_meta": q_meta,
                    "input_messages": input_msgs,
                    "response": response_text,
                    "expected_tokens": list(expected),
                }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eagle3", required=True,
                    help="Source agent_results_eagle3.json")
    ap.add_argument("--trajectory", required=True,
                    help="Path to write trajectory JSON (consumed by SGLang)")
    ap.add_argument("--output", required=True,
                    help="Path to write replay-capture JSON")
    ap.add_argument("--server-url", default="http://localhost:30000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokenizer-path",
                    help="HF tokenizer path (defaults to --model)")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--dataset", default=None,
                    help="Dataset JSONL (used to reconstruct specbench prompts)")
    ap.add_argument("--build-trajectory-only", action="store_true",
                    help="Stage 1 only: extract turns + write trajectory file")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="Flush partial output JSON every N turns")
    args = ap.parse_args()

    eagle3 = Path(args.eagle3)
    if not eagle3.exists():
        print(f"ERROR: eagle3 not found: {eagle3}", file=sys.stderr)
        return 1

    print(f"[replay] reading {eagle3}", file=sys.stderr)
    dataset = None
    if args.dataset:
        dataset = []
        with open(args.dataset) as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(json.loads(line))
        print(f"[replay] loaded dataset: {len(dataset)} entries from {args.dataset}",
              file=sys.stderr)

    # Need tokenizer up-front because we tokenize captured response text
    # to obtain expected_tokens.
    from transformers import AutoTokenizer
    tok_path = args.tokenizer_path or args.model
    print(f"[replay] loading tokenizer: {tok_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    turns = list(stream_turns(eagle3, tokenizer, dataset=dataset,
                              max_questions=args.max_questions))
    print(f"[replay] {len(turns)} turns", file=sys.stderr)
    if not turns:
        print("ERROR: no turns extracted", file=sys.stderr)
        return 1

    # ---- Stage 1: build trajectory (submission-order keys) ----
    traj = {}
    for t in turns:
        key = f"q{t['q_idx']:06d}_t{t['t_idx']:03d}"
        traj[key] = t["expected_tokens"]
    Path(args.trajectory).parent.mkdir(parents=True, exist_ok=True)
    with open(args.trajectory, "w") as f:
        json.dump(traj, f)
    total_tokens = sum(len(v) for v in traj.values())
    print(f"[replay] trajectory: {len(traj)} keys, "
          f"{total_tokens:,} tokens, "
          f"file={Path(args.trajectory).stat().st_size/1024/1024:.1f}MB",
          file=sys.stderr)

    if args.build_trajectory_only:
        return 0

    # ---- Stage 2: submit each turn ----
    # Oracle log API (need to use the same path as the SGLang server).
    sys.path.insert(0, "/workspace")
    from simulation.oracle.oracle_patch import (
        get_oracle_log_position,
        read_oracle_log,
    )

    client = OpenAI(base_url=args.server_url, api_key="dummy",
                    timeout=600.0)

    # Per-question accumulator.
    per_q = {}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def _flush():
        out = {"questions": [per_q[k] for k in sorted(per_q.keys())]}
        with open(args.output, "w") as f:
            # default=str handles Decimal etc. that swebench's q_meta may carry.
            json.dump(out, f, default=str)

    t_start = time.time()
    n_done = 0
    n_failed = 0
    for ti, t in enumerate(turns):
        msgs = _lc_to_openai(t["input_messages"])
        # Build prompt token IDs via the chat template.
        try:
            prompt_text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            prompt_ids = tokenizer.encode(prompt_text,
                                          add_special_tokens=False)
        except Exception as e:
            print(f"[replay] turn {ti} (q{t['q_idx']}.t{t['t_idx']}) "
                  f"chat-template FAILED: {e}", file=sys.stderr)
            n_failed += 1
            continue

        n_expected = len(t["expected_tokens"])
        before = get_oracle_log_position()
        try:
            resp = client.completions.create(
                model=args.model,
                prompt=prompt_ids,
                max_tokens=n_expected,
                temperature=0.0,
                stop=None,
                # ignore_eos: keep decoding through any trajectory EOS so
                # that exactly n_expected tokens are produced.
                extra_body={"ignore_eos": True},
            )
            _ = resp  # response content irrelevant — replay forced it
        except Exception as e:
            print(f"[replay] turn {ti} (q{t['q_idx']}.t{t['t_idx']}) "
                  f"completion FAILED: {e}", file=sys.stderr)
            n_failed += 1
            continue

        # Read entries appended during this submission.
        # batch=1 + serial submission ⇒ no other entries can interleave.
        entries = read_oracle_log(before)

        q_idx = t["q_idx"]
        if q_idx not in per_q:
            per_q[q_idx] = {**t["q_meta"], "turns": [],
                            "agent_metrics": {"steps": []}}
        unit = {
            "t_idx": t["t_idx"],
            "response": t["response"],
            "expected_token_count": n_expected,
            "captured_entry_count": len(entries),
            "spec_decode": {"oracle_vanilla_entries": entries},
        }
        # Mirror under both `turns` (specbench-style _extract_online path)
        # and `agent_metrics.steps` (bfcl/swebench _extract_bfcl path) so
        # downstream assemble_records can use either selector.
        per_q[q_idx]["turns"].append(unit)
        per_q[q_idx]["agent_metrics"]["steps"].append(unit)
        n_done += 1

        if n_done % 5 == 0:
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 0.1)
            eta = (len(turns) - n_done) / max(rate, 1e-6)
            print(f"[replay] {n_done}/{len(turns)} done "
                  f"(failed={n_failed}, "
                  f"{rate:.2f} turn/s, ETA {eta/60:.1f}m)",
                  file=sys.stderr)

        if n_done % args.checkpoint_every == 0:
            _flush()

    _flush()
    print(f"[replay] DONE — {n_done} turns "
          f"(failed={n_failed}) in {time.time() - t_start:.1f}s",
          file=sys.stderr)
    print(f"[replay] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
