"""Send workload prompts to a running MTP/NEXTN SGLang server (with
Oracle-Vanilla logging enabled) and pull the per-step draft entries.

The server is expected to be launched via launch_h100_server.sh which
exports SGLANG_ORACLE_VANILLA=1 and writes per-step JSONL to
$SGLANG_ORACLE_LOG (default /tmp/sglang_oracle_vanilla.jsonl).

We tag each request with a UUID and a `bfcl_id` derived from the
dataset entry, then for every request:
  1. Snapshot the current oracle log byte offset.
  2. POST {input_ids, sampling_params} to the server.
  3. After the response, read everything the server appended since the
     snapshot — those are this request's per-step entries.
  4. Save (request_id, prompt, output, oracle_entries[]) to the output
     JSONL.

Force-accept (accept_length=0) is on, so output_token_ids equals the
greedy decode of the target model and each step's ground-truth-future
is exactly the bonus tokens emitted by subsequent steps.

Workloads supported (matches our existing capture sets):
  * specbench    — turns are list[str] (single-turn user msgs)
  * bfcl_v4      — chat-style multi-turn; we feed the FIRST turn only
                   (same single-prompt approximation as the original
                   MTP attempt)
  * swebench_verified — prompt = first turn's user content

For accept-rate purposes we only need a representative decode trace,
so single-turn capture is sufficient.

Usage:
  python3 -m simulation.scripts.mtp.collect_mtp_drafts \\
      --workload specbench \\
      --dataset data/specbench/dataset.jsonl \\
      --model Qwen/Qwen3.5-9B \\
      --server-url http://localhost:31010 \\
      --max-questions 80 \\
      --output simulation/results/qwen35_mtp/mtp_drafts_specbench.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import requests as http_requests

# Reuse the byte-offset reader from oracle_patch — works for any oracle log.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.oracle.oracle_patch import (  # noqa: E402
    get_oracle_log_position,
    read_oracle_log,
)


def _build_prompts_specbench(dataset_path: str) -> Iterable[dict]:
    """Yield {req_id, prompt, raw} for each specbench question.
    Specbench schema: {question_id, category, turns: [str, ...]}.
    First turn = user request. We feed that as a single user message.
    """
    with open(dataset_path) as f:
        for line in f:
            q = json.loads(line)
            qid = str(q["question_id"])
            turns = q.get("turns") or []
            if not turns:
                continue
            yield {
                "req_id": f"specbench__{qid}",
                "prompt": turns[0],
                "raw": q,
            }


def _build_prompts_bfcl(dataset_path: str) -> Iterable[dict]:
    """Yield first-turn prompts from BFCL agent dataset (best-effort).
    BFCL entries have a list of `turns` whose first element is typically
    a user message dict {"role": "user", "content": "..."}.
    """
    with open(dataset_path) as f:
        for line in f:
            q = json.loads(line)
            qid = q.get("bfcl_id") or q.get("id") or q.get("question_id")
            turns = q.get("turns") or q.get("messages") or []
            if not turns:
                continue
            t0 = turns[0]
            content = (t0 if isinstance(t0, str)
                       else t0.get("content") or "")
            if not content:
                continue
            yield {
                "req_id": f"bfcl_v4__{qid}",
                "prompt": content,
                "raw": q,
            }


def _build_prompts_swebench(dataset_path: str) -> Iterable[dict]:
    """SWE-Bench-verified entries: schema mirrors BFCL's first-turn."""
    with open(dataset_path) as f:
        for line in f:
            q = json.loads(line)
            qid = q.get("instance_id") or q.get("id")
            turns = q.get("turns") or q.get("messages") or []
            if not turns:
                continue
            t0 = turns[0]
            content = (t0 if isinstance(t0, str)
                       else t0.get("content") or "")
            if not content:
                continue
            yield {
                "req_id": f"swebench__{qid}",
                "prompt": content,
                "raw": q,
            }


WORKLOAD_BUILDERS = {
    "specbench": _build_prompts_specbench,
    "bfcl_v4":   _build_prompts_bfcl,
    "swebench_verified": _build_prompts_swebench,
}


def _tokenize_chat(prompt: str, tokenizer) -> list[int]:
    """Wrap a single user message in the chat template and tokenize."""
    msgs = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True)
        return list(ids)
    # Fallback: raw tokenize.
    return tokenizer.encode(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True,
                    choices=list(WORKLOAD_BUILDERS.keys()))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True,
                    help="HF model id or path. Used only for the chat "
                         "template + tokenizer (must match server).")
    ap.add_argument("--server-url", default="http://localhost:31010")
    ap.add_argument("--oracle-log",
                    default=os.environ.get(
                        "SGLANG_ORACLE_LOG",
                        "/tmp/sglang_oracle_vanilla.jsonl"))
    ap.add_argument("--max-questions", type=int, default=None,
                    help="Cap total questions sent (None = all)")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="Server-side decode cap per request.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", required=True,
                    help="JSONL — one line per request with prompt, "
                         "output_token_ids, oracle_entries[].")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    print(f"[mtp-client] workload={args.workload} server={args.server_url}",
          file=sys.stderr)
    print(f"[mtp-client] oracle log: {args.oracle_log}", file=sys.stderr)

    # Ping server.
    try:
        ping = http_requests.get(f"{args.server_url}/get_model_info",
                                 timeout=10).json()
        print(f"[mtp-client] server model: {ping.get('model_path', '?')}",
              file=sys.stderr)
    except Exception as e:
        sys.exit(f"server ping failed: {e}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    builder = WORKLOAD_BUILDERS[args.workload]

    n_done = 0
    t_start = time.time()
    with open(args.output, "w") as out_fp:
        for entry in builder(args.dataset):
            if args.max_questions and n_done >= args.max_questions:
                break

            input_ids = _tokenize_chat(entry["prompt"], tokenizer)
            if not input_ids:
                continue

            # Snapshot oracle log offset just before the request.
            os.environ["SGLANG_ORACLE_LOG"] = args.oracle_log
            pos_before = get_oracle_log_position()

            try:
                resp = http_requests.post(
                    f"{args.server_url}/generate",
                    json={
                        "input_ids": input_ids,
                        "sampling_params": {
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature,
                        },
                    },
                    timeout=args.timeout,
                ).json()
            except Exception as e:
                print(f"[mtp-client] WARN req={entry['req_id']} failed: {e}",
                      file=sys.stderr)
                continue

            output_ids = resp.get("output_ids") or resp.get("token_ids") or []
            output_text = resp.get("text") or ""

            # Read oracle entries appended since pos_before — these are
            # the per-step draft trees for THIS request.
            oracle_entries = read_oracle_log(pos_before)

            out_fp.write(json.dumps({
                "request_id": entry["req_id"],
                "prompt": entry["prompt"],
                "input_ids": input_ids,
                "output_ids": output_ids,
                "output_text": output_text,
                "oracle_entries": oracle_entries,
            }) + "\n")
            n_done += 1

            if n_done % 5 == 0:
                elapsed = time.time() - t_start
                avg_steps = (
                    sum(len(json.loads(line)["oracle_entries"])
                        for line in open(args.output))
                    / max(1, n_done))
                print(f"[mtp-client] {n_done} done in {elapsed:.0f}s "
                      f"(avg {avg_steps:.0f} steps/req)",
                      file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"[mtp-client] DONE — {n_done} requests in {elapsed:.0f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
