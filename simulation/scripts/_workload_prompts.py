"""Workload prompt loader shared by the measure_*_cost.py scripts.

For each workload (specbench / bfcl_v4 / swebench) extract the first
``n_samples`` user-visible prompts as chat-style message lists. Output is
uniform across workloads so callers can feed them to OpenAI-compatible
``/v1/chat/completions`` endpoints or arbitrary tokenizers.

Dataset path conventions (produced by ``simulation/scripts/prepare_*_data.py``):

* specbench → ``data/specbench/dataset.jsonl``
  Schema: ``{question_id, category, turns: [str, ...]}``
  ``turns[0]`` is used as the user message.

* bfcl_v4 → ``data/bfcl_agent/dataset.jsonl``
  Schema: ``{question_id, category, bfcl_id, question: [[{role, content}, ...], ...], ...}``
  ``question[0][0]["content"]`` (first round, first message).

* swebench → ``data/swebench/dataset.jsonl``
  Schema varies; we read the ``problem_statement`` or ``issue`` field.
  If the file is missing the caller receives an empty list + warning (SWE-Bench
  prep is not automated in this repo).

Intentionally minimal — these prompts are used only for GPU warm-up and
single-prompt latency measurement, not full agent loops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional


WORKLOAD_FILES = {
    "specbench": "data/specbench/dataset.jsonl",
    "bfcl_v4": "data/bfcl_agent/dataset.jsonl",
    "swebench": "data/swebench/dataset.jsonl",
}


def _extract_specbench(row: dict) -> Optional[str]:
    turns = row.get("turns")
    if isinstance(turns, list) and turns and isinstance(turns[0], str):
        return turns[0]
    return None


def _extract_bfcl_v4(row: dict) -> Optional[str]:
    q = row.get("question")
    if not isinstance(q, list) or not q:
        return None
    first_round = q[0]
    if not isinstance(first_round, list) or not first_round:
        return None
    first_msg = first_round[0]
    if isinstance(first_msg, dict):
        return first_msg.get("content")
    return None


def _extract_swebench(row: dict) -> Optional[str]:
    for key in ("problem_statement", "issue", "prompt", "text"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


_EXTRACTORS = {
    "specbench": _extract_specbench,
    "bfcl_v4": _extract_bfcl_v4,
    "swebench": _extract_swebench,
}


def load_workload_prompts(
    workload: str,
    n_samples: int = 2,
    root: Optional[Path] = None,
) -> List[dict]:
    """Return the first ``n_samples`` prompts from ``workload`` as chat messages.

    Each returned dict:
        {"id": str, "category": str,
         "messages": [{"role": "user", "content": str}]}

    Unknown workload → ``ValueError``.
    Missing dataset file (e.g. swebench not prepared) → empty list + stderr warning.
    """
    if workload not in WORKLOAD_FILES:
        raise ValueError(
            f"Unknown workload: {workload}. "
            f"Valid: {sorted(WORKLOAD_FILES.keys())}")

    root = root or Path.cwd()
    path = root / WORKLOAD_FILES[workload]
    if not path.exists():
        print(f"WARN: {workload} dataset not found at {path}; skipping.",
              file=sys.stderr)
        return []

    extractor = _EXTRACTORS[workload]
    out: List[dict] = []
    with open(path) as f:
        for line in f:
            if len(out) >= n_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = extractor(row)
            if not content:
                continue
            out.append({
                "id": str(row.get("bfcl_id")
                          or row.get("question_id")
                          or row.get("instance_id")
                          or f"{workload}_{len(out)}"),
                "category": str(row.get("category") or workload),
                "messages": [{"role": "user", "content": content}],
            })
    if len(out) < n_samples:
        print(f"WARN: {workload} only yielded {len(out)}/{n_samples} prompts.",
              file=sys.stderr)
    return out
