"""Merge per-step draft trees from multiple proposers into a union trie (Stage 4).

Consumes:
  * --agent-results     : Stage 1 EAGLE3 output (provides EAGLE3 trees + prompts)
  * --suffix-drafts     : Stage 3a output (per-step suffix trees)
  * --draft-model-drafts: Stage 3b output (per-step draft-model chains)  [optional]
  * --mtp-agent-results : Stage 3c output (MTP round agent_results)      [optional]

For every decoding step it emits one JSONL record with the merged union
trie, per-proposer sub-trees, context_token_ids (needed by Stage 5), and
the ground-truth future suffix.

Usage:
    python3 -m simulation.pipeline.collect_union_trie \\
        --agent-results results/.../agent_results_eagle3.json \\
        --suffix-drafts results/.../suffix_drafts.jsonl \\
        --draft-model-drafts results/.../draft_model_drafts.jsonl \\
        --mtp-agent-results results/.../agent_results_mtp.json \\
        --output simulation/results/.../union_trie_data.jsonl \\
        --model zai-org/GLM-4.7-Flash
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from simulation.evaluation.run_oracle_sim import (
    _flat_to_tree,
    extract_requests,
    load_exclude_ids,
)


# ---------------------------------------------------------------------------
# Lightweight trie builder (no dependency on tree_utils / arctic_inference)
# ---------------------------------------------------------------------------

class _TrieNode:
    """Minimal trie node for union trie construction."""
    __slots__ = ("token_id", "children", "sources", "node_id")

    def __init__(self, token_id: int, node_id: int):
        self.token_id = token_id
        self.children: Dict[int, _TrieNode] = {}  # token_id → child
        self.sources: set = set()
        self.node_id = node_id


def _paths_from_flat_tree(
    token_ids: Sequence[int],
    parents: Sequence[int],
) -> List[List[int]]:
    """Extract all root-to-leaf paths from a flat (token_ids, parents) tree."""
    n = len(token_ids)
    if n == 0:
        return []

    children: Dict[int, List[int]] = {-1: []}
    for i in range(n):
        children.setdefault(i, [])
        children.setdefault(parents[i], []).append(i)

    paths = []

    def _dfs(node: int, path: List[int]):
        ch = children.get(node, [])
        if not ch:
            if path:
                paths.append(list(path))
            return
        for c in ch:
            path.append(token_ids[c])
            _dfs(c, path)
            path.pop()

    _dfs(-1, [])
    return paths


def build_union_trie(
    proposer_trees: Dict[str, Tuple[List[int], List[int]]],
) -> Tuple[List[int], List[int], List[List[str]]]:
    """Build union trie from multiple proposers' draft trees.

    Parameters
    ----------
    proposer_trees : dict
        Mapping from proposer name to (token_ids, parents).

    Returns
    -------
    (flat_token_ids, flat_parents, source_map)
        flat_token_ids: Merged trie token ids.
        flat_parents: Parent indices (-1 for root children).
        source_map: For each node, sorted list of contributing proposers.
    """
    root = _TrieNode(token_id=-1, node_id=0)
    next_id = 1

    for proposer_name, (token_ids, parents) in proposer_trees.items():
        if not token_ids:
            continue
        paths = _paths_from_flat_tree(token_ids, parents)
        for path in paths:
            node = root
            for token_id in path:
                if token_id in node.children:
                    child = node.children[token_id]
                    child.sources.add(proposer_name)
                else:
                    child = _TrieNode(token_id, next_id)
                    next_id += 1
                    child.sources.add(proposer_name)
                    node.children[token_id] = child
                node = child

    # BFS flatten
    flat_tokens: List[int] = []
    flat_parents: List[int] = []
    source_map: List[List[str]] = []

    # node_id → flat_index mapping
    id_to_idx: Dict[int, int] = {}
    queue = [(root, -1)]  # (node, parent_flat_idx)

    while queue:
        next_queue = []
        for node, parent_idx in queue:
            if node is root:
                id_to_idx[root.node_id] = -1
                for child in sorted(node.children.values(), key=lambda n: n.node_id):
                    next_queue.append((child, -1))
            else:
                idx = len(flat_tokens)
                id_to_idx[node.node_id] = idx
                flat_tokens.append(node.token_id)
                flat_parents.append(parent_idx)
                source_map.append(sorted(node.sources))
                for child in sorted(node.children.values(), key=lambda n: n.node_id):
                    next_queue.append((child, idx))
        queue = next_queue

    # BFS guarantees: parent[i] < i for all i (enables safe truncation)
    assert all(p == -1 or p < i for i, p in enumerate(flat_parents)), \
        "BFS order violated: parent must precede child"

    return flat_tokens, flat_parents, source_map


# ---------------------------------------------------------------------------
# Per-step draft loaders (Stage 3a / 3b outputs)
# ---------------------------------------------------------------------------

def load_per_step_drafts(path: str) -> Dict[Tuple[str, int, int], dict]:
    """Load a JSONL of per-step drafts into a (rid, call_idx, step_idx) → record dict.

    Each line is expected to have keys: request_id, call_idx, step_idx,
    token_ids, parents (+ optional extras like score). Any extra fields are
    preserved in the returned dict so callers can propagate them onward.
    """
    out: Dict[Tuple[str, int, int], dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["request_id"], int(r["call_idx"]), int(r["step_idx"]))
            out[key] = r
    return out


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_union_tries(
    requests: List[dict],
    suffix_by_key: Optional[Dict[Tuple[str, int, int], dict]] = None,
    dm_by_key: Optional[Dict[Tuple[str, int, int], dict]] = None,
    mtp_requests: Optional[List[dict]] = None,
    include_union_trie: bool = True,
) -> List[dict]:
    """Collect per-step union trie data for all requests.

    Parameters
    ----------
    requests : list[dict]
        Primary requests (EAGLE3 round) from extract_requests.
    suffix_by_key, dm_by_key : dict, optional
        Per-step draft records keyed by (request_id, call_idx, step_idx).
    mtp_requests : list[dict], optional
        MTP round requests (same bfcl_ids, same token sequences). If
        provided, MTP drafts are included in the union trie.
    include_union_trie : bool
        When True (default), each record includes the merged ``union_trie``
        and ``source_map`` fields. When False, those fields are omitted;
        records still carry ``per_proposer`` / ``context_token_ids`` /
        ``ground_truth_future``. Used by Stage 6 runs with ``UNION_TRIE=0``.
    """
    suffix_by_key = suffix_by_key or {}
    dm_by_key = dm_by_key or {}

    mtp_by_id: Dict[str, dict] = {}
    if mtp_requests:
        for mr in mtp_requests:
            mtp_by_id[mr["bfcl_id"]] = mr

    records = []

    for ri, req in enumerate(requests):
        bfcl_id = req["bfcl_id"]
        prompt_ids_list = req.get("per_call_prompt_ids")
        mtp_req = mtp_by_id.get(bfcl_id)

        for call_idx in range(len(req["per_call_tokens"])):
            tokens = req["per_call_tokens"][call_idx]
            eagle3s = req["per_call_eagle3s"][call_idx]
            N = len(tokens)
            if N == 0:
                continue

            if prompt_ids_list and call_idx < len(prompt_ids_list):
                prompt = np.array(prompt_ids_list[call_idx], dtype=np.int32)
            else:
                prompt = np.array([], dtype=np.int32)
            decoded: List[int] = []

            for pos in range(N):
                future = tokens[pos:]
                if len(future) <= 1:  # only current token, nothing to verify
                    decoded.append(tokens[pos])
                    continue

                proposer_trees: Dict[str, Tuple[List[int], List[int]]] = {}

                # EAGLE3: prefer full tree, else flat chain
                eagle3_p_t = None
                eagle3_path_draft_p_t = None
                e3_trees = req.get("per_call_eagle3_trees")
                e3_p_ts = req.get("per_call_eagle3_tree_p_ts")
                e3_draft_p_ts = req.get(
                    "per_call_eagle3_tree_path_draft_p_ts")
                e3_attached = False
                if e3_trees and call_idx < len(e3_trees):
                    call_trees = e3_trees[call_idx]
                    if pos < len(call_trees) and call_trees[pos] is not None:
                        et = call_trees[pos]
                        proposer_trees["eagle3"] = (et["token_ids"], et["parents"])
                        e3_attached = True
                        if e3_p_ts and call_idx < len(e3_p_ts):
                            call_p_ts = e3_p_ts[call_idx]
                            if pos < len(call_p_ts) and call_p_ts[pos] is not None:
                                eagle3_p_t = call_p_ts[pos]
                        if e3_draft_p_ts and call_idx < len(e3_draft_p_ts):
                            call_draft_p_ts = e3_draft_p_ts[call_idx]
                            if (pos < len(call_draft_p_ts)
                                    and call_draft_p_ts[pos] is not None):
                                eagle3_path_draft_p_t = call_draft_p_ts[pos]
                if not e3_attached:
                    e_draft = eagle3s[pos] if pos < len(eagle3s) else []
                    if e_draft:
                        e_parents, e_tokens = _flat_to_tree(e_draft)
                        proposer_trees["eagle3"] = (e_tokens, e_parents)

                # Suffix: looked up from Stage 3a output
                key = (bfcl_id, call_idx, pos)
                suffix_rec = suffix_by_key.get(key)
                suffix_score = 0.0
                if suffix_rec and suffix_rec.get("token_ids"):
                    proposer_trees["suffix"] = (
                        list(suffix_rec["token_ids"]),
                        list(suffix_rec["parents"]),
                    )
                    suffix_score = float(suffix_rec.get("score", 0.0))

                # Draft model: looked up from Stage 3b output
                dm_rec = dm_by_key.get(key)
                if dm_rec and dm_rec.get("token_ids"):
                    proposer_trees["draft_model"] = (
                        list(dm_rec["token_ids"]),
                        list(dm_rec["parents"]),
                    )

                # MTP: from Stage 3c replay
                if mtp_req:
                    mtp_eagle3s = mtp_req.get("per_call_eagle3s", [])
                    mtp_e3_trees = mtp_req.get("per_call_eagle3_trees")
                    if call_idx < len(mtp_eagle3s):
                        mtp_call_eagle3s = mtp_eagle3s[call_idx]
                        if (mtp_e3_trees and call_idx < len(mtp_e3_trees)
                                and pos < len(mtp_e3_trees[call_idx])
                                and mtp_e3_trees[call_idx][pos] is not None):
                            mt = mtp_e3_trees[call_idx][pos]
                            proposer_trees["mtp"] = (mt["token_ids"], mt["parents"])
                        elif pos < len(mtp_call_eagle3s) and mtp_call_eagle3s[pos]:
                            m_parents, m_tokens = _flat_to_tree(mtp_call_eagle3s[pos])
                            proposer_trees["mtp"] = (m_tokens, m_parents)

                if not proposer_trees:
                    decoded.append(tokens[pos])
                    continue

                # Context for p_t collection: ends at tokens[pos-1] so that
                # logits at context[-1] predict tokens[pos], matching what
                # tree depth-1 nodes predict.
                context_for_pt = list(decoded)  # = tokens[0:pos]
                if len(prompt) > 0:
                    context = (
                        np.concatenate(
                            [prompt, np.array(context_for_pt, dtype=np.int32)])
                        if context_for_pt else prompt.copy())
                else:
                    context = np.array(context_for_pt, dtype=np.int32)

                per_proposer = {}
                for name, (tids, pids) in proposer_trees.items():
                    entry = {
                        "token_ids": tids,
                        "parents": pids,
                        "size": len(tids),
                    }
                    if name == "eagle3" and eagle3_p_t is not None:
                        entry["p_t"] = eagle3_p_t
                    if name == "eagle3" and eagle3_path_draft_p_t is not None:
                        entry["path_draft_p_t"] = eagle3_path_draft_p_t
                    if name == "suffix":
                        entry["score"] = suffix_score
                    per_proposer[name] = entry

                record = {
                    "request_id": bfcl_id,
                    "call_idx": call_idx,
                    "step_idx": pos,
                    "per_proposer": per_proposer,
                    "ground_truth_future": list(future),
                    "context_token_ids": context.tolist(),
                }
                if include_union_trie:
                    flat_tokens, flat_parents, source_map = build_union_trie(
                        proposer_trees)
                    record["union_trie"] = {
                        "token_ids": flat_tokens,
                        "parents": flat_parents,
                    }
                    record["source_map"] = source_map
                records.append(record)

                decoded.append(tokens[pos])

        if (ri + 1) % 10 == 0:
            print(f"  Processed {ri + 1}/{len(requests)} requests, "
                  f"{len(records)} steps", file=sys.stderr)

    return records


def assemble_records_from_artifacts(
    agent_results_path: str,
    suffix_drafts_path: Optional[str] = None,
    draft_model_drafts_path: Optional[str] = None,
    mtp_agent_results_path: Optional[str] = None,
    exclude_path: Optional[str] = None,
    model: Optional[str] = None,
    dataset_path: Optional[str] = None,
    responses_path: Optional[str] = None,
    include_union_trie: bool = True,
) -> List[dict]:
    """End-to-end loader: read Stage 1/3 artifacts, return per-step records.

    Mirrors the side effects of ``main()`` (tokenizer load, BFCL/SpecBench
    prompt reconstruction, per-step JSONL loading, MTP extraction) and
    produces the same per-step record list that Stage 4 would write.
    When ``include_union_trie`` is False, records omit the ``union_trie``
    and ``source_map`` fields so callers who only need per-proposer data
    (e.g. ``UNION_TRIE=0`` path in run_tree_oracle_sim) can bypass the
    Stage 4 build entirely.
    """
    exclude_ids = load_exclude_ids(exclude_path) if exclude_path else set()

    print(f"Loading: {agent_results_path}", file=sys.stderr)
    with open(agent_results_path) as f:
        data = json.load(f)

    tokenizer = None
    if model:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {model}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model)

    bfcl_dataset: Optional[dict] = None
    resp_by_id: Optional[dict] = None
    specbench_dataset: Optional[dict] = None
    if dataset_path and responses_path:
        try:
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(PROJECT_ROOT))
            from simulation.agents.bfcl_agent import preprocess_bfcl_requests
            entries = []
            with open(dataset_path) as f:
                for line in f:
                    entries.append(json.loads(line))
            preprocess_bfcl_requests(entries)
            bfcl_dataset = {e["bfcl_id"]: e for e in entries}
            with open(responses_path) as f:
                resp_data = json.load(f)
            resp_by_id = {r["bfcl_id"]: r for r in resp_data}
        except Exception as e:
            print(f"WARN: BFCL prompt reconstruction failed: {e}",
                  file=sys.stderr)
    elif dataset_path:
        try:
            specbench_dataset = {}
            with open(dataset_path) as f:
                for line in f:
                    entry = json.loads(line)
                    specbench_dataset[entry["question_id"]] = entry
        except Exception as e:
            print(f"WARN: SpecBench dataset load failed: {e}",
                  file=sys.stderr)

    all_requests = extract_requests(
        data, exclude_ids, None,
        tokenizer, bfcl_dataset, resp_by_id, specbench_dataset)
    print(f"Requests: {len(all_requests)}", file=sys.stderr)

    suffix_by_key: Dict[Tuple[str, int, int], dict] = {}
    if suffix_drafts_path:
        print(f"Loading suffix drafts: {suffix_drafts_path}", file=sys.stderr)
        suffix_by_key = load_per_step_drafts(suffix_drafts_path)
        print(f"  {len(suffix_by_key)} suffix drafts", file=sys.stderr)

    dm_by_key: Dict[Tuple[str, int, int], dict] = {}
    if draft_model_drafts_path:
        print(f"Loading draft-model drafts: {draft_model_drafts_path}",
              file=sys.stderr)
        dm_by_key = load_per_step_drafts(draft_model_drafts_path)
        print(f"  {len(dm_by_key)} draft-model drafts", file=sys.stderr)

    mtp_all_requests = None
    if mtp_agent_results_path:
        print(f"Loading MTP data: {mtp_agent_results_path}", file=sys.stderr)
        with open(mtp_agent_results_path) as f:
            mtp_data = json.load(f)
        mtp_all_requests = extract_requests(mtp_data, exclude_ids)
        print(f"MTP requests: {len(mtp_all_requests)}", file=sys.stderr)

    t0 = time.time()
    mode = "union tries" if include_union_trie else "per-proposer records"
    print(f"Building {mode} for {len(all_requests)} requests...",
          file=sys.stderr)
    records = collect_union_tries(
        all_requests,
        suffix_by_key=suffix_by_key,
        dm_by_key=dm_by_key,
        mtp_requests=mtp_all_requests,
        include_union_trie=include_union_trie,
    )
    elapsed = time.time() - t0
    print(f"  assembled {len(records)} step records in {elapsed:.1f}s",
          file=sys.stderr)
    return records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-results", required=True,
                        help="Path to agent_results_eagle3.json (Stage 1)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path for union trie data")
    parser.add_argument("--suffix-drafts", default=None,
                        help="Path to suffix_drafts.jsonl (Stage 3a)")
    parser.add_argument("--draft-model-drafts", default=None,
                        help="Path to draft_model_drafts.jsonl (Stage 3b)")
    parser.add_argument("--mtp-agent-results", default=None,
                        help="Path to agent_results_mtp.json (Stage 3c)")
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--model", default=None,
                        help="Target model name for tokenizer")
    parser.add_argument("--responses", default=None,
                        help="Path to agent_results_responses.json (BFCL)")
    parser.add_argument("--dataset", default=None,
                        help="Path to dataset.jsonl (BFCL/SpecBench)")
    args = parser.parse_args()

    records = assemble_records_from_artifacts(
        agent_results_path=args.agent_results,
        suffix_drafts_path=args.suffix_drafts,
        draft_model_drafts_path=args.draft_model_drafts,
        mtp_agent_results_path=args.mtp_agent_results,
        exclude_path=args.exclude,
        model=args.model,
        dataset_path=args.dataset,
        responses_path=args.responses,
        include_union_trie=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    total_union_nodes = sum(
        len(r["union_trie"]["token_ids"]) for r in records)
    proposer_counts: Dict[str, int] = {}
    for r in records:
        for name in r["per_proposer"]:
            proposer_counts[name] = proposer_counts.get(name, 0) + 1

    print(f"\nSteps: {len(records)}", file=sys.stderr)
    print(f"Total union trie nodes: {total_union_nodes:,} "
          f"(avg {total_union_nodes / max(len(records), 1):.1f}/step)",
          file=sys.stderr)
    for name, count in sorted(proposer_counts.items()):
        print(f"  {name}: {count} steps with drafts", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
