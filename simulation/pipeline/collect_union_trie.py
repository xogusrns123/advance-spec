"""Build per-step union tries from agent_results.json.

For each decoding step, merges draft trees from all proposers
(EAGLE3, Suffix, Draft Model) into a union trie and writes the
result to a JSONL file for downstream oracle simulation.

Usage:
    python3 -m simulation.pipeline.collect_union_trie \
        --agent-results results/.../agent_results.json \
        --output results/.../union_trie_data.jsonl \
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

    paths: List[List[int]] = []

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
# Main collection loop
# ---------------------------------------------------------------------------

def collect_union_tries(
    requests: List[dict],
    suffix_cache,
    max_spec_tokens: int = 256,
    max_spec_factor: float = 4.0,
    min_token_prob: float = 0.0,
    mtp_requests: Optional[List[dict]] = None,
) -> List[dict]:
    """Collect per-step union trie data for all requests.

    Parameters
    ----------
    requests : list[dict]
        Primary requests (e.g. EAGLE3 round).
    suffix_cache :
        SuffixDecodingCache instance.
    mtp_requests : list[dict], optional
        MTP round requests (same bfcl_ids, same token sequences).
        If provided, MTP drafts are included in the union trie.
    """
    # Index MTP requests by bfcl_id for fast lookup
    mtp_by_id: Dict[str, dict] = {}
    if mtp_requests:
        for mr in mtp_requests:
            mtp_by_id[mr["bfcl_id"]] = mr

    records = []
    req_id = 0

    for ri, req in enumerate(requests):
        bfcl_id = req["bfcl_id"]
        dm_drafts = req.get("draft_model_drafts")
        prompt_ids_list = req.get("per_call_prompt_ids")
        mtp_req = mtp_by_id.get(bfcl_id)
        global_pos = 0

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
            suffix_cache.start_request(req_id, prompt)
            decoded = []

            for pos in range(N):
                # All proposers now predict tokens[pos:] (aligned)
                future = tokens[pos:]
                if len(future) <= 1:  # only current token, nothing to verify
                    decoded.append(tokens[pos])
                    suffix_cache.add_active_response(req_id, [tokens[pos]])
                    continue

                proposer_trees: Dict[str, Tuple[List[int], List[int]]] = {}

                # EAGLE3: use full tree if available, else flat chain
                eagle3_p_t = None  # p_t from verification logits
                e3_trees = req.get("per_call_eagle3_trees")
                e3_p_ts = req.get("per_call_eagle3_tree_p_ts")
                if e3_trees and call_idx < len(e3_trees):
                    call_trees = e3_trees[call_idx]
                    if pos < len(call_trees) and call_trees[pos] is not None:
                        et = call_trees[pos]
                        proposer_trees["eagle3"] = (et["token_ids"], et["parents"])
                        # Get p_t if available
                        if e3_p_ts and call_idx < len(e3_p_ts):
                            call_p_ts = e3_p_ts[call_idx]
                            if pos < len(call_p_ts) and call_p_ts[pos] is not None:
                                eagle3_p_t = call_p_ts[pos]
                    else:
                        e_draft = eagle3s[pos] if pos < len(eagle3s) else []
                        if e_draft:
                            e_parents, e_tokens = _flat_to_tree(e_draft)
                            proposer_trees["eagle3"] = (e_tokens, e_parents)
                else:
                    e_draft = eagle3s[pos] if pos < len(eagle3s) else []
                    if e_draft:
                        e_parents, e_tokens = _flat_to_tree(e_draft)
                        proposer_trees["eagle3"] = (e_tokens, e_parents)

                # Suffix: speculate from context BEFORE tokens[pos]
                # so that suffix predicts tokens[pos:] (same as EAGLE3)
                response_so_far = list(decoded)
                if len(prompt) > 0:
                    suffix_context = np.concatenate(
                        [prompt, np.array(response_so_far, dtype=np.int32)]) if response_so_far else prompt.copy()
                else:
                    suffix_context = np.array(response_so_far, dtype=np.int32)
                suffix_draft = suffix_cache.speculate(
                    req_id, suffix_context,
                    max_spec_tokens=max_spec_tokens,
                    max_spec_factor=max_spec_factor,
                    min_token_prob=min_token_prob,
                    use_tree_spec=True)

                # Context for p_t collection: ends at tokens[pos-1]
                # so that logits at context[-1] predict tokens[pos],
                # matching what tree depth-1 nodes predict.
                context_for_pt = list(decoded)  # = tokens[0:pos]
                if len(prompt) > 0:
                    context = np.concatenate(
                        [prompt, np.array(context_for_pt, dtype=np.int32)]) if context_for_pt else prompt.copy()
                else:
                    context = np.array(context_for_pt, dtype=np.int32)
                suffix_score = getattr(suffix_draft, "score", 0.0)
                if suffix_draft.token_ids:
                    proposer_trees["suffix"] = (
                        list(suffix_draft.token_ids),
                        list(suffix_draft.parents),
                    )

                # MTP: from Round 2 replay data
                if mtp_req:
                    mtp_eagle3s = mtp_req.get("per_call_eagle3s", [])
                    mtp_e3_trees = mtp_req.get("per_call_eagle3_trees")
                    if call_idx < len(mtp_eagle3s):
                        mtp_call_eagle3s = mtp_eagle3s[call_idx]
                        # Prefer full tree if available
                        if (mtp_e3_trees and call_idx < len(mtp_e3_trees)
                                and pos < len(mtp_e3_trees[call_idx])
                                and mtp_e3_trees[call_idx][pos] is not None):
                            mt = mtp_e3_trees[call_idx][pos]
                            proposer_trees["mtp"] = (mt["token_ids"], mt["parents"])
                        elif pos < len(mtp_call_eagle3s) and mtp_call_eagle3s[pos]:
                            m_parents, m_tokens = _flat_to_tree(mtp_call_eagle3s[pos])
                            proposer_trees["mtp"] = (m_tokens, m_parents)

                # Draft model (flat chain → tree)
                dm_d = (dm_drafts[global_pos + pos]
                        if dm_drafts and (global_pos + pos) < len(dm_drafts)
                        else [])
                if dm_d:
                    dm_parents, dm_tokens = _flat_to_tree(dm_d)
                    proposer_trees["draft_model"] = (dm_tokens, dm_parents)

                if not proposer_trees:
                    decoded.append(tokens[pos])
                    suffix_cache.add_active_response(req_id, [tokens[pos]])
                    continue

                flat_tokens, flat_parents, source_map = build_union_trie(
                    proposer_trees)

                per_proposer = {}
                for name, (tids, pids) in proposer_trees.items():
                    entry = {
                        "token_ids": tids,
                        "parents": pids,
                        "size": len(tids),
                    }
                    # Include p_t from verification logits if available
                    if name == "eagle3" and eagle3_p_t is not None:
                        entry["p_t"] = eagle3_p_t
                    if name == "suffix":
                        entry["score"] = float(suffix_score)
                    per_proposer[name] = entry

                records.append({
                    "request_id": bfcl_id,
                    "call_idx": call_idx,
                    "step_idx": pos,
                    "union_trie": {
                        "token_ids": flat_tokens,
                        "parents": flat_parents,
                    },
                    "source_map": source_map,
                    "per_proposer": per_proposer,
                    # ground_truth_future: tokens[pos:] — all proposers predict from here
                    "ground_truth_future": list(future),
                    "context_token_ids": context.tolist(),
                })

                decoded.append(tokens[pos])
                suffix_cache.add_active_response(req_id, [tokens[pos]])

            suffix_cache.stop_request(req_id)
            global_pos += N
            req_id += 1

        if (ri + 1) % 10 == 0:
            print(f"  Processed {ri + 1}/{len(requests)} requests, "
                  f"{len(records)} steps", file=sys.stderr)

    return records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-results", required=True,
                        help="Path to agent_results.json")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path for union trie data")
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--mtp-agent-results", default=None,
                        help="Path to MTP round agent_results.json (Round 2 replay)")
    parser.add_argument("--draft-model-drafts", default=None,
                        help="Path to draft_model_drafts.json")
    parser.add_argument("--model", default=None,
                        help="Target model name for tokenizer")
    parser.add_argument("--responses", default=None,
                        help="Path to agent_results_responses.json")
    parser.add_argument("--dataset", default=None,
                        help="Path to dataset.jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.0,
                        help="Fraction for cache warmup (default: 0)")
    args = parser.parse_args()

    from arctic_inference.suffix_decoding import SuffixDecodingCache

    exclude_ids = load_exclude_ids(args.exclude) if args.exclude else set()

    print(f"Loading: {args.agent_results}", file=sys.stderr)
    with open(args.agent_results) as f:
        data = json.load(f)

    dm_by_id = {}
    if args.draft_model_drafts:
        with open(args.draft_model_drafts) as f:
            dm_data = json.load(f)
        dm_by_id = {r["bfcl_id"]: r["drafts"] for r in dm_data["requests"]}
        print(f"Draft model drafts: {len(dm_by_id)} requests", file=sys.stderr)

    tokenizer = None
    if args.model:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.model}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    bfcl_dataset = None
    resp_by_id = None
    specbench_dataset = None
    if args.dataset and args.responses:
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
        except Exception as e:
            print(f"WARN: BFCL prompt reconstruction failed: {e}",
                  file=sys.stderr)
    elif args.dataset:
        try:
            specbench_dataset = {}
            with open(args.dataset) as f:
                for line in f:
                    entry = json.loads(line)
                    specbench_dataset[entry["question_id"]] = entry
        except Exception as e:
            print(f"WARN: SpecBench dataset load failed: {e}",
                  file=sys.stderr)

    all_requests = extract_requests(data, exclude_ids, dm_by_id,
                                    tokenizer, bfcl_dataset, resp_by_id,
                                    specbench_dataset)
    print(f"Requests: {len(all_requests)}", file=sys.stderr)

    # Load MTP round data if provided
    mtp_all_requests = None
    if args.mtp_agent_results:
        print(f"Loading MTP data: {args.mtp_agent_results}", file=sys.stderr)
        with open(args.mtp_agent_results) as f:
            mtp_data = json.load(f)
        mtp_all_requests = extract_requests(mtp_data, exclude_ids)
        print(f"MTP requests: {len(mtp_all_requests)}", file=sys.stderr)

    # Train/test split
    if args.train_ratio > 0:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for req in all_requests:
            by_cat[req["category"]].append(req)
        train_requests, test_requests = [], []
        for cat in sorted(by_cat):
            reqs = by_cat[cat]
            n_train = int(len(reqs) * args.train_ratio)
            train_requests.extend(reqs[:n_train])
            test_requests.extend(reqs[n_train:])
    else:
        train_requests = []
        test_requests = all_requests

    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
    if train_requests:
        print(f"Warming up suffix cache with {len(train_requests)} requests...",
              file=sys.stderr)
        warmup_id = 0
        for req in train_requests:
            for call_idx in range(len(req["per_call_tokens"])):
                tokens = req["per_call_tokens"][call_idx]
                if not tokens:
                    continue
                prompt_ids_list = req.get("per_call_prompt_ids")
                if prompt_ids_list and call_idx < len(prompt_ids_list):
                    prompt = np.array(prompt_ids_list[call_idx], dtype=np.int32)
                else:
                    prompt = np.array([], dtype=np.int32)
                cache.start_request(warmup_id, prompt)
                cache.add_active_response(warmup_id, tokens)
                cache.stop_request(warmup_id)
                warmup_id += 1

    # Filter MTP requests to match test set
    mtp_test = None
    if mtp_all_requests:
        test_ids = {r["bfcl_id"] for r in test_requests}
        mtp_test = [r for r in mtp_all_requests if r["bfcl_id"] in test_ids]
        print(f"MTP test requests: {len(mtp_test)}", file=sys.stderr)

    t0 = time.time()
    print(f"Collecting union tries for {len(test_requests)} requests...",
          file=sys.stderr)
    records = collect_union_tries(test_requests, cache, mtp_requests=mtp_test)
    elapsed = time.time() - t0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    total_union_nodes = sum(
        len(r["union_trie"]["token_ids"]) for r in records)
    proposer_counts = {}
    for r in records:
        for name in r["per_proposer"]:
            proposer_counts[name] = proposer_counts.get(name, 0) + 1

    print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)
    print(f"Steps: {len(records)}", file=sys.stderr)
    print(f"Total union trie nodes: {total_union_nodes:,} "
          f"(avg {total_union_nodes / max(len(records), 1):.1f}/step)",
          file=sys.stderr)
    for name, count in sorted(proposer_counts.items()):
        print(f"  {name}: {count} steps with drafts", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
