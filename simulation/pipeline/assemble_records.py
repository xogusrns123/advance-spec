"""Per-step record assembly library used by Stage 3 (oracle simulation).

The Stage 3 driver (``run_tree_oracle_sim.py``) calls
``assemble_records_from_artifacts()`` to read Stage 1's
``agent_results_eagle3.json`` plus Stage 2's ``draft_model_drafts.jsonl``
and emit an in-memory list of per-(request, call, step) records ready for
simulation. Stage 3 draws suffix candidates live inside the simulator
(``_live_suffix_draft``); no per-step suffix file needs to exist.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.pipeline._agent_io import (
    _flat_to_tree,
    extract_requests,
    load_exclude_ids,
)


# ---------------------------------------------------------------------------
# Per-step draft loaders (Stage 2 outputs)
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

def collect_step_records(
    requests: List[dict],
    suffix_by_key: Optional[Dict[Tuple[str, int, int], dict]] = None,
    dm_by_key: Optional[Dict[Tuple[str, int, int], dict]] = None,
    mtp_requests: Optional[List[dict]] = None,
    eagle3_reslice: Optional[Tuple[int, int, int, int]] = None,
) -> List[dict]:
    """Assemble per-step records for all requests.

    Each emitted record carries ``per_proposer`` (eagle3 / suffix /
    draft_model / mtp draft trees), ``ground_truth_future`` (the remaining
    target tokens at this step), and ``context_token_ids`` (prompt + tokens
    already decoded). Stage 3 consumes this list directly.

    Parameters
    ----------
    requests : list[dict]
        Primary requests (EAGLE3 round) from extract_requests.
    suffix_by_key, dm_by_key : dict, optional
        Per-step draft records keyed by (request_id, call_idx, step_idx).
    mtp_requests : list[dict], optional
        MTP round requests (same bfcl_ids, same token sequences). If
        provided, MTP drafts are attached to per_proposer.
    eagle3_reslice : tuple, optional
        ``(S_orig, K_orig, s_prime, k_prime)`` — when set and the request
        carries ``per_call_eagle3_pool_fulls`` (= captured with
        ``SGLANG_CAPTURE_FULL_POOL=1``), the per-step EAGLE3 tree is
        REPLACED with one resliced from the full pool to (s', k').
        ``S_orig`` and ``K_orig`` must match the capture's config (e.g.
        ``(8, 16)``). Steps where pool data is missing fall back to the
        original truncated tree.
    """
    suffix_by_key = suffix_by_key or {}
    dm_by_key = dm_by_key or {}

    reslice_args = None
    if eagle3_reslice is not None:
        S_orig, K_orig, s_prime, k_prime = eagle3_reslice
        from simulation.pipeline.pool_reslicer import reslice_eagle3_pool
        reslice_args = (reslice_eagle3_pool, S_orig, K_orig, s_prime, k_prime)

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

                # EAGLE3: prefer resliced full pool (when reslice config is set
                # AND pool was captured), else the truncated tree, else flat chain.
                eagle3_p_t = None
                eagle3_path_draft_p_t = None
                e3_trees = req.get("per_call_eagle3_trees")
                e3_p_ts = req.get("per_call_eagle3_tree_p_ts")
                e3_draft_p_ts = req.get(
                    "per_call_eagle3_tree_path_draft_p_ts")
                e3_pool_fulls = req.get("per_call_eagle3_pool_fulls")
                e3_attached = False

                # Reslice path: when (s', k') override active and full pool present
                if (reslice_args is not None and e3_pool_fulls
                        and call_idx < len(e3_pool_fulls)):
                    call_pools = e3_pool_fulls[call_idx]
                    if (pos < len(call_pools)
                            and call_pools[pos] is not None):
                        fp = call_pools[pos]
                        try:
                            (reslicer_fn, S_orig, K_orig,
                             s_p, k_p) = reslice_args
                            sub_ids, sub_par, sub_pp = reslicer_fn(
                                fp["draft_tokens"], fp["parent_list"],
                                fp["path_probs"], fp["pool_size"],
                                S_orig, K_orig, s_p, k_p)
                            proposer_trees["eagle3"] = (sub_ids, sub_par)
                            eagle3_path_draft_p_t = sub_pp
                            e3_attached = True
                        except Exception as e:
                            # Reslicer failed (corrupt entry?) — fall through
                            # to original tree path; logged once.
                            if not getattr(collect_step_records,
                                           "_reslice_warned", False):
                                print(f"WARN: pool reslice failed at "
                                      f"({bfcl_id}, call {call_idx}, "
                                      f"step {pos}): {e}", file=sys.stderr)
                                collect_step_records._reslice_warned = True

                if not e3_attached and e3_trees and call_idx < len(e3_trees):
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

                # Suffix: looked up from a precomputed draft file (legacy
                # path; current pipeline draws suffix live inside Stage 3).
                key = (bfcl_id, call_idx, pos)
                suffix_rec = suffix_by_key.get(key)
                suffix_score = 0.0
                if suffix_rec and suffix_rec.get("token_ids"):
                    proposer_trees["suffix"] = (
                        list(suffix_rec["token_ids"]),
                        list(suffix_rec["parents"]),
                    )
                    suffix_score = float(suffix_rec.get("score", 0.0))

                # Draft model: looked up from Stage 2 output
                dm_rec = dm_by_key.get(key)
                if dm_rec and dm_rec.get("token_ids"):
                    proposer_trees["draft_model"] = (
                        list(dm_rec["token_ids"]),
                        list(dm_rec["parents"]),
                    )

                # MTP: from optional MTP-replay round
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
    eagle3_reslice: Optional[Tuple[int, int, int, int]] = None,
) -> List[dict]:
    """End-to-end loader: read Stage 1/2 artifacts, return per-step records.

    Mirrors the side effects of ``main()`` (tokenizer load, BFCL/SpecBench
    prompt reconstruction, per-step JSONL loading, MTP extraction) and
    produces the per-step record list that Stage 3 consumes directly.
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
    print(f"Assembling per-step records for {len(all_requests)} requests...",
          file=sys.stderr)
    records = collect_step_records(
        all_requests,
        suffix_by_key=suffix_by_key,
        dm_by_key=dm_by_key,
        mtp_requests=mtp_all_requests,
        eagle3_reslice=eagle3_reslice,
    )
    elapsed = time.time() - t0
    print(f"  assembled {len(records)} step records in {elapsed:.1f}s",
          file=sys.stderr)
    return records
