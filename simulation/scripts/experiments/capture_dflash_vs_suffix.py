#!/usr/bin/env python3
"""OFFLINE capture of DFlash-vs-suffix per-depth decisions for chain-hybrid
selection-accuracy + MAT analysis. DFlash is a block drafter (no live select-1),
but the decision surface + counterfactual select-1 MAT ARE measurable offline
against ground truth.

Two collection modes:
  --mode agentic (default): run the SAME bfcl_v4 multi-turn tool-calling agent
      loop as the MTP/EAGLE3 chain-hybrid captures (process_request), but with
      generation done by an offline HF DFlash decoder instead of an sglang
      server. Each turn's block-draft per-position probs are captured; after the
      trajectory finishes, the suffix trie is replayed over the full committed
      token stream (prompt + every turn's generation + tool results) and queried
      PER-DEPTH autoregressively (temporary_extension), identical to
      chain_hybrid_patch. This makes DFlash trajectories agentic + comparable to
      MTP/EAGLE3 (which run through the agent).
  --mode legacy: single-turn greedy decode of the turn-1 prompt only (no tools).

Row: {rid, round, depth, dflash_p, dflash_tok, suffix_p, suffix_tok,
      suffix_match_len, gt_tok, oracle_hit}.

Run INSIDE sglang-bench on Blackwell GPU0 (sdpa target):
  CUDA_VISIBLE_DEVICES=0 python3 simulation/scripts/experiments/capture_dflash_vs_suffix.py \
    --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 --cell qwen3_8b
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

_VENDOR = "/workspace/vendor/ddtree"
sys.path.insert(0, _VENDOR)
sys.path.insert(0, "/workspace/simulation/agents")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from model import DFlashDraftModel, extract_context_feature  # noqa: E402

REPO = Path("/workspace")
DATASET = REPO / "data/bfcl_agent/dataset_stratified_interleaved.jsonl"


def slug(model: str) -> str:
    return model.split("/")[-1].lower().replace(".", "").replace("-", "_")


def load_draft(draft_path, dev):
    cfg = AutoConfig.from_pretrained(draft_path, trust_remote_code=True)
    if getattr(cfg, "block_size", None) is None:
        bs = (getattr(cfg, "dflash_config", None) or {}).get("block_size")
        if bs is not None:
            cfg.block_size = bs
    return DFlashDraftModel.from_pretrained(
        draft_path, config=cfg, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()


def load_web_search_entries(n_tasks, offset):
    """Full BFCLv4 web_search dataset entries [offset:offset+n_tasks]
    (question + function + initial_config + involved_classes), for the agent."""
    out = []; seen = 0
    with open(DATASET) as f:
        for line in f:
            e = json.loads(line)
            if "web_search" not in (e.get("category") or ""):
                continue
            if seen < offset:
                seen += 1; continue
            out.append(e)
            if len(out) >= n_tasks:
                break
    if not out:
        raise SystemExit("no web_search tasks at this offset")
    return out


def _chat_ids(tok, messages, dev):
    """Render messages -> prompt token ids. Split tokenize=False + encode to
    avoid the BatchEncoding return of apply_chat_template(tokenize=True) on
    Qwen3.5/transformers 5.5 (see feedback_chat_template_tokenize)."""
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return torch.tensor([tok.encode(text)], device=dev), text


@torch.inference_mode()
def dflash_decode(model, target, tok, prompt_ids, max_new, stop_strings, eos_ids,
                  num_input_offset=0):
    """Greedy DFlash decode from prompt_ids. Captures, per round, the whole
    drafted block (depths 0..b-2) with softmax prob + the round's start offset
    (relative to the generated span). Stops at eos or when a stop string appears
    in the generated text (round-aligned). Returns (content_str, gen_token_ids,
    rounds[{start_in_turn,dtoks,dps}])."""
    dev = model.device; b = model.block_size; mask_id = model.mask_token_id
    num_input = prompt_ids.shape[1]; max_len = num_input + max_new
    out_ids = torch.full((1, max_len + b), mask_id, dtype=torch.long, device=dev)
    pos = torch.arange(out_ids.shape[1], device=dev).unsqueeze(0)
    eos_t = torch.tensor(eos_ids, device=dev)
    pkv_t = DynamicCache(); pkv_d = DynamicCache()

    o = target(prompt_ids, position_ids=pos[:, :num_input], past_key_values=pkv_t,
               use_cache=True, logits_to_keep=1, output_hidden_states=b > 1)
    out_ids[:, :num_input] = prompt_ids
    out_ids[:, num_input:num_input + 1] = torch.argmax(o.logits, dim=-1)
    th = extract_context_feature(o.hidden_states, model.target_layer_ids) if b > 1 else None

    rounds = []; start = num_input
    while start < max_len:
        block = out_ids[:, start:start + b].clone()
        if b > 1:
            ne = target.model.embed_tokens(block)
            dh = model(target_hidden=th, noise_embedding=ne,
                       position_ids=pos[:, pkv_d.get_seq_length():start + b],
                       past_key_values=pkv_d, use_cache=True, is_causal=False)[:, -b + 1:, :]
            dl = target.lm_head(dh); pkv_d.crop(start)
            probs = torch.softmax(dl.float(), dim=-1); dpmax, dtok = probs.max(dim=-1)
            block[:, 1:] = dtok
            rounds.append({"start_in_turn": int(start - num_input),
                           "dtoks": [int(x) for x in dtok[0].tolist()],
                           "dps": [float(x) for x in dpmax[0].tolist()]})
        o = target(block, position_ids=pos[:, start:start + b], past_key_values=pkv_t,
                   use_cache=True, output_hidden_states=b > 1)
        post = torch.argmax(o.logits, dim=-1)
        acc = int((block[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item())
        out_ids[:, start:start + acc + 1] = block[:, :acc + 1]
        out_ids[:, start + acc + 1] = post[:, acc]
        start += acc + 1
        pkv_t.crop(start)
        if b > 1:
            th = extract_context_feature(o.hidden_states, model.target_layer_ids)[:, :acc + 1, :]
        gen = out_ids[0, num_input:start]
        if torch.isin(gen[-(acc + 2):], eos_t).any():
            break
        if stop_strings and any(s in tok.decode(gen.tolist()) for s in stop_strings):
            break
    gen_tokens = [int(x) for x in out_ids[0, num_input:start].tolist()]
    return tok.decode(gen_tokens), gen_tokens, rounds


# ---- suffix phase-B over a committed token stream ------------------------------
def suffix_replay(rid, stream, global_rounds, cache, block_size, D, rows, warn):
    """Per-depth autoregressive suffix query + GT join over an ordered committed
    token stream. global_rounds = [{root, dtoks, dps}] (root = absolute position
    of the block root in `stream`)."""
    cache.start_request(rid, np.asarray(stream[:1], dtype=np.int32))
    warmed = 1
    for gr in sorted(global_rounds, key=lambda r: r["root"]):
        root = gr["root"]
        while warmed <= root and warmed < len(stream):
            cache.add_active_response(rid, [int(stream[warmed])]); warmed += 1
        base_tail = stream[max(0, root + 1 - D):root + 1]
        for d in range(block_size - 1):
            gtpos = root + 1 + d
            if gtpos >= len(stream) or d >= len(gr["dtoks"]):
                break
            gt = int(stream[gtpos]); d_tok = gr["dtoks"][d]; d_p = gr["dps"][d]
            chain = stream[root + 1:root + 1 + d]
            ctx = np.asarray((base_tail + chain)[-D:], dtype=np.int32)
            s_tok = s_p = s_ml = None
            try:
                with cache.temporary_extension(rid, chain):
                    sd = cache.speculate(rid, ctx, max_spec_tokens=1, use_tree_spec=False)
                s_ml = int(getattr(sd, "match_len", 0) or 0)
                if sd.token_ids is not None and len(sd.token_ids):
                    s_tok = int(sd.token_ids[0])
                    s_p = float(sd.probs[0]) if (sd.probs is not None and len(sd.probs)) else 0.0
            except Exception as e:
                if warn[0]:
                    print(f"  WARN suffix: {type(e).__name__}: {e}", flush=True); warn[0] = False
            e_hit = (d_tok == gt); s_hit = (s_tok is not None and s_tok == gt)
            oh = "both" if (e_hit and s_hit) else "eagle" if e_hit else "suffix" if s_hit else "none"
            rows.append({"rid": rid, "round": gr["root"], "depth": d, "dflash_p": d_p,
                         "dflash_tok": d_tok, "suffix_p": s_p, "suffix_tok": s_tok,
                         "suffix_match_len": s_ml, "gt_tok": gt, "oracle_hit": oh})
    cache.stop_request(rid)


def capture_agentic(draft, target, tok, cache, entries, max_new, max_iter, eos_ids, rows):
    """Run the bfcl_v4 agent loop with DFlash generation; reconstruct the
    committed token stream per task; suffix-replay it."""
    from bfcl_v4_agent import process_request, STOP_AFTER_CALL
    b = draft.block_size; D = cache.max_tree_depth; dev = draft.device; warn = [True]
    for i, entry in enumerate(entries):
        turns = []

        def gen(messages, step, _t=turns):
            pids, _ = _chat_ids(tok, messages, dev)
            content, ctoks, rnds = dflash_decode(draft, target, tok, pids, max_new,
                                                  STOP_AFTER_CALL, eos_ids)
            _t.append({"ctoks": ctoks, "rounds": rnds})
            return content

        result = process_request(client=None, model="dflash", request=entry,
                                 max_iterations=max_iter, collect_oracle=False, generate_fn=gen)
        steps = result.get("agent_metrics", {}).get("steps", [])
        if not steps or not turns:
            print(f"  task {i}: no steps", flush=True); continue

        # committed stream = turn-0 prompt + each turn's generation + its tool results
        _, p0 = _chat_ids(tok, steps[0]["messages"], dev)
        stream = list(tok.encode(p0)); global_rounds = []
        for k, turn in enumerate(turns):
            base = len(stream); stream.extend(turn["ctoks"])
            for r in turn["rounds"]:
                global_rounds.append({"root": base + r["start_in_turn"],
                                      "dtoks": r["dtoks"], "dps": r["dps"]})
            if k < len(steps):
                for er in (steps[k].get("exec_results") or []):
                    stream.extend(tok.encode(str(er)))
        suffix_replay(i, stream, global_rounds, cache, b, D, rows, warn)
        print(f"  task {i}: {len(turns)} turns, stream={len(stream)} tok, rows={len(rows)}", flush=True)


# ---- legacy single-turn (no agent / tools) -------------------------------------
def capture_legacy(draft, target, tok, cache, prompts, max_new, eos_ids, rows):
    b = draft.block_size; D = cache.max_tree_depth; dev = draft.device; warn = [True]
    for i, text in enumerate(prompts):
        pids = torch.tensor([tok.encode(text)], device=dev)
        _, gen_tokens, rounds = dflash_decode(draft, target, tok, pids, max_new, None, eos_ids)
        prompt_tokens = tok.encode(text)
        stream = list(prompt_tokens) + gen_tokens
        base = len(prompt_tokens)
        global_rounds = [{"root": base + r["start_in_turn"], "dtoks": r["dtoks"], "dps": r["dps"]}
                         for r in rounds]
        suffix_replay(i, stream, global_rounds, cache, b, D, rows, warn)
        print(f"  task {i}: {len(rounds)} rounds, rows={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cell", default="qwen3_8b")
    ap.add_argument("--mode", choices=["agentic", "legacy"], default="agentic")
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--offset", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--out", default="simulation/results/chain_hybrid_perdepth")
    args = ap.parse_args()

    dev = "cuda"
    print(f"loading target {args.target} (sdpa, bf16) ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    print(f"loading draft {args.draft} ...", flush=True)
    draft = load_draft(args.draft, dev)
    tok = AutoTokenizer.from_pretrained(args.target)
    print(f"block_size={draft.block_size} mask_token_id={draft.mask_token_id} mode={args.mode}", flush=True)

    from arctic_inference.suffix_decoding import SuffixDecodingCache
    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000, enable_undo=True)
    eos_ids = [tok.eos_token_id]

    rows = []
    if args.mode == "agentic":
        entries = load_web_search_entries(args.n_tasks, args.offset)
        print(f"{len(entries)} web_search entries (offset {args.offset})", flush=True)
        capture_agentic(draft, target, tok, cache, entries, args.max_new_tokens,
                        args.max_iter, eos_ids, rows)
    else:
        ents = load_web_search_entries(args.n_tasks, args.offset)
        prompts = [tok.apply_chat_template(e["question"][0], tokenize=False,
                   add_generation_prompt=True, enable_thinking=False) for e in ents]
        capture_legacy(draft, target, tok, cache, prompts, args.max_new_tokens, eos_ids, rows)

    out_dir = (REPO / args.out) / f"{args.cell}_dflash"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decisions_dflash_suffix.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"-> {out_path}  ({len(rows)} rows)\nDFLASH_SUFFIX_CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
