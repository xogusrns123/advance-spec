#!/usr/bin/env python3
"""OFFLINE DFlash block-draft over a gt trajectory, re-speculating a block at every
committed root — the DFlash-aux proposer for the 27B 3-way ceiling (MTP + DFlash +
suffix on Qwen3.5-27B). DFlash-27B cannot serve as the sglang main worker (Mamba
hybrid: the verify needs a KV crop the linear-attn cache can't do), so under option
B (gt-substitution, all arms on the identical gt path) we get DFlash's per-position
proposals by running it DECOUPLED on the gt trajectory: one use_cache=False target
forward over gt (captures the DFlash target layers), then mirror the vendor
spec_generate block draft (DynamicCache + crop) but TEACHER-FORCE the gt commits
(the served accept pattern from the main-worker log) instead of DFlash's own accept.

Faithfulness is validated on the 8B (where DFlash DOES serve): compare these offline
proposals to a served DFlash ceiling log's eagle_token (= DFlash's served proposal),
just as eagle3_offline.py was validated. Then apply to 27B.

  # validate on 8B:
  python3 dflash_offline.py --validate --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
      --record-dir <8B dflash ceiling dir> --decisions-file decisions_select1_oracle.jsonl
  # emit 27B proposals merged-by (rid,decode_step,depth):
  python3 dflash_offline.py --emit --target Qwen/Qwen3.5-27B --draft z-lab/Qwen3.5-27B-DFlash \
      --record-dir <27B mtp ceiling dir> --decisions-file decisions_select1_oracle.jsonl \
      --out <dir>/dflash_proposals.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/vendor/ddtree")
sys.path.insert(0, "/workspace/vendor/ddtree/model")


def _load_record(record_dir, decisions_file=None):
    """gt_tokens.jsonl + per-(rid) decision/step rows (same schema as the served
    chain-hybrid logs). Returns gt{input_ids->output_ids}, reqs, decisions, steps."""
    dd = Path(record_dir)
    if decisions_file:
        log = dd / decisions_file
    else:
        log = next((dd / c for c in ("decisions_select1_oracle.jsonl",
                                     "decisions_record.jsonl",
                                     "decisions_select1.jsonl") if (dd / c).exists()), None)
        if log is None:
            hits = sorted(dd.glob("decisions_*.jsonl")); log = hits[0] if hits else None
    if log is None or not log.exists():
        raise SystemExit(f"no decisions log in {record_dir}")
    gt = {}
    if (dd / "gt_tokens.jsonl").exists():
        for line in open(dd / "gt_tokens.jsonl"):
            r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
    reqs, decisions, steps = {}, {}, {}
    for line in open(log):
        r = json.loads(line); t = r.get("type")
        if t == "req":
            reqs[r["rid"]] = r["input_ids"]
        elif t == "decision":
            decisions.setdefault(r["rid"], []).append(r)
        elif t == "step":
            steps.setdefault(r["rid"], {})[r["decode_step"]] = r.get("accept_len")
    return gt, reqs, decisions, steps, log


class DFlashOffline:
    def __init__(self, target_name, draft_name, device="cuda", dtype=torch.bfloat16,
                 _skip_target=False):
        from model import DFlashDraftModel  # vendor/ddtree/model
        self.device = device; self.dtype = dtype
        self.target = None
        if not _skip_target:
            print(f"loading target {target_name} (sdpa, bf16) ...", flush=True)
            self.target = AutoModelForCausalLM.from_pretrained(
                target_name, attn_implementation="sdpa", dtype=dtype).to(device).eval()
            self.embed = self.target.model.embed_tokens
            self.lm_head_weight = self.target.lm_head.weight
            self.vocab = int(self.target.get_input_embeddings().num_embeddings)
        if draft_name is not None:
            print(f"loading draft {draft_name} (sdpa, bf16) ...", flush=True)
            self.draft = DFlashDraftModel.from_pretrained(
                draft_name, attn_implementation="sdpa", dtype=dtype).to(device).eval()
            self.block_size = self.draft.block_size
            self.mask_token_id = self.draft.mask_token_id
            self.layer_ids = self.draft.target_layer_ids
            print(f"block_size={self.block_size} mask={self.mask_token_id} "
                  f"target_layer_ids={self.layer_ids}", flush=True)

    @classmethod
    def from_served(cls, draft_name, served_model, device="cuda", dtype=torch.bfloat16):
        """In-serving: reuse the ALREADY-LOADED sglang target's embedding + lm_head
        weight (no second 27B target -> no OOM); only the small DFlash draft is
        loaded. Target FEATURES are captured live via forward hooks on the served
        target (see chain_hybrid_patch), NOT via this object's target forward."""
        self = cls(None, draft_name, device=device, dtype=dtype, _skip_target=True)
        self.embed = served_model.get_input_embeddings()
        self.lm_head_weight = served_model.lm_head.weight
        # Clamp the argmax to the BASE ("org") vocab shard, NOT config.vocab_size.
        # sglang's ParallelLMHead / VocabParallelEmbedding store the vocab as a
        # SHARDED, PADDED weight: [base tokens (0..num_org) | padding | added/special
        # tokens (at a SHIFTED offset)]. A raw `hidden @ weight.T` argmax returns a
        # WEIGHT ROW INDEX that equals a valid contiguous token id ONLY in the base
        # region; a row index in the padding/added region is not a token id and, once
        # injected into the MTP draft chain, OOBs the draft's embed_tokens
        # (qwen3_5_mtp.py:166 -> F.embedding -> indexSelectSmallIndex device assert).
        # Mirror _install_dflash_prob_stash (chain_hybrid_patch.py): use the lm_head's
        # shard_indices.num_org_elements. Added/special tokens are excluded
        # (approximate for a draft proposer, but matches the existing dflash_p path
        # and is crash-safe). Fallback to min(config, embed, lm_head) if unavailable.
        cfg = getattr(served_model, "config", None)
        cvocab = getattr(cfg, "vocab_size", None) if cfg is not None else None
        emb_n = int(self.embed.num_embeddings)
        lm_n = int(self.lm_head_weight.shape[0])
        num_org = None
        try:
            num_org = int(served_model.lm_head.shard_indices.num_org_elements)
        except Exception:
            num_org = None
        cands = [v for v in (num_org, cvocab, emb_n, lm_n) if v]
        self.vocab = min(cands) if cands else min(emb_n, lm_n)
        print(f"DFlash-aux from_served: embed={emb_n} lm_head={lm_n} "
              f"config_vocab={cvocab} num_org={num_org} -> using vocab={self.vocab}",
              flush=True)
        return self

    def _lm_logits(self, hidden):
        # Slice to the REAL vocab: the served (sglang) lm_head pads the vocab to a
        # multiple, and an argmax into a padded slot (>= embed vocab) would later
        # OOB the embedding. HF lm_head is exact so this is a no-op offline.
        logits = hidden.to(self.lm_head_weight.dtype) @ self.lm_head_weight.T
        return logits[..., :self.vocab]

    def _ctx_feature(self, hidden_states):
        """extract_context_feature: cat(hidden_states[L+1]) for L in target_layer_ids."""
        return torch.cat([hidden_states[L + 1] for L in self.layer_ids], dim=-1)

    @torch.inference_mode()
    def _capture_target_hidden(self, t):
        """Capture ONLY the DFlash target layers via forward hooks (avoids
        output_hidden_states=True holding all num_layers+1 hidden states, which
        OOMs on long sequences). Hooking layer L's OUTPUT == HF hidden_states[L+1]
        == extract_context_feature's hidden_states[L+1]."""
        captured = {}
        hooks = []
        for L in self.layer_ids:
            layer = self.target.model.layers[L]

            def hook(mod, inp, out, _L=L):
                h = out[0] if isinstance(out, tuple) else out
                captured[_L] = h.detach()
            hooks.append(layer.register_forward_hook(hook))
        try:
            self.target(t, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return torch.cat([captured[L] for L in self.layer_ids], dim=-1)  # [1,S,K*H]

    @torch.inference_mode()
    def proposals_for_seq(self, seq, prompt_len, commit_lens, max_seq=20000,
                          with_p2=False):
        """Re-speculate a DFlash block at every committed root along `seq` (a list of
        ids = prompt + gt output). commit_lens[s] = accept_len_s + 1 (the served main
        worker's commit length at decode step s) drives the TEACHER-FORCED advance, so
        the blocks are rooted at the SAME positions as the served arm.
        Returns list of (decode_step, depth, token, prob) where depth d is the block's
        prediction for position root+1+d. with_p2: append the SECOND-highest prob
        (7th tuple slot) — the margin signal for argmax-agreement analysis."""
        from transformers.cache_utils import DynamicCache
        if max(seq) >= self.vocab:
            print(f"  [skip] sequence has OOB token (max={max(seq)} >= vocab "
                  f"{self.vocab}); corrupted request, skipping", flush=True)
            return []
        if len(seq) > max_seq:
            print(f"  [skip] sequence len {len(seq)} > {max_seq}; would OOM the "
                  f"single full-seq target forward, skipping", flush=True)
            return []
        t = torch.tensor([seq], device=self.device)
        th_full = self._capture_target_hidden(t)                # [1, S, K*H]
        S = t.shape[1]
        position_ids = torch.arange(S + self.block_size, device=self.device).unsqueeze(0)
        draft_kv = DynamicCache()
        recs = []
        start = prompt_len                                      # root of block 0 (prefill bonus pos)
        target_hidden = th_full[:, :prompt_len, :]              # prefill context features
        for ds, clen in enumerate(commit_lens, start=1):
            if start >= S:
                break
            end = min(start + self.block_size, S + self.block_size)
            # block_output_ids: seed at index 0 (= seq[start]); rest masked.
            block_ids = torch.full((1, self.block_size), self.mask_token_id,
                                   dtype=torch.long, device=self.device)
            avail = min(self.block_size, S - start)
            block_ids[0, :avail] = torch.tensor(seq[start:start + avail], device=self.device)
            block_ids[0, 1:] = self.mask_token_id              # only the seed is given
            block_ids[0, 0] = seq[start]
            noise_emb = self.embed(block_ids)
            pos = position_ids[:, draft_kv.get_seq_length(): start + self.block_size]
            hs = self.draft(
                target_hidden=target_hidden, noise_embedding=noise_emb,
                position_ids=pos, past_key_values=draft_kv, use_cache=True)
            draft_logits = self._lm_logits(hs[:, -self.block_size + 1:, :])
            draft_kv.crop(start)
            probs = torch.softmax(draft_logits.float(), dim=-1)
            pmax, toks = probs.max(dim=-1)                      # [1, block-1]
            top2v = probs.topk(2, dim=-1).values if with_p2 else None
            for d in range(self.block_size - 1):
                pos_pred = start + 1 + d
                if pos_pred >= S:
                    break
                # TREE (DDTree) signal: rank of the gt token in the depth-d block
                # distribution. tree accept with top-k budget = first depth where
                # gt_rank >= k (gt not among the top-k candidates). rank 0 = argmax
                # (== the linear-chain accept). gt_p = the gt token's prob mass.
                gt_tok = seq[pos_pred]
                logits_d = draft_logits[0, d]
                gt_rank = int((logits_d > logits_d[gt_tok]).sum().item())
                gt_p = float(probs[0, d, gt_tok].item())
                row = (ds, d, int(toks[0, d].item()),
                       float(pmax[0, d].item()), gt_rank, gt_p)
                if with_p2:
                    row = row + (float(top2v[0, d, 1].item()),)
                recs.append(row)
            # teacher-force commit: advance by the served commit length
            nxt = start + clen
            target_hidden = th_full[:, start:nxt, :]           # newly committed features
            start = nxt
        return recs


class DFlashInLoop:
    """Stateful in-serving DFlash block drafter — the same incremental block draft
    as DFlashOffline.proposals_for_seq (validated vs served at 99.75% hi-conf on 8B),
    split into per-step calls so it can be driven by the LIVE served loop:
      reset(prompt_aux)                     # request start; prompt DFlash features
      draft_block(seed_token, root)         # -> [(depth, token, prob)] for root+1+d
      commit(committed_aux, clen)           # feed the just-committed features, advance
    `*_aux` tensors are the EAGLE3/DFlash target-layer concat (extract_context_feature)
    captured live. Mirrors vendor spec_generate's DynamicCache + crop exactly."""

    def __init__(self, dfo):
        from transformers.cache_utils import DynamicCache
        self._DynamicCache = DynamicCache
        self.dfo = dfo
        self.block_size = dfo.block_size
        self.reset(None, 0)

    def reset(self, prompt_aux, prompt_len):
        self.draft_kv = self._DynamicCache()
        self.start = prompt_len                 # current block root (abs position)
        self.pending = prompt_aux               # committed features fed to the next block
        dev = self.dfo.device
        self.position_ids = torch.arange(1 << 20, device=dev).unsqueeze(0) \
            if prompt_aux is not None else None

    @torch.inference_mode()
    def draft_block(self, seed_token):
        """Draft one block from self.start (seed = last committed token). Returns
        [(depth d, token, prob)] for predictions of positions start+1+d."""
        if self.pending is None or self.position_ids is None:
            return []
        b = self.block_size
        # Consistency guard: the block forward needs pending(ctx) + b queries to be
        # contiguous after the cropped KV, i.e. pending_len == start - kv_len. If a
        # commit desynced this (rare trajectory edge), skip WITHOUT touching the KV
        # so one bad step can't corrupt the rest of the request.
        if self.pending.shape[1] != self.start - self.draft_kv.get_seq_length():
            import os as _os
            if _os.environ.get("SGLANG_DFA_DEBUG") == "1":
                print(f"[DFADBG SKIP] start={self.start} kv_len={self.draft_kv.get_seq_length()} "
                      f"pending_len={self.pending.shape[1]} (need {self.start - self.draft_kv.get_seq_length()})",
                      flush=True)
            return []
        block_ids = torch.full((1, b), self.dfo.mask_token_id, dtype=torch.long,
                               device=self.dfo.device)
        block_ids[0, 0] = int(seed_token)
        noise = self.dfo.embed(block_ids).reshape(1, b, -1)   # served embed may flatten
        pos = self.position_ids[:, self.draft_kv.get_seq_length(): self.start + b]
        import os as _os
        if _os.environ.get("SGLANG_DFA_DEBUG") == "1":
            print(f"[DFADBG draft] start={self.start} kv_len={self.draft_kv.get_seq_length()} "
                  f"pos_len={pos.shape[1]} pending_len={self.pending.shape[1]} b={b}", flush=True)
        hs = self.dfo.draft(target_hidden=self.pending, noise_embedding=noise,
                            position_ids=pos, past_key_values=self.draft_kv, use_cache=True)
        logits = self.dfo._lm_logits(hs[:, -b + 1:, :])
        self.draft_kv.crop(self.start)
        probs = torch.softmax(logits.float(), dim=-1)
        pmax, toks = probs.max(dim=-1)
        return [(d, int(toks[0, d].item()), float(pmax[0, d].item())) for d in range(b - 1)]

    def commit(self, committed_aux, clen):
        """Feed the features of the tokens committed THIS step; advance the root."""
        self.pending = committed_aux
        self.start += clen


def _commit_lens(rid, steps):
    """ordered commit lengths (accept_len+1) per decode step."""
    sd = steps.get(rid, {})
    return [(sd[s] + 1) if sd.get(s) is not None else 1 for s in sorted(sd)]


def validate(args):
    dfo = DFlashOffline(args.target, args.draft)
    gt, reqs, decisions, steps, log = _load_record(args.record_dir, args.decisions_file)
    print(f"record: {log} | reqs={len(reqs)} gt_rows={len(gt)}", flush=True)
    per = defaultdict(lambda: {"m": 0, "n": 0, "him": 0, "hin": 0, "pabs": 0.0, "np": 0})
    for rid, ids in reqs.items():
        out = gt.get(tuple(ids))
        if out is None:
            continue
        seq = list(ids) + list(out)
        clens = _commit_lens(rid, steps)
        recs = dfo.proposals_for_seq(seq, len(ids), clens)
        mine = {(ds, d): (tok, p) for ds, d, tok, p in recs}
        bystep = defaultdict(dict)
        for r in decisions.get(rid, []):
            bystep[r["decode_step"]][r["depth"]] = r
        for ds in bystep:
            for d, r in bystep[ds].items():
                srv = r.get("eagle_token")           # DFlash served proposal (legacy field)
                if srv is None or (ds, d) not in mine:
                    continue
                tok, p = mine[(ds, d)]
                pe = per[d]; pe["n"] += 1; pe["m"] += (tok == int(srv))
                sp = r.get("eagle_p")
                if sp is not None:
                    pe["pabs"] += abs(p - float(sp)); pe["np"] += 1
                    if float(sp) > 0.5:
                        pe["hin"] += 1; pe["him"] += (tok == int(srv))
    tm = tn = thm = thn = 0
    print(f"\n{'depth':>5} {'match':>13} {'rate':>7} {'hi-conf':>13} {'hi-rate':>7} {'|dp|':>7}")
    for d in sorted(per):
        pe = per[d]; rate = pe["m"]/max(pe["n"],1); hr = pe["him"]/max(pe["hin"],1)
        dp = pe["pabs"]/max(pe["np"],1)
        print(f"{d:>5} {pe['m']:>6}/{pe['n']:<6} {rate:>7.4f} {pe['him']:>6}/{pe['hin']:<6} {hr:>7.4f} {dp:>7.4f}")
        tm += pe["m"]; tn += pe["n"]; thm += pe["him"]; thn += pe["hin"]
    print(f"\n=== OVERALL token match: {tm}/{tn} = {tm/max(tn,1):.4f} ===")
    if thn:
        print(f"=== high-conf(served p>0.5) match: {thm}/{thn} = {thm/max(thn,1):.4f} ===")
    print("VALIDATE_OK" if tm/max(tn,1) > 0.9 else "VALIDATE_LOW")


def emit(args):
    dfo = DFlashOffline(args.target, args.draft)
    gt, reqs, decisions, steps, log = _load_record(args.record_dir, args.decisions_file)
    n = 0
    nreq = 0
    with open(args.out, "w") as f:
        for rid, ids in reqs.items():
            if args.limit_requests and nreq >= args.limit_requests:
                break
            out = gt.get(tuple(ids))
            if out is None:
                continue
            seq = list(ids) + list(out)
            # --dense: re-speculate a block at EVERY gt position (commit advances by 1,
            # ignoring the served accept pattern) so the chain-handoff oracle gets a
            # DFlash proposal rooted at every decode position, mirroring the oracle-
            # vanilla (accept_length=0) dense capture used by the eagle/MTP handoff.
            clens = [1] * len(out) if args.dense else _commit_lens(rid, steps)
            recs = dfo.proposals_for_seq(seq, len(ids), clens)
            for ds, d, tok, p, gt_rank, gt_p in recs:
                f.write(json.dumps({"rid": rid, "decode_step": ds, "depth": d,
                                    "dflash_token": tok, "dflash_p": round(p, 6),
                                    "gt_rank": gt_rank,
                                    "gt_p": round(gt_p, 6)}) + "\n")
                n += 1
            nreq += 1
            print(f"  [{nreq}/{len(reqs)}] rid={rid} out_len={len(out)} "
                  f"cum_rows={n}", flush=True)
            torch.cuda.empty_cache()
    print(f"wrote {n} DFlash proposals -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--decisions-file", default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--dense", action="store_true",
                    help="emit a block at every gt position (commit_lens=1), for "
                         "the chain-handoff dense oracle instead of the served "
                         "accept pattern")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-requests", type=int, default=0,
                    help="smoke mode: only emit the first N requests (0 = all)")
    args = ap.parse_args()
    if args.validate:
        validate(args)
    elif args.emit:
        if not args.out:
            raise SystemExit("--emit needs --out")
        emit(args)
    else:
        raise SystemExit("use --validate or --emit")


if __name__ == "__main__":
    main()
