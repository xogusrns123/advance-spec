#!/usr/bin/env python3
"""OFFLINE teacher-forced forward for an AngelSlim EAGLE3 draft head, faithful to
sglang's LlamaForCausalLMEagle3 (single midlayer). Used as the EAGLE3-aux proposer
for the 3-way chain-hybrid select study (EAGLE3 + DFlash + suffix on Qwen3-8B).

The served 3-way runs DFlash as the sglang MAIN worker; this module is the
hand-built EAGLE3-aux forward, VALIDATED here offline against a sglang EAGLE3-8B
chain-hybrid `record` log before being ported into the served DFlash hook.

Arch (AngelSlim/Qwen3-8B_eagle3, model_type=llama, num_hidden_layers=1):
  fused   = fc( concat[ target_hs[L] for L in aux_layers ] )          # 3*4096 -> 4096
  x       = cat([ input_layernorm(embed_tok), hidden_norm(fused) ])   # -> 8192
  attn    = o_proj( SelfAttn_RoPE_GQA(x) )                            # 8192 -> 4096
  res2    = fused + attn ; h = post_attention_layernorm(res2)
  res3    = res2 + mlp(h) ; logits = lm_head( norm(res3) )            # -> draft_vocab
  draft_id= argmax(logits) ; target_tok = draft_id + d2t[draft_id]
  aux carried to depth+1 = res3 (pre-norm)

sglang captures the "input to layer i" residual for i in layers_to_capture; the
default for EAGLE3 (no eagle_config) is [2, num_layers//2, num_layers-3] =
[2,18,33] for the 36-layer Qwen3-8B. In HF output_hidden_states (index 0 =
embeddings, index k = input to layer k) that is hidden_states[2|18|33]. The exact
indices are confirmed by --validate (token/prob match vs the sglang record log).

Run INSIDE sglang-bench on Blackwell GPU0 (after the record server has exited):
  CUDA_VISIBLE_DEVICES=0 python3 simulation/scripts/experiments/eagle3_offline.py \
    --validate --record-dir simulation/results/chain_hybrid_perdepth/qwen3_8b_eagle3_3way_val
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_HUB = "/root/.cache/huggingface/hub"


def _eagle3_snapshot(repo="models--AngelSlim--Qwen3-8B_eagle3") -> str:
    snaps = sorted(glob.glob(f"{HF_HUB}/{repo}/snapshots/*"))
    if not snaps:
        raise SystemExit(f"eagle3 snapshot not found under {HF_HUB}/{repo}")
    return snaps[-1]


def _rmsnorm(x, w, eps=1e-6):
    """Llama/Qwen RMSNorm (compute in fp32)."""
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (w * x.to(dt))


def _rope_cos_sin(positions, head_dim, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]      # [S, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                     # [S, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(q, k, cos, sin):
    # q [S,Hq,D] k [S,Hk,D] ; cos/sin [S,D]
    cos = cos[:, None, :]; sin = sin[:, None, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class Eagle3Offline:
    """Faithful single-layer EAGLE3 draft forward, teacher-forced."""

    def __init__(self, snapshot: str, device="cuda", dtype=torch.bfloat16,
                 aux_layers=(2, 18, 33)):
        self.device = device; self.dtype = dtype
        cfg = json.load(open(Path(snapshot) / "config.json"))
        self.H = cfg["hidden_size"]; self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]; self.hd = cfg["head_dim"]
        self.theta = cfg["rope_theta"]; self.eps = cfg["rms_norm_eps"]
        self.aux_layers = list(aux_layers)
        sd = torch.load(Path(snapshot) / "pytorch_model.bin", map_location=device,
                        weights_only=False)
        g = lambda k: sd[k].to(device=device, dtype=dtype)
        self.W = {
            "fc": g("fc.weight"),
            "q": g("midlayer.self_attn.q_proj.weight"),
            "k": g("midlayer.self_attn.k_proj.weight"),
            "v": g("midlayer.self_attn.v_proj.weight"),
            "o": g("midlayer.self_attn.o_proj.weight"),
            "gate": g("midlayer.mlp.gate_proj.weight"),
            "up": g("midlayer.mlp.up_proj.weight"),
            "down": g("midlayer.mlp.down_proj.weight"),
            "hidden_norm": g("midlayer.hidden_norm.weight"),
            "input_ln": g("midlayer.input_layernorm.weight"),
            "post_ln": g("midlayer.post_attention_layernorm.weight"),
            "norm": g("norm.weight"),
            "lm_head": g("lm_head.weight"),
        }
        self.d2t = sd["d2t"].to(device)   # [draft_vocab] int

    def _mlp(self, h):
        return (F.silu(h @ self.W["gate"].T) * (h @ self.W["up"].T)) @ self.W["down"].T

    def _make_qkv(self, fused, embeds, positions):
        """One EAGLE3 layer up to (post-RoPE) q,k,v. fused [S,H] is the carried
        hidden (= fc(aux) at depth 0, or the previous draft res3 at depth>0);
        embeds [S,H] the token embeddings; positions [S]. The single midlayer is
        always `is_input_layer` (num_hidden_layers=1), so it concats the normed
        embed with the normed fused hidden into the 2H attn input."""
        hn = _rmsnorm(fused, self.W["hidden_norm"], self.eps)
        e = _rmsnorm(embeds.to(self.dtype), self.W["input_ln"], self.eps)
        x = torch.cat([e, hn], dim=-1)                                   # [S,2H]
        S = x.shape[0]
        q = (x @ self.W["q"].T).view(S, self.nh, self.hd)
        k = (x @ self.W["k"].T).view(S, self.nkv, self.hd)
        v = (x @ self.W["v"].T).view(S, self.nkv, self.hd)
        cos, sin = _rope_cos_sin(positions, self.hd, self.theta, x.device, x.dtype)
        q, k = _apply_rope(q, k, cos, sin)
        return q, k, v                                                   # post-RoPE

    def _attn_out(self, q, k_all, v_all):
        """q [Sq,nh,hd] post-RoPE; k_all/v_all [Sk,nkv,hd] post-RoPE (full cache,
        new keys appended last). Causal: the Sq new queries occupy the LAST Sq
        slots of the cache (attend to all earlier + themselves)."""
        rep = self.nh // self.nkv
        k = k_all.repeat_interleave(rep, dim=1); v = v_all.repeat_interleave(rep, dim=1)
        Sk = k.shape[0]; Sq = q.shape[0]
        qh = q.transpose(0, 1); kh = k.transpose(0, 1); vh = v.transpose(0, 1)
        scores = (qh.float() @ kh.float().transpose(-1, -2)) / (self.hd ** 0.5)
        offset = Sk - Sq
        qi = torch.arange(Sq, device=q.device)[:, None] + offset
        ki = torch.arange(Sk, device=q.device)[None, :]
        scores = scores + torch.where(ki <= qi, 0.0, float("-inf"))[None]
        out = (F.softmax(scores, dim=-1) @ vh.float()).to(q.dtype)       # [H,Sq,D]
        out = out.transpose(0, 1).reshape(Sq, self.nh * self.hd)
        return out @ self.W["o"].T

    def _finish(self, fused, attn):
        """Residual + post-attn-norm + MLP + final-norm + lm_head. Returns
        (res3 pre-norm hidden carried to next depth, draft logits)."""
        res2 = fused + attn
        h = _rmsnorm(res2, self.W["post_ln"], self.eps)
        res3 = res2 + self._mlp(h)
        logits = _rmsnorm(res3, self.W["norm"], self.eps) @ self.W["lm_head"].T
        return res3, logits

    def _logits_to_tok(self, logits):
        probs = torch.softmax(logits.float(), dim=-1)
        pmax, draft_id = probs.max(dim=-1)
        target_tok = draft_id + self.d2t[draft_id]
        return target_tok, pmax

    @torch.inference_mode()
    def prefill_sequence(self, target_all_hidden, embeds, positions):
        """EAGLE3 draft PREFILL over a committed sequence (depth-0 at every pos).
        target_all_hidden: HF output_hidden_states (len num_layers+1), each [S,H].
        embeds: [S,H] target token embeddings. positions: [S] absolute positions.
        Returns dict {k,v: post-RoPE draft cache [S,nkv,hd]; res3: [S,H] carried
        hidden; tok,prob: [S] depth-0 proposal (token[t] = proposal for pos t+1)}.

        EAGLE3 alignment (VALIDATED 99.5% hi-conf vs sglang): the draft at position
        t consumes embed e(t) with the target feature f_{t-1} (ONE position behind),
        so the aux target hidden is rolled +1 along the sequence."""
        aux = torch.cat([torch.roll(target_all_hidden[L], 1, dims=0)
                         for L in self.aux_layers], dim=-1)
        fused = aux.to(self.dtype) @ self.W["fc"].T                      # [S,H]
        q, k, v = self._make_qkv(fused, embeds, positions)
        attn = self._attn_out(q, k, v)                                  # full causal
        res3, logits = self._finish(fused, attn)
        tok, prob = self._logits_to_tok(logits)
        return {"k": k, "v": v, "res3": res3, "tok": tok, "prob": prob}

    @torch.inference_mode()
    def depth0_over_sequence(self, target_all_hidden, embeds, positions):
        """Back-compat thin wrapper: returns (tok, prob, res3) for depth 0."""
        pf = self.prefill_sequence(target_all_hidden, embeds, positions)
        return pf["tok"], pf["prob"], pf["res3"]

    # ----------------------------- SERVING API ----------------------------- #
    # The served 3-way runs EAGLE3 as an in-process aux proposer next to the
    # sglang DFlash main worker. These methods reuse the SAME validated layer
    # primitives but operate on an INCREMENTAL committed K/V cache (DFlash is the
    # main worker, so sglang does not maintain EAGLE3's committed-context KV).
    # Caller supplies aux features ALREADY sliced to EAGLE3's 3 chunks and ALREADY
    # lag-shifted (KV at position p uses the target feature at p-1).

    @torch.inference_mode()
    def extend_kv(self, aux_lagged, tokens, positions, embed_fn,
                  k_old=None, v_old=None):
        """Append committed positions to the EAGLE3 draft K/V cache and compute
        their depth-0 proposals. aux_lagged [N,3H] = fc input (target features,
        already chunk-sliced and lag-rolled). tokens [N] long; positions [N].
        k_old/v_old: existing committed cache ([M,nkv,hd]) or None. Returns
        {k,v: full cache [M+N,...]; res3 [N,H]; tok,prob [N] depth-0 proposals}."""
        fused = aux_lagged.to(self.dtype) @ self.W["fc"].T               # [N,H]
        emb = embed_fn(tokens).view(-1, self.H)
        q, k, v = self._make_qkv(fused, emb, positions)
        k_all = k if k_old is None else torch.cat([k_old, k], dim=0)
        v_all = v if v_old is None else torch.cat([v_old, v], dim=0)
        attn = self._attn_out(q, k_all, v_all)                          # new q attend all
        res3, logits = self._finish(fused, attn)
        tok, prob = self._logits_to_tok(logits)
        return {"k": k_all, "v": v_all, "res3": res3, "tok": tok, "prob": prob}

    @torch.inference_mode()
    def chain_step(self, h_prev, token_id, position, embed_fn,
                   k_committed, v_committed, chain_k, chain_v):
        """One depth>0 EAGLE3 chain step. h_prev [1,H] = res3 from the previous
        depth; token_id = the CHOSEN token at the previous depth (sits at
        `position`); k_committed/v_committed [M,nkv,hd] = committed cache [0..root];
        chain_k/chain_v = the within-block chain cache so far (None at depth 1,
        grows by 1 each step). Returns (tok, prob, res3 [1,H], chain_k', chain_v')."""
        emb = embed_fn(torch.tensor([token_id], device=self.device)).view(1, self.H)
        pos = torch.tensor([position], device=self.device)
        q, k, v = self._make_qkv(h_prev, emb, pos)                       # fused=h_prev (no fc)
        chain_k = k if chain_k is None else torch.cat([chain_k, k], dim=0)
        chain_v = v if chain_v is None else torch.cat([chain_v, v], dim=0)
        k_all = torch.cat([k_committed, chain_k], dim=0)
        v_all = torch.cat([v_committed, chain_v], dim=0)
        attn = self._attn_out(q, k_all, v_all)                          # 1 new q attend all
        res3, logits = self._finish(h_prev, attn)
        tok, prob = self._logits_to_tok(logits)
        return int(tok.item()), float(prob.item()), res3, chain_k, chain_v

    @torch.inference_mode()
    def draft_chain(self, pf, root, chain_tokens, embed_fn, num_steps):
        """Roll a topk=1 EAGLE3 chain from root position `root`.

        pf: prefill_sequence() output (committed K/V + res3 + depth-0 proposal).
        root: last committed position; depth-0 proposal is pf['tok'][root].
        chain_tokens: the tokens INJECTED at the head of each depth (the chosen /
          pinned tokens). chain_tokens[d] is fed as input at depth d+1 and sits at
          position root+d+1. (Under --pin-trajectory these are the gt continuation.)
        embed_fn(token_id_tensor)->[*,H] token embedding (shared target embedding).
        num_steps: max chain length (EAGLE speculative_num_steps).

        Recurrence (matches sglang eagle_worker.draft_forward, topk=1): depth d>0
          consumes embed(chain_tokens[d-1]) + the draft res3 from depth d-1, at
          position root+d, attending to committed[0..root] + chain[root+1..root+d].
        Returns (tokens[D], probs[D]) where D<=num_steps, depth 0 first."""
        toks = [int(pf["tok"][root].item())]
        probs = [float(pf["prob"][root].item())]
        k_cache = pf["k"][:root + 1]                  # committed positions 0..root
        v_cache = pf["v"][:root + 1]
        h_prev = pf["res3"][root:root + 1]            # [1,H] feature at position root
        for d in range(1, num_steps):
            tok_in = chain_tokens[d - 1]              # chosen token at depth d-1
            if tok_in is None:
                break
            emb = embed_fn(torch.tensor([tok_in], device=self.device)).view(1, self.H)
            pos = torch.tensor([root + d], device=self.device)
            q, k, v = self._make_qkv(h_prev, emb, pos)   # fused = carried res3 (no fc)
            k_cache = torch.cat([k_cache, k], dim=0)
            v_cache = torch.cat([v_cache, v], dim=0)
            attn = self._attn_out(q, k_cache, v_cache)
            res3, logits = self._finish(h_prev, attn)
            tok, prob = self._logits_to_tok(logits)
            toks.append(int(tok.item())); probs.append(float(prob.item()))
            h_prev = res3
        return toks, probs


# --------------------------------------------------------------------------- #
def _chosen_token(rec):
    """The token actually injected into the draft chain at this depth (= the
    select-1 pick). Mirrors chain_hybrid_patch's decision body. Labels: 'suffix'
    -> suffix_token; 'e3' -> e3_token (3-way EAGLE3-aux); 'oracle' -> gt_token
    (3-way oracle arm picks gt when any proposer hits); else ('dflash'/'eagle3'/
    default) -> eagle_token (the main worker's proposal: DFlash or EAGLE/MTP)."""
    c = rec.get("chosen")
    if c == "suffix" and rec.get("suffix_token") is not None:
        return rec["suffix_token"]
    if c == "e3" and rec.get("e3_token") is not None:
        return rec["e3_token"]
    if c == "oracle" and rec.get("gt_token") is not None:
        return rec["gt_token"]
    return rec.get("eagle_token")


def _load_record(record_dir, decisions_file=None):
    """Parse a chain-hybrid record run: gt rows + per-(rid) decision/step rows."""
    dd = Path(record_dir)
    log = None
    if decisions_file:
        log = dd / decisions_file if not os.path.isabs(decisions_file) else Path(decisions_file)
        if not log.exists():
            raise SystemExit(f"decisions file not found: {log}")
    if log is None:
        for cand in ["decisions_record.jsonl", "decisions_select1.jsonl"]:
            if (dd / cand).exists():
                log = dd / cand; break
    if log is None:
        hits = sorted(dd.glob("decisions_*.jsonl"))
        log = hits[0] if hits else None
    if log is None:
        raise SystemExit(f"no decisions_*.jsonl in {record_dir}")
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


def validate(args):
    """Compare offline EAGLE3 predictions to the sglang record log, AT ALL DEPTHS.

    Depth 0 is the draft prefill at the last committed position. Depths d>0 are the
    EAGLE3 chain recurrence: the chain is fed the pinned (gt) tokens at its head and
    carries the draft's own res3 as the hidden, exactly as sglang's draft loop does
    under --pin-trajectory. Per-depth token/prob match vs the log's eagle_token."""
    dev = "cuda"
    gt, reqs, decisions, steps, log = _load_record(args.record_dir, args.decisions_file)
    print(f"record log: {log}  | reqs={len(reqs)} gt_rows={len(gt)}", flush=True)
    print(f"loading target {args.target} ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    embed_fn = target.model.embed_tokens
    snap = args.eagle3 or _eagle3_snapshot()
    print(f"loading eagle3 {snap} aux_layers={args.aux_layers} ...", flush=True)
    e3 = Eagle3Offline(snap, device=dev, aux_layers=tuple(args.aux_layers))

    tok_key = "eagle_token" if args.compare_field == "eagle" else "e3_token"
    p_key = "eagle_p" if args.compare_field == "eagle" else "e3_p"
    print(f"comparing offline forward against log field: {tok_key}/{p_key}", flush=True)
    from collections import defaultdict
    per = defaultdict(lambda: {"m": 0, "n": 0, "him": 0, "hin": 0, "pabs": 0.0, "np": 0})
    pos_chk_ok = pos_chk_tot = 0
    for rid, ids in reqs.items():
        out = gt.get(tuple(ids))
        if out is None:
            continue
        seq = list(ids) + list(out)
        t = torch.tensor([seq], device=dev)
        with torch.inference_mode():
            o = target(t, output_hidden_states=True, use_cache=False)
        allh = [h[0] for h in o.hidden_states]                # each [S,H]
        emb = target.model.embed_tokens(t)[0]                 # [S,H]
        pos = torch.arange(len(seq), device=dev)
        pf = e3.prefill_sequence(allh, emb, pos)
        # reconstruct absolute output position per decode_step (depth 0).
        # off = committed output tokens BEFORE this decode step; starts at 1
        # because target prefill commits output[0], so decode step 1 predicts
        # output[1]. off advances by accept_len+1 (accepted drafts + 1 bonus).
        L = len(ids); off = 1
        bystep = {}
        for d in decisions.get(rid, []):
            bystep.setdefault(d["decode_step"], {})[d["depth"]] = d
        for s in sorted(bystep):
            depths = bystep[s]
            root = L + off - 1                 # last committed position
            maxd = max(depths)
            if 0 <= root < len(seq):
                # chain head tokens = the CHOSEN proposer's token at each depth
                # (NOT gt): --pin-trajectory forces only the committed/verify
                # trajectory to gt; the draft chain is built from the select-1
                # pick (chain_hybrid_patch._decide_and_inject appends chosen_tok).
                # chosen[depth i] is fed at depth i+1 (sits at position root+1+i).
                chain_tokens = [_chosen_token(depths[i]) if i in depths else None
                                for i in range(maxd)]
                # position-reconstruction self-check: gt_token[d] == seq[root+1+d]
                for i in range(maxd):
                    ap = root + 1 + i
                    if i in depths and depths[i].get("gt_token") is not None and ap < len(seq):
                        pos_chk_tot += 1
                        pos_chk_ok += (int(depths[i]["gt_token"]) == int(seq[ap]))
                toks, probs = e3.draft_chain(pf, root, chain_tokens, embed_fn, maxd + 1)
                for d in sorted(depths):
                    rec = depths[d]
                    if rec.get(tok_key) is None or d >= len(toks):
                        continue
                    mine = toks[d]; sgl = int(rec[tok_key])
                    pe = per[d]
                    pe["n"] += 1; pe["m"] += (mine == sgl)
                    if rec.get(p_key) is not None:
                        pe["pabs"] += abs(probs[d] - float(rec[p_key])); pe["np"] += 1
                        if float(rec[p_key]) > 0.5:
                            pe["hin"] += 1; pe["him"] += (mine == sgl)
            al = steps.get(rid, {}).get(s)
            off += (al + 1) if al is not None else 1

    tot_m = tot_n = tot_him = tot_hin = 0; tot_pabs = 0.0; tot_np = 0
    print(f"\n=== aux_layers={args.aux_layers} per-depth EAGLE3 token match ===")
    print(f"{'depth':>5} {'match':>14} {'rate':>7} {'hi-conf':>14} {'hi-rate':>7} {'|dp|':>7}")
    for d in sorted(per):
        pe = per[d]
        rate = pe["m"] / max(pe["n"], 1)
        hrate = pe["him"] / max(pe["hin"], 1)
        dp = pe["pabs"] / max(pe["np"], 1)
        print(f"{d:>5} {pe['m']:>6}/{pe['n']:<7} {rate:>7.4f} "
              f"{pe['him']:>6}/{pe['hin']:<7} {hrate:>7.4f} {dp:>7.4f}")
        tot_m += pe["m"]; tot_n += pe["n"]; tot_him += pe["him"]; tot_hin += pe["hin"]
        tot_pabs += pe["pabs"]; tot_np += pe["np"]
    orate = tot_m / max(tot_n, 1)
    print(f"\n=== OVERALL token match: {tot_m}/{tot_n} = {orate:.4f} ===")
    if tot_hin:
        print(f"=== high-conf (eagle_p>0.5) match: {tot_him}/{tot_hin} = {tot_him/tot_hin:.4f} ===")
    if tot_np:
        print(f"=== mean |prob - sglang_prob|: {tot_pabs/tot_np:.4f}  (n={tot_np}) ===")
    if pos_chk_tot:
        print(f"=== position self-check gt_token==seq: {pos_chk_ok}/{pos_chk_tot} "
              f"= {pos_chk_ok/pos_chk_tot:.4f} ===")
    d0rate = per[0]["m"] / max(per[0]["n"], 1)
    print("VALIDATE_OK" if d0rate > 0.95 else "VALIDATE_LOW")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--eagle3", default=None, help="eagle3 snapshot dir (default: HF cache)")
    ap.add_argument("--aux-layers", type=int, nargs=3, default=[2, 18, 33])
    ap.add_argument("--compare-field", choices=["eagle", "e3"], default="eagle",
                    help="log token field to compare the offline forward against: "
                         "'eagle' (native EAGLE3 record) or 'e3' (served 3-way "
                         "DFlash-main EAGLE3-aux log).")
    ap.add_argument("--decisions-file", default=None,
                    help="decisions log filename within --record-dir (e.g. "
                         "decisions_select1_oracle.jsonl); default auto-detects.")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--record-dir",
                    default="simulation/results/chain_hybrid_perdepth/qwen3_8b_eagle3_3way_val")
    args = ap.parse_args()
    if args.validate:
        validate(args)
    else:
        raise SystemExit("only --validate is implemented in this milestone")


if __name__ == "__main__":
    main()
