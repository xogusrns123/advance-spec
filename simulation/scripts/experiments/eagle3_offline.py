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

    def _attn(self, x, positions):
        """x [S,8192] -> [S,H]. Causal GQA + RoPE."""
        S = x.shape[0]
        q = (x @ self.W["q"].T).view(S, self.nh, self.hd)
        k = (x @ self.W["k"].T).view(S, self.nkv, self.hd)
        v = (x @ self.W["v"].T).view(S, self.nkv, self.hd)
        cos, sin = _rope_cos_sin(positions, self.hd, self.theta, x.device, x.dtype)
        q, k = _apply_rope(q, k, cos, sin)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        # [S,H,D] -> [H,S,D]
        q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)
        scores = (q.float() @ k.float().transpose(-1, -2)) / (self.hd ** 0.5)
        mask = torch.full((S, S), float("-inf"), device=x.device).triu(1)
        scores = scores + mask[None]
        out = (F.softmax(scores, dim=-1) @ v.float()).to(x.dtype)   # [H,S,D]
        out = out.transpose(0, 1).reshape(S, self.nh * self.hd)
        return out @ self.W["o"].T

    def _mlp(self, h):
        return (F.silu(h @ self.W["gate"].T) * (h @ self.W["up"].T)) @ self.W["down"].T

    @torch.inference_mode()
    def depth0_over_sequence(self, target_all_hidden, embeds, positions):
        """Depth-0 (one-step) EAGLE3 prediction at EVERY position of a sequence.
        target_all_hidden: HF output_hidden_states tuple/list (len num_layers+1),
                           each [S,H]. embeds: [S,H] target token embeddings.
        positions: [S] absolute positions. Returns (target_tokens[S], probs[S],
                   aux_hidden[S,H]) where token[t] is the proposal for position t+1.

        EAGLE3 alignment (VALIDATED 99.5% hi-conf vs sglang): the draft at position
        t consumes embed e(t) together with the target feature f_{t-1} (ONE position
        behind), predicting token t+1. So the aux target hidden is rolled +1 along
        the sequence (position t uses target_all_hidden[L][t-1]); position 0 wraps
        but is never a compared output position."""
        aux = torch.cat([torch.roll(target_all_hidden[L], 1, dims=0)
                         for L in self.aux_layers], dim=-1)
        fused = aux.to(self.dtype) @ self.W["fc"].T                      # [S,H]
        hn = _rmsnorm(fused, self.W["hidden_norm"], self.eps)
        e = _rmsnorm(embeds.to(self.dtype), self.W["input_ln"], self.eps)
        x = torch.cat([e, hn], dim=-1)                                  # [S,8192]
        attn = self._attn(x, positions)
        res2 = fused + attn
        h = _rmsnorm(res2, self.W["post_ln"], self.eps)
        res3 = res2 + self._mlp(h)
        logits = _rmsnorm(res3, self.W["norm"], self.eps) @ self.W["lm_head"].T
        probs = torch.softmax(logits.float(), dim=-1)
        pmax, draft_id = probs.max(dim=-1)
        target_tok = draft_id + self.d2t[draft_id]
        return target_tok, pmax, res3


# --------------------------------------------------------------------------- #
def _load_record(record_dir):
    """Parse a chain-hybrid record run: gt rows + per-(rid) decision/step rows."""
    dd = Path(record_dir)
    log = None
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
    """Compare offline depth-0 EAGLE3 predictions to the sglang record log."""
    dev = "cuda"
    gt, reqs, decisions, steps, log = _load_record(args.record_dir)
    print(f"record log: {log}  | reqs={len(reqs)} gt_rows={len(gt)}", flush=True)
    print(f"loading target {args.target} ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    snap = args.eagle3 or _eagle3_snapshot()
    print(f"loading eagle3 {snap} aux_layers={args.aux_layers} ...", flush=True)
    e3 = Eagle3Offline(snap, device=dev, aux_layers=tuple(args.aux_layers))

    n_match = n_tot = 0; p_abs = 0.0; n_p = 0
    hi_match = hi_tot = 0          # conditioned on sglang eagle_p > 0.5
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
        tok, pmax, _ = e3.depth0_over_sequence(allh, emb, pos)
        # reconstruct absolute output position per decode_step (depth 0).
        # off = committed output tokens BEFORE this decode step; starts at 1
        # because target prefill commits output[0], so decode step 1 predicts
        # output[1]. off advances by accept_len+1 (accepted drafts + 1 bonus).
        L = len(ids); off = 1
        per_step = decisions.get(rid, [])
        bystep = {}
        for d in per_step:
            bystep.setdefault(d["decode_step"], {})[d["depth"]] = d
        for s in sorted(bystep):
            d0 = bystep[s].get(0)
            if d0 is not None and d0.get("eagle_token") is not None:
                # depth-0 of step s predicts committed output[off] = seq[L+off],
                # made FROM position L+off-1. tok[t] = prediction for seq[t+1].
                abspos = L + off - 1
                if 0 <= abspos < len(seq):
                    mine = int(tok[abspos].item())
                    sgl = int(d0["eagle_token"])
                    n_tot += 1; n_match += (mine == sgl)
                    if d0.get("eagle_p") is not None:
                        p_abs += abs(float(pmax[abspos].item()) - float(d0["eagle_p"])); n_p += 1
                        if float(d0["eagle_p"]) > 0.5:
                            hi_tot += 1; hi_match += (mine == sgl)
            al = steps.get(rid, {}).get(s)
            off += (al + 1) if al is not None else 1
    rate = n_match / max(n_tot, 1)
    print(f"\n=== aux_layers={args.aux_layers} depth-0 token match: {n_match}/{n_tot} = {rate:.4f} ===")
    if hi_tot:
        print(f"=== high-conf (sglang eagle_p>0.5) match: {hi_match}/{hi_tot} = {hi_match/hi_tot:.4f} ===")
    if n_p:
        print(f"=== mean |prob - sglang_prob|: {p_abs/n_p:.4f}  (n={n_p}) ===")
    print("VALIDATE_OK" if rate > 0.95 else "VALIDATE_LOW")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--eagle3", default=None, help="eagle3 snapshot dir (default: HF cache)")
    ap.add_argument("--aux-layers", type=int, nargs=3, default=[2, 18, 33])
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
