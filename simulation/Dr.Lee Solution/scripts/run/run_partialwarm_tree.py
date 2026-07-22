#!/usr/bin/env python3
"""Partial-warm crossover with the CUDA tree-verify: does the ungated TREE composition beat the
gated CHAIN composition AND both standalones, as novel slots grow?

Workload multislot_k{0,1,2,4,8}: a shared Python-fn skeleton with k VARYING slots (novel values,
held out of warming but stated in the prompt). Warm the Suffix on the 12 warm outputs, then on the
8 eval prompts measure eval K (accepted/round) for four proposers, all verified by our hybrid
tree-verify:
  dflash   : DFlash full block (Predictor only) — no warming, ~flat
  suffix   : Suffix continuation (Memorizer only) — stalls at each novel slot, falls with k
  chain    : DFlash head(k*) + Suffix tail AFTER block (compCONF, GATED)
  tree     : DFlash head(k*) + Suffix tail at EVERY prefix (UNGATED)

Saves results/partialwarm_tree/pw.json (K per proposer per k + measured coverage) for plotting.

GPU, slow. Usage:  /home/ina_spark1/.venvs/vllm/bin/python scripts/run_partialwarm_tree.py
  KS="0 1 2 4 8" NEVAL=4 ROUNDS=6 NWARM=8 ... to scope.
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from specedge_serving.common.tree import DraftTree                          # noqa: E402
from fusion_tree import (build_extension_chain, build_extension_tree,       # noqa: E402
                         adaptive_nhead)
import tree_verify_hybrid as TVH                                            # noqa: E402
import dflash_candidates as DC                                              # noqa: E402
from measure_k_fusion import ArcticSuffix, dflash_block_logp                # noqa: E402


@torch.no_grad()
def greedy(target, tok, prompt, max_tokens):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", enable_thinking=False)
    ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(target.device)
    # PERF-ONLY reproduction change (identical greedy trace): the original used
    # use_cache=False (O(n²) recompute per token). On this host the GatedDeltaNet
    # fast-path kernels (flash-linear-attention / causal-conv1d) are NOT installed,
    # so the hybrid 27B falls back to a slow torch impl and the coverage-greedy
    # precompute (5·20·96 tokens) took >1h. KV-caching yields the SAME argmax trace.
    out = []
    past, cur = None, ids
    for _ in range(max_tokens):
        o = target(cur, use_cache=True, past_key_values=past)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        out.append(nxt)
        cur = torch.tensor([[nxt]], device=ids.device)
        ids = torch.cat([ids, cur], dim=1)
        if nxt == tok.eos_token_id:
            break
    return ids, out


def _bestfirst_caps(rem, k, conf, tails):
    """Allocate `rem` tail tokens across k+1 prefixes proportional to branch probability p_j:
    p_j = S_j·(1-a_j) (head survives to j then breaks) for j<k, p_k = S_k (head fully survives),
    with a_j = clip(0.69·conf_j+0.29), S_j = ∏a. Capped at each tail's natural length; leftover
    goes greedily to the highest-p_j prefixes that still have room."""
    a = [min(1.0, max(0.0, 0.69 * float(conf[j]) + 0.29)) for j in range(min(k, len(conf)))]
    S = [1.0]
    for aj in a:
        S.append(S[-1] * aj)
    p = []
    for j in range(k + 1):
        Sj = S[j] if j < len(S) else S[-1]
        p.append(Sj * (1.0 - (a[j] if j < len(a) else 1.0)) if j < k else Sj)
    tot = sum(p) or 1.0
    caps = [min(len(tails[j]), int(rem * p[j] / tot)) for j in range(k + 1)]
    left = rem - sum(caps)
    order = sorted(range(k + 1), key=lambda j: -p[j])
    while left > 0:
        progress = False
        for j in order:
            if left > 0 and caps[j] < len(tails[j]):
                caps[j] += 1; left -= 1; progress = True
        if not progress:
            break
    return caps


def build_draft(proposer, block_full, suf, tails, k, num_spec, budget=None, alloc="even", conf=None):
    """Build a draft under a total-node BUDGET B (default num_spec). Since verify V(N) is
    shape-invariant (measured: tree≈chain per node), holding B equal makes chain vs tree a fair
    'which shape uses the budget better' comparison. Tree allocation policy (alloc):
      even       (a) — split B-k evenly across k+1 prefix tails
      bestfirst  (b) — split proportional to branch prob p_j (needs conf)
      none       (c) — no cap, each tail its full natural length (node blow-up)"""
    B = num_spec if budget is None else budget
    if proposer == "dflash":
        return build_extension_chain(block_full[:B], [])          # Predictor only (block <= 15)
    if proposer == "suffix":
        return build_extension_chain([], suf[:B])                # Memorizer only
    if proposer == "chain":
        tb = max(0, B - k)                                       # all remaining budget in ONE tail
        return build_extension_chain(block_full[:k], tails[k][:tb])   # gated (linear)
    # tree (ungated): head(k) + a suffix tail at every prefix; distribute B-k per `alloc`
    rem = max(0, B - k)
    if alloc == "none":
        caps = [num_spec] * (k + 1)                             # uncapped -> natural tail lengths
    elif alloc == "bestfirst" and conf is not None:
        caps = _bestfirst_caps(rem, k, conf, tails)
    else:                                                       # even (a)
        per = rem // (k + 1) if k + 1 else 0
        caps = [per] * (k + 1)
    return build_extension_tree(block_full[:k], [t[:c] for t, c in zip(tails, caps)])


def eval_K(target, draft, tok, cfg, cfg_d, suffix, prompt, proposer, max_rounds, num_spec,
           budget=None, alloc="even"):
    """Returns (mean_K, tokens, wall_s, mean_nhead, mean_T, mean_N, mean_D_ms, mean_V_ms).
    D = draft-forward cost, V = the DIRECTLY-timed tree_verify forward (cuda-synced, no model),
    N = draft node budget actually built. budget caps total nodes; alloc = tree budget policy
    (even/bestfirst/none) — see build_draft."""
    import time
    dev = target.device
    ids, _ = greedy(target, tok, prompt, 0)                    # just the prompt ids
    suffix.new_eval(ids[0].tolist())                          # start a fresh Arctic eval request
    ctx = ids
    Ks = []; toks = 0; wall = 0.0; NHs = []; Ts = []; Ns = []; Ds = []; Vs = []
    for _ in range(max_rounds):
        # --- draft cost D: DFlash block forward + controller + suffix build ---
        torch.cuda.synchronize(); t0 = time.perf_counter()
        root, logp = dflash_block_logp(target, draft, ctx, cfg_d)
        ctx_root = torch.cat([ctx, torch.tensor([[root]], device=dev)], dim=1)
        ctx_list = ctx[0].tolist() + [root]
        W = len(logp)                                          # DFlash draft_horizon (head cap)
        block_full = [int(logp[i].argmax()) for i in range(W)]
        suf, T = suffix.probe(ctx_list, num_spec)             # T = probe.score (ext_suffix controller)
        conf = logp.max(dim=-1).values.exp().tolist()
        k = adaptive_nhead(conf, T=T, num_spec=W)             # head length k* <= draft_horizon
        NHs.append(k); Ts.append(T)
        tails = [suffix.speculate(ctx_list + block_full[:j], num_spec) for j in range(k + 1)]
        tree = build_draft(proposer, block_full, suf, tails, k, num_spec, budget, alloc, conf)
        torch.cuda.synchronize(); t_draft = time.perf_counter() - t0
        Ns.append(len(tree.tokens))                           # draft node budget N -> verify cost
        # --- verify cost V: the actual tree_verify forward, timed alone (no model) ---
        torch.cuda.synchronize(); t1 = time.perf_counter()
        acc, bonus, _ = TVH.tree_verify(target, ctx_root, tree, cfg)
        torch.cuda.synchronize(); t_verify = time.perf_counter() - t1
        Ds.append(t_draft * 1e3); Vs.append(t_verify * 1e3)   # ms
        wall += t_draft + t_verify
        Ks.append(len(acc))
        if os.environ.get("DEBUG"):
            head_acc = sum(1 for i in acc if i < k)              # accepted head nodes
            tl = len(tails[k]) if proposer in ("chain", "tree") else len(suf)
            print(f"    [{proposer}] k={k} T={T:.1f} tail_len={tl} acc={len(acc)} "
                  f"head_acc={head_acc}/{k} {'HEAD_BROKE(gate!)' if head_acc < k and k > 0 else 'head_ok'}",
                  flush=True)
        nxt = [root] + [tree.tokens[i] for i in acc] + [bonus]   # accepted + root + bonus
        toks += len(nxt)
        suffix.add_response(nxt)                                  # eval-time warming (match vLLM)
        ctx = torch.cat([ctx, torch.tensor([nxt], device=dev)], dim=1)
        if tok.eos_token_id in nxt:
            break
    import statistics as _st
    mean = lambda a: _st.mean(a) if a else 0.0
    return (sum(Ks) / max(len(Ks), 1), toks, wall, mean(NHs), mean(Ts),
            mean(Ns), mean(Ds), mean(Vs))


def coverage(eval_outs, warm_toks, n=4):
    """% of eval-output n-grams that appear in the warm corpus (the master variable)."""
    warm_ng = set()
    for s in warm_toks:
        for i in range(len(s) - n):
            warm_ng.add(tuple(s[i:i + n]))
    tot = hit = 0
    for s in eval_outs:
        for i in range(len(s) - n):
            tot += 1
            hit += tuple(s[i:i + n]) in warm_ng
    return hit / max(tot, 1)


def main():
    from transformers import AutoTokenizer
    tgt = os.environ.get("TGT", "Qwen/Qwen3.5-4B")
    target, draft, cfg_d = DC.load(tgt, os.environ.get("DFT", "z-lab/Qwen3.5-4B-DFlash"))
    tok = AutoTokenizer.from_pretrained(tgt)
    c = target.config.get_text_config()
    cfg = dict(Hk=c.linear_num_key_heads, Hv=c.linear_num_value_heads,
               Dk=c.linear_key_head_dim, Dv=c.linear_value_head_dim, Keep=c.linear_conv_kernel_dim - 1)

    KS = [int(x) for x in os.environ.get("KS", "0 1 2 4 8").split()]
    NWARM = int(os.environ.get("NWARM", "8"))
    NEVAL = int(os.environ.get("NEVAL", "4"))
    ROUNDS = int(os.environ.get("ROUNDS", "6"))
    MAXTOK = int(os.environ.get("MAXTOK", "80"))
    NUM_SPEC = int(os.environ.get("NUM_SPEC", "32"))          # total draft budget (match vLLM)
    props = os.environ.get("PROPS", "dflash suffix chain tree").split()
    # throughput model from experiments/.../verify_cost_real.json (27B, real vLLM verify path):
    # V(N) = V_BASE + V_SLOPE·N ms ; t_draft ~ D_MS. tp_model = (K+1)/(D + V(N)) in tok/s.
    V_BASE = float(os.environ.get("V_BASE", "235"))   # base verify (prefix/model forward) ms
    V_SLOPE = float(os.environ.get("V_SLOPE", "1.2"))  # ms per extra draft node
    D_MS = float(os.environ.get("D_MS", "113"))        # DFlash block draft cost ms
    data = {"model": tgt, "KS": KS, "K": {p: [] for p in props},
            "tps": {p: [] for p in props}, "nhead": {p: [] for p in props},
            "T": {p: [] for p in props}, "nodes": {p: [] for p in props},
            "tp_model": {p: [] for p in props}, "coverage": [],
            "vmodel": dict(V_BASE=V_BASE, V_SLOPE=V_SLOPE, D_MS=D_MS)}

    # precompute greedy warm/eval outputs for every k (both modes reuse these)
    per_k = {}
    for k in KS:
        rows = [json.loads(l) for l in open(f"scripts/bench_prompts_multislot_k{k}.jsonl")]
        warm = [r for r in rows if r["role"] == "warm"][:NWARM]
        ev = [r for r in rows if r["role"] == "eval"][:NEVAL]
        warm_toks = [greedy(target, tok, r["prompt"], MAXTOK)[1] for r in warm]
        eval_toks = [greedy(target, tok, r["prompt"], MAXTOK)[1] for r in ev]
        per_k[k] = dict(warm=warm, ev=ev, warm_toks=warm_toks, eval_toks=eval_toks)
        data["coverage"].append(coverage(eval_toks, warm_toks))
    if os.environ.get("DUMP"):
        k0 = KS[0]; wt = per_k[k0]["warm_toks"]; et = per_k[k0]["eval_toks"]
        print(f"=== k={k0} generated outputs (check thinking/format) ===")
        print("WARM[0]:", repr(tok.decode(wt[0])[:500]))
        print("EVAL[0]:", repr(tok.decode(et[0])[:500]))
        return

    def _eval_set(suffix, ev, p, budget=None):
        acc = [0.0] * 7                                  # K, toks, wall, nh, T, N, D, V sums (7 means)
        Vsum = 0.0
        for r in ev:
            out = eval_K(target, draft, tok, cfg, cfg_d, suffix, r["prompt"], p, ROUNDS, NUM_SPEC, budget)
            mK, toks, wall, mnh, mT, mN, mD, mV = out
            acc[0] += mK; acc[1] += toks; acc[2] += wall; acc[3] += mnh
            acc[4] += mT; acc[5] += mN; acc[6] += mD; Vsum += mV
        n = len(ev)
        mK, mN, mD, mV = acc[0] / n, acc[5] / n, acc[6] / n, Vsum / n
        tp_meas = 1000.0 * (mK + 1) / (mD + mV)          # (K+1)/(D+V) with DIRECTLY-timed D,V
        return mK, acc[1] / acc[2], acc[3] / n, acc[4] / n, mN, mD, mV, tp_meas

    def _record(p, k, r):
        mK, mtps, mnh, mT, mN, mD, mV, tp_meas = r
        data["K"][p].append(mK); data["tps"][p].append(mtps)
        data["nhead"][p].append(mnh); data["T"][p].append(mT)
        data["nodes"][p].append(mN); data["tp_model"][p].append(tp_meas)
        return f"{p}=K{mK:.1f}/N{mN:.0f}/V{mV:.0f}ms/D{mD:.0f}ms/tp{tp_meas:.1f}"

    global_warm = os.environ.get("GLOBAL_WARM") == "1"
    if global_warm:
        # vLLM protocol: ONE global suffix tree per proposer, warmed by msw{k} tasks in KS order,
        # eval-time warming persists across tasks -> mse4's tree has already absorbed msw0/mse0/msw...
        print("[GLOBAL_WARM] one accumulating suffix tree per proposer (mirrors vLLM sequential warming)")
        for p in props:
            suffix = ArcticSuffix()                          # fresh per proposer, accumulates across k
            for k in KS:
                suffix.fit(per_k[k]["warm_toks"])            # warm task msw{k} into the global tree
                seg = _record(p, k, _eval_set(suffix, per_k[k]["ev"], p))
                print(f"  k={k} cov={data['coverage'][KS.index(k)]:.2f}  {seg}", flush=True)
    else:
        for k in KS:
            line = f"k={k}  cov={data['coverage'][KS.index(k)]:.2f} | "
            for p in props:
                suffix = ArcticSuffix(); suffix.fit(per_k[k]["warm_toks"])  # isolated: this k only
                line += _record(p, k, _eval_set(suffix, per_k[k]["ev"], p)) + "  "
            print(line, flush=True)
        # throughput verdict: does the tree's K gain beat its verify tax? (V DIRECTLY timed)
        if "chain" in props and "tree" in props:
            print("\n=== throughput verdict (V = directly-timed tree_verify, tp=(K+1)/(D+V)) ===")
            for i, k in enumerate(KS):
                tc, tt = data["tp_model"]["chain"][i], data["tp_model"]["tree"][i]
                win = "TREE" if tt > tc else "CHAIN"
                print(f"  k={k}: chain tp={tc:.1f} (K{data['K']['chain'][i]:.1f}/N{data['nodes']['chain'][i]:.0f})  "
                      f"tree tp={tt:.1f} (K{data['K']['tree'][i]:.1f}/N{data['nodes']['tree'][i]:.0f})  "
                      f"-> {win} wins ({max(tc, tt)/min(tc, tt):.2f}x)")

    out = Path(os.environ.get("OUT", "results/partialwarm_tree/pw.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out, "w"), indent=1)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
