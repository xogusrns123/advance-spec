"""ArcticSuffix (Memorizer wrapper) + dflash_block_logp (Predictor block draft).

controller.md §3 / §2. ArcticSuffix wraps Arctic ``SuffixDecodingCache`` with the
``fit / new_eval / probe / speculate / add_response`` surface the round loop uses,
at the Extension defaults **msf=4.0, mtp=0.1, max_tree_depth=64** (NOT arctic's
msf=1.0 default). Warming folds each trace into the global count-ordered tree via
``start_request(seq[:1]) → add_active_response(seq[1:]) → stop_request`` (drops
the duplicate first token). ``probe(ctx)`` returns the ctx-only warmth
``T = draft.score``; ``speculate(ctx)`` returns the flat copy-tail to graft.

``dflash_block_logp(target, draft, ctx, cfg_d)`` runs ONE DFlash block draft
rooted at the target-greedy token after ctx (teacher-forced from the cached
trace) and returns ``(root, logp)`` where ``logp`` is the per-position
log-softmax over vocab, shape ``[W, vocab]`` with W = block_size-1. The round
loop reads ``conf = logp.max(-1).exp()`` and ``block[i] = logp[i].argmax()``.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

# Extension suffix defaults (controller.md §3).
MSF = 4.0          # max_spec_factor  (tail ≤ msf × match_length)
MTP = 0.1          # min_token_prob   (candidate cut)
MAX_TREE_DEPTH = 64


class ArcticSuffix:
    """Model-free memorizer over Arctic SuffixDecodingCache."""

    def __init__(self, msf: float = MSF, mtp: float = MTP,
                 max_tree_depth: int = MAX_TREE_DEPTH, max_cached: int = 100000):
        from arctic_inference.suffix_decoding import SuffixDecodingCache
        self.cache = SuffixDecodingCache(max_tree_depth=max_tree_depth,
                                         max_cached_requests=max_cached)
        self.msf = float(msf)
        self.mtp = float(mtp)
        self._rid = 0
        self._eval_rid = None

    def _next(self) -> int:
        self._rid += 1
        return self._rid

    # -- warming (global tree) -------------------------------------------------
    def fit(self, traces: Sequence[Sequence[int]]) -> None:
        """Warm the global tree with each target-greedy trace (controller.md §3):
        start_request(seq[:1]) → add_active_response(seq[1:]) → stop_request."""
        for seq in traces:
            seq = [int(t) for t in seq]
            if not seq:
                continue
            rid = self._next()
            self.cache.start_request(rid, np.asarray(seq[:1], dtype=np.int32))
            if len(seq) > 1:
                self.cache.add_active_response(rid, seq[1:])
            self.cache.stop_request(rid)

    # -- eval request lifecycle ------------------------------------------------
    def new_eval(self, ids: Sequence[int]) -> None:
        """Fresh Arctic eval request (LOCAL tree reset). The prior eval request
        is stopped first so its eval-time-warmed tokens fold into the global tree
        (matches vLLM eval-time warming persisting across the eval set)."""
        if self._eval_rid is not None:
            try:
                self.cache.stop_request(self._eval_rid)
            except Exception:
                pass
        self._eval_rid = self._next()
        self.cache.start_request(self._eval_rid, np.asarray([int(t) for t in ids], dtype=np.int32))

    # score_mode: "raw" = arctic ML score (sum of survival of k/n edge probs);
    # "succ"/"kt" = rescore with the Bayesian rule-of-succession posterior
    # (k+1)/(n+2) (Laplace) or (k+.5)/(n+1) (KT/Jeffreys) per edge — the
    # zero-parameter finite-sample correction of the tree's own count estimates.
    # "dfprior" = DFlash-informed Beta prior: replace the uniform 0.5 in the
    # succession rule with DFlash's per-position probability q, encoded as
    # pseudo-count Beta(q*s,(1-q)*s) -> posterior mean (k + q*s)/(n + s). Falls
    # back to Laplace when q=0.5,s=2. Needs df_tok/df_conf aligned to tail edges.
    score_mode = "raw"
    df_s = 4.0        # DFlash prior concentration (pseudo-count strength)
    df_u = 1.4        # head-confidence calibration scalar q_agree = min(1, u*conf)
    df_qdis = 0.0     # prior for the suffix token when DFlash disagrees (veto)
    gb_a = 1.0        # genbeta: per-edge posterior (k+a)/(n+b), Beta(a,b-a) prior
    gb_b = 2.0        # (a=1,b=2 = Laplace; a=.5,b=1 = KT; prior mean a/b, strength b)
    ws_w0 = 1.0       # wsplit: depth-split scalars — score = Σ_d w_d·S_d with raw ML
    ws_w1 = 1.0       # edges; w_d = ws_w0 at d=0 else ws_w1 (w0=w1 = the flat scalar)
    aux_succ = False  # opt-in: stash (raw_score, succession_score) per _spec call
    last_aux_succ = None   # the AdaptiveScalar succratio arm's label-free signal

    def _spec(self, ctx: Sequence[int], num_spec: int,
              df_tok=None, df_conf=None) -> Tuple[List[int], float]:
        draft = self.cache.speculate(
            self._eval_rid, np.asarray([int(t) for t in ctx], dtype=np.int32),
            max_spec_tokens=int(num_spec), max_spec_factor=self.msf,
            min_token_prob=self.mtp, use_tree_spec=False)   # flat copy-tail
        toks = [int(t) for t in draft.token_ids] if getattr(draft, "token_ids", None) is not None else []
        score = float(getattr(draft, "score", 0.0) or 0.0)
        if self.aux_succ:
            # label-free succratio signal: (raw ML score, Laplace-succession
            # rescore) of THIS draft, regardless of score_mode
            aux = None
            if getattr(draft, "probs", None):
                surv, sc, prev = 1.0, 0.0, 1.0
                for pi, kc in zip(draft.probs, draft.counts):
                    p = pi / prev if prev > 0 else 0.0
                    n = int(round(kc / p)) if p > 0 else 0
                    q = (kc + 1.0) / (n + 2.0) if n > 0 else 0.0
                    surv *= q
                    sc += surv
                    prev = pi
                aux = (score, sc)
            self.last_aux_succ = aux
        if self.score_mode == "wsplit":
            # depth-split scalars WITHOUT loop reconstruction: the bulk term uses the
            # exact raw score (same quantity the fixed-scalar arm scales), only the
            # depth-0 survival S_0 = probs[0] is re-weighted. Immune to rounds where
            # probs is missing but score > 0 (there score = w1*raw, a bounded bias).
            pr = getattr(draft, "probs", None)
            p0 = max(0.0, min(1.0, float(pr[0]))) if pr is not None and len(pr) else 0.0
            return toks, self.ws_w1 * score + (self.ws_w0 - self.ws_w1) * p0
        if self.score_mode != "raw" and getattr(draft, "probs", None):
            surv, sc, prev = 1.0, 0.0, 1.0
            for j, (pi, k) in enumerate(zip(draft.probs, draft.counts)):
                p = pi / prev if prev > 0 else 0.0
                n = int(round(k / p)) if p > 0 else 0
                if n <= 0:
                    q = 0.0
                elif self.score_mode == "succ":
                    q = (k + 1.0) / (n + 2.0)
                elif self.score_mode == "kt":
                    q = (k + 0.5) / (n + 1.0)
                elif self.score_mode == "genbeta":      # general 2-param Beta posterior
                    q = min(1.0, (k + self.gb_a) / (n + self.gb_b))
                else:                                   # dfprior: DFlash-informed Beta prior
                    s = self.df_s
                    if df_tok is not None and df_conf is not None \
                            and j < len(df_tok) and j < len(df_conf):
                        # agree -> calibrated DFlash conf; disagree -> veto prior
                        qpr = (min(1.0, self.df_u * float(df_conf[j]))
                               if j < len(toks) and toks[j] == df_tok[j] else self.df_qdis)
                    else:
                        qpr = 0.5                       # no DFlash opinion beyond the block
                    q = (k + qpr * s) / (n + s)
                surv *= q
                sc += surv
                prev = pi
            score = sc
        return toks, score

    def probe(self, ctx: Sequence[int], num_spec: int) -> Tuple[List[int], float]:
        """(suffix continuation, T=score). T = ctx-only expected accept length."""
        return self._spec(ctx, num_spec)

    def speculate(self, ctx: Sequence[int], num_spec: int) -> List[int]:
        toks, _ = self._spec(ctx, num_spec)
        return toks

    def add_response(self, toks: Sequence[int]) -> None:
        """Eval-time warming: fold this round's committed tokens back in."""
        toks = [int(t) for t in toks]
        if toks and self._eval_rid is not None:
            self.cache.add_active_response(self._eval_rid, toks)


@torch.inference_mode()
def dflash_block_logp(target, draft, ctx, cfg_d) -> Tuple[int, torch.Tensor]:
    """One DFlash block draft rooted at the target-greedy token after ctx.

    target : TargetWrapper (cached trace + features)
    draft  : DFlashOffline  (holds .draft, .embed, ._lm_logits)
    ctx    : LongTensor [1, L]
    Returns (root, logp[W, vocab]),  W = block_size - 1.
    """
    b = int(cfg_d["block_size"])
    dev = draft.device
    ids = ctx[0].tolist()
    pos = len(ids)
    target._ensure(ids)
    root = target.trace_token(pos)                      # teacher-forced greedy seed
    target_hidden = target.features_upto(pos)           # [1, pos, K*H]
    block_ids = torch.full((1, b), int(cfg_d["mask_token_id"]), dtype=torch.long, device=dev)
    block_ids[0, 0] = int(root)
    noise = draft.embed(block_ids).reshape(1, b, -1)
    position_ids = torch.arange(pos + b, device=dev).unsqueeze(0)
    draft_kv = DynamicCache()
    hs = draft.draft(target_hidden=target_hidden, noise_embedding=noise,
                     position_ids=position_ids, past_key_values=draft_kv, use_cache=True)
    draft_logits = draft._lm_logits(hs[:, -b + 1:, :])  # [1, W, vocab]
    logp = F.log_softmax(draft_logits[0].float(), dim=-1)  # [W, vocab]
    return int(root), logp
