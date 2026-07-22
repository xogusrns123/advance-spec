"""DC.load — target + DFlash draft loader (controller.md §2), reusing the repo's
validated ``DFlashOffline`` (scripts/experiments/dflash_offline.py, faithfulness-
checked vs served DFlash at 99.75% on 8B).

``load(target, draft)`` returns ``(TargetWrapper, DFlashOffline, cfg_d)`` — the
exact 3-tuple ``run_partialwarm_tree.main`` unpacks. ``cfg_d`` carries the
draft-config values (block_size / mask_token_id / target_layer_ids), read from
the DFlash config, NOT hardcoded (controller.md §2).

TargetWrapper is the offline-GT-walk engine. It proxies the HF target model (so
``run_partialwarm_tree.greedy`` and ``target.config`` / ``target.device`` keep
working) and, per prompt, memoizes:
  * the full target-greedy trace ``prompt + greedy_output`` (regenerated the
    first time it sees a new prompt; extended on demand), and
  * the concatenated DFlash target-layer features over that full sequence
    (one ``use_cache=False`` forward via DFlashOffline's hooks).
Because greedy spec-decode commits only greedy tokens, every ctx the round loop
passes is a prefix of this cached trace — so ``dflash_block_logp`` slices cached
features (teacher-forcing root = trace token) and ``tree_verify`` walks against
the cached trace, with just ONE 27B generate + ONE feature forward per prompt.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, "/workspace/vendor/ddtree")
sys.path.insert(0, "/workspace/vendor/ddtree/model")
sys.path.insert(0, "/workspace/simulation/scripts/experiments")

from dflash_offline import DFlashOffline  # noqa: E402  (repo, validated)


class TargetWrapper:
    """Proxies the HF target model and serves cached greedy-trace + features."""

    def __init__(self, dfo: DFlashOffline, eos_token_id: int, base_horizon: int = 96):
        self.dfo = dfo
        self.model = dfo.target
        self.eos_token_id = int(eos_token_id)
        self.base_horizon = int(base_horizon)
        # per-prompt cache
        self._prompt: Optional[List[int]] = None
        self._full: Optional[List[int]] = None          # prompt + greedy output
        self._feat: Optional[torch.Tensor] = None        # [1, len(full), K*H]

    # -- proxy so greedy()/target.device/target.config keep working -------------
    def __call__(self, *a, **k):
        return self.model(*a, **k)

    def __getattr__(self, name):
        # only reached for attrs not set on the wrapper (device, config, lm_head…)
        return getattr(self.__dict__["model"], name)

    # -- greedy generation (KV-cached) -----------------------------------------
    @torch.inference_mode()
    def _generate(self, prompt_ids: List[int], horizon: int) -> List[int]:
        dev = self.model.device
        out: List[int] = []
        cur = torch.tensor([prompt_ids], device=dev)
        past = None
        for _ in range(horizon):
            o = self.model(cur, use_cache=True, past_key_values=past)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            out.append(nxt)
            if nxt == self.eos_token_id:
                break
            cur = torch.tensor([[nxt]], device=dev)
        return out

    def _build(self, prompt_ids: List[int], horizon: int) -> None:
        out = self._generate(prompt_ids, horizon)
        self._prompt = list(prompt_ids)
        self._full = list(prompt_ids) + out
        t = torch.tensor([self._full], device=self.model.device)
        self._feat = self.dfo._capture_target_hidden(t)   # [1, S, K*H]

    def _ensure(self, ids: List[int]) -> None:
        """Guarantee self._full covers `ids` (a prefix of the current trace),
        (re)generating for a new prompt or extending past the horizon."""
        n = len(ids)
        if self._full is not None:
            fl = len(self._full)
            if n <= fl and ids == self._full[:n]:
                return                                     # hit
            if n >= fl and self._full == ids[:fl]:
                self._build(self._prompt, n - len(self._prompt) + 64)  # extend
                return
        self._build(ids, self.base_horizon)               # new prompt

    # -- API used by dflash_block_logp / tree_verify ---------------------------
    def features_upto(self, pos: int) -> torch.Tensor:
        return self._feat[:, :pos, :]

    def trace_token(self, pos: int) -> int:
        if pos >= len(self._full):
            self._build(self._prompt, pos - len(self._prompt) + 64)
        return int(self._full[pos])

    @torch.inference_mode()
    def greedy_continuation(self, ctx_root: torch.Tensor, max_len: int) -> List[int]:
        ids = ctx_root[0].tolist() if hasattr(ctx_root, "tolist") else list(ctx_root)
        self._ensure(ids)
        L = len(ids)
        end = L + int(max_len)
        if end > len(self._full):
            self._build(self._prompt, end - len(self._prompt) + 8)
        return [int(t) for t in self._full[L:end]]


def load(target_name: str, draft_name: str, device: str = "cuda") -> Tuple[TargetWrapper, DFlashOffline, dict]:
    from transformers import AutoTokenizer
    dfo = DFlashOffline(target_name, draft_name, device=device)
    tok = AutoTokenizer.from_pretrained(target_name)
    eos = tok.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    horizon = int(os.environ.get("WRAP_HORIZON", os.environ.get("MAXTOK", "96")))
    tw = TargetWrapper(dfo, eos_token_id=eos, base_horizon=horizon)
    cfg_d = {
        "block_size": dfo.block_size,
        "mask_token_id": dfo.mask_token_id,
        "target_layer_ids": list(dfo.layer_ids),
    }
    return tw, dfo, cfg_d
