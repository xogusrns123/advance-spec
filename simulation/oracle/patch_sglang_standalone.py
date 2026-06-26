"""Make STANDALONE speculative decoding work with a Qwen3.5 (multimodal/hybrid)
draft model, the proper way.

Two independent sglang limitations block `Qwen/Qwen3.5-0.8B` (draft) + a
Qwen3.5 target under `--speculative-algorithm STANDALONE`:

1. ModelConfig._config_draft_model() remaps ANY draft architecture to its
   *MTP/*NextN head variant (Qwen3_5ForConditionalGeneration ->
   Qwen3_5ForCausalLMMTP) WITHOUT checking the algorithm. Correct for EAGLE/MTP
   (draft == target, shared hidden), wrong for STANDALONE (independent base LM,
   different hidden size) -> the MTP head's cat([input_embeds, target_hidden])
   fails ("Expected size 5120 but got size 1024").

2. The Qwen3.5 base architecture is multimodal (Qwen3_5ForConditionalGeneration
   = Qwen3VL). Its forward unconditionally runs the multimodal embed routine,
   which crashes on the text-only draft-extend forward ("shape mismatch:
   value tensor of shape [64, 1024] cannot be broadcast to ... [0, 1024]").
   The native TEXT causal LM (Qwen3_5ForCausalLM, which also handles the
   GatedDeltaNet hybrid layers) is the right class but is not registered.

Fix (proper, upstreamable):
- Register the native text classes Qwen3_5ForCausalLM / Qwen3_5MoeForCausalLM
  as loadable architectures (add to qwen3_5.py EntryClass).
- Thread server_args.speculative_algorithm into ModelConfig and, for STANDALONE
  drafts, skip the MTP/NextN remap and instead drop a multimodal
  *ForConditionalGeneration draft to its text *ForCausalLM counterpart.

Idempotent. Targets the installed sglang in the active environment.
"""

import importlib.util
import py_compile


def _path(modname: str) -> str:
    spec = importlib.util.find_spec(modname)
    assert spec and spec.origin, f"{modname} not found"
    return spec.origin


# (anchor, replacement, sentinel-that-means-already-applied)
QWEN35_PATCHES = [
    (
        "EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]",
        "EntryClass = [\n"
        "    Qwen3_5MoeForConditionalGeneration,\n"
        "    Qwen3_5ForConditionalGeneration,\n"
        "    Qwen3_5ForCausalLM,\n"
        "    Qwen3_5MoeForCausalLM,\n"
        "]",
        "Qwen3_5ForConditionalGeneration,\n    Qwen3_5ForCausalLM,",
    ),
]

MODEL_CONFIG_PATCHES = [
    (
        '        model_config_parser: str = "auto",\n    ) -> None:',
        '        model_config_parser: str = "auto",\n'
        "        speculative_algorithm: Optional[str] = None,\n"
        "    ) -> None:",
        "speculative_algorithm: Optional[str] = None,",
    ),
    (
        "        self.model_config_parser = model_config_parser\n",
        "        self.model_config_parser = model_config_parser\n"
        "        self.speculative_algorithm = speculative_algorithm\n",
        "self.speculative_algorithm = speculative_algorithm",
    ),
    (
        "            model_config_parser=server_args.model_config_parser,\n"
        "            **kwargs,",
        "            model_config_parser=server_args.model_config_parser,\n"
        "            speculative_algorithm=(\n"
        "                server_args.speculative_algorithm if is_draft_model else None\n"
        "            ),\n"
        "            **kwargs,",
        "speculative_algorithm=(",
    ),
    (
        "    def _config_draft_model(self):\n"
        "        is_draft_model = self.is_draft_model\n",
        "    def _config_draft_model(self):\n"
        "        is_draft_model = self.is_draft_model\n"
        "        # STANDALONE drafts are independent base LMs (their hidden size may\n"
        "        # differ from the target); they are NOT MTP/NextN heads. Skip the\n"
        "        # head remap, and for a multimodal *ForConditionalGeneration draft\n"
        "        # drop to its text *ForCausalLM counterpart AND use the text\n"
        "        # sub-config (hf_text_config) as the model config so it runs as a\n"
        "        # pure text LM (the text class expects config.hidden_size, which\n"
        "        # lives on text_config for the Qwen3VL multimodal config).\n"
        '        if is_draft_model and getattr(self, "speculative_algorithm", None) == "STANDALONE":\n'
        "            _to_text = {\n"
        '                "Qwen3_5ForConditionalGeneration": "Qwen3_5ForCausalLM",\n'
        '                "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeForCausalLM",\n'
        "            }\n"
        "            _arch = self.hf_config.architectures[0]\n"
        "            if _arch in _to_text and self.hf_config is not self.hf_text_config:\n"
        "                self.hf_config = self.hf_text_config\n"
        "                self.hf_config.architectures = [_to_text[_arch]]\n"
        "            return\n",
        "STANDALONE drafts are independent base LMs",
    ),
]


def _apply(path, patches):
    text = open(path).read()
    changed = False
    for anchor, replacement, sentinel in patches:
        if sentinel in text:
            continue
        if anchor not in text:
            raise SystemExit(f"anchor missing in {path} (version drift): {anchor[:50]!r}")
        text = text.replace(anchor, replacement, 1)
        changed = True
    if changed:
        open(path, "w").write(text)
    py_compile.compile(path, doraise=True)
    return changed


def main() -> int:
    q = _apply(_path("sglang.srt.models.qwen3_5"), QWEN35_PATCHES)
    m = _apply(_path("sglang.srt.configs.model_config"), MODEL_CONFIG_PATCHES)
    print(f"qwen3_5.py: {'patched' if q else 'already patched'}")
    print(f"model_config.py: {'patched' if m else 'already patched'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
