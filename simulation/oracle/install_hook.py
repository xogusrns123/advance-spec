"""
Install SUFFIX as a speculative decoding algorithm in SGLang.

Patches SGLang source files on disk so that all spawned subprocesses
(scheduler, tp workers) also recognize the SUFFIX algorithm.

Usage:
    python3 -m simulation.oracle.install_hook [sglang args...]

Example:
    python3 -m simulation.oracle.install_hook \\
        --model-path zai-org/GLM-4.7-Flash \\
        --tp-size 4 \\
        --speculative-algorithm SUFFIX \\
        --speculative-num-draft-tokens 16 \\
        --mem-fraction-static 0.8 \\
        --disable-cuda-graph \\
        --host 0.0.0.0 --port 30000
"""

from __future__ import annotations

# Limit torch.compile workers BEFORE any torch import. Covers the parent process.
import os as _os
if "TORCHINDUCTOR_COMPILE_THREADS" not in _os.environ:
    _os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import logging
import re
import site
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_installed = False


def _get_sglang_root() -> Path:
    """Find the installed sglang package root."""
    import sglang
    return Path(sglang.__file__).parent


def install_suffix_algorithm() -> None:
    """
    Patch SGLang source files on disk to support --speculative-algorithm SUFFIX.

    Patches (idempotent):
    1. spec_info.py: SpeculativeAlgorithm enum + is_suffix() + create_worker()
    2. spec_info.py: SpecInputType enum (not strictly needed since we reuse NGRAM_VERIFY)
    3. server_args.py: argparse choices for --speculative-algorithm
    4. server_args.py: validation (__post_init__) to treat SUFFIX like NGRAM
    """
    global _installed
    if _installed:
        return

    root = _get_sglang_root()

    _patch_spec_info(root / "srt" / "speculative" / "spec_info.py")
    _patch_server_args(root / "srt" / "server_args.py")
    _patch_scheduler(root / "srt" / "managers" / "scheduler.py")

    _installed = True
    logger.info("Installed SUFFIX speculative algorithm into SGLang (on-disk patches)")


def _patch_spec_info(path: Path) -> None:
    """Add SUFFIX to SpeculativeAlgorithm and create_worker()."""
    text = path.read_text()

    if "SUFFIX" in text:
        return  # already patched

    # 1. Add SUFFIX member to the enum (after NGRAM)
    text = text.replace(
        "    NGRAM = auto()\n    NONE = auto()",
        "    NGRAM = auto()\n    SUFFIX = auto()\n    NONE = auto()",
    )

    # 2. Add is_suffix() method (after is_ngram)
    text = text.replace(
        "    def is_ngram(self) -> bool:\n        return self == SpeculativeAlgorithm.NGRAM",
        "    def is_ngram(self) -> bool:\n        return self == SpeculativeAlgorithm.NGRAM\n\n"
        "    def is_suffix(self) -> bool:\n        return self == SpeculativeAlgorithm.SUFFIX",
    )

    # 3. Add SUFFIX dispatch in create_worker() (before the final raise)
    text = text.replace(
        '        raise ValueError("Unreachable code path in create_worker.")',
        "        elif self.is_suffix():\n"
        "            if enable_overlap:\n"
        "                raise ValueError(\n"
        '                    f"Speculative algorithm {self.name} does not support overlap worker creation."\n'
        "                )\n"
        "            from hybrid_spec_decoding.sglang_integration.suffix_worker import SuffixWorker\n"
        "            return SuffixWorker\n\n"
        '        raise ValueError("Unreachable code path in create_worker.")',
    )

    path.write_text(text)
    logger.info(f"Patched {path}")


def _patch_server_args(path: Path) -> None:
    """Add SUFFIX to argparse choices and validation."""
    text = path.read_text()

    if '"SUFFIX"' in text:
        return  # already patched

    # 1. Add SUFFIX to argparse choices
    text = text.replace(
        'choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM"]',
        'choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM", "SUFFIX"]',
    )

    # 2. In validation (__post_init__), SUFFIX should be treated like NGRAM.
    #    The key validation is: if speculative_algorithm == "NGRAM" → skip draft model checks.
    #    We add SUFFIX to the same conditional.
    #    Find: if self.speculative_algorithm == "NGRAM":
    #    Replace with: if self.speculative_algorithm in ("NGRAM", "SUFFIX"):
    text = text.replace(
        'if self.speculative_algorithm == "NGRAM":',
        'if self.speculative_algorithm in ("NGRAM", "SUFFIX"):',
    )

    # 3. Also handle the draft model requirement check:
    #    speculative_algorithm != "NGRAM" → speculative_algorithm not in ("NGRAM", "SUFFIX")
    text = text.replace(
        'self.speculative_algorithm != "NGRAM"',
        'self.speculative_algorithm not in ("NGRAM", "SUFFIX")',
    )

    path.write_text(text)
    logger.info(f"Patched {path}")


def _patch_scheduler(path: Path) -> None:
    """Make scheduler treat SUFFIX like NGRAM (no draft KV cache)."""
    text = path.read_text()

    if "is_suffix()" in text:
        return  # already patched

    # In init_disaggregation: skip draft KV pool for SUFFIX too
    text = text.replace(
        "self.spec_algorithm.is_ngram()",
        "self.spec_algorithm.is_ngram() or self.spec_algorithm.is_suffix()",
    )

    path.write_text(text)
    logger.info(f"Patched {path}")


def install_oracle_patch() -> None:
    """
    Install oracle vanilla patch that hooks EAGLEWorker and MultiLayerEagleWorker
    __init__ to auto-patch instances for draft token logging.

    Activated by SGLANG_ORACLE_VANILLA=1 env var.
    Both workers use the same EagleVerifyInput format, so the same
    patch_eagle_worker_full() works for both.
    """
    import os
    if os.environ.get("SGLANG_ORACLE_VANILLA", "0") != "1":
        return

    root = _get_sglang_root()

    # Patch EAGLEWorker (EAGLE3)
    _inject_oracle_into_worker(
        root / "srt" / "speculative" / "eagle_worker.py",
        "EAGLEWorker",
    )

    # Patch MultiLayerEagleWorker (MTP) — same verify interface
    _inject_oracle_into_worker(
        root / "srt" / "speculative" / "multi_layer_eagle_worker.py",
        "MultiLayerEagleWorker",
    )

    # Patch StandaloneWorker (STANDALONE algorithm) — does NOT call super().__init__()
    # so eagle_worker.py injection alone misses it. Inject directly.
    _inject_oracle_into_worker(
        root / "srt" / "speculative" / "standalone_worker.py",
        "StandaloneWorker",
    )


NEW_ORACLE_IMPORT = "from simulation.oracle.oracle_patch import patch_eagle_worker_full"
OLD_ORACLE_IMPORT = (
    "from hybrid_spec_decoding.sglang_integration.oracle_patch "
    "import patch_eagle_worker_full"
)


def _inject_oracle_into_worker(worker_path: Path, worker_name: str) -> None:
    """Inject oracle patch call at the end of a worker's __init__."""
    if not worker_path.exists():
        logger.warning(f"{worker_path} not found, skipping oracle patch for {worker_name}")
        return

    text = worker_path.read_text()

    # Migration: prior versions of the patch used the now-removed
    # ``hybrid_spec_decoding.sglang_integration.oracle_patch`` import path.
    # Rewrite it in place so older on-disk patches keep working after the
    # module moved to simulation.oracle.
    if OLD_ORACLE_IMPORT in text:
        text = text.replace(OLD_ORACLE_IMPORT, NEW_ORACLE_IMPORT)
        worker_path.write_text(text)
        logger.info(f"Migrated oracle import path in {worker_path} ({worker_name})")
        return

    if NEW_ORACLE_IMPORT in text:
        return  # already patched with current path

    # Ensure `from __future__ import annotations` so runtime type hints
    # (e.g. -> ModelRunner in TYPE_CHECKING block) don't raise NameError.
    if "from __future__ import annotations" not in text:
        # Insert after the license header comment block
        lines = text.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if not line.startswith("#") and line.strip():
                insert_idx = i
                break
        lines.insert(insert_idx, "from __future__ import annotations")
        text = "\n".join(lines)

    # Find the sentinel: last assignment in __init__
    # EAGLEWorker: self.extend_lens = torch.empty(...)
    # MultiLayerEagleWorker: may differ, try common patterns
    sentinel = "self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)"
    if sentinel not in text:
        # Fallback: try to find end of __init__ by looking for the pattern
        # "self.extend_lens = ..." which both workers have
        import re
        match = re.search(r'(self\.extend_lens\s*=\s*[^\n]+)', text)
        if match:
            sentinel = match.group(1)
        else:
            logger.warning(f"Could not find __init__ sentinel in {worker_path} for oracle patch")
            return

    patch_code = (
        sentinel + "\n\n"
        f"        # Oracle vanilla patch: log draft tokens per step ({worker_name})\n"
        "        import os as _os\n"
        "        if _os.environ.get('SGLANG_ORACLE_VANILLA', '0') == '1':\n"
        "            from simulation.oracle.oracle_patch import patch_eagle_worker_full\n"
        "            patch_eagle_worker_full(self)\n"
    )
    text = text.replace(sentinel, patch_code)
    worker_path.write_text(text)
    logger.info(f"Installed oracle vanilla patch into {worker_path} ({worker_name})")


def _start_process_watchdog(max_children: int = 50, check_interval: int = 5):
    """Background thread that kills the server if child processes explode."""
    import os
    import signal
    import threading

    my_pid = os.getpid()

    def _watchdog():
        while True:
            try:
                children = os.popen(f"pgrep -c -P {my_pid}").read().strip()
                n = int(children) if children else 0
                if n > max_children:
                    print(
                        f"\n[WATCHDOG] Too many child processes ({n} > {max_children}). "
                        f"Killing server to prevent fork bomb.",
                        flush=True,
                    )
                    os.killpg(os.getpgid(my_pid), signal.SIGKILL)
            except Exception:
                pass
            threading.Event().wait(check_interval)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()


if __name__ == "__main__":
    _start_process_watchdog(max_children=50, check_interval=5)

    install_suffix_algorithm()
    install_oracle_patch()

    sglang_args = sys.argv[1:]
    if "--" in sglang_args:
        sglang_args = sglang_args[sglang_args.index("--") + 1:]

    if sglang_args:
        from sglang.launch_server import prepare_server_args, run_server
        server_args = prepare_server_args(sglang_args)
        run_server(server_args)
    else:
        print("Hooks installed (SUFFIX algorithm + oracle patch).")
        print(f"Example: python -m {__name__} "
              "--model-path zai-org/GLM-4.7-Flash --tp-size 4 "
              "--speculative-algorithm EAGLE3 ...")
