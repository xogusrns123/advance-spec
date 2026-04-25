"""Shared result saving: full (pipeline) + response-only (human-readable),
plus per-request checkpoint helpers for agent resume.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path


def _atomic_write_json(data: dict, path: Path) -> None:
    """Write JSON atomically (tmpfile + rename) so a crash mid-write
    can never corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def checkpoint_path(output_path: str | os.PathLike) -> Path:
    """Path used for per-request incremental writes."""
    p = Path(output_path)
    return p.with_name(p.name + ".partial")


def load_checkpoint(output_path: str | os.PathLike) -> dict | None:
    """Return previously-saved progress as ``{metadata, questions: [...]}``.

    Reads ``<output>.partial`` first (live during a run) and falls back to
    the finalized ``<output>`` file. The fall-back lets a coordinator that
    invokes the agent multiple times in --num-requests=N --resume mode
    keep skipping already-done IDs across invocations: each invocation
    finalizes the current set into the final file, and the next one reads
    that file as its starting checkpoint.
    """
    p = checkpoint_path(output_path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Corrupt partial — discard and start fresh
            pass
    final = Path(output_path)
    if final.exists():
        try:
            with open(final) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def done_ids(checkpoint: dict | None, id_keys=("bfcl_id", "instance_id",
                                               "question_id")) -> set:
    """Extract already-processed request ids from a checkpoint dict."""
    if not checkpoint:
        return set()
    out = set()
    for q in checkpoint.get("questions", []):
        for k in id_keys:
            v = q.get(k)
            if v is not None:
                out.add(str(v))
                break
    return out


def append_to_checkpoint(output_path: str | os.PathLike, question: dict,
                         metadata: dict | None = None) -> None:
    """Append a finished question to the checkpoint and atomically rewrite.

    Cheap enough for per-request use because the partial is reasonably
    small (oracle entries dominate but a few hundred questions stays
    well under tens of MB).
    """
    p = checkpoint_path(output_path)
    cp = load_checkpoint(output_path) or {"metadata": {}, "questions": []}
    if metadata:
        cp["metadata"] = metadata
    cp["questions"].append(question)
    _atomic_write_json(cp, p)


def finalize_checkpoint(output_path: str | os.PathLike,
                        metadata: dict | None = None) -> dict | None:
    """Move the partial to the final path via save_agent_results.

    Returns the saved dict, or None if no checkpoint existed.
    """
    cp = load_checkpoint(output_path)
    if cp is None:
        return None
    if metadata:
        cp["metadata"] = metadata
    save_agent_results(cp, output_path)
    try:
        checkpoint_path(output_path).unlink()
    except FileNotFoundError:
        pass
    return cp


def save_agent_results(data: dict, output_path: str) -> None:
    """Save agent results as two files:

    1. {name}.json — full data with oracle entries (for pipeline)
    2. {name}_response.json — oracle stripped, human-readable

    Args:
        data: Full agent results dict with questions[].
        output_path: Path for the full output file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Full output (pipeline) — atomic so concurrent reads see a
    #    coherent file.
    _atomic_write_json(data, path)

    # 2. Response-only (human-readable)
    light = copy.deepcopy(data)
    for q in light.get("questions", []):
        # BFCL format: agent_metrics.steps[].spec_decode
        if "agent_metrics" in q:
            for s in q["agent_metrics"].get("steps", []):
                sd = s.pop("spec_decode", None)
                if sd:
                    s["oracle_entries_count"] = len(
                        sd.get("oracle_vanilla_entries", []))
        # SpecBench format: turns[].spec_decode
        if "turns" in q:
            for t in q["turns"]:
                if isinstance(t, dict):
                    sd = t.pop("spec_decode", None)
                    if sd:
                        t["oracle_entries_count"] = len(
                            sd.get("oracle_vanilla_entries", []))

    response_path = path.with_name(path.stem + "_response.json")
    _atomic_write_json(light, response_path)

    full_kb = path.stat().st_size / 1024
    resp_kb = response_path.stat().st_size / 1024
    print(f"  Full:     {path.name} ({full_kb:.0f} KB)")
    print(f"  Response: {response_path.name} ({resp_kb:.0f} KB)")
