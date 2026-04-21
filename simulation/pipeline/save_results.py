"""Shared result saving: full (pipeline) + response-only (human-readable)."""

from __future__ import annotations

import copy
import json
from pathlib import Path


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

    # 1. Full output (pipeline)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
    with open(response_path, "w") as f:
        json.dump(light, f, indent=2, ensure_ascii=False)

    full_kb = path.stat().st_size / 1024
    resp_kb = response_path.stat().st_size / 1024
    print(f"  Full:     {path.name} ({full_kb:.0f} KB)")
    print(f"  Response: {response_path.name} ({resp_kb:.0f} KB)")
