"""Extract trajectory from Round 1 oracle agent_results.json.

Reads oracle_vanilla_entries and builds a per-request token sequence
that Round 2 (MTP) can follow exactly.

Usage:
    python3 -m simulation.pipeline.extract_trajectory \
        --agent-results results/glm4_flash/oracle_vanilla/agent_results.json \
        --output /tmp/oracle_trajectory.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def extract_trajectories(agent_results_path: str) -> dict[str, list[int]]:
    """Extract per-request token sequences from oracle vanilla entries.

    Returns:
        Dict mapping req_id → list of token IDs in generation order.
    """
    with open(agent_results_path) as f:
        data = json.load(f)

    trajectories: dict[str, list[int]] = defaultdict(list)

    for question in data["questions"]:
        if "agent_metrics" in question:
            # BFCL / SWE-bench format: agent_metrics.steps[].spec_decode
            for step in question["agent_metrics"]["steps"]:
                entries = step.get("spec_decode", {}).get("oracle_vanilla_entries", [])
                for entry in entries:
                    req_id = entry.get("req_id", "")
                    tokens = entry.get("tokens", [[]])
                    if tokens and tokens[0]:
                        trajectories[req_id].extend(tokens[0])
        elif "turns" in question:
            # SpecBench format: turns[].spec_decode
            for turn in question["turns"]:
                if isinstance(turn, dict):
                    entries = turn.get("spec_decode", {}).get("oracle_vanilla_entries", [])
                    for entry in entries:
                        req_id = entry.get("req_id", "")
                        tokens = entry.get("tokens", [[]])
                        if tokens and tokens[0]:
                            trajectories[req_id].extend(tokens[0])

    return dict(trajectories)


def main():
    parser = argparse.ArgumentParser(
        description="Extract trajectory from oracle agent_results.json"
    )
    parser.add_argument("--agent-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trajectories = extract_trajectories(args.agent_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(trajectories, f)

    total_tokens = sum(len(t) for t in trajectories.values())
    print(f"Extracted {len(trajectories)} trajectories, {total_tokens} total tokens")
    for req_id, tokens in trajectories.items():
        print(f"  {req_id}: {len(tokens)} tokens")


if __name__ == "__main__":
    main()
