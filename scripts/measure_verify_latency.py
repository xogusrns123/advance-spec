"""Measure verify latency at budget 1-15 via verify server."""

import json
import sys
import requests


def main():
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "results/qwen3_8b/pipeline_test/union_trie_data.jsonl"

    # Pick a record with ~200 token context
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            ctx_len = len(rec.get("context_token_ids", []))
            trie_len = len(rec["union_trie"]["token_ids"])
            if 150 < ctx_len < 250 and trie_len >= 15:
                break

    print(f"Context: {len(rec['context_token_ids'])} tokens, Trie: {len(rec['union_trie']['token_ids'])} nodes")

    resp = requests.post(f"{server_url}/benchmark_verify", json={
        "context_token_ids": rec["context_token_ids"],
        "tree_token_ids": rec["union_trie"]["token_ids"],
        "tree_parents": rec["union_trie"]["parents"],
        "budgets": list(range(1, 16)),
        "n_trials": 50,
    }, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    vanilla = data["vanilla_step_ms"]
    latencies = data["verify_latencies"]

    print(f"\nVanilla step: {vanilla:.3f} ms")
    print()
    print(f"{'Budget':>6} | {'Verify (ms)':>11} | {'Overhead vs vanilla':>20}")
    print("-" * 45)

    for B in range(1, 16):
        v = latencies[str(B)]
        overhead = v / vanilla
        print(f"{B:>6} | {v:>11.3f} | {overhead:>19.2f}x")

    # Output as JSON for simulation
    output = {
        "vanilla_step_ms": vanilla,
        "verify_latencies_ms": {int(k): v for k, v in latencies.items()},
    }
    out_path = "results/qwen3_8b/pipeline_test/latency_config.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
