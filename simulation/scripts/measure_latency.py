"""Measure tree verification latency at different budget sizes."""

import json
import statistics
import sys
import time

import requests


def main():
    server_url = "http://localhost:8100/verify_tree"
    data_path = "simulation/results/qwen3_8b/pipeline_test/union_trie_data_server_pt.jsonl"

    # Load records, pick one with moderate context length
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Find record with ~200 token context and >=10 trie nodes
    rec = None
    for r in records:
        ctx_len = len(r.get("context_token_ids", []))
        trie_len = len(r["union_trie"]["token_ids"])
        if 150 < ctx_len < 250 and trie_len >= 10:
            rec = r
            break
    if rec is None:
        rec = records[len(records) // 2]

    ctx = rec["context_token_ids"]
    full_tids = rec["union_trie"]["token_ids"]
    full_pids = rec["union_trie"]["parents"]
    print(f"Context len={len(ctx)}, full trie={len(full_tids)} nodes")

    # Warmup
    for _ in range(5):
        requests.post(server_url, json={
            "context_token_ids": ctx,
            "tree_token_ids": full_tids[:1],
            "tree_parents": [-1],
        })

    # Measure at different budget sizes
    budgets = [1, 2, 3, 4, 5, 6, 8, 10, 15]
    N_TRIALS = 30

    print()
    print(f"{'Budget':>8} | {'Latency (ms)':>14} | {'Std (ms)':>10}")
    print("-" * 40)

    results = {}
    for B in budgets:
        tids = full_tids[:B]
        pids = full_pids[:B]
        pids = [p if p < B else -1 for p in pids]

        times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            resp = requests.post(server_url, json={
                "context_token_ids": ctx,
                "tree_token_ids": tids,
                "tree_parents": pids,
            })
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)

        avg = statistics.mean(times)
        std = statistics.stdev(times)
        results[B] = avg
        print(f"{B:>8} | {avg:>14.2f} | {std:>10.2f}")

    # Vanilla baseline (1 token)
    vanilla = results[1]

    # MAT values from previous analysis (Qwen3-8B, 3338 steps)
    mat_eu = {1: 0.4643, 2: 0.6905, 3: 0.8074, 4: 0.8709, 5: 0.9068,
              6: 0.9296, 8: 0.9554, 10: 0.9682, 15: 0.9799}
    mat_c1 = {1: 0.4374, 2: 0.6615, 3: 0.7942, 4: 0.8628, 5: 0.9053,
              6: 0.9365, 8: 0.9703, 10: 0.9805, 15: 0.9949}

    print()
    print("=" * 80)
    print("LATENCY-AWARE SPEEDUP ANALYSIS")
    print("=" * 80)
    print()
    print(f"{'Budget':>6} | {'Verify':>8} | {'EU MAT':>7} | {'C1 MAT':>7} | "
          f"{'EU ms/tok':>10} | {'C1 ms/tok':>10} | {'EU speedup':>11} | {'C1 speedup':>11}")
    print("-" * 90)

    for B in budgets:
        v_ms = results[B]
        eu_mat = mat_eu[B]
        c1_mat = mat_c1[B]
        eu_ms_tok = v_ms / max(eu_mat, 0.01)
        c1_ms_tok = v_ms / max(c1_mat, 0.01)
        eu_speedup = vanilla / eu_ms_tok
        c1_speedup = vanilla / c1_ms_tok
        print(f"{B:>6} | {v_ms:>7.1f}ms | {eu_mat:>7.4f} | {c1_mat:>7.4f} | "
              f"{eu_ms_tok:>9.1f}ms | {c1_ms_tok:>9.1f}ms | {eu_speedup:>10.2f}x | {c1_speedup:>10.2f}x")

    print()
    print(f"Vanilla: {vanilla:.1f}ms/tok (1.00x)")
    print()

    # Best budget for each method
    best_eu_B = max(budgets, key=lambda B: vanilla / (results[B] / max(mat_eu[B], 0.01)))
    best_c1_B = max(budgets, key=lambda B: vanilla / (results[B] / max(mat_c1[B], 0.01)))
    best_eu_speedup = vanilla / (results[best_eu_B] / mat_eu[best_eu_B])
    best_c1_speedup = vanilla / (results[best_c1_B] / mat_c1[best_c1_B])
    print(f"Best EU Oracle: budget={best_eu_B}, speedup={best_eu_speedup:.2f}x")
    print(f"Best Choose-One: budget={best_c1_B}, speedup={best_c1_speedup:.2f}x")
    print(f"EU advantage: {(best_eu_speedup / best_c1_speedup - 1) * 100:.1f}% faster")


if __name__ == "__main__":
    main()
