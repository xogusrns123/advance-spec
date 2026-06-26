"""Node-budget sweep of REAL per-method spec-decoding serving.

For each (model, method) cell and each node budget B in --budgets, boot one
SGLang server in real speculative mode with the oracle timing hook, run the
SAME fast BFCLv4 web_search workload (task=1, 2 turns) used by
measure_methods_latency.py, and reduce the per-step timing log to:

    accept_length_mean = mean(accept_lengths)              # = tau (mat)
    per_token_ms       = sum(step_total_ms) / sum(committed_tokens)
    step_ms            = median(step_total_ms)

This produces, per (model, method), a budget -> (tau, ms/tok) table for the
dual-axis "Node Budget" tradeoff figure (plot_budget_tradeoff.py).

Node budget B maps to `--speculative-num-draft-tokens` (the number of draft
tree nodes verified per step). For the tree methods (EAGLE3/EAGLE/STANDALONE)
we also raise topk/steps so the candidate pool can admit B nodes
(max_tree_capacity); DFlash/Suffix take B directly.

Reuses the validated per-method server configs (model ids, draft paths, mem
fractions, env, common flags, workload) from measure_methods_latency.py, and
max_tree_capacity from measure_eagle3_cost.py.

Run INSIDE the sglang-bench container (Blackwell GPU0), cwd /workspace:
    docker exec sglang-bench bash -lc \
      ". /opt/venv/bin/activate && cd /workspace && \
       python3 simulation/scripts/experiments/measure_budget_sweep.py \
         --cells 8b_suffix --budgets 16,256 --port 30066"
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Sibling (experiments/) and parent (scripts/) imports.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))            # measure_methods_latency.py
sys.path.insert(0, str(_HERE.parent))     # measure_eagle3_cost.py

import measure_methods_latency as mml      # noqa: E402
from measure_eagle3_cost import max_tree_capacity  # noqa: E402

from openai import OpenAI                   # noqa: E402


def slug(model: str) -> str:
    """Same convention as plot_method_latency.slug (Qwen/Qwen3-8B -> qwen3_8b)."""
    return (model.split("/")[-1].lower().replace(".", "").replace("-", "_"))


def _flag_value(spec: list[str], flag: str) -> str | None:
    """Return the value following `flag` in a spec flag list, or None."""
    try:
        return spec[spec.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _min_steps_for_budget(topk: int, budget: int, max_steps: int = 16) -> int:
    """Smallest num-steps S such that max_tree_capacity(topk, S) >= budget."""
    for s in range(1, max_steps + 1):
        if max_tree_capacity(topk, s) >= budget:
            return s
    return max_steps


def budget_to_spec(cell: dict, budget: int, topk: int = 16) -> tuple[list[str], int]:
    """Build the speculative spec-flag list for this cell at node budget `budget`.

    Returns (spec_flags, effective_budget). effective_budget may be < budget if
    the method caps it (e.g. DFlash block size).
    """
    spec = cell["spec"]
    algo = _flag_value(spec, "--speculative-algorithm")
    draft = _flag_value(spec, "--speculative-draft-model-path")

    base = ["--speculative-algorithm", algo]
    if draft is not None:
        base += ["--speculative-draft-model-path", draft]

    if algo in ("EAGLE3", "EAGLE", "STANDALONE"):
        # Tree methods: verify width = budget; pool must admit it.
        steps = _min_steps_for_budget(topk, budget)
        spec_out = base + [
            "--speculative-num-steps", str(steps),
            "--speculative-eagle-topk", str(topk),
            "--speculative-num-draft-tokens", str(budget),
            "--speculative-draft-attention-backend", "triton",
        ]
        return spec_out, budget

    if algo == "DFLASH":
        # DFlash block size (verify window) = --speculative-num-draft-tokens.
        # SGLang uses this value directly as block_size; if it differs from the
        # drafter's trained block it only warns (dflash_worker.py:172-186) and
        # runs, so the node-budget axis sweeps B as-is.
        return base + ["--speculative-num-draft-tokens", str(budget)], budget

    if algo == "SUFFIX":
        # Model-free trie: num-draft-tokens = max speculated tokens.
        return base + ["--speculative-num-draft-tokens", str(budget)], budget

    raise SystemExit(f"unknown speculative algorithm {algo!r} for cell")


def reduce_timing(entries: list[dict]) -> dict:
    """Reduce decode entries to tau (accept_length_mean) + exact per-token ms.

    Each decode entry is one (possibly batched) decode step:
      accept_lengths[i]   = #accepted draft tokens for sub-step i
      committed_tokens[i] = total committed for sub-step i (= 1 + accept_lengths[i])
      step_total_ms       = wall time for the whole entry
    Falls back to num_tokens when the lists are empty.
    """
    accept_flat: list[float] = []
    step_list: list[float] = []
    total_step_ms = 0.0
    total_committed = 0.0
    for e in entries:
        st = e.get("step_total_ms")
        if st is None:
            continue
        accs = list(e.get("accept_lengths") or [])
        comm = list(e.get("committed_tokens") or [])
        if not accs and not comm:
            nt = e.get("num_tokens")
            if nt is not None:
                comm = [nt]
                accs = [max(float(nt) - 1.0, 0.0)]
        if not comm and accs:
            comm = [a + 1 for a in accs]
        accept_flat.extend(float(a) for a in accs if isinstance(a, (int, float)))
        total_committed += sum(float(c) for c in comm if isinstance(c, (int, float)))
        total_step_ms += float(st)
        step_list.append(float(st))

    out: dict = {"n_samples": len(step_list)}
    if accept_flat:
        out["accept_length_mean"] = round(statistics.mean(accept_flat), 4)
    if step_list:
        out["step_ms"] = round(statistics.median(step_list), 3)
    vt = [e["verify_total_ms"] for e in entries if "verify_total_ms" in e]
    draft = [e["eagle3_draft_ms"] for e in entries if "eagle3_draft_ms" in e]
    if vt:
        out["target_cost_ms"] = round(statistics.median(vt), 3)
    if draft:
        out["draft_cost_ms"] = round(statistics.median(draft), 3)
    if total_committed > 0:
        out["per_token_ms"] = round(total_step_ms / total_committed, 3)
        out["committed_tokens_total"] = total_committed
    return out


def measure_budget(name: str, cell: dict, B: int, args, turn1, turn2,
                   env_base: dict) -> dict | None:
    """Boot one server at budget B, run the 2-turn workload, return metrics."""
    spec, eff_B = budget_to_spec(cell, B, topk=args.topk)
    model = cell["model"]
    mem = args.mem_fraction or cell["mem"]
    timing_log = Path(f"/tmp/timing_{name}_b{B}.jsonl")
    timing_log.unlink(missing_ok=True)

    env = os.environ.copy()
    env.update(env_base)
    env["SGLANG_ORACLE_TIMING_LOG"] = str(timing_log)

    # Idempotently patch the oracle hook into the SGLang worker sources.
    subprocess.run([sys.executable, "-m", "simulation.oracle.install_hook"],
                   env={**os.environ, "SGLANG_ORACLE_VANILLA": "1"},
                   cwd=str(mml.REPO), check=False)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--mem-fraction-static", str(mem),
        "--host", "0.0.0.0", "--port", str(args.port),
    ] + mml.COMMON_FLAGS + spec

    log_path = Path(f"/tmp/server_{name}_b{B}.log")
    print(f"\n{'='*70}\n[{name} B={B} (eff {eff_B})] {model} / {cell['method']}\n{'='*70}",
          flush=True)
    print(" ".join(cmd), flush=True)
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh)
    try:
        import requests as _rq
        booted = False
        t_boot = time.time()
        while time.time() - t_boot < 900:
            if proc.poll() is not None:
                print(f"  ERROR: server exited (code {proc.returncode}) during boot",
                      flush=True)
                break
            try:
                if _rq.get(f"http://localhost:{args.port}/health",
                           timeout=5).status_code == 200:
                    booted = True
                    break
            except Exception:
                pass
            time.sleep(3)
        if not booted:
            mml.kill_server(proc)
            print(f"  ERROR: server failed to boot. tail {log_path}:", flush=True)
            for ln in log_path.read_text(errors="ignore").splitlines()[-30:]:
                print("   ", ln, flush=True)
            return None

        client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="dummy")
        try:
            client.chat.completions.create(model=model, messages=turn1,
                                           max_tokens=8, temperature=0.0)
        except Exception as e:
            print(f"  WARN warmup: {e}", flush=True)

        offset = timing_log.stat().st_size if timing_log.exists() else 0
        r1 = client.chat.completions.create(model=model, messages=turn1,
                                            max_tokens=args.max_tokens, temperature=0.0)
        a1 = r1.choices[0].message.content or ""
        msgs2 = list(turn1) + [{"role": "assistant", "content": a1}] + list(turn2)
        client.chat.completions.create(model=model, messages=msgs2,
                                       max_tokens=args.max_tokens, temperature=0.0)

        window = mml.read_timing_window(timing_log, offset)
        summ = reduce_timing(window)
    finally:
        mml.kill_server(proc)
        time.sleep(3)

    if not summ.get("n_samples"):
        print("  ERROR: no decode timing captured", flush=True)
        return None

    summ.update({"budget": B, "effective_budget": eff_B})
    print(f"  tau={summ.get('accept_length_mean')} "
          f"per_tok={summ.get('per_token_ms')}ms step={summ.get('step_ms')}ms "
          f"(n={summ['n_samples']})", flush=True)
    return summ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="8b_eagle3,8b_small,8b_dflash,8b_suffix",
                    help=f"Comma list from {list(mml.CELLS)}")
    ap.add_argument("--budgets", default="16,32,64,128,256,512,1024")
    ap.add_argument("--topk", type=int, default=16,
                    help="eagle-topk for tree methods (pool width)")
    ap.add_argument("--port", type=int, default=30066)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--mem-fraction", default=None,
                    help="override cell mem-fraction-static (e.g. 0.85 for 27B large budgets)")
    ap.add_argument("--out", default="simulation/results/budget_tradeoff")
    ap.add_argument("--no-latency-only", action="store_true",
                    help="drop SGLANG_LATENCY_ONLY (full end-to-end timing)")
    ap.add_argument("--no-vanilla", action="store_true",
                    help="drop SGLANG_ORACLE_VANILLA")
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for c in cells:
        if c not in mml.CELLS:
            raise SystemExit(f"unknown cell {c}; choices {list(mml.CELLS)}")
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    env_base = dict(mml.BLACKWELL_ENV)
    if args.no_latency_only:
        env_base.pop("SGLANG_LATENCY_ONLY", None)
    if args.no_vanilla:
        env_base.pop("SGLANG_ORACLE_VANILLA", None)

    turn1, turn2 = mml.load_2turn_messages()
    out_root = (mml.REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    for name in cells:
        cell = mml.CELLS[name]
        model_dir = out_root / slug(cell["model"])
        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / f"{cell['method'].replace('/', '_')}.json"

        # Resume: load existing budgets so reruns skip completed points.
        doc = {"model": cell["model"], "method": cell["method"],
               "cell": name, "budgets": budgets, "results": []}
        # Resume: GOOD points are skipped; error/None points are RETRIED (e.g.
        # a transient OOM that a higher --mem-fraction may now clear).
        results_by_b: dict[int, dict] = {}
        if out_path.exists():
            try:
                prev = json.loads(out_path.read_text())
                if prev.get("cell") == name:
                    doc = prev
                    results_by_b = {int(r["budget"]): r
                                    for r in prev.get("results", [])}
            except (json.JSONDecodeError, KeyError):
                pass
        doc["budgets"] = budgets

        def _good(r) -> bool:
            return (isinstance(r, dict) and "error" not in r
                    and r.get("accept_length_mean") is not None)

        # Dedup by effective budget: methods that clamp (e.g. DFlash block size)
        # would otherwise re-measure the same config under many B labels and draw
        # a misleading flat line.
        seen_eff = {r.get("effective_budget") for r in results_by_b.values()
                    if _good(r)}

        # Early-stop: OOM/capacity failures are monotonic in budget, so once a
        # cell fails to MEASURE (not skip) twice in a row, higher budgets are
        # infeasible — stop this cell to avoid wasted boots.
        consec_fail = 0
        for B in budgets:
            if B in results_by_b and _good(results_by_b[B]):
                print(f"[{name} B={B}] cached — skip", flush=True)
                continue
            try:
                _, eff = budget_to_spec(cell, B, topk=args.topk)
            except SystemExit:
                eff = B
            if eff in seen_eff:
                print(f"[{name} B={B}] effective budget {eff} already measured — skip",
                      flush=True)
                continue
            try:
                res = measure_budget(name, cell, B, args, turn1, turn2, env_base)
            except Exception as e:  # server crash mid-request etc. — keep going
                print(f"  EXCEPTION at {name} B={B}: {type(e).__name__}: {e}",
                      flush=True)
                res = None
            if res is None:
                res = {"budget": B, "error": "measure_failed"}
            if _good(res):
                seen_eff.add(res.get("effective_budget"))
                consec_fail = 0
            else:
                consec_fail += 1
            results_by_b[B] = res
            doc["results"] = [results_by_b[b] for b in sorted(results_by_b)]
            out_path.write_text(json.dumps(doc, indent=2))
            print(f"  -> {out_path}", flush=True)
            if consec_fail >= 2:
                print(f"[{name}] {consec_fail} consecutive failures at B={B}; "
                      f"higher budgets infeasible — stopping cell.", flush=True)
                break

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
