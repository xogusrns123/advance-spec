"""Spider 2.0-DBT (AgenticSQL) workload via the OFFICIAL mini-swe-agent (textbased).

The repo's original spider2_dbt_agent uses langchain `llm.bind_tools()`, which fails
with Qwen3.5 thinking models (reasoning output isn't parsed into OpenAI tool_calls —
same failure that killed the old swebench_agent). This driver reuses the PROVEN
mini-swe-agent textbased machinery (build_live_agent from minisweagent_agent) that
already produced valid SWEBench trajectories with Qwen3.5-27B: the model emits a
```bash ...``` action parsed from TEXT (thinking disabled), run in a LocalEnvironment
inside each instance directory (dbt_project.yml + profiles.yml + models/ + <db>.duckdb).

Each instance dir is a real dbt/duckdb project; the agent explores it, edits the SQL
models, runs dbt/duckdb via bash, and submits. Captures conversation (+ per-model-call
oracle entries) in the SAME schema as minisweagent_agent so the readable/merge tooling
is shared.

CLI mirrors minisweagent_agent so measure_chain_hybrid / the collect script can launch it.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.oracle.oracle_patch import is_oracle_enabled
from simulation.agents.minisweagent_agent import build_live_agent


DBT_TASK_PREFIX = (
    "You are working inside a dbt project that uses a local DuckDB database "
    "(see dbt_project.yml and profiles.yml; the .duckdb file is in this directory). "
    "Complete the task by editing/creating the dbt SQL models and verifying with dbt/duckdb "
    "(e.g. `dbt run`, or `python3 -c \"import duckdb; ...\"`). Task:\n\n"
)


def run_live(url, model, instances, instances_dir, max_steps, collect_oracle):
    out = []
    for inst in instances:
        iid = inst["instance_id"]
        instr = inst.get("instruction", "")
        cat = inst.get("type", "")
        wd = os.path.abspath(os.path.join(instances_dir, iid))
        if not os.path.isdir(wd):
            print(f"  skip {iid}: instance dir not found ({wd})", flush=True)
            continue
        agent, steps = build_live_agent(url, model, wd, max_steps, collect_oracle)
        task = DBT_TASK_PREFIX + instr
        print(f"  [{iid}] cat={cat} agent={type(agent).__name__} task_len={len(task)}", flush=True)
        t0 = time.perf_counter()
        try:
            agent.run(task)
        except Exception as e:
            import traceback
            print(f"  agent.run({iid}) raised: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in agent.messages]
        out.append({
            "instance_id": iid, "category": cat, "num_turns": len(steps),
            "total_latency": time.perf_counter() - t0,
            "turns": [{"messages": msgs}],
            "agent_metrics": {"steps": [
                {"step": k, "content": s["content"],
                 "spec_decode": {"oracle_vanilla_entries": s["oracle_vanilla_entries"]}}
                for k, s in enumerate(steps)]}})
        n_oracle = sum(len(s["oracle_vanilla_entries"]) for s in steps)
        print(f"  {iid}: {len(steps)} model-calls, {n_oracle} oracle entries", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Spider2.0-DBT workload (mini-swe-agent textbased)")
    ap.add_argument("--url", default="http://localhost:30000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--instances-dir", default="data/spider2_dbt/instances")
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=1)  # serial (oracle log is global)
    args = ap.parse_args()
    collect_oracle = is_oracle_enabled()
    print(f"spider2-dbt workload (oracle={'on' if collect_oracle else 'off'})", flush=True)

    rows = [json.loads(l) for l in open(args.input_file)]
    rows = rows[args.offset:]
    if args.num_requests is not None:
        rows = rows[:args.num_requests]
    questions = run_live(args.url, args.model, rows, args.instances_dir,
                         args.max_iterations, collect_oracle)
    meta = {"model": args.model, "benchmark": "spider2_dbt_minisweagent",
            "num_requests": len(questions)}
    json.dump({"metadata": meta, "questions": questions}, open(args.output_file, "w"))
    print(f"Saved {len(questions)} instances -> {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
