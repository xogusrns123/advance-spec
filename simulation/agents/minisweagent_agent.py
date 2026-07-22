"""SWE-Bench workload agent backed by the OFFICIAL mini-swe-agent (latest, v2.4.x).

WHY: our previous custom swebench_agent used llm.bind_tools() and relied on the server to parse
Qwen3.5's thinking+tool output into OpenAI tool_calls; with reasoning_parser=None this failed and
the agentic loop died at turn 1. mini-swe-agent's LitellmTextbasedModel parses the bash ACTION from
the model's TEXT via regex (ignoring <think>), so it is robust to thinking models — the official,
correct harness.

LIVE (record / select1 / oracle without --replay): drive the official mini-swe-agent DefaultAgent
(swebench.yaml config + LocalEnvironment on our pre-cloned repo + textbased model -> our chain-hybrid
server). Captures the conversation (for later replay) + per-model-call oracle entries.
REPLAY (--replay): re-send the recorded conversation turn-by-turn via the OpenAI client (no bash
exec) so the chain-hybrid server (oracle/select1 mode) logs per-position proposer decisions on the
SAME trajectory -> realized raw/oracle/calib ladder.

CLI mirrors the other workload agents so measure_chain_hybrid can launch it.
"""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path

from simulation.oracle.oracle_patch import (
    get_oracle_log_position, read_oracle_log, is_oracle_enabled)


def setup_repo(repos_dir: str, instance_id: str, base_commit: str) -> str | None:
    wd = os.path.join(repos_dir, instance_id)
    # `.git` is a directory for a full clone but a FILE (gitlink) for a git
    # worktree — accept both so disk-efficient worktree checkouts are runnable.
    if not os.path.exists(os.path.join(wd, ".git")):
        return None
    for cmd in (["git", "reset", "--hard", base_commit], ["git", "clean", "-fd"]):
        subprocess.run(cmd, cwd=wd, capture_output=True, timeout=120)
    # swebench_backticks.yaml tells the model its working dir is /testbed (the
    # official SWE-bench Docker convention, where the repo IS mounted at
    # /testbed). Our LocalEnvironment runs in `wd` instead, so without this the
    # model wastes ~2 steps (`find /testbed` -> "No such file", then `pwd` to
    # discover the real path) before adapting -> diverges from the canonical
    # harness. Point /testbed at wd (best-effort; skipped if / isn't writable or
    # /testbed is a real directory we shouldn't clobber).
    try:
        if os.path.islink("/testbed"):
            os.remove("/testbed")
        if not os.path.exists("/testbed"):
            os.symlink(wd, "/testbed")
    except OSError:
        pass
    return wd


def load_problem_statements(instance_ids: set[str]) -> dict:
    """Official problem_statement per instance_id (HF SWE-bench Verified); {} if unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        return {r["instance_id"]: r["problem_statement"] for r in ds
                if r["instance_id"] in instance_ids}
    except Exception as e:
        print(f"  (HF dataset unavailable: {e}; falling back to dataset turns)")
        return {}


def build_live_agent(url: str, model: str, workdir: str, max_steps: int, collect_oracle: bool):
    from minisweagent.config import get_config_from_spec, builtin_config_dir
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.agents import get_agent

    # swebench_backticks.yaml instructs the ```mswea_bash_command action format that the
    # LitellmTextbasedModel parses (swebench.yaml is for the tool-call model -> format mismatch).
    cfg = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench_backticks.yaml"))
    cfg.setdefault("agent", {})["step_limit"] = max_steps
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"  # our local model isn't in litellm's price map

    steps: list = []

    class CapturingModel(LitellmTextbasedModel):
        def query(self, messages, **kw):
            pos = get_oracle_log_position() if collect_oracle else None
            r = super().query(messages, **kw)
            steps.append({"content": r.get("content", ""),       # always count the model call
                          "oracle_vanilla_entries": (read_oracle_log(pos) if pos is not None else [])})
            return r

    model_obj = CapturingModel(model_name=f"openai/{model}", cost_tracking="ignore_errors",
        model_kwargs={"api_base": url, "api_key": "dummy", "temperature": 0.0,
                      "max_tokens": int(os.environ.get("SWEB_MAX_TOKENS", "4096")), "timeout": 900, "drop_params": True,
                      # thinking ON = official setting for SWE-bench/agentic runs. The
                      # LitellmTextbasedModel parses the bash action from TEXT, ignoring the
                      # <think>...</think> block, so reasoning output is fine.
                      "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}})
    env = LocalEnvironment(cwd=workdir, timeout=120)
    agent = get_agent(model_obj, env, cfg.get("agent", {}), default_type="default")  # agent sub-config
    return agent, steps


def run_live(url, model, instances, repos_dir, max_steps, collect_oracle):
    pstmts = load_problem_statements({i["instance_id"] for i in instances})
    out = []
    for inst in instances:
        iid = inst["instance_id"]
        wd = setup_repo(repos_dir, iid, inst.get("base_commit", ""))
        if wd is None:
            print(f"  skip {iid}: repo not cloned"); continue
        task = pstmts.get(iid) or (inst["turns"][0] if isinstance(inst["turns"], list) else inst["turns"])
        agent, steps = build_live_agent(url, model, wd, max_steps, collect_oracle)
        print(f"  [{iid}] agent={type(agent).__name__} model={type(agent.model).__name__} task_len={len(task)}")
        t0 = time.perf_counter()
        # per-instance wall-clock safety net: a pathological big-context instance
        # (thinking-ON, slow long-context decode) can otherwise block for hours.
        # SIGALRM interrupts agent.run(); partial trajectory (agent.messages) is kept.
        import signal
        _to = int(os.environ.get("SWEB_INSTANCE_TIMEOUT", "1800"))
        # NOTE: derive from BaseException, NOT Exception — mini-swe-agent's per-step
        # `except Exception` swallows a plain-Exception timeout (instance then runs on
        # for hours despite the alarm). BaseException escapes that catch and reaches
        # our handler below, so the 30-min wall-clock hard-cut actually fires.
        class _InstTimeout(BaseException): pass
        def _on_alarm(signum, frame): raise _InstTimeout()
        _old = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(_to)
        try:
            agent.run(task)
        except _InstTimeout:
            print(f"  {iid}: INSTANCE TIMEOUT after {_to}s -> keeping partial trajectory")
        except Exception as e:
            import traceback
            print(f"  agent.run({iid}) raised: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, _old)
        print(f"  [{iid}] post-run: messages={len(agent.messages)} "
              f"roles={[m.get('role') for m in agent.messages][:6]} steps_captured={len(steps)}")
        if agent.messages:
            print(f"  [{iid}] last msg ({agent.messages[-1].get('role')}): "
                  f"{str(agent.messages[-1].get('content',''))[:200]}")
        msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in agent.messages]
        out.append({"instance_id": iid, "category": "swebench",
                    "num_turns": len(steps), "total_latency": time.perf_counter() - t0,
                    "turns": [{"messages": msgs}],
                    "agent_metrics": {"steps": [
                        {"step": k, "content": s["content"],
                         "spec_decode": {"oracle_vanilla_entries": s["oracle_vanilla_entries"]}}
                        for k, s in enumerate(steps)]}})
        n_oracle = sum(len(s["oracle_vanilla_entries"]) for s in steps)
        print(f"  {iid}: {len(steps)} model-calls, {n_oracle} oracle entries")
    return out


def run_replay(url, model, record_path, collect_oracle):
    """Re-send each recorded instance's conversation turn-by-turn; the server (oracle/select1
    mode) logs per-position decisions on the same trajectory. No bash execution."""
    from openai import OpenAI
    client = OpenAI(base_url=url, api_key="dummy", timeout=14400.0)
    rec = json.load(open(record_path))
    out = []
    for q in rec.get("questions", []):
        iid = q["instance_id"]
        msgs = q["turns"][0]["messages"]
        # rebuild assistant-turn boundaries: re-generate at each assistant message position
        steps = []
        ctx = []
        for m in msgs:
            if m["role"] == "assistant":
                pos = get_oracle_log_position() if collect_oracle else None
                try:
                    client.chat.completions.create(model=model, messages=ctx,
                                                   temperature=0.0, max_tokens=4096)
                except Exception as e:
                    pass
                if pos is not None:
                    steps.append({"oracle_vanilla_entries": read_oracle_log(pos)})
            ctx.append({"role": m["role"], "content": m["content"]})
        out.append({"instance_id": iid, "category": "swebench", "num_turns": len(steps),
                    "turns": [{"messages": msgs}],
                    "agent_metrics": {"steps": [
                        {"step": k, "spec_decode": {"oracle_vanilla_entries": s["oracle_vanilla_entries"]}}
                        for k, s in enumerate(steps)]}})
        print(f"  {iid}: replayed {len(steps)} assistant turns")
    return out


def main():
    ap = argparse.ArgumentParser(description="mini-swe-agent SWE-Bench workload (official harness)")
    ap.add_argument("--url", default="http://localhost:30000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--repos-dir", default="data/swebench/repos")
    ap.add_argument("--max-iterations", type=int, default=250)  # official mini-swe-agent swebench.yaml step_limit
    ap.add_argument("--num-workers", type=int, default=1)  # serial (oracle log is global)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--tool-style", default=None)  # ignored (mini-swe-agent is textbased)
    args = ap.parse_args()
    collect_oracle = is_oracle_enabled()
    print(f"mini-swe-agent workload (oracle={'on' if collect_oracle else 'off'}) "
          f"{'REPLAY' if args.replay else 'LIVE'}")

    if args.replay:
        questions = run_replay(args.url, args.model, args.replay, collect_oracle)
    else:
        rows = [json.loads(l) for l in open(args.input_file)]
        rows = rows[args.offset:]
        if args.num_requests is not None:
            rows = rows[:args.num_requests]
        questions = run_live(args.url, args.model, rows, args.repos_dir,
                             args.max_iterations, collect_oracle)

    meta = {"model": args.model, "benchmark": "swebench_minisweagent",
            "num_requests": len(questions)}
    json.dump({"metadata": meta, "questions": questions}, open(args.output_file, "w"))
    print(f"Saved {len(questions)} instances -> {args.output_file}")


if __name__ == "__main__":
    main()
