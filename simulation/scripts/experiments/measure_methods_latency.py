"""Per-method decoding-latency comparison on ONE BFCLv4 web_search task (2 turns).

For each (model, method) cell, boot one SGLang server on GPU 0 with the oracle
LATENCY-only instrumentation (SGLANG_ORACLE_VANILLA=1 + SGLANG_LATENCY_ONLY=1),
run a 2-turn web_search conversation, and reduce the per-step timing log to:

  verify-step scope (one decode step = verify + draft + others, ms):
      verify_ms = median(verify_total_ms)            # target forward + verify overhead
      draft_ms  = median(eagle3_draft_ms)            # draft proposal
      others_ms = median(step_total_ms) - verify_ms - draft_ms
  draft-step scope (per single draft token, ms):
      per_draft_token_ms = draft_ms / num_draft_tokens_proposed   (=draft_ms here, =1)

Methods: EAGLE-3, MTP, small-model, DFlash, Suffix. Chain structure, num_draft_tokens=1.

Output: <out>/summary.json keyed by model -> method -> metrics (merged across runs).

Run INSIDE the sglang-bench container (cwd /workspace):
    docker exec sglang-bench bash -lc \
      ". /opt/venv/bin/activate && cd /workspace && \
       python3 simulation/scripts/experiments/measure_methods_latency.py \
         --cells 8b_eagle3,8b_small,27b_mtp,27b_small --port 30055"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI


# --- self-contained helpers (inlined from measure_eagle3_cost so this script
# --- has no dependency on _workload_prompts/datasets in a minimal image) ---
def kill_server(proc: subprocess.Popen):
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def read_timing_window(log_path: Path, start_offset: int) -> list[dict]:
    entries: list[dict] = []
    try:
        with open(log_path, "rb") as f:
            f.seek(start_offset)
            tail = f.read().decode("utf-8", errors="ignore")
    except FileNotFoundError:
        return entries
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("phase", "decode") == "decode":
            entries.append(e)
    return entries


def summarize_entries(entries: list[dict]) -> dict:
    vt = [e["verify_total_ms"] for e in entries if "verify_total_ms" in e]
    draft = [e["eagle3_draft_ms"] for e in entries if "eagle3_draft_ms" in e]
    step = [e["step_total_ms"] for e in entries if "step_total_ms" in e]
    tf = [e["target_forward_ms"] for e in entries if "target_forward_ms" in e]
    accept_flat: list[float] = []
    for e in entries:
        for a in (e.get("accept_lengths") or []):
            if isinstance(a, (int, float)):
                accept_flat.append(float(a))
    out: dict = {"n_samples": len(entries)}
    if vt:
        out["target_cost_ms"] = statistics.median(vt)
    if draft:
        out["draft_cost_ms"] = statistics.median(draft)
    if step:
        out["step_ms"] = statistics.median(step)
    if tf:
        out["target_forward_ms"] = statistics.median(tf)
    if accept_flat:
        out["accept_length_mean"] = statistics.mean(accept_flat)
    return out

REPO = Path("/workspace")
DATASET = REPO / "data/bfcl_agent/dataset_stratified_interleaved.jsonl"

# Blackwell GPU-0 + measurement env, applied to every cell.
# Official lmsysorg/sglang:*-cu130 image is self-contained for CUDA 13; no
# cu12.8 toolkit / cu13 LD hacks needed. deep_gemm JIT disabled (bf16/dense
# models never use the fp8 path) to skip startup compile.
BLACKWELL_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    "SGLANG_ORACLE_VANILLA": "1",
    "SGLANG_LATENCY_ONLY": "1",
}
# Common server flags (Blackwell: triton attn + pytorch sampling; eager for per-step hook).
# --disable-overlap-schedule forces the V1 spec workers (EAGLEWorker/StandaloneWorker/
# MultiLayerEagleWorker) where the oracle latency patch is injected; DFLASH also requires
# overlap off.
COMMON_FLAGS = [
    "--tp-size", "1",
    "--disable-cuda-graph",
    "--disable-piecewise-cuda-graph",
    "--disable-radix-cache",
    "--disable-overlap-schedule",
    "--max-running-requests", "1",
    "--attention-backend", "triton",
    "--sampling-backend", "pytorch",
    "--watchdog-timeout", "600",
]

# num_draft_tokens=1 chain: 1 proposed draft token per step.
#   EAGLE3/EAGLE(MTP)/STANDALONE -> num-steps 1, topk 1, verify width (num-draft-tokens) 2
#   DFLASH -> block size (num-draft-tokens) 1
CELLS: dict[str, dict] = {
    "8b_eagle3": {
        "model": "Qwen/Qwen3-8B", "method": "EAGLE-3",
        "spec": ["--speculative-algorithm", "EAGLE3",
                 "--speculative-draft-model-path", "AngelSlim/Qwen3-8B_eagle3",
                 "--speculative-num-steps", "1", "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", "2",
                 "--speculative-draft-attention-backend", "triton"],
        "mem": "0.70", "ndt": 1,
    },
    "8b_small": {
        "model": "Qwen/Qwen3-8B", "method": "small-model",
        "spec": ["--speculative-algorithm", "STANDALONE",
                 "--speculative-draft-model-path", "Qwen/Qwen3-0.6B",
                 "--speculative-num-steps", "1", "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", "2",
                 "--speculative-draft-attention-backend", "triton"],
        "mem": "0.60", "ndt": 1,
    },
    "27b_mtp": {
        "model": "Qwen/Qwen3.5-27B", "method": "MTP",
        "spec": ["--speculative-algorithm", "EAGLE",
                 "--speculative-draft-model-path", "Qwen/Qwen3.5-27B",
                 "--speculative-num-steps", "1", "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", "2",
                 "--speculative-draft-attention-backend", "triton"],
        "mem": "0.70", "ndt": 1,
    },
    "27b_small": {
        "model": "Qwen/Qwen3.5-27B", "method": "small-model",
        # Qwen3.5 base is a multimodal (Qwen3VL) arch. The model_config
        # STANDALONE patch keeps the draft on its base arch AND forces
        # enable_multimodal=False for it, so the draft runs as a pure text LM
        # (no multimodal embed routine).
        "spec": ["--speculative-algorithm", "STANDALONE",
                 "--speculative-draft-model-path", "Qwen/Qwen3.5-0.8B",
                 "--speculative-num-steps", "1", "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", "2",
                 "--speculative-draft-attention-backend", "triton"],
        "mem": "0.70", "ndt": 1,
    },
    # --- Qwen3.5-35B-A3B MoE, UNQUANTIZED bf16 (~70GB) — the largest Qwen3.5
    # MoE that fits one 96GB Blackwell WITHOUT quantization (122B bf16=234GB,
    # 397B bigger). MTP + DFlash (EAGLE-3/small-model unavailable for Qwen3.5).
    # bf16 => no deep_gemm needed.
    "35b_mtp": {
        "model": "Qwen/Qwen3.5-35B-A3B", "method": "MTP",
        "spec": ["--speculative-algorithm", "EAGLE",
                 "--speculative-draft-model-path", "Qwen/Qwen3.5-35B-A3B",
                 "--speculative-num-steps", "1", "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", "2",
                 "--speculative-draft-attention-backend", "triton"],
        "mem": "0.82", "ndt": 1,
    },
    "35b_dflash": {
        "model": "Qwen/Qwen3.5-35B-A3B", "method": "DFlash",
        "spec": ["--speculative-algorithm", "DFLASH",
                 "--speculative-draft-model-path", "z-lab/Qwen3.5-35B-A3B-DFlash",
                 "--speculative-num-draft-tokens", "1"],
        # 70GB bf16 weights need mem>=0.73; keep KV minimal so the DFlash
        # drafter + draft-block buffers fit in the ~25GB headroom.
        "mem": "0.74", "ndt": 1,
    },
    # DFlash cells require SGLang >=0.5.11 + the Spec-V2 instrumentation (run later).
    "8b_dflash": {
        "model": "Qwen/Qwen3-8B", "method": "DFlash",
        "spec": ["--speculative-algorithm", "DFLASH",
                 "--speculative-draft-model-path", "z-lab/Qwen3-8B-DFlash-b16",
                 "--speculative-num-draft-tokens", "1"],
        "mem": "0.70", "ndt": 1, "spec_v2": True,
    },
    "27b_dflash": {
        "model": "Qwen/Qwen3.5-27B", "method": "DFlash",
        "spec": ["--speculative-algorithm", "DFLASH",
                 "--speculative-draft-model-path", "z-lab/Qwen3.5-27B-DFlash",
                 "--speculative-num-draft-tokens", "1"],
        "mem": "0.70", "ndt": 1, "spec_v2": True,
    },
    # --- SUFFIX decoding (ArcticInference SuffixDecodingCache). Model-free: the
    # draft is a trie lookup over the generated context (no draft model forward),
    # so the draft cost is tiny (sub-ms / us). num_draft_tokens=2 => max_spec=1,
    # i.e. 1 proposed draft token, matching the EAGLE/MTP "1 draft token" framing.
    # The SuffixWorker emits the same timing JSONL schema (eagle3_draft_ms /
    # verify_total_ms / step_total_ms) directly. Works for any target model.
    "8b_suffix": {
        "model": "Qwen/Qwen3-8B", "method": "Suffix",
        "spec": ["--speculative-algorithm", "SUFFIX",
                 "--speculative-num-draft-tokens", "2"],
        "mem": "0.70", "ndt": 1,
    },
    "27b_suffix": {
        "model": "Qwen/Qwen3.5-27B", "method": "Suffix",
        "spec": ["--speculative-algorithm", "SUFFIX",
                 "--speculative-num-draft-tokens", "2"],
        "mem": "0.70", "ndt": 1,
    },
    "35b_suffix": {
        "model": "Qwen/Qwen3.5-35B-A3B", "method": "Suffix",
        "spec": ["--speculative-algorithm", "SUFFIX",
                 "--speculative-num-draft-tokens", "2"],
        "mem": "0.82", "ndt": 1,
    },
}


def load_2turn_messages() -> tuple[list[dict], list[dict]]:
    """First web_search task -> (turn1 messages, follow-up turn2 user msg).

    All web_search tasks in the dataset are single-turn, so turn 2 is a generic
    follow-up that grows the context and triggers a second decode round.
    """
    with open(DATASET) as f:
        for line in f:
            e = json.loads(line)
            if "web_search" in (e.get("category") or ""):
                turn1 = e["question"][0]
                return turn1, [{
                    "role": "user",
                    "content": "Based on your previous answer, give a detailed "
                               "step-by-step explanation and list any caveats.",
                }]
    raise SystemExit("no web_search task found")


def run_cell(name: str, cell: dict, args, turn1: list[dict],
             turn2: list[dict]) -> dict | None:
    model = cell["model"]
    timing_log = Path(f"/tmp/timing_{name}.jsonl")
    timing_log.unlink(missing_ok=True)

    env = os.environ.copy()
    env.update(BLACKWELL_ENV)
    # FP8 MoE (block-fp8 GEMM) needs deep_gemm; the default BLACKWELL_ENV disables
    # JIT deep_gemm (fine for bf16/dense). Re-enable it for FP8-MoE cells.
    if cell.get("deepgemm"):
        env.pop("SGLANG_ENABLE_JIT_DEEPGEMM", None)
    env["SGLANG_ORACLE_TIMING_LOG"] = str(timing_log)
    # DFLASH uses the non-overlap DFlashWorker (selected by --disable-overlap-schedule
    # in COMMON_FLAGS), which the oracle dflash patch instruments — no spec-v2 env needed.

    # Ensure the oracle hook is patched into the SGLang worker sources (idempotent).
    subprocess.run([sys.executable, "-m", "simulation.oracle.install_hook"],
                   env={**os.environ, "SGLANG_ORACLE_VANILLA": "1"},
                   cwd=str(REPO), check=False)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--mem-fraction-static", (args.mem or cell["mem"]),
        "--host", "0.0.0.0", "--port", str(args.port),
    ] + COMMON_FLAGS + cell["spec"]

    url = f"http://localhost:{args.port}"
    log_path = Path(f"/tmp/server_{name}.log")
    print(f"\n{'='*70}\n[{name}] {model} / {cell['method']}\n{'='*70}", flush=True)
    print(" ".join(cmd), flush=True)
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh)
    try:
        import requests as _rq
        booted = False
        t_boot = time.time()
        while time.time() - t_boot < 900:
            if proc.poll() is not None:
                print(f"  ERROR: server process exited (code {proc.returncode}) "
                      "during boot", flush=True)
                break
            try:
                if _rq.get(f"{url}/health", timeout=5).status_code == 200:
                    booted = True
                    break
            except Exception:
                pass
            time.sleep(3)
        if not booted:
            kill_server(proc)
            print(f"  ERROR: server failed to boot. tail {log_path}:", flush=True)
            for ln in Path(log_path).read_text(errors="ignore").splitlines()[-30:]:
                print("   ", ln, flush=True)
            return None

        client = OpenAI(base_url=f"{url}/v1", api_key="dummy")
        # Warmup (JIT + KV), result discarded.
        try:
            client.chat.completions.create(model=model, messages=turn1,
                                           max_tokens=8, temperature=0.0)
        except Exception as e:
            print(f"  WARN warmup: {e}", flush=True)

        offset = timing_log.stat().st_size if timing_log.exists() else 0

        # Turn 1 (measured)
        r1 = client.chat.completions.create(model=model, messages=turn1,
                                            max_tokens=args.max_tokens, temperature=0.0)
        a1 = r1.choices[0].message.content or ""
        # Turn 2 (measured) with growing context
        msgs2 = list(turn1) + [{"role": "assistant", "content": a1}] + list(turn2)
        client.chat.completions.create(model=model, messages=msgs2,
                                       max_tokens=args.max_tokens, temperature=0.0)

        window = read_timing_window(timing_log, offset)
        summ = summarize_entries(window)
    finally:
        kill_server(proc)
        time.sleep(3)

    if not summ.get("n_samples"):
        print("  ERROR: no decode timing captured", flush=True)
        return None

    verify_ms = summ.get("target_cost_ms", 0.0)
    draft_ms = summ.get("draft_cost_ms", 0.0)
    step_ms = summ.get("step_ms", 0.0)
    others_ms = round(step_ms - verify_ms - draft_ms, 3)
    per_draft_token_ms = round(draft_ms / max(cell["ndt"], 1), 3)
    out = {
        "model": model, "method": cell["method"],
        "verify_ms": round(verify_ms, 3), "draft_ms": round(draft_ms, 3),
        "others_ms": others_ms, "step_ms": round(step_ms, 3),
        "per_draft_token_ms": per_draft_token_ms,
        "n_samples": summ["n_samples"],
        "accept_length_mean": round(summ.get("accept_length_mean", 0.0), 3),
    }
    print(f"  verify={verify_ms:.2f}ms draft={draft_ms:.2f}ms others={others_ms:.2f}ms "
          f"step={step_ms:.2f}ms per_draft_tok={per_draft_token_ms:.2f}ms "
          f"(n={summ['n_samples']})", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="8b_eagle3,8b_small,27b_mtp,27b_small",
                    help=f"Comma list from {list(CELLS)}")
    ap.add_argument("--port", type=int, default=30055)
    ap.add_argument("--mem", default="", help="override --mem-fraction-static for all cells")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="simulation/results/latency/method_compare")
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for c in cells:
        if c not in CELLS:
            raise SystemExit(f"unknown cell {c}; choices {list(CELLS)}")

    turn1, turn2 = load_2turn_messages()

    out_dir = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary: dict = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    for name in cells:
        cell = CELLS[name]
        res = run_cell(name, cell, args, turn1, turn2)
        if res is None:
            continue
        summary.setdefault(cell["model"], {})[cell["method"]] = res
        summary_path.write_text(json.dumps(summary, indent=2))
        # archive raw timing log
        tl = Path(f"/tmp/timing_{name}.jsonl")
        if tl.exists():
            shutil.copyfile(tl, out_dir / f"timing_{name}.jsonl")

    print(f"\nSummary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
