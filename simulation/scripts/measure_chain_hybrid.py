"""A/B driver for the chain-hybrid (EAGLE3 + suffix) serving PoC.

Per arm: boot one SGLang server (chain mode: topk=1, num_draft_tokens=S+1),
run the workload agent for N tasks against it (the server stays alive within
an arm so the global suffix trie warms across tasks), aggregate per-step
timings from the oracle timing log, and emit one JSON block per arm.

Arms:
  baseline  eagle3-only chain   (SGLANG_ORACLE_VANILLA=1 SGLANG_LATENCY_ONLY=1)
  select1   per-depth select-1  (+ SGLANG_CHAIN_HYBRID=1; at every depth
                                 eagle3 top-1 and suffix top-1 are re-drawn
                                 and compete by probability, decision JSONL)
  suffix    suffix-only CHAIN   (--speculative-algorithm SUFFIX +
                                 SGLANG_SUFFIX_CHAIN=1: top-1 path spec, no
                                 length cap beyond --suffix-num-draft-tokens;
                                 SuffixWorker has no per-step timing ->
                                 wall-clock only)

Usage (inside the sglang-bench container, cwd = repo root):
    python simulation/scripts/measure_chain_hybrid.py \\
        --preset qwen3_14b --workload bfcl_v4 --steps 8 --n-tasks 10 \\
        --arms baseline,select1,suffix \\
        --output simulation/results/chain_hybrid_poc/poc.json

Analysis: simulation/scripts/analyze_chain_hybrid.py joins the decision and
timing logs (copied next to --output) into the per-arm comparison table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from measure_eagle3_cost import (  # noqa: E402
    kill_server,
    read_timing_window,
    summarize_entries,
    wait_for_server,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PRESETS: dict[str, dict] = {
    "qwen3_14b": {
        "model": "Qwen/Qwen3-14B",
        "draft_model": "AngelSlim/Qwen3-14B_eagle3",
        "tool_call_parser": "qwen25",
    },
    "qwen3_8b": {
        "model": "Qwen/Qwen3-8B",
        "draft_model": "AngelSlim/Qwen3-8B_eagle3",
        "tool_call_parser": "qwen25",
    },
    # Qwen3.5-27B native MTP head: speculative_algorithm=EAGLE with
    # draft == target path triggers SGLang's MTP loading (same convention as
    # run_experiment.py's qwen35_27b_mtp preset). The Mamba-hybrid arch
    # requires --disable-radix-cache with speculative decoding; 131072
    # context matches the rr capture configs (KV is cheap on this arch —
    # only the few full-attention layers pay).
    "qwen35_27b_mtp": {
        "model": "Qwen/Qwen3.5-27B",
        "draft_model": "Qwen/Qwen3.5-27B",
        "tool_call_parser": "qwen25",
        "speculative_algorithm": "EAGLE",
        "extra_server_args": ["--disable-radix-cache",
                              "--context-length", "131072"],
    },
    # DFlash (z-lab block/diffusion drafter) on Blackwell GPU-0. DFlash drafts a
    # whole block per step (no eagle topk / num-steps); block_size ==
    # --speculative-num-draft-tokens. The chain-hybrid patch hooks DFlashWorker
    # (sole worker for DFLASH), so --disable-overlap-schedule is required.
    # Blackwell needs triton attention + pytorch sampling (project_blackwell_env)
    # and the cu130 image (project_dflash_cu130_official_image). 8B fits at
    # mem 0.70 (project_dflash_vs_suffix). Trained block = 16.
    "qwen3_8b_dflash": {
        "model": "Qwen/Qwen3-8B",
        "draft_model": "z-lab/Qwen3-8B-DFlash-b16",
        "tool_call_parser": "qwen25",
        "speculative_algorithm": "DFLASH",
        "is_dflash": True,
        "block_size": 16,
        "extra_server_args": ["--attention-backend", "triton",
                              "--sampling-backend", "pytorch",
                              "--disable-piecewise-cuda-graph",
                              "--disable-radix-cache"],
    },
}

WORKLOAD_REGISTRY: dict[str, dict] = {
    "bfcl_v4": {
        "agent_module": "simulation.agents.bfcl_v4_agent",
        "dataset": "data/bfcl_agent/dataset_stratified_interleaved.jsonl",
    },
    "specbench": {
        "agent_module": "simulation.agents.specbench_agent",
        "dataset": "data/specbench/dataset_interleaved.jsonl",
    },
}

ARM_NAMES = ("baseline", "hybrid_e3", "record",
             "select1", "select1_calib", "select1_calib_jeffreys",
             "select1_calib_offline", "select1_calib_jeffreys_offline",
             "select1_calib_histogram", "select1_calib_isotonic",
             "select1_calib_logistic", "select1_calib_beta",
             "select1_online_histogram", "select1_online_isotonic",
             "select1_online_logistic", "select1_online_beta",
             "select1_online_multifeat",
             "select1_disc_logistic", "select1_disc_beta",
             "select1_mono", "select1_bayes",
             "select1_multifeat",
             "select1_oracle", "suffix")

# ONLINE calibration arms: NO pre-fit JSON. The per-(group,depth) calibration
# map is learned at serving time from a sliding window of the preceding prefix,
# calibrated to target_p (q_target read off the verify logits post-verify). One
# algorithm per arm. Deliberately NOT in CALIB_MAP_FILE (no map-existence check,
# no fit phase) — build_env sets SGLANG_CHAIN_HYBRID_ONLINE_* instead.
ONLINE_METHOD = {
    "select1_online_histogram": "histogram",
    "select1_online_isotonic": "isotonic",
    "select1_online_logistic": "logistic",
    "select1_online_beta": "beta",
    # COMBO: windowed MULTI-FEATURE online calibrator (suffix prob+count+total+
    # match_len, eagle prob; depth feature) — serving-time multivariate fit.
    "select1_online_multifeat": "multifeat",
}

# Calibrated arms: each loads a frozen isotonic suffix-prob map fitted from the
# raw select1 arm's decision log (so select1 must run first in the same dir).
# The *_offline arms evaluate IN-SAMPLE: same frozen maps (fit on the train
# tasks), agent runs on those SAME train tasks (offset 0) — the map is never
# updated at serving time (predict-only), so this isolates the in-sample vs
# out-of-sample calibration effect.
CALIB_MAP_FILE = {
    "select1_calib": "calib_noshrink.json",
    "select1_calib_jeffreys": "calib_jeffreys.json",
    "select1_calib_offline": "calib_noshrink.json",
    "select1_calib_jeffreys_offline": "calib_jeffreys.json",
    # PER-POSITION per-method maps (meta.per_position=true), fit in the fit phase
    # by fit_chain_hybrid_calib_perpos.py. The serving calibrator selects the
    # per-depth curve. These are the "O4" calibrated-selection arms.
    "select1_calib_histogram": "calib_pp_histogram.json",
    "select1_calib_isotonic": "calib_pp_isotonic.json",
    "select1_calib_logistic": "calib_pp_logistic.json",
    "select1_calib_beta": "calib_pp_beta.json",
}

# Discriminator arms: JOINT logistic/beta model fitted on the comparative label
# (which proposer == GT) with BOTH proposers' features [suffix_p, eagle_p,
# match_len, suffix_count, suffix_total]. Unlike calibration (per-proposer,
# marginal accept-prob), the discriminator sees both scores + trie evidence at
# once. Maps are fit by fit_chain_hybrid_discriminator.py from a TRAIN-slice
# oracle decision log (labels exist only in oracle mode) and placed in out_dir.
DISC_MAP_FILE = {
    "select1_disc_logistic": "disc_logistic.json",
    "select1_disc_beta": "disc_beta.json",
    "select1_mono": "disc_gbm_mono.json",
    "select1_bayes": "disc_gbm_bayes.json",
}


def is_eagle_arm(arm: str) -> bool:
    """Every arm except the SuffixWorker-only 'suffix' arm is an EAGLE3 chain."""
    return arm != "suffix"


def is_select1_family(arm: str) -> bool:
    return arm.startswith("select1")


def is_chain_hybrid_arm(arm: str) -> bool:
    """Arms that run the chain-hybrid patch (per-depth select-1 family, the
    Arctic-style score-fallback hybrid baseline, and the GT record arm)."""
    return is_select1_family(arm) or arm in ("hybrid_e3", "record")


def build_server_cmd(args, arm: str, preset: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", preset["model"],
        "--tp-size", str(args.tp_size),
    ]
    if arm == "suffix":
        # SuffixWorker path (install_hook's on-disk SUFFIX patch). No draft
        # model. Chain mode: the draft length is uncapped up to the verify
        # tensor size (--suffix-num-draft-tokens), not tied to eagle steps.
        cmd += [
            "--speculative-algorithm", "SUFFIX",
            "--speculative-num-draft-tokens", str(args.suffix_num_draft_tokens),
        ]
    elif preset.get("is_dflash"):
        # DFlash block drafter: block_size == --speculative-num-draft-tokens.
        # No eagle topk / num-steps. The select-1 is applied by substituting
        # tokens into the verify block (chain_hybrid_patch.patch_chain_hybrid_dflash).
        block = args.spec_num_draft_tokens or preset.get("block_size", 16)
        cmd += [
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", preset["draft_model"],
            "--speculative-num-draft-tokens", str(block),
        ]
    else:
        cmd += [
            "--speculative-algorithm",
            preset.get("speculative_algorithm", "EAGLE3"),
            "--speculative-draft-model-path", preset["draft_model"],
            "--speculative-num-steps", str(args.steps),
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens",
            str(args.spec_num_draft_tokens or (args.steps + 1)),
        ]
    cmd += [
        "--tool-call-parser", preset["tool_call_parser"],
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--max-running-requests", "1",
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--disable-cuda-graph",
        # sglang >=0.5.12 routes EAGLE/EAGLE3/MTP to the V2 spec workers
        # (eagle_worker_v2.EAGLEWorkerV2) whenever overlap scheduling is on
        # (enable_overlap = not disable_overlap_schedule, spec_info.py). The
        # oracle/chain-hybrid install_hook only patches the LEGACY EAGLEWorker,
        # so with overlap on the patch never engages (zero instrumentation, no
        # GT dump). Forcing overlap off selects the legacy worker the patch
        # targets. (No-op on pre-V2 sglang; overlap gives ~nothing at bs=1.)
        "--disable-overlap-schedule",
        # RTX 4090s have no P2P peer access; SGLang falls back anyway but
        # warns loudly per rank — disable explicitly.
        "--disable-custom-all-reduce",
        "--watchdog-timeout", "600",
        "--host", "0.0.0.0", "--port", str(args.port),
    ]
    if args.context_length:
        cmd += ["--context-length", str(args.context_length)]
    cmd += preset.get("extra_server_args", [])
    cmd += args.extra_args
    return cmd


def build_env(args, arm: str, timing_log: Path, decision_log: Path,
              calib_map: Path | None = None, gt_out: Path | None = None,
              gt_file: Path | None = None, disc_map: Path | None = None,
              pin_file: Path | None = None) -> dict:
    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    # SGLang 2.9.1 + CuDNN 9.10 raises on a Conv3d perf bug; Qwen3.5 (Mamba)
    # trips it, text-only is safe to bypass (see project_qwen35_cudnn_check;
    # run_experiment.py does the same).
    env.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    # DFlash runs on Blackwell GPU-0 (cu130 image). Pin GPU 0 (GPU 1 = um3maru)
    # and skip the deep_gemm fp8 JIT compile (bf16 dense models never use it).
    if MODEL_PRESETS[args.preset].get("is_dflash"):
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
    # Eagle arms get the LATENCY_ONLY instrumentation via the worker-init
    # hook; the suffix arm uses SuffixWorker (not an EAGLEWorker), so the
    # oracle env vars are irrelevant there and left unset.
    if is_eagle_arm(arm):
        env["SGLANG_ORACLE_VANILLA"] = "1"
        env["SGLANG_LATENCY_ONLY"] = "1"
        env["SGLANG_ORACLE_TIMING_LOG"] = str(timing_log)
    # select1 (raw), the calibrated variants, and the hybrid_e3 baseline all
    # run the chain-hybrid patch; calibrated arms additionally point it at a
    # frozen suffix-prob map; hybrid_e3 switches the patch to per-step
    # score-fallback mode (sim hybrid_e3:t semantics).
    if is_chain_hybrid_arm(arm):
        env["SGLANG_CHAIN_HYBRID"] = "1"
        env["SGLANG_CHAIN_HYBRID_LOG"] = str(decision_log)
        if calib_map is not None:
            env["SGLANG_CHAIN_HYBRID_CALIB"] = str(calib_map)
        # Direction-2 multi-feature per-proposer calibrator (its own env; the
        # serving patch prefers it over a 1-D CALIB map).
        if arm == "select1_multifeat" and getattr(args, "multifeat_map", None):
            env["SGLANG_CHAIN_HYBRID_MULTIFEAT"] = str(args.multifeat_map)
        if disc_map is not None:
            env["SGLANG_CHAIN_HYBRID_DISC"] = str(disc_map)
        if arm == "hybrid_e3":
            env["SGLANG_CHAIN_HYBRID_MODE"] = "score_fallback"
            env["SGLANG_CHAIN_HYBRID_SCORE_THRESHOLD"] = str(
                args.hybrid_score_threshold)
            env["SGLANG_CHAIN_HYBRID_FB_FACTOR"] = str(args.hybrid_fb_factor)
            env["SGLANG_CHAIN_HYBRID_FB_MIN_PROB"] = str(
                args.hybrid_fb_min_prob)
        if arm == "record":
            env["SGLANG_CHAIN_HYBRID_MODE"] = "record"
            env["SGLANG_CHAIN_HYBRID_GT_OUT"] = str(gt_out)
        if arm == "select1_oracle":
            env["SGLANG_CHAIN_HYBRID_MODE"] = "oracle"
            env["SGLANG_CHAIN_HYBRID_GT"] = str(gt_file)
        # PIN: force committed tokens onto the standalone trajectory (select1/
        # calib/hybrid_e3 arms). The patch ignores this in oracle mode and
        # rejects it in record mode, so the caller only sets pin_file for the
        # pinnable arms.
        if pin_file is not None and arm not in ("record", "select1_oracle"):
            env["SGLANG_CHAIN_HYBRID_PIN"] = str(pin_file)
        # ONLINE calibration arms: learn the per-(group,depth) map at serving
        # time from a sliding window (target_p). No frozen JSON; the patch reads
        # q_target off the verify logits post-verify.
        if arm in ONLINE_METHOD:
            env["SGLANG_CHAIN_HYBRID_ONLINE_CALIB"] = ONLINE_METHOD[arm]
            env["SGLANG_CHAIN_HYBRID_ONLINE_WINDOW"] = str(args.online_window)
            env["SGLANG_CHAIN_HYBRID_ONLINE_MIN_SAMPLES"] = str(
                args.online_min_samples)
            env["SGLANG_CHAIN_HYBRID_ONLINE_REFIT_K"] = str(args.online_refit_k)
            env["SGLANG_CHAIN_HYBRID_ONLINE_SCOPE"] = args.online_scope
            env["SGLANG_CHAIN_HYBRID_ONLINE_LABEL"] = args.online_label
            if args.online_conditional:
                env["SGLANG_CHAIN_HYBRID_ONLINE_CONDITIONAL"] = "1"
            env["SGLANG_CHAIN_HYBRID_ONLINE_PAIRS"] = str(
                decision_log.parent / f"online_pairs_{arm}.jsonl")
        # Suffix tail append (route b) is part of every chain-hybrid arm by
        # default; --tail-max-tokens 0 turns it off (ablation).
        if args.tail_max_tokens > 0:
            env["SGLANG_CHAIN_HYBRID_TAIL"] = str(args.tail_max_tokens)
            env["SGLANG_CHAIN_HYBRID_TAIL_FACTOR"] = str(args.tail_factor)
            env["SGLANG_CHAIN_HYBRID_TAIL_MIN_PROB"] = str(args.tail_min_prob)
            if args.tail_check:
                env["SGLANG_CHAIN_HYBRID_TAIL_CHECK"] = "1"
    if arm == "suffix":
        env["SGLANG_SUFFIX_CHAIN"] = "1"
        # SuffixWorker writes per-step accept/timing JSONL in the same
        # schema as the oracle instrumentation -> MAT/survival for free.
        env["SGLANG_ORACLE_TIMING_LOG"] = str(timing_log)
    return env


def run_agent(args, workload: dict, out_file: Path, env: dict,
              n_tasks: int | None = None, offset: int = 0,
              replay: Path | None = None) -> tuple[int, float]:
    """Run the workload agent once against the live server. Returns
    (returncode, wall_time_s). n_tasks/offset carve a task slice (offset lets
    the fit phase and eval phase use disjoint train/test tasks). replay
    follows a previous run's conversation (byte-identical prompts — required
    by the oracle arm so GT trajectories match)."""
    preset = MODEL_PRESETS[args.preset]
    cmd = [
        sys.executable, "-m", workload["agent_module"],
        "--url", f"http://localhost:{args.port}/v1",
        "--model", preset["model"],
        "--input-file", workload["dataset"],
        "--output-file", str(out_file),
        "--num-requests", str(n_tasks if n_tasks is not None else args.n_tasks),
        "--num-workers", "1",
    ]
    if replay is not None:
        cmd += ["--replay", str(replay)]
    if args.workload == "bfcl_v4":
        if args.max_iterations:
            cmd += ["--max-iterations", str(args.max_iterations)]
        if args.include_category:
            cmd += ["--include-category", args.include_category]
        if offset:
            cmd += ["--offset", str(offset)]
        if getattr(args, "exclude_ids", None):
            cmd += ["--exclude-ids", args.exclude_ids]

    log_path = out_file.parent / f"{out_file.stem}_agent.log"
    t0 = time.perf_counter()
    with open(log_path, "a") as lf:
        rc = subprocess.call(cmd, env=env, stdout=lf,
                             stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    wall = time.perf_counter() - t0
    if rc != 0:
        print(f"  agent rc={rc} (see {log_path})", file=sys.stderr)
    return rc, wall


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="qwen3_14b",
                        choices=sorted(MODEL_PRESETS.keys()))
    parser.add_argument("--workload", default="bfcl_v4",
                        choices=sorted(WORKLOAD_REGISTRY.keys()))
    parser.add_argument("--steps", type=int, default=8,
                        help="Chain depth S (num_draft_tokens is forced to S+1)")
    parser.add_argument("--n-tasks", type=int, default=10,
                        help="Agent tasks per arm (--num-requests), evaluated "
                             "on the test slice [train-n-tasks : +n-tasks]")
    parser.add_argument("--train-n-tasks", type=int, default=0,
                        help="If >0 and calibrated arms requested: fit the "
                             "calibration maps from a raw select1 run on the "
                             "first N (train) tasks, then evaluate all arms on "
                             "the next --n-tasks (test) tasks — disjoint, no "
                             "in-sample leakage. 0 = fit in-sample (legacy).")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="bfcl_v4 agent --max-iterations")
    parser.add_argument("--include-category", default=None,
                        help="bfcl_v4 agent --include-category substring "
                             "filter (e.g. 'web_search')")
    parser.add_argument("--exclude-ids", default=None,
                        help="bfcl_v4 agent --exclude-ids: comma-separated "
                             "bfcl_ids to drop (e.g. repetition loopers).")
    parser.add_argument("--arms", default="baseline,select1,suffix",
                        help=f"Comma list from {ARM_NAMES}")
    parser.add_argument("--suffix-num-draft-tokens", type=int, default=64,
                        help="Verify tensor size for the suffix arm (the "
                             "chain draft is uncapped up to this minus 1)")
    parser.add_argument("--tail-max-tokens", type=int, default=32,
                        help="Suffix tail append budget T_max for all "
                             "select1-family arms (route b); 0 disables "
                             "(ablation = pre-tail behavior)")
    parser.add_argument("--tail-factor", type=float, default=4.0,
                        help="max_spec_factor for the tail suffix query "
                             "(run length cap = match_len * factor)")
    parser.add_argument("--tail-min-prob", type=float, default=0.1,
                        help="min_token_prob for the tail suffix query "
                             "(cumulative path prob cutoff)")
    parser.add_argument("--tail-check", action="store_true",
                        help="Enable per-step reconstruction bit-check "
                             "(SGLANG_CHAIN_HYBRID_TAIL_CHECK=1, debug)")
    parser.add_argument("--online-window", type=int, default=256,
                        help="select1_online_* arms: sliding-window size in "
                             "decode steps (clock units) for the online "
                             "target_p calibrator")
    parser.add_argument("--online-min-samples", type=int, default=50,
                        help="select1_online_* arms: per-(group,depth) samples "
                             "required before the online map leaves cold-start "
                             "(raw fallback) ")
    parser.add_argument("--online-refit-k", type=int, default=0,
                        help="select1_online_* arms: refit cadence in steps for "
                             "logistic/beta (0 = method default: 1 hist/iso, 8 "
                             "logistic/beta); histogram is always incremental")
    parser.add_argument("--online-scope", default="continuous",
                        choices=["continuous", "per_request"],
                        help="select1_online_* arms: window rolls across the "
                             "whole task stream (continuous) or resets per "
                             "request (per_request)")
    parser.add_argument("--online-label", default="target_p",
                        choices=["target_p", "accept_rate"],
                        help="select1_online_* arms: regress onto q_target "
                             "(continuous) or the binary accept event (drafted "
                             "token == target argmax at that row)")
    parser.add_argument("--multifeat-map", default=None,
                        help="select1_multifeat arm: path to a "
                             "fit_chain_hybrid_calib_multifeat map (suffix uses "
                             "prob+count+total+match_len, eagle uses prob, depth "
                             "as feature). Served via SGLANG_CHAIN_HYBRID_MULTIFEAT.")
    parser.add_argument("--online-conditional", action="store_true",
                        help="select1_online_* arms: ingest a depth only if the "
                             "realized chain prefix was accepted through it "
                             "(accept_len >= depth) -> per-step conditional accept")
    parser.add_argument("--hybrid-score-threshold", type=float, default=5.0,
                        help="hybrid_e3 arm: use the suffix run iff its "
                             "score >= this (sim hybrid_e3:t)")
    parser.add_argument("--hybrid-fb-factor", type=float, default=1.0,
                        help="hybrid_e3 arm: suffix max_spec_factor "
                             "(paper-faithful default 1.0)")
    parser.add_argument("--hybrid-fb-min-prob", type=float, default=0.1,
                        help="hybrid_e3 arm: suffix min_token_prob "
                             "(paper-faithful default 0.1)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--kv-cache-dtype", default="fp8_e5m2")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--output", required=True,
                        help="Summary JSON; raw logs are copied next to it")
    parser.add_argument("--replay-all", action="store_true",
                        help="Every measured arm REPLAYS the record arm's "
                             "conversation (byte-identical prompts + tool "
                             "outputs), so all arms decode the SAME trajectory "
                             "and MAT isolates draft/selection quality (removes "
                             "live-tool/FP trajectory variance). Requires the "
                             "'record' arm, which is forced to run first.")
    parser.add_argument("--replay-existing", action="store_true",
                        help="Replay the EXISTING out_dir/agent_results_record.json "
                             "for every measured arm WITHOUT re-running the record "
                             "arm — to re-run a single arm (e.g. a crashed one) "
                             "within-run-fair vs a prior full run. Don't pass "
                             "'record' in --arms.")
    parser.add_argument("--pin-trajectory", action="store_true",
                        help="Force EVERY chain-hybrid arm's committed tokens onto "
                             "the standalone trajectory in out_dir/gt_tokens.jsonl "
                             "(the same file the oracle arm uses), so all arms follow "
                             "the IDENTICAL token path despite greedy FP-tie flips "
                             "while each still builds its own draft chain and "
                             "recomputes its own eagle features. Requires --replay-all "
                             "or --replay-existing (prompts must match the trajectory's "
                             "input_ids). The 'record'/'select1_oracle' arms are "
                             "unaffected (oracle pins via its own GT).")
    parser.add_argument("--spec-num-draft-tokens", type=int, default=None,
                        help="Override --speculative-num-draft-tokens (default "
                             "steps+1). Use to OVERSIZE the static Mamba spec "
                             "cache for the suffix tail on Qwen3.5/27B (set to "
                             "steps+1+tail_max). Requires the server_args topk==1 "
                             "force-reset bypass (auto via SGLANG_CHAIN_HYBRID_TAIL).")
    parser.add_argument("--skip-calib-fit", action="store_true",
                        help="Do NOT re-fit calibration maps in the fit phase; "
                             "use the maps already placed in out_dir (e.g. fit "
                             "offline from an oracle-trajectory log). Still applies "
                             "the test offset = --train-n-tasks.")
    parser.add_argument("--jeffreys", action="store_true",
                        help="Fit the per-position calibration maps on the "
                             "Jeffreys-shrunk suffix prob (c+0.5)/(n+1) instead "
                             "of the raw count ratio c/n (meta.shrink=jeffreys; "
                             "serving applies the same shrink before lookup). "
                             "Affects the select1_calib_<method> arms.")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Passed through to sglang.launch_server")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_NAMES:
            parser.error(f"unknown arm '{a}' (choices: {ARM_NAMES})")
    # Calibrated arms need a frozen map; it is auto-fit from the raw select1
    # arm's decision log, so order: non-select1 arms, then select1, then the
    # remaining select1-family arms (calibrated last) — preserving every
    # requested arm.
    calib_arms = [a for a in arms if a in CALIB_MAP_FILE]
    arms = ([a for a in arms if not is_select1_family(a)]
            + [a for a in arms if a == "select1"]
            + [a for a in arms if is_select1_family(a) and a != "select1"
               and a not in CALIB_MAP_FILE]
            + calib_arms)
    if args.replay_all:
        if "record" not in arms:
            parser.error("--replay-all requires the 'record' arm (it produces "
                         "the canonical conversation every other arm replays)")
        # record must run first so its conversation exists for the replays.
        arms = ["record"] + [a for a in arms if a != "record"]
    if args.pin_trajectory and not (args.replay_all or args.replay_existing):
        parser.error("--pin-trajectory requires --replay-all or --replay-existing "
                     "(prompts must match the trajectory's input_ids to pin)")

    preset = MODEL_PRESETS[args.preset]
    workload = WORKLOAD_REGISTRY[args.workload]
    url = f"http://localhost:{args.port}"

    output_path = Path(args.output).resolve()
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure SGLang worker sources carry the oracle init hook (idempotent).
    hook_env = os.environ.copy()
    hook_env["SGLANG_ORACLE_VANILLA"] = "1"
    subprocess.run(
        [sys.executable, "-m", "simulation.oracle.install_hook"],
        env=hook_env, check=True, cwd=str(REPO_ROOT))

    summary: dict = {
        "preset": args.preset,
        "model": preset["model"],
        "draft_model": preset["draft_model"],
        "workload": args.workload,
        "steps": args.steps,
        "topk": 1,
        "num_draft_tokens": args.steps + 1,
        "n_tasks": args.n_tasks,
        "tail": {
            "max_tokens": args.tail_max_tokens,
            "factor": args.tail_factor,
            "min_prob": args.tail_min_prob,
        },
        "arms": {},
    }

    def _save():
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    _save()

    # ---- Fit phase (disjoint train/test) --------------------------------
    # With calibrated arms and --train-n-tasks > 0, fit the maps from a raw
    # select1 run on the first N (train) tasks, then evaluate every arm on the
    # next --n-tasks (test) tasks via offset. offset 0 keeps legacy in-sample.
    # --train-n-tasks > 0 WITHOUT calib arms still applies the test offset
    # (no fit phase) so single-arm add-on runs share the test slice of a
    # previous train/test run.
    eval_offset = args.train_n_tasks if args.train_n_tasks > 0 else 0
    if calib_arms and args.train_n_tasks > 0 and args.skip_calib_fit:
        print("=" * 72, file=sys.stderr)
        print("SKIP-CALIB-FIT: using pre-placed calib maps in out_dir; "
              f"eval offset = {eval_offset} (no fit-phase re-fit)", file=sys.stderr)
        summary["train_n_tasks"] = args.train_n_tasks
        summary["eval_offset"] = eval_offset
        _save()
    if calib_arms and args.train_n_tasks > 0 and not args.skip_calib_fit:
        print("=" * 72, file=sys.stderr)
        print(f"FIT PHASE: raw select1 on {args.train_n_tasks} train tasks "
              f"(offset 0) -> calibration maps", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        fit_timing = Path(f"/tmp/sglang_ch_timing_fit_p{args.port}.jsonl")
        fit_dec = Path(f"/tmp/sglang_ch_decisions_fit_p{args.port}.jsonl")
        for p in (fit_timing, fit_dec):
            p.unlink(missing_ok=True)
        fit_env = build_env(args, "select1", fit_timing, fit_dec)  # raw, no map
        fit_cmd = build_server_cmd(args, "select1", preset)
        fit_server_log = out_dir / "server_fit.log"
        log_fh = open(fit_server_log, "w")
        proc = subprocess.Popen(fit_cmd, env=fit_env, stdout=log_fh,
                                stderr=log_fh, cwd=str(REPO_ROOT))
        try:
            if not wait_for_server(url):
                kill_server(proc)
                sys.exit(f"FIT PHASE: server boot failed. See {fit_server_log}")
            run_agent(args, workload, out_dir / "agent_results_fit.json",
                      fit_env, n_tasks=args.train_n_tasks, offset=0)
        finally:
            kill_server(proc)
            try:
                log_fh.close()
            except Exception:
                pass
            time.sleep(3)
        kept_fit_dec = out_dir / "decisions_select1_train.jsonl"
        if fit_dec.exists():
            shutil.copyfile(fit_dec, kept_fit_dec)
        fit_cmd2 = [sys.executable,
                    str(REPO_ROOT / "simulation/scripts/fit_chain_hybrid_calib.py"),
                    "--decision-log", str(kept_fit_dec),
                    "--out-noshrink", str(out_dir / "calib_noshrink.json"),
                    "--out-jeffreys", str(out_dir / "calib_jeffreys.json")]
        print("  fitting calibration maps from train decision log", file=sys.stderr)
        if subprocess.call(fit_cmd2, cwd=str(REPO_ROOT)) != 0:
            sys.exit("FIT PHASE: calibration fit failed")
        # Per-position per-method maps for any select1_calib_<method> arm
        # (calib_pp_*.json). Depth-indexed; see fit_chain_hybrid_calib_perpos.py.
        if any(CALIB_MAP_FILE.get(a, "").startswith("calib_pp_")
               for a in calib_arms):
            pp_cmd = [sys.executable,
                      str(REPO_ROOT
                          / "simulation/scripts/fit_chain_hybrid_calib_perpos.py"),
                      "--decision-log", str(kept_fit_dec),
                      "--out-dir", str(out_dir)]
            if args.jeffreys:
                pp_cmd.append("--jeffreys")
            print(f"  fitting per-position calibration maps"
                  f"{' (Jeffreys suffix)' if args.jeffreys else ''}",
                  file=sys.stderr)
            if subprocess.call(pp_cmd, cwd=str(REPO_ROOT)) != 0:
                sys.exit("FIT PHASE: per-position calibration fit failed")
        eval_offset = args.train_n_tasks
        summary["train_n_tasks"] = args.train_n_tasks
        summary["eval_offset"] = eval_offset
        _save()

    for arm in arms:
        # *_offline arms evaluate in-sample on the fit (train) slice; all
        # other arms evaluate on the disjoint test slice.
        is_offline = arm.endswith("_offline")
        arm_offset = 0 if is_offline else eval_offset
        arm_n_tasks = (args.train_n_tasks
                       if is_offline and args.train_n_tasks > 0
                       else args.n_tasks)
        print("=" * 72, file=sys.stderr)
        print(f"ARM {arm}: {args.workload} x {arm_n_tasks} tasks "
              f"(offset {arm_offset}{', in-sample' if is_offline else ''}), "
              f"S={args.steps}, topk=1", file=sys.stderr)
        print("=" * 72, file=sys.stderr)

        timing_log = Path(f"/tmp/sglang_ch_timing_{arm}_p{args.port}.jsonl")
        decision_log = Path(f"/tmp/sglang_ch_decisions_{arm}_p{args.port}.jsonl")
        # build_env sets the online-pairs dump to decision_log.parent (= /tmp);
        # clear stale copies so the per-arm online_pairs file is fresh.
        online_pairs_log = decision_log.parent / f"online_pairs_{arm}.jsonl"
        for p in (timing_log, decision_log, online_pairs_log):
            p.unlink(missing_ok=True)

        calib_map = None
        if arm in CALIB_MAP_FILE:
            calib_map = out_dir / CALIB_MAP_FILE[arm]
            if not calib_map.exists():
                arm_row = {"error": f"calib map missing: {calib_map} "
                                    f"(run the select1 arm first)"}
                summary["arms"][arm] = arm_row
                _save()
                print(f"ERROR: {arm}: {arm_row['error']}", file=sys.stderr)
                continue

        disc_map = None
        if arm in DISC_MAP_FILE:
            disc_map = out_dir / DISC_MAP_FILE[arm]
            if not disc_map.exists():
                arm_row = {"error": f"disc map missing: {disc_map} "
                                    f"(fit with fit_chain_hybrid_discriminator.py "
                                    f"from a train-slice oracle log)"}
                summary["arms"][arm] = arm_row
                _save()
                print(f"ERROR: {arm}: {arm_row['error']}", file=sys.stderr)
                continue

        # GT plumbing: the record arm dumps gt_tokens.jsonl; the oracle arm
        # consumes it AND replays the record arm's conversation.
        gt_path = out_dir / "gt_tokens.jsonl"
        gt_out = gt_file = replay_file = pin_file = None
        if arm == "record":
            gt_path.unlink(missing_ok=True)  # no stale mixing across runs
            gt_out = gt_path
        else:
            # Oracle needs the GT; it also replays the record conversation so
            # GT positions align. Under --replay-all EVERY measured arm replays
            # that same conversation, so all arms decode the identical
            # trajectory (fair MAT — no live-tool/FP divergence).
            if arm == "select1_oracle":
                gt_file = gt_path
            if arm == "select1_oracle" or args.replay_all or args.replay_existing:
                replay_file = out_dir / "agent_results_record.json"
            # --pin-trajectory: force this arm's committed tokens onto the same
            # standalone gt_tokens.jsonl (oracle pins via its own GT, so skip it).
            if (args.pin_trajectory and is_chain_hybrid_arm(arm)
                    and arm != "select1_oracle"):
                pin_file = gt_path
            missing = [str(p) for p in (gt_file, replay_file, pin_file)
                       if p is not None and not p.exists()]
            if missing:
                arm_row = {"error": f"inputs missing: {missing} "
                                    f"(run the record arm first)"}
                summary["arms"][arm] = arm_row
                _save()
                print(f"ERROR: {arm}: {arm_row['error']}", file=sys.stderr)
                continue

        env = build_env(args, arm, timing_log, decision_log, calib_map,
                        gt_out=gt_out, gt_file=gt_file, disc_map=disc_map,
                        pin_file=pin_file)
        cmd = build_server_cmd(args, arm, preset)

        server_log = out_dir / f"server_{arm}.log"
        agent_out = out_dir / f"agent_results_{arm}.json"
        log_fh = open(server_log, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh,
                                cwd=str(REPO_ROOT))
        arm_row: dict = {"server_log": str(server_log)}
        try:
            if not wait_for_server(url):
                kill_server(proc)
                arm_row["error"] = "server_failed"
                summary["arms"][arm] = arm_row
                _save()
                print(f"ERROR: {arm} server failed. See {server_log}",
                      file=sys.stderr)
                continue

            rc, wall = run_agent(args, workload, agent_out, env,
                                 n_tasks=arm_n_tasks, offset=arm_offset,
                                 replay=replay_file)
            arm_row["agent_rc"] = rc
            arm_row["wall_time_s"] = round(wall, 1)
            arm_row["agent_output"] = str(agent_out)
            arm_row["task_offset"] = arm_offset
            arm_row["n_tasks"] = arm_n_tasks
            arm_row["in_sample"] = is_offline

            if timing_log.exists():
                entries = read_timing_window(timing_log, 0)
                arm_row.update(summarize_entries(entries))
                kept_timing = out_dir / f"timing_{arm}.jsonl"
                shutil.copyfile(timing_log, kept_timing)
                arm_row["timing_log"] = str(kept_timing)
            if is_chain_hybrid_arm(arm) and decision_log.exists():
                kept_dec = out_dir / f"decisions_{arm}.jsonl"
                shutil.copyfile(decision_log, kept_dec)
                arm_row["decision_log"] = str(kept_dec)
            # Online-calibration arms also dump (raw_prob, q_target) pairs for
            # the per-depth graphs; copy them next to the decision log.
            if arm in ONLINE_METHOD and online_pairs_log.exists():
                kept_pairs = out_dir / f"online_pairs_{arm}.jsonl"
                shutil.copyfile(online_pairs_log, kept_pairs)
                arm_row["online_pairs"] = str(kept_pairs)

            summary["arms"][arm] = arm_row
            _save()

            # In-sample mode only (no train/test split): after the RAW select1
            # arm, fit the calibration maps from its own decision log so the
            # calibrated arms (later in the order) can load them. With a train
            # phase (--train-n-tasks>0) the maps are already fit on disjoint
            # train tasks, so skip.
            if arm == "select1" and calib_arms and args.train_n_tasks == 0:
                kept_dec = out_dir / "decisions_select1.jsonl"
                fit_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "simulation/scripts/fit_chain_hybrid_calib.py"),
                    "--decision-log", str(kept_dec),
                    "--out-noshrink", str(out_dir / "calib_noshrink.json"),
                    "--out-jeffreys", str(out_dir / "calib_jeffreys.json"),
                ]
                print(f"  fitting calibration maps -> {out_dir}", file=sys.stderr)
                fit_rc = subprocess.call(fit_cmd, cwd=str(REPO_ROOT))
                if fit_rc != 0:
                    print(f"  WARNING: calib fit rc={fit_rc}; calibrated arms "
                          f"will error on missing maps", file=sys.stderr)
            print(f"  {arm}: wall={arm_row.get('wall_time_s')}s "
                  f"n={arm_row.get('n_samples', 0)} "
                  f"accept_mean={arm_row.get('accept_length_mean', float('nan')):.3f} "
                  f"step_ms={arm_row.get('step_ms', float('nan')):.2f}"
                  if arm != "suffix" else
                  f"  {arm}: wall={arm_row.get('wall_time_s')}s (reference)",
                  file=sys.stderr)
        finally:
            kill_server(proc)
            try:
                log_fh.close()
            except Exception:
                pass
            time.sleep(3)

    _save()
    print(f"\nSummary: {output_path}", file=sys.stderr)
    print("Next: python simulation/scripts/analyze_chain_hybrid.py "
          f"--summary {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
