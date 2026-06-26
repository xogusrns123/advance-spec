#!/bin/bash
# RIGOROUS objective comparison on a SHARED trajectory (no code changes).
#
# Problem: the target_p run (qwen3_14b_tp) and token_gt run (qwen3_14b_def) each
# recorded their OWN live web_search trajectory, so their raw/oracle differ
# (raw 1.384 vs 1.294, oracle 1.823 vs 1.778). Cross-run calib deltas therefore
# mix the OBJECTIVE effect with a TRAJECTORY confound (different decision points).
#
# Fix: --replay-existing replays a frozen out_dir/agent_results_record.json for
# every arm WITHOUT re-recording (deterministic). So we evaluate BOTH objectives'
# calib maps on the SAME trajectory, with raw/oracle recomputed on it as anchors:
#   B) token_gt maps  on T_tp  (target_p's trajectory)  -> xtraj_def_on_tp
#   A) target_p maps  on T_def (token_gt's trajectory)  -> xtraj_tp_on_def
# raw'/oracle' in each must reproduce the source run's raw/oracle (faithfulness).
#
# Calib maps are trajectory-independent x->y lookups (fit offline), so dropping
# the other objective's map onto a fixed trajectory is exactly "what if we'd used
# that calibration on these identical decisions" -- the apples-to-apples test.
#
# Waits for the token_gt run's completion marker THEN GPU0 free (no race, no kill).
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
: "${CUDA_HOME:?no CUDA toolkit with nvcc found under /usr/local}"
export PATH="$CUDA_HOME/bin:$PATH"
echo "using CUDA_HOME=$CUDA_HOME"

BASE=simulation/results/o4_perdepth
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
PORT=30021
TAIL=64
# select1(raw') + 4 calib + oracle' on the shared trajectory. No record/baseline:
# record would re-record live (defeats the purpose); baseline is objective-free.
ARMS="select1,select1_calib_histogram,select1_calib_isotonic,select1_calib_logistic,select1_calib_beta,select1_oracle"

echo "=== waiting for token_gt run completion marker (O4_DEFAULT_qwen3_14b_def_DONE) ==="
while ! grep -q "O4_DEFAULT_qwen3_14b_def_DONE" "$BASE/qwen3_14b_def.runlog" 2>/dev/null; do sleep 60; done
echo "token_gt run complete; waiting for GPU0 (<15GB)"
while true; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "  GPU0 used=${U} MiB $(date +%H:%M:%S)"; [ "${U:-99999}" -lt 15000 ] && break; sleep 30
done
sleep 20

# $1=outdir  $2=trajectory src dir (agent_results_record.json+gt_tokens.jsonl)
# $3=calib maps src dir (calib_pp_*.json = the objective under test)
setup_dir () {
  mkdir -p "$1"
  cp -f "$2/agent_results_record.json" "$1/agent_results_record.json"
  cp -f "$2/gt_tokens.jsonl"           "$1/gt_tokens.jsonl"
  cp -f "$3"/calib_pp_*.json           "$1/"
  echo "setup $1: traj<-$2  maps<-$3 ($(python3 -c "import json;print(json.load(open('$1/calib_pp_isotonic.json'))['meta'].get('label'))"))"
}

run_dir () {  # $1=outdir
  python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b $COMMON \
    --train-n-tasks 30 --n-tasks 20 --port "$PORT" --tail-max-tokens "$TAIL" \
    --replay-existing --skip-calib-fit --arms "$ARMS" \
    --output "$1/run.json" $EXTRA
  echo "RC_$1=$?"
}

echo "=== B) token_gt maps on T_tp (the user's ask: token_gt on target_p's trajectory) ==="
setup_dir "$BASE/xtraj_def_on_tp" "$BASE/qwen3_14b_tp" "$BASE/qwen3_14b_def"
run_dir "$BASE/xtraj_def_on_tp"

echo "=== A) target_p maps on T_def (symmetric check) ==="
setup_dir "$BASE/xtraj_tp_on_def" "$BASE/qwen3_14b_def" "$BASE/qwen3_14b_tp"
run_dir "$BASE/xtraj_tp_on_def"

echo "SHARED_TRAJ_DONE"
