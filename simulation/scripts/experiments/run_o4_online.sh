#!/bin/bash
# ONLINE sliding-window calibration (target_p) experiment.
#
# References (raw/oracle + recorded conversation) are measured ONCE on 40 tasks;
# then the 4 online-calib arms (histogram/isotonic/logistic/beta) are served per
# window W via --replay-existing on that SAME conversation (so all windows and
# the raw/oracle refs share one trajectory). The online calibrator learns the
# per-(group,depth) map at serving time from a sliding window of the preceding
# prefix (q_target read off the verify logits) — NO train/fit/capture step.
# Then per (window, arm): adapter -> pairs.gz -> calib_verify PER-DEPTH graphs.
#
# Usage:  run_o4_online.sh <PRESET> <PORT> <DIR> <TAIL> <NDT> [WINDOWS]
#   14B:  run_o4_online.sh qwen3_14b      30021 qwen3_14b_online    64 0
#   27B:  run_o4_online.sh qwen35_27b_mtp 30022 qwen35_27b_online   32 49
#   WINDOWS default "64 256 1024". Run inside sglang-bench as root.
set -uo pipefail
PRESET="${1:?preset}"; PORT="${2:?port}"; DIR="${3:?dir}"; TAIL="${4:?tail}"; NDT="${5:?ndt(0=default)}"
WINDOWS="${6:-64 256 1024}"
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$_c/bin/nvcc" ] && export CUDA_HOME="$_c" && break
done
: "${CUDA_HOME:?no CUDA toolkit with nvcc found}"
export PATH="$CUDA_HOME/bin:$PATH"
echo "using CUDA_HOME=$CUDA_HOME"
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
BASE="simulation/results/o4_perdepth"
NDTARG=""; [ "$NDT" != "0" ] && NDTARG="--spec-num-draft-tokens $NDT"
REF="$BASE/${DIR}_ref"
ONLINE_ARMS="select1_online_histogram,select1_online_isotonic,select1_online_logistic,select1_online_beta"

wait_gpu0() {
  echo "=== waiting for GPU0 (<15GB) ==="
  while true; do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    echo "  GPU0 used=${U} MiB $(date +%H:%M:%S)"; [ "${U:-99999}" -lt 15000 ] && break; sleep 60
  done; sleep 15
}

# STEP R: references once (record + raw + oracle), 40 tasks, replay-all.
wait_gpu0
echo "=== STEP R: references (record,baseline,select1,oracle) 40 tasks tail=$TAIL ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
  --n-tasks 40 --train-n-tasks 0 --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
  --replay-all --arms record,baseline,select1,select1_oracle \
  --output "$REF/run.json" $EXTRA
echo "STEP_R_RC=$?"
[ -s "$REF/agent_results_record.json" ] || { echo "ABORT: no recorded conversation"; exit 1; }

# Per-window online arms via replay-existing on the shared conversation.
for W in $WINDOWS; do
  WD="$BASE/${DIR}_w${W}"
  mkdir -p "$WD"
  cp -f "$REF/agent_results_record.json" "$WD/"
  [ -s "$REF/gt_tokens.jsonl" ] && cp -f "$REF/gt_tokens.jsonl" "$WD/"
  wait_gpu0
  echo "=== STEP O[w=$W]: 4 online-calib arms (replay-existing) tail=$TAIL ==="
  python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
    --n-tasks 40 --train-n-tasks 0 --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
    --replay-existing --online-window "$W" --online-scope continuous \
    --arms "$ONLINE_ARMS" --output "$WD/run.json" $EXTRA
  echo "STEP_O_w${W}_RC=$?"
  # per-depth calib_verify-style graphs per arm
  for ARM in histogram isotonic logistic beta; do
    A="select1_online_${ARM}"
    PAIRS="$WD/online_pairs_${A}.jsonl"
    if [ ! -s "$PAIRS" ]; then echo "WARN: no online pairs for $A (w=$W)"; continue; fi
    python3 simulation/scripts/online_pairs_to_pairs.py \
      --in "$PAIRS" --out "$WD/pairs_${A}.jsonl.gz"
    python3 simulation/scripts/calib_perposition.py \
      --pairs "$WD/pairs_${A}.jsonl.gz" --out-dir "$WD/figs_${ARM}" \
      --max-positions 16 || echo "perposition fail $A"
    python3 simulation/scripts/plot_calib_scatter_box.py \
      --pairs "$WD/pairs_${A}.jsonl.gz" --out-dir "$WD/figs_${ARM}" \
      --model-label EAGLE3 || echo "scatterbox fail $A"
  done
done
echo "O4_ONLINE_${DIR}_DONE"
