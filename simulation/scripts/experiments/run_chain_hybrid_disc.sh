#!/bin/bash
# Real-serving test of the OUR-BAYES joint discriminator (alive-conditioned + depth),
# the faithful serving version of the offline Bayes selector. Mirrors STEP G2 of
# run_chain_hybrid_perdepth.sh: fit the disc map from the TRAIN-oracle log, then
# serve select1_disc_{logistic,beta} on the SAME eval trajectory with --pin so the
# only difference vs the calib arms is the selector. Eagle is re-generated freshly
# per arm (incl. eagle-after-suffix) -> this is what offline reconstruction cannot do.
#
# Usage: run_chain_hybrid_disc.sh <PRESET> <PORT> <DIR> <TAIL> <NDT> [legacy]
#   14B target_p: run_chain_hybrid_disc.sh qwen3_14b 30021 qwen3_14b_tp 64 0
#   add a 6th arg "legacy" to ALSO fit/serve the pooled-no-depth baseline for A/B.
set -uo pipefail
PRESET="${1:?preset}"; PORT="${2:?port}"; DIR="${3:?dir}"; TAIL="${4:?tail}"
NDT="${5:?ndt(0=default)}"; WANT_LEGACY="${6:-}"
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
: "${CUDA_HOME:?no CUDA toolkit with nvcc found under /usr/local}"
export PATH="$CUDA_HOME/bin:$PATH"
echo "using CUDA_HOME=$CUDA_HOME"
BASE="simulation/results/chain_hybrid_perdepth"
D="$BASE/$DIR"; TRAIN="$BASE/${DIR}_train"
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
NDTARG=""; [ "$NDT" != "0" ] && NDTARG="--spec-num-draft-tokens $NDT"
TRAIN_N="${TRAIN_N:-30}"; EVAL_N="${EVAL_N:-20}"
[ -s "$TRAIN/decisions_select1_oracle.jsonl" ] || { echo "ABORT: $TRAIN/decisions_select1_oracle.jsonl (train-oracle log) missing"; exit 1; }
[ -s "$D/agent_results_record.json" ] || { echo "ABORT: $D/agent_results_record.json missing (need a prior record for --replay-existing)"; exit 1; }
[ -s "$D/gt_tokens.jsonl" ] || { echo "ABORT: $D/gt_tokens.jsonl missing (need it for --pin-trajectory)"; exit 1; }

serve () {  # $1=tag  (disc maps must already be in $D)
  local TAG="$1"
  echo "=== SERVE disc ($TAG): replay-existing + pin, arms=disc_logistic,disc_beta ($(date +%H:%M:%S)) ==="
  python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
    --train-n-tasks "$TRAIN_N" --n-tasks "$EVAL_N" --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
    --replay-existing --skip-calib-fit --pin-trajectory \
    --arms select1_disc_logistic,select1_disc_beta \
    --output "$D/run_disc_${TAG}.json" $EXTRA
  echo "DISC_${TAG}_RC=$?"
  for m in logistic beta; do for k in timing decisions; do
    [ -f "$D/${k}_select1_disc_${m}.jsonl" ] && mv -f "$D/${k}_select1_disc_${m}.jsonl" "$D/${k}_select1_disc_${m}_${TAG}.jsonl"
  done; done
}

# --- OUR-BAYES (accept-conditioned + depth) ---
echo "=== FIT our-Bayes disc maps from $TRAIN ==="
python3 simulation/scripts/fit_chain_hybrid_discriminator.py \
  --oracle-log "$TRAIN/decisions_select1_oracle.jsonl" --out-dir "$D" || { echo "FIT FAILED"; exit 1; }
serve ourbayes

# --- optional A/B: legacy pooled-no-depth ---
if [ "$WANT_LEGACY" = "legacy" ]; then
  echo "=== FIT legacy (pooled,no-depth) disc maps ==="
  python3 simulation/scripts/fit_chain_hybrid_discriminator.py \
    --oracle-log "$TRAIN/decisions_select1_oracle.jsonl" --out-dir "$D" \
    --no-accept-conditioned --no-with-depth || { echo "FIT FAILED"; exit 1; }
  serve legacy
fi
echo "DISC_DONE $DIR (compare run_disc_*.json MAT vs served raw 1.318 / calib 1.351 / oracle 1.761)"
