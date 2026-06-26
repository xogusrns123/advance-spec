#!/bin/bash
# Serve the Panel-B BOUNDARY arms as real, PINNED, train-fit classifiers:
#   select1_mono  = best-monotone GBM   (calibration-framework ceiling, deployable)
#   select1_bayes = unconstrained GBM   (0.5-Bayes, deployable)
# Both fit on the TRAIN oracle (held-out) over (suffix_p, eagle_p, depth) ONLY,
# then served --pin-trajectory on the SAME eval trajectory as raw/calib/oracle so
# the MAT/selacc comparison is fair (differences come only from selection).
# NOTE: served (train-fit) numbers sit BELOW the offline OOF ceiling estimate by
# the train->eval gap -- same gap calib pays -> fair apples-to-apples.
#
# Args: PRESET PORT EVAL_DIR TRAIN_DIR TAIL
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$c/bin/nvcc" ] && export CUDA_HOME="$c" && break; done
export PATH="$CUDA_HOME/bin:$PATH"
PRESET="${1:?preset}"; PORT="${2:?port}"; D="${3:?eval dir}"; TRAIN="${4:?train dir}"; TAIL="${5:?tail}"
R=simulation/results/chain_hybrid_perdepth
[ -s "$R/$TRAIN/decisions_select1_oracle.jsonl" ] || { echo "ABORT: no train oracle $TRAIN"; exit 1; }
[ -s "$R/$D/agent_results_record.json" ] || { echo "ABORT: no eval record $D"; exit 1; }
[ -s "$R/$D/gt_tokens.jsonl" ] || { echo "ABORT: no gt_tokens $D"; exit 1; }
echo "using CUDA_HOME=$CUDA_HOME"

# 1. fit the two GBM boundary arms on the train oracle (CPU)
python3 simulation/scripts/fit_chain_hybrid_discriminator.py --gbm \
  --oracle-log "$R/$TRAIN/decisions_select1_oracle.jsonl" --out-dir "$R/$D" \
  || { echo "GBM FIT FAILED"; exit 1; }

# 2. serve both arms PINNED (measure_chain_hybrid reboots a server per arm with
#    that arm's disc blob); reuse the recorded eval trajectory.
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" \
  --workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1 \
  --train-n-tasks 30 --n-tasks 20 --port "$PORT" --tail-max-tokens "$TAIL" \
  --replay-existing --skip-calib-fit --pin-trajectory \
  --arms select1_mono,select1_bayes --output "$R/$D/run_boundary.json" \
  --extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch
echo "BOUNDARY_SERVE_RC=$?"
echo "BOUNDARY_SERVE_DONE_${D}"
