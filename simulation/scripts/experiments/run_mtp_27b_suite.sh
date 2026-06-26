#!/bin/bash
# MTP (Qwen3.5-27B) chain-hybrid suite — mirrors the 14B accept_rate study so the
# MTP draft path can be compared on the SAME techniques:
#   raw, oracle, baseline, suffix, calib all-trained x4, calib cond-trained x4,
#   online windowed calib (window), multi-feature calib (multifeat).
#
# Stage 1  accept_rate cell  (run_chain_hybrid_perdepth.sh)  -> qwen35_27b_ar(+_train)
# Stage 2  online window sweep (run_online_sweep.sh, staged on the cell trajectory)
# Stage 3  multifeat fit + serve (fit_..._multifeat.py + run_multifeat.sh)
#
# MTP serving knobs: tail=0, ndt=0 (=> num_draft_tokens=steps+1=17), port 30022.
# tail is DISABLED on MTP: the Mamba-hybrid mamba spec cache must be oversized to
# steps+1+tail_max at server-args time, but that oversized num_draft_tokens then
# breaks the eagle draft path (organize_draft_results does topk(ndt-1) on a chain
# that only has ~steps scores -> "selected index k out of range"). The per-depth
# select-1 study (raw/oracle/calib/window/multifeat) is tail-INDEPENDENT, so we
# run tail=0; only the post-chain suffix-tail MAT boost is absent.
# Env: TRAIN_N (default 30), EVAL_N (default 20) for a fast smoke override.
set -uo pipefail
cd /workspace
B=simulation/results/chain_hybrid_perdepth
PRESET=qwen35_27b_mtp; PORT=30022; TAIL=0; NDT=0
CELL="${CELL:-qwen35_27b_ar}"
AR=$B/$CELL; ONLINE=$B/qwen35_27b_online; MF=$B/qwen35_27b_multifeat
TRAIN_N="${TRAIN_N:-30}"; EVAL_N="${EVAL_N:-20}"
WINDOWS="${WINDOWS:-256 1024}"; LABELS="${LABELS:-accept_rate}"

echo "########## STAGE 1: accept_rate cell ($CELL) ##########"
TRAIN_N="$TRAIN_N" EVAL_N="$EVAL_N" \
  bash simulation/scripts/experiments/run_chain_hybrid_perdepth.sh "$PRESET" "$PORT" "$CELL" "$TAIL" "$NDT" accept_rate
echo "STAGE1_RC=$?"
[ -s "$AR/decisions_select1_oracle.jsonl" ] || { echo "ABORT: stage1 oracle log missing"; exit 1; }

echo "########## STAGE 2: online window sweep ##########"
PRESET="$PRESET" SRC="$AR" DIR="$ONLINE" PORT="$PORT" TAIL="$TAIL" NDT="$NDT" \
  TRAIN_N="$TRAIN_N" EVAL_N="$EVAL_N" LABELS="$LABELS" WINDOWS="$WINDOWS" \
  bash simulation/scripts/experiments/run_online_sweep.sh
echo "STAGE2_RC=$?"

echo "########## STAGE 3: multifeat fit + serve ##########"
mkdir -p "$MF"
for L in $LABELS; do
  python3 simulation/scripts/fit_chain_hybrid_calib_multifeat.py \
    --decision-log "$B/${CELL}_train/decisions_select1_oracle.jsonl" \
    --out "$MF/multifeat_${L}.json"
  echo "MF_FIT_${L}_RC=$?"
done
PRESET="$PRESET" DIR="$ONLINE" MF="$MF" PORT="$PORT" TAIL="$TAIL" NDT="$NDT" \
  TRAIN_N="$TRAIN_N" EVAL_N="$EVAL_N" LABELS="$LABELS" \
  bash simulation/scripts/experiments/run_multifeat.sh
echo "STAGE3_RC=$?"
echo "MTP_27B_SUITE_DONE"
