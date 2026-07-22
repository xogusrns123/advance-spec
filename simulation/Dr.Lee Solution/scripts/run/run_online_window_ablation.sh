#!/usr/bin/env bash
# ============================================================================
# ONLINE-CALIB WINDOW × (head×tail) ABLATION (2026-07-15)
#
# The deployable online arm (replay_extension.py --online-calib) warms its
# calibrators on the serving stream itself over a sliding window. This driver
# sweeps the FULL cross product:
#     head ∈ {logistic, beta, linear}  ×  tail ∈ {isotonic, linear}   (6 combos)
#     window ∈ {no-window(=1e9), 16000, 8000, 4000, 2000, 1000}       (6 sizes)
# across all 5 deployable workloads (specbench bfcl swebench spider tau2).
# = 6 × 6 × 5 = 180 CPU replays. Tail window = head window / 4 (OnlineCalib).
#
# Eval protocol matches the canonical ONLINE_grid deliverable: --three-way
# --group-mode convlabel, MAT measured on the TEST half (fit-free stream).
# CPU-only; no GPU. Safe alongside GPU0 serving.
#
#   docker exec -d sglang-bench bash \
#     "/workspace/simulation/Dr.Lee Solution/scripts/run/run_online_window_ablation.sh"
#
# Outputs
#   replay logs  readable_outputs/figures/replay_logs/
#                  mat_{ds}_4way_calib_onl_{head}_{tail}_w{W}_split.replay.txt
#   status       results/online_window_ablation/{driver.log,DONE,FAILED}
# ============================================================================
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RLOG=readable_outputs/figures/replay_logs
ST=results/online_window_ablation
GM=convlabel
JOBS=${JOBS:-10}                       # concurrent replays
mkdir -p "$ST" "$RLOG"
rm -f "$ST/DONE" "$ST/FAILED"

log() { echo "[$(TZ=Asia/Seoul date +%H:%M)KST] $*" >> "$ST/driver.log"; }
fail() { log "FAILED: $*"; echo "$*" > "$ST/FAILED"; exit 1; }

declare -A REC=(
  [specbench]=results/perpos_specbench_alleval/specbench_4way.jsonl
  [bfcl]=results/perpos_bfcl_alleval/bfcl_4way.jsonl
  [swebench]=results/perpos_swebench_alleval/swebench_4way.jsonl
  [spider]=results/perpos_spider_alleval/spider_4way.jsonl
  [tau2]=results/perpos_tau2_alleval/tau2_4way.jsonl)
DSS="specbench bfcl swebench spider tau2"
HEADS="logistic beta linear"
TAILS="isotonic linear"
WINDOWS="1000000000 16000 8000 4000 2000 1000"

for ds in $DSS; do [ -f "${REC[$ds]}" ] || fail "record ${REC[$ds]} missing"; done

# ---- build the job list --------------------------------------------------
JOBLIST="$ST/jobs.txt"; : > "$JOBLIST"
for ds in $DSS; do
  for h in $HEADS; do for t in $TAILS; do for w in $WINDOWS; do
    echo "$ds $h $t $w" >> "$JOBLIST"
  done; done; done
done
NJOBS=$(wc -l < "$JOBLIST")
log "start: $NJOBS jobs (5 wl × 6 combos × 6 windows), JOBS=$JOBS parallel"

run_one() {
  local ds=$1 h=$2 t=$3 w=$4
  local out="$RLOG/mat_${ds}_4way_calib_onl_${h}_${t}_w${w}_split.replay.txt"
  local err="$ST/re_${ds}_${h}_${t}_w${w}.err"
  python3 scripts/replay_extension.py --record "${REC[$ds]}" --max-rounds 4096 \
    --props calib --three-way --group-mode $GM \
    --online-calib --online-window "$w" --online-head "$h" --online-tail "$t" \
    > "$out" 2> "$err"
  if grep -q "calib: K=" "$out"; then rm -f "$err"; else
    echo "$ds $h $t $w" >> "$ST/failed_jobs.txt"
  fi
}
export -f run_one
export REC RLOG ST GM

# ---- run with bounded parallelism ----------------------------------------
: > "$ST/failed_jobs.txt"
running=0
while read -r ds h t w; do
  run_one "$ds" "$h" "$t" "$w" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
done < "$JOBLIST"
wait

NFAIL=$(wc -l < "$ST/failed_jobs.txt")
if [ "$NFAIL" -ne 0 ]; then
  log "WARNING: $NFAIL jobs produced no K (see failed_jobs.txt)"
  fail "$NFAIL replays failed"
fi
log "ALL $NJOBS ONLINE WINDOW-ABLATION REPLAYS DONE"
touch "$ST/DONE"
