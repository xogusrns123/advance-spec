#!/usr/bin/env bash
# ============================================================================
# ADAPTIVE (FEEDBACK-LOOP) COMPOSE SCALARS (2026-07-18)
#
# User ask: replace the FIXED tail weight w (and the two-scalar u/w) with a
# feedback loop — sliding-window / EWMA online adaptation of the scalars from
# verify-time signals. PRIMARY tier = deployable information only (censored
# counterfactuals / chosen-arm ratio / label-free succ discount); full-info
# FTL runs as the adaptation CEILING reference.
#
# Fixed-champion references (test half, mean over 5 wl):
#   rawhead fixed w=0.075        3.862   per-wl best-of-grid UB 3.908
#   two-scalar u=1.4/w=0.125     3.874
#   online-beta + fixed w=0.08   3.898   per-wl best-of-grid UB 3.926
#   SOTA onlbeta+gbscale+w0.16   3.916
#
# Arms (all --props calib, TEST half of the 3-way convlabel split):
#   deg*        validation gates — degenerate 1-candidate grids MUST exactly
#               reproduce the fixed champion logs (V1/V3)
#   cftl*       PRIMARY: censored follow-the-leader over the w-grid
#   cftl2d      PRIMARY: censored FTL over the (u,w)-grid
#   ratio*      PRIMARY: w_t = gamma * windowed E[tail_acc]/E[tail_score]
#   succratio*  PRIMARY: LABEL-FREE w_t = gamma * windowed E[succ]/E[raw]
#   ftlw*       CEILING: full-info FTL (uncensored counterfactual credits)
#   onlb_*      E4: adaptive w under the online-beta head (+ gbscale stack)
#
#   docker exec -d sglang-bench bash \
#     "/workspace/simulation/Dr.Lee Solution/scripts/run/run_adaptive_scalar.sh"
#
# Outputs
#   replay logs  readable_outputs/figures/replay_logs/mat_{ds}_adw_{arm}_split.replay.txt
#   traces       results/adaptive_scalar/trace_{ds}_{arm}.json
#   status       results/adaptive_scalar/{driver.log,DONE,FAILED,failed_jobs.txt}
# ============================================================================
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RLOG=readable_outputs/figures/replay_logs
ST=results/adaptive_scalar
GM=convlabel
JOBS=${JOBS:-12}
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

GRID="0.02,0.03,0.04,0.05,0.0625,0.075,0.09,0.11,0.14,0.18"
GRID2D="0.03,0.05,0.075,0.11,0.16"
UGRID="1.0,1.2,1.4,1.7,2.0"
OGRID="0.02,0.03,0.04,0.055,0.075,0.10,0.125,0.16,0.20"   # online-beta head frame
GBGRID="0.08,0.11,0.16,0.22,0.30"                          # gbscale-rescored frame

declare -A ARMS=(
  # --- validation gates (degenerate grids == fixed champions) --------------
  [deg0075]="--adaptive ftl --adaptive-head raw --adaptive-grid 0.075"
  [onlb_deg008]="--adaptive ftl --adaptive-head beta --adaptive-grid 0.08 --online-window 8000"
  # --- PRIMARY: deployable censored-FTL (window / memory sweep) ------------
  [cftl_w2k]="--adaptive ftl --adaptive-head raw --adaptive-grid $GRID --adaptive-window 2000 --adaptive-censor"
  [cftl_w8k]="--adaptive ftl --adaptive-head raw --adaptive-grid $GRID --adaptive-window 8000 --adaptive-censor"
  [cftl_w32k]="--adaptive ftl --adaptive-head raw --adaptive-grid $GRID --adaptive-window 32000 --adaptive-censor"
  [cftl_ewma2k]="--adaptive ftl --adaptive-head raw --adaptive-grid $GRID --adaptive-ewma-halflife 2000 --adaptive-censor"
  [cftl2d_w8k]="--adaptive ftl2d --adaptive-grid $GRID2D --adaptive-ugrid $UGRID --adaptive-window 8000 --adaptive-censor"
  # --- PRIMARY: deployable feedback scalars ---------------------------------
  [ratio_g02]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.2 --adaptive-window 8000"
  [ratio_g03]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.3 --adaptive-window 8000"
  [succratio_h14]="--adaptive succratio --adaptive-head scale --head-scale 1.4 --adaptive-gamma 0.6 --adaptive-window 8000 --adaptive-init 0.125"
  [succratio_raw]="--adaptive succratio --adaptive-head raw --adaptive-gamma 0.375 --adaptive-window 8000"
  # --- CEILING: full-info FTL (replay tier) ---------------------------------
  [ftlw_inf]="--adaptive ftl --adaptive-head raw --adaptive-grid $GRID --adaptive-window 0"
  [ftlw_w8k]="--adaptive ftl --adaptive-head raw --adaptive-grid $GRID --adaptive-window 8000"
  # --- E4: online-beta head composition (+ SOTA gbscale stack) --------------
  [onlb_cftl_w8k]="--adaptive ftl --adaptive-head beta --adaptive-grid $OGRID --adaptive-window 8000 --adaptive-censor --online-window 8000 --adaptive-init 0.08"
  [onlb_gb_cftl]="--adaptive ftl --adaptive-head beta --adaptive-grid $GBGRID --adaptive-window 8000 --adaptive-censor --online-window 8000 --tail-cal gbscale --gb-a 0 --gb-b 0.125 --adaptive-init 0.16"
)

for ds in $DSS; do [ -f "${REC[$ds]}" ] || fail "record ${REC[$ds]} missing"; done

JOBLIST="$ST/jobs.txt"; : > "$JOBLIST"
for ds in $DSS; do for arm in "${!ARMS[@]}"; do echo "$ds $arm"; done; done \
  | shuf --random-source=<(yes 42) > "$JOBLIST"   # mix fast/slow workloads across lanes
NJOBS=$(wc -l < "$JOBLIST")
log "start: $NJOBS jobs (${#ARMS[@]} arms x 5 workloads), JOBS=$JOBS parallel"

: > "$ST/failed_jobs.txt"
run_one() {
  local ds=$1 arm=$2
  local out="$RLOG/mat_${ds}_adw_${arm}_split.replay.txt"
  local err="$ST/re_${ds}_${arm}.err"
  # shellcheck disable=SC2086
  python3 scripts/replay_extension.py --record "${REC[$ds]}" --max-rounds 4096 \
    --props calib --three-way --group-mode $GM ${ARMS[$arm]} \
    --adaptive-trace "$ST/trace_${ds}_${arm}.json" \
    > "$out" 2> "$err"
  if grep -q "calib: K=" "$out"; then rm -f "$err"; else
    echo "$ds $arm" >> "$ST/failed_jobs.txt"
  fi
}

running=0
while read -r ds arm; do
  run_one "$ds" "$arm" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
done < "$JOBLIST"
wait

NFAIL=$(wc -l < "$ST/failed_jobs.txt")
if [ "$NFAIL" -ne 0 ]; then
  log "WARNING: $NFAIL jobs produced no K (see failed_jobs.txt)"
  fail "$NFAIL replays failed"
fi
log "ALL $NJOBS ADAPTIVE-SCALAR REPLAYS DONE"
touch "$ST/DONE"
