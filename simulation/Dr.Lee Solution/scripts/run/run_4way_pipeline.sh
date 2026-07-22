#!/usr/bin/env bash
# ============================================================================
# 2026-07-05 recapture -> replay -> figures pipeline for the Dr.Lee calibrated
# 4-arm study (dflash / suffix / compose-calibrated / oracle). Run inside
# sglang-bench (root, GPU0):
#
#   docker exec -d sglang-bench bash -c \
#     'nohup bash "/workspace/simulation/Dr.Lee Solution/scripts/run_4way_pipeline.sh" \
#      > /workspace/simulation/results/pipeline_4way/driver.out 2>&1'
#
# WORKLOADS env selects the datasets (default: the three whose collections are
# final — spider specbench swebench, longest-replay first). When the user
# confirms the BFCL collection is complete, rerun with WORKLOADS=bfcl: it
# captures/replays bfcl and REGENERATES both figures with all four workloads
# (plot_mat_4way.py skips datasets whose replay logs are absent).
#
# Stages (restartable; each skips when its output is fresh):
#   W  wait for GPU0 to be free (the running BFCL web_search collection's
#      server; its wrapper kills the server via EXIT trap when it converges)
#   1  build_conv_map (exact conversation/task map on the NEW gt_tokens)
#   2  capture_traj sequential on GPU0
#   3  replay_extension per workload, 3 parallel procs (singles/calib/oracle),
#      TWO-WAY IN-SAMPLE protocol (no --three-way; calib = --calib-insample
#      --hazard-fit beta), --max-rounds 4096 (score whole calls)
#   4  move old MAT figures -> legacy/pre_20260705 (once; *_4way* excluded),
#      plot_mat_4way.py
#
# Status: results/pipeline_4way/pipeline.log + DONE / FAILED marker files.
# ============================================================================
set -uo pipefail
cd "/workspace/simulation/Dr.Lee Solution"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace

WORKLOADS="${WORKLOADS:-spider specbench swebench}"

R=/workspace/simulation/results
ST=$R/pipeline_4way
FIG=readable_outputs/figures
RLOG=$FIG/replay_logs
mkdir -p "$ST" "$RLOG"
rm -f "$ST/DONE" "$ST/FAILED"

BF=$R/bfcl_v4_full_traj/qwen35_27b_dflash
SP=$R/specbench_full_traj/qwen35_27b_dflash
SD=$R/spider_dbt_full_traj/qwen35_27b_dflash
SW=$R/swebench_full_traj/qwen35_27b_dflash

log() { echo "[$(date -u +%F' '%T)UTC] $*" | tee -a "$ST/pipeline.log"; }
fail() { log "FAILED: $*"; echo "$*" > "$ST/FAILED"; exit 1; }

# fresh(out, src): out exists and is newer than src
fresh() { [ -f "$1" ] && [ "$1" -nt "$2" ]; }

# ---- stage W: wait for GPU0 to be free ---------------------------------------
log "stage W: workloads='$WORKLOADS' — waiting for GPU0 (sglang server on :30071)"
t0=$SECONDS
while pgrep -f "sglang.launch_server.*30071" >/dev/null 2>&1; do
  [ $((SECONDS - t0)) -gt $((6 * 3600)) ] && fail "stage W: 6h timeout waiting for GPU0"
  sleep 60
done
sleep 15
log "stage W: GPU0 free after $(((SECONDS-t0)/60))min"

# ---- stage 1: conv maps ------------------------------------------------------
declare -A CM_KIND=( [bfcl]=bfcl [specbench]=specbench [spider]=spider [swebench]=swe )
declare -A CM_JSON=( [bfcl]=$BF/agent_results_record.json [specbench]=$SP/agent_results_record.json
                     [spider]=$SD/agent_results_all.json  [swebench]=$SW/agent_results_all.json )
declare -A GT=( [bfcl]=$BF/gt_tokens.jsonl [specbench]=$SP/gt_tokens.jsonl
                [spider]=$SD/gt_tokens.jsonl [swebench]=$SW/gt_tokens.jsonl )
declare -A PP=( [bfcl]=results/perpos_bfcl_full [specbench]=results/perpos_specbench_full
                [spider]=results/perpos_spider [swebench]=results/perpos_swebench )

for ds in $WORKLOADS; do
  [ "$ds" = "bfcl" ] && [ ! -f "${CM_JSON[$ds]}" ] \
    && fail "stage 1[bfcl]: final agent_results_record.json missing"
  cm=${PP[$ds]}/conv_map_0705.json
  if fresh "$cm" "${GT[$ds]}"; then log "stage 1[$ds]: conv_map fresh, skip"; continue; fi
  log "stage 1[$ds]: build_conv_map"
  python3 scripts/build_conv_map.py --kind "${CM_KIND[$ds]}" \
    --agent-json "${CM_JSON[$ds]}" --gt-tokens "${GT[$ds]}" --out "$cm" \
    >> "$ST/convmap_$ds.log" 2>&1 || fail "stage 1[$ds]: build_conv_map (see $ST/convmap_$ds.log)"
done
log "stage 1: conv maps done"

# ---- stages 2+3: capture (sequential, GPU) then replay (parallel, CPU) ------
declare -A CAP_ARGS=(
  [bfcl]="--task bfcl_v4_full --split-mode label-rank"
  [specbench]="--task specbench_full --split-mode label-rank"
  [spider]="--task spider_dbt"
  [swebench]="--task swebench --per-task-eval-convs 2 --max-calls-per-conv 8"
)
REPLAY_PIDS=()

start_replays() {                       # $1 = ds, $2 = record
  local ds=$1 rec=$2
  local common=(--record "$rec" --max-rounds 4096)
  log "stage 3[$ds]: launching replays (singles/calib/oracle)"
  python3 scripts/replay_extension.py "${common[@]}" --props dflash suffix \
    > "$RLOG/mat_${ds}_4way_singles.replay.txt" 2> "$ST/replay_${ds}_singles.err" &
  REPLAY_PIDS+=($!)
  python3 scripts/replay_extension.py "${common[@]}" --props calib \
    --calib-insample --hazard-fit beta \
    > "$RLOG/mat_${ds}_4way_calib.replay.txt" 2> "$ST/replay_${ds}_calib.err" &
  REPLAY_PIDS+=($!)
  python3 scripts/replay_extension.py "${common[@]}" --props oracle \
    > "$RLOG/mat_${ds}_4way_oracle.replay.txt" 2> "$ST/replay_${ds}_oracle.err" &
  REPLAY_PIDS+=($!)
}

for ds in $WORKLOADS; do
  rec=${PP[$ds]}/${ds}_4way.jsonl
  tr=${PP[$ds]}/${ds}_4way.traces.json
  if fresh "$rec" "${GT[$ds]}" && fresh "$tr" "${GT[$ds]}"; then
    log "stage 2[$ds]: capture fresh, skip"
  else
    log "stage 2[$ds]: capture_traj start (${CAP_ARGS[$ds]})"
    python3 scripts/capture_traj.py --gt-tokens "${GT[$ds]}" \
      --conv-map "${PP[$ds]}/conv_map_0705.json" ${CAP_ARGS[$ds]} --out "$rec" \
      > "${PP[$ds]}/capture_0705.log" 2>&1 \
      || fail "stage 2[$ds]: capture_traj (see ${PP[$ds]}/capture_0705.log)"
    log "stage 2[$ds]: capture done ($(wc -l < "$rec") records)"
  fi
  start_replays "$ds" "$rec"
done

log "stage 3: waiting for ${#REPLAY_PIDS[@]} replay procs"
RC=0
for pid in "${REPLAY_PIDS[@]}"; do wait "$pid" || RC=1; done
[ $RC -ne 0 ] && fail "stage 3: one or more replays failed (see $ST/replay_*.err)"
log "stage 3: all replays done"

# ---- stage 4: figures --------------------------------------------------------
LEG=$FIG/legacy/pre_20260705
mkdir -p "$LEG"
moved=0
for f in "$FIG"/mat/*.png "$FIG"/ladders/*.png; do
  [ -f "$f" ] || continue
  case "$(basename "$f")" in *_4way*) continue;; esac
  mv "$f" "$LEG/"; moved=$((moved+1))
done
log "stage 4: moved $moved old MAT figures -> $LEG"
python3 scripts/plot_mat_4way.py >> "$ST/pipeline.log" 2>&1 \
  || fail "stage 4: plot_mat_4way.py"
python3 scripts/plot_mat_subtask.py >> "$ST/pipeline.log" 2>&1 \
  || fail "stage 4: plot_mat_subtask.py"
log "PIPELINE COMPLETE (workloads: $WORKLOADS)"
touch "$ST/DONE"
