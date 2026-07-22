#!/usr/bin/env bash
# BFCL leg of the 4-way pipeline (separate file so the RUNNING 3-workload
# run_4way_pipeline.sh instance is never mutated mid-execution). Waits until
# GPU0 is free of BOTH the sglang server and any capture_traj.py (the 3-workload
# captures run sequentially first), then captures + replays bfcl and regenerates
# the figures with all four workloads. conv_map_0707.json was built beforehand
# (lookahead 500 to skip failed-web_search retry residue).
#
#   docker exec -d sglang-bench bash -c \
#     'nohup bash "/workspace/simulation/Dr.Lee Solution/scripts/run_4way_pipeline_bfcl.sh" \
#      > /workspace/simulation/results/pipeline_4way/driver_bfcl.out 2>&1'
#
# Markers: results/pipeline_4way/DONE_BFCL / FAILED_BFCL.
set -uo pipefail
cd "/workspace/simulation/Dr.Lee Solution"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace

R=/workspace/simulation/results
ST=$R/pipeline_4way
FIG=readable_outputs/figures
RLOG=$FIG/replay_logs
mkdir -p "$ST" "$RLOG"
rm -f "$ST/DONE_BFCL" "$ST/FAILED_BFCL"

BF=$R/bfcl_v4_full_traj/qwen35_27b_dflash
GT=$BF/gt_tokens.jsonl
PP=results/perpos_bfcl_full
REC=$PP/bfcl_4way.jsonl

log() { echo "[$(date -u +%F' '%T)UTC] $*" | tee -a "$ST/pipeline_bfcl.log"; }
fail() { log "FAILED: $*"; echo "$*" > "$ST/FAILED_BFCL"; exit 1; }
fresh() { [ -f "$1" ] && [ "$1" -nt "$2" ]; }

[ -f "$PP/conv_map_0707.json" ] || fail "conv_map_0707.json missing"

# ---- stage W: GPU0 free of server AND of the 3-workload captures -------------
log "stage W(bfcl): waiting for GPU0 (server + capture_traj)"
t0=$SECONDS
while pgrep -f "sglang.launch_server.*30071" >/dev/null 2>&1 \
   || pgrep -f "capture_traj.py" >/dev/null 2>&1; do
  [ $((SECONDS - t0)) -gt $((8 * 3600)) ] && fail "stage W: 8h timeout waiting for GPU0"
  sleep 60
done
sleep 15
log "stage W(bfcl): GPU0 free after $(((SECONDS-t0)/60))min"

# ---- stage 2: capture --------------------------------------------------------
if fresh "$REC" "$GT" && fresh "$PP/bfcl_4way.traces.json" "$GT"; then
  log "stage 2(bfcl): capture fresh, skip"
else
  log "stage 2(bfcl): capture_traj start"
  python3 scripts/capture_traj.py --gt-tokens "$GT" \
    --conv-map "$PP/conv_map_0707.json" \
    --task bfcl_v4_full --split-mode label-rank --out "$REC" \
    > "$PP/capture_0705.log" 2>&1 \
    || fail "stage 2(bfcl): capture_traj (see $PP/capture_0705.log)"
  log "stage 2(bfcl): capture done ($(wc -l < "$REC") records)"
fi

# ---- stage 3: replays --------------------------------------------------------
log "stage 3(bfcl): launching replays (singles/calib/oracle)"
PIDS=()
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props dflash suffix \
  > "$RLOG/mat_bfcl_4way_singles.replay.txt" 2> "$ST/replay_bfcl_singles.err" &
PIDS+=($!)
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props calib --calib-insample --hazard-fit beta \
  > "$RLOG/mat_bfcl_4way_calib.replay.txt" 2> "$ST/replay_bfcl_calib.err" &
PIDS+=($!)
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props oracle \
  > "$RLOG/mat_bfcl_4way_oracle.replay.txt" 2> "$ST/replay_bfcl_oracle.err" &
PIDS+=($!)
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
[ $RC -ne 0 ] && fail "stage 3(bfcl): replay failed (see $ST/replay_bfcl_*.err)"
log "stage 3(bfcl): replays done"

# ---- stage 4: figures (wait for any other replays so logs are complete) ------
while pgrep -f "replay_extension.py" >/dev/null 2>&1; do sleep 60; done
python3 scripts/plot_mat_4way.py >> "$ST/pipeline_bfcl.log" 2>&1 \
  || fail "stage 4(bfcl): plot_mat_4way.py"
log "PIPELINE COMPLETE (bfcl joined — figures now 4 workloads)"
touch "$ST/DONE_BFCL"
