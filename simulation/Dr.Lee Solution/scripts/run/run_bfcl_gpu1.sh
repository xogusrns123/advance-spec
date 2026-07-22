#!/usr/bin/env bash
# BFCL 4-way + segment leg, EXCEPTIONALLY on GPU1 (H100), with a hard KST-09:00
# deadline. capture_traj checkpoints .traces.json every 25 rows and stops
# cleanly at --deadline-epoch (08:40 KST) leaving a consistent partial capture;
# a separate watchdog hard-kills everything at 09:00 KST no matter what.
#
#   docker exec -d sglang-bench bash -c \
#     'nohup bash "/workspace/simulation/Dr.Lee Solution/scripts/run_bfcl_gpu1.sh" \
#      > /workspace/simulation/results/pipeline_4way/driver_bfcl_gpu1.out 2>&1'
#
# Markers: pipeline_4way/DONE_BFCL_GPU1 / FAILED_BFCL_GPU1 / STOPPED_BFCL_GPU1.
set -uo pipefail
cd "/workspace/simulation/Dr.Lee Solution"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace

CAP_DEADLINE=1783294800      # 08:40 KST — capture stops cleanly (checkpoint)
HARD_KILL=1783296000         # 09:00 KST — watchdog kills everything

R=/workspace/simulation/results
ST=$R/pipeline_4way
SEG=$ST/segments
FIG=readable_outputs/figures
RLOG=$FIG/replay_logs
mkdir -p "$ST" "$SEG" "$RLOG"
rm -f "$ST/DONE_BFCL_GPU1" "$ST/FAILED_BFCL_GPU1" "$ST/STOPPED_BFCL_GPU1"

BF=$R/bfcl_v4_full_traj/qwen35_27b_dflash
GT=$BF/gt_tokens.jsonl
PP=results/perpos_bfcl_full
REC=$PP/bfcl_4way.jsonl
TR=$PP/bfcl_4way.traces.json

log() { echo "[$(TZ=Asia/Seoul date +%F' '%T)KST] $*" | tee -a "$ST/bfcl_gpu1.log"; }
fail() { log "FAILED: $*"; echo "$*" > "$ST/FAILED_BFCL_GPU1"; exit 1; }

[ -f "$PP/conv_map_0705.json" ] || fail "conv_map_0705.json missing"

# ---- hard-kill watchdog (independent; survives this driver) ------------------
nohup bash -c "
  while [ \$(date +%s) -lt $HARD_KILL ]; do sleep 30; done
  echo '[watchdog] 09:00 KST hard kill' >> '$ST/bfcl_gpu1.log'
  pkill -KILL -f 'capture_traj.py'
  pkill -KILL -f 'replay_extension.py'
  pkill -KILL -f 'replay_segments_5way.py'
  pkill -KILL -f 'replay_fallback_sweep.py'
  pkill -KILL -f 'run_bfcl_gpu1.sh'
  touch '$ST/STOPPED_BFCL_GPU1'
" > /dev/null 2>&1 &
WATCHDOG=$!
log "watchdog $WATCHDOG armed: hard kill at 09:00 KST (epoch $HARD_KILL)"

# ---- stage 1: capture on GPU1 (checkpointed, deadline 08:40 KST) -------------
log "stage 1: capture on GPU1 (H100), checkpoint/25 rows, deadline 08:40 KST"
python3 scripts/capture_traj.py --gt-tokens "$GT" \
  --conv-map "$PP/conv_map_0705.json" \
  --task bfcl_v4_full --split-mode label-rank --out "$REC" \
  --checkpoint-every 25 --deadline-epoch "$CAP_DEADLINE" \
  > "$PP/capture_gpu1.log" 2>&1 \
  || fail "stage 1: capture_traj (see $PP/capture_gpu1.log)"
NREC=$(wc -l < "$REC"); NEVAL=$(python3 -c "import json;print(len(json.load(open('$TR'))['eval_traces']))")
if grep -q "STOPPED EARLY at deadline" "$PP/capture_gpu1.log"; then
  log "stage 1: DEADLINE-STOPPED — partial capture ($NREC records / $NEVAL eval calls)"
  echo "deadline-stopped: $NREC rec / $NEVAL eval" > "$ST/STOPPED_BFCL_GPU1"
else
  log "stage 1: capture COMPLETE ($NREC records / $NEVAL eval calls)"
fi
[ "$NEVAL" -ge 1 ] || fail "stage 1: no eval calls captured"

# GPU no longer needed — everything below is CPU. Disarm the (GPU) worry early
# by continuing; the watchdog still guards the 09:00 wall clock.

# ---- stage 2: 4-way replays (bfcl) -> plot_mat_4way (all 4 workloads) --------
log "stage 2: bfcl 4-way replays (singles/calib/oracle)"
PIDS=()
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props dflash suffix \
  > "$RLOG/mat_bfcl_4way_singles.replay.txt" 2> "$ST/replay_bfcl_singles.err" & PIDS+=($!)
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props calib --calib-insample --hazard-fit beta \
  > "$RLOG/mat_bfcl_4way_calib.replay.txt" 2> "$ST/replay_bfcl_calib.err" & PIDS+=($!)
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props oracle \
  > "$RLOG/mat_bfcl_4way_oracle.replay.txt" 2> "$ST/replay_bfcl_oracle.err" & PIDS+=($!)
RC=0; for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
[ $RC -ne 0 ] && fail "stage 2: 4-way replay (see $ST/replay_bfcl_*.err)"
python3 scripts/plot_mat_4way.py \
  >> "$ST/bfcl_gpu1.log" 2>&1 || fail "stage 2: plot_mat_4way"
python3 scripts/plot_mat_subtask.py \
  >> "$ST/bfcl_gpu1.log" 2>&1 || fail "stage 2: plot_mat_subtask"
log "stage 2: per-workload + per-subtask figures regenerated"

# ---- stage 3: bfcl segment leg (sweep -> tau -> 5-arm segment replays) -------
log "stage 3: bfcl fallback tau sweep"
python3 scripts/replay_fallback_sweep.py --records "bfcl=$REC" \
  --taus 0.5 1 2 4 8 16 32 --out "$SEG/fallback_sweep_fresh_bfcl.json" \
  > "$SEG/sweep_fresh_bfcl.log" 2>&1 || fail "stage 3: bfcl sweep"
TAU=$(python3 -c "import json;d=next(iter(json.load(open('$SEG/fallback_sweep_fresh_bfcl.json')).values()));print(max(d.items(),key=lambda kv:kv[1]['K'])[0])")
log "stage 3: bfcl best tau=$TAU; segment replays (dflash/suffix/fallback/calib/oracle)"
PIDS=()
python3 scripts/replay_segments_5way.py --record "$REC" --kind bfcl \
  --arms dflash suffix fallback --tau "$TAU" --out-dir "$SEG" \
  > "$SEG/run_fresh_bfcl_A.log" 2>&1 & PIDS+=($!)
python3 scripts/replay_segments_5way.py --record "$REC" --kind bfcl \
  --arms calib --out-dir "$SEG" > "$SEG/run_fresh_bfcl_B.log" 2>&1 & PIDS+=($!)
python3 scripts/replay_segments_5way.py --record "$REC" --kind bfcl \
  --arms oracle --out-dir "$SEG" > "$SEG/run_fresh_bfcl_C.log" 2>&1 & PIDS+=($!)
RC=0; for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
[ $RC -ne 0 ] && fail "stage 3: bfcl segment replay (see $SEG/run_fresh_bfcl_*.log)"

# merge taus (all 4) then plot all four segment figures fresh
python3 - "$SEG" <<'EOF'
import json, sys
seg = sys.argv[1]; taus = {}
for ds in ("spider", "specbench", "swebench", "bfcl"):
    try:
        d = next(iter(json.load(open(f"{seg}/fallback_sweep_fresh_{ds}.json")).values()))
        taus[ds] = float(max(d.items(), key=lambda kv: kv[1]["K"])[0])
    except FileNotFoundError:
        pass
json.dump(taus, open(f"{seg}/taus_fresh.json", "w")); print("taus:", taus)
EOF
python3 scripts/plot_mat_segments.py \
  --note "offline replay · records: 2026-07-05 captures" \
  --taus-json "$SEG/taus_fresh.json" >> "$ST/bfcl_gpu1.log" 2>&1 \
  || fail "stage 3: plot_mat_segments"
log "stage 3: segment figures regenerated (4 workloads)"

kill "$WATCHDOG" 2>/dev/null || true
log "BFCL GPU1 PIPELINE COMPLETE"
touch "$ST/DONE_BFCL_GPU1"
