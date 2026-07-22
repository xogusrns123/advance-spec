#!/usr/bin/env bash
# swebench step250 (max_steps=250) re-collection, SELF-TERMINATED instances only
# (COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT marker; turn-limit-truncated dropped at
# conv_map time). Waits for ANY GPU to have >=58 GB free (sustained), then captures
# there, replays (4-way + fallback sweep + 5-arm segments) and regenerates ALL
# figures with the corrected swebench.
#
#   docker exec -d sglang-bench bash -c \
#     'nohup bash "/workspace/simulation/Dr.Lee Solution/scripts/run_swebench_step250.sh" \
#      > /workspace/simulation/results/pipeline_4way/driver_swe250.out 2>&1'
#
# Markers: pipeline_4way/DONE_SWE250 / FAILED_SWE250.
set -uo pipefail
cd "/workspace/simulation/Dr.Lee Solution"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace

R=/workspace/simulation/results
ST=$R/pipeline_4way
SEG=$ST/segments
FIG=readable_outputs/figures
RLOG=$FIG/replay_logs
D=$R/swebench_full_traj/qwen35_27b_dflash_step250
FROZEN=$D/gt_tokens_frozen_0707.jsonl
PP=results/perpos_swebench_step250
CMAP=$PP/conv_map_0707.json
REC=$PP/swebench_4way.jsonl
TR=$PP/swebench_4way.traces.json
NEED=58000                                  # MiB free required for a 27B capture
mkdir -p "$SEG" "$RLOG" "$PP"
rm -f "$ST/DONE_SWE250" "$ST/FAILED_SWE250"

log() { echo "[$(TZ=Asia/Seoul date +%F' '%T)KST] $*" | tee -a "$ST/swe250.log"; }
fail() { log "FAILED: $*"; echo "$*" > "$ST/FAILED_SWE250"; exit 1; }

[ -f "$FROZEN" ] || fail "frozen gt_tokens missing"
[ -f "$CMAP" ] || fail "conv_map missing"

freemib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits; }

# ---- stage W: wait for a GPU with >=NEED MiB free, stable over two checks -----
log "stage W: waiting for a GPU with >=${NEED}MiB free (sustained)"
GPU=-1; t0=$SECONDS
while :; do
  for g in ${CAND_GPUS:-0}; do          # GPU0 only by default (standing rule)
    f=$(freemib "$g" 2>/dev/null || echo 0)
    if [ "${f:-0}" -ge "$NEED" ]; then
      sleep 45
      f2=$(freemib "$g" 2>/dev/null || echo 0)
      if [ "${f2:-0}" -ge "$NEED" ]; then GPU=$g; break; fi
    fi
  done
  [ "$GPU" -ge 0 ] && break
  [ $((SECONDS - t0)) -gt $((18 * 3600)) ] && fail "stage W: 18h timeout waiting for a free GPU"
  sleep 120
done
log "stage W: GPU$GPU free (${f2}MiB); capturing there"
export CUDA_VISIBLE_DEVICES=$GPU

# ---- stage 1: capture (self-terminated 19 instances) -------------------------
python3 scripts/capture_traj.py --gt-tokens "$FROZEN" --conv-map "$CMAP" \
  --task swebench --split-mode label-rank --checkpoint-every 5 --out "$REC" \
  > "$PP/capture_0707.log" 2>&1 || fail "stage 1: capture (see $PP/capture_0707.log)"
NEVAL=$(python3 -c "import json;print(len(json.load(open('$TR'))['eval_traces']))")
log "stage 1: capture done ($(wc -l < "$REC") records / $NEVAL eval calls)"
[ "$NEVAL" -ge 1 ] || fail "stage 1: no eval calls"

# ---- stage 2: 4-way replays (overwrite the standard swebench logs) -----------
PIDS=()
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 --props dflash suffix \
  > "$RLOG/mat_swebench_4way_singles.replay.txt" 2> "$ST/replay_swe250_singles.err" & PIDS+=($!)
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props calib --calib-insample --hazard-fit beta \
  > "$RLOG/mat_swebench_4way_calib.replay.txt" 2> "$ST/replay_swe250_calib.err" & PIDS+=($!)
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 --props oracle \
  > "$RLOG/mat_swebench_4way_oracle.replay.txt" 2> "$ST/replay_swe250_oracle.err" & PIDS+=($!)
RC=0; for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
[ $RC -ne 0 ] && fail "stage 2: 4-way replay"
log "stage 2: 4-way replays done"

# ---- stage 3: fallback sweep + 5-arm segment replays -------------------------
python3 scripts/replay_fallback_sweep.py --records "swebench=$REC" \
  --taus 0.5 1 2 4 8 16 32 --out "$SEG/fallback_sweep_fresh_swebench.json" \
  > "$SEG/sweep_fresh_swebench.log" 2>&1 || fail "stage 3: sweep"
TAU=$(python3 -c "import json;d=next(iter(json.load(open('$SEG/fallback_sweep_fresh_swebench.json')).values()));print(max(d.items(),key=lambda kv:kv[1]['K'])[0])")
log "stage 3: swebench best tau=$TAU; segment replays"
PIDS=()
python3 scripts/replay_segments_5way.py --record "$REC" --kind swebench \
  --arms dflash suffix fallback --tau "$TAU" --out-dir "$SEG" \
  > "$SEG/run_fresh_swebench_A.log" 2>&1 & PIDS+=($!)
python3 scripts/replay_segments_5way.py --record "$REC" --kind swebench \
  --arms calib --out-dir "$SEG" > "$SEG/run_fresh_swebench_B.log" 2>&1 & PIDS+=($!)
python3 scripts/replay_segments_5way.py --record "$REC" --kind swebench \
  --arms oracle --out-dir "$SEG" > "$SEG/run_fresh_swebench_C.log" 2>&1 & PIDS+=($!)
RC=0; for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
[ $RC -ne 0 ] && fail "stage 3: segment replay"
log "stage 3: segment replays done"

# ---- stage 4: merge taus + regenerate ALL figures ----------------------------
python3 - "$SEG" <<'EOF'
import json, sys
seg = sys.argv[1]; taus = {}
for ds in ("spider", "specbench", "swebench", "bfcl", "tau2"):
    try:
        d = next(iter(json.load(open(f"{seg}/fallback_sweep_fresh_{ds}.json")).values()))
        taus[ds] = float(max(d.items(), key=lambda kv: kv[1]["K"])[0])
    except FileNotFoundError:
        pass
json.dump(taus, open(f"{seg}/taus_fresh.json", "w")); print("taus:", taus)
EOF
python3 scripts/plot_mat_4way.py >> "$ST/swe250.log" 2>&1 || fail "stage 4: plot_mat_4way"
python3 scripts/plot_mat_subtask.py >> "$ST/swe250.log" 2>&1 || fail "stage 4: plot_mat_subtask"
python3 scripts/plot_mat_segments.py --taus-json "$SEG/taus_fresh.json" \
  >> "$ST/swe250.log" 2>&1 || fail "stage 4: plot_mat_segments"
log "SWE250 PIPELINE COMPLETE (swebench = self-terminated step250)"
touch "$ST/DONE_SWE250"
