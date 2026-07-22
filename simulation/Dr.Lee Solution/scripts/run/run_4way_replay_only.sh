#!/usr/bin/env bash
# CPU-ONLY immediate pass: replay the 4 arms on the EXISTING (2026-07-03
# snapshot) perpos records and draw the 4-way figures right away — no GPU, no
# capture. The main run_4way_pipeline.sh later recaptures from the updated
# gt_tokens and regenerates the same figures (its logs use the un-suffixed
# names; this pass writes *_pre logs, so the two never collide).
#
#   docker exec -i sglang-bench bash "/workspace/simulation/Dr.Lee Solution/scripts/run_4way_replay_only.sh"
set -uo pipefail
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace

ST=/workspace/simulation/results/pipeline_4way
FIG=readable_outputs/figures
RLOG=$FIG/replay_logs
mkdir -p "$ST" "$RLOG"

declare -A REC=(
  [spider]=results/perpos_spider/spider.jsonl
  [specbench]=results/perpos_specbench_full/specbench.jsonl
  [swebench]=results/perpos_swebench/swebench_quick.jsonl
)

log() { echo "[$(date -u +%F' '%T)UTC] $*" | tee -a "$ST/replay_only.log"; }

PIDS=()
for ds in spider specbench swebench; do
  rec=${REC[$ds]}
  [ -f "$rec" ] || { log "SKIP $ds: record $rec missing"; continue; }
  log "replays[$ds] start on $rec"
  python3 scripts/replay_extension.py --record "$rec" --max-rounds 4096 \
    --props dflash suffix \
    > "$RLOG/mat_${ds}_4way_singles_pre.replay.txt" 2> "$ST/replay_${ds}_singles_pre.err" &
  PIDS+=($!)
  python3 scripts/replay_extension.py --record "$rec" --max-rounds 4096 \
    --props calib --calib-insample --hazard-fit beta \
    > "$RLOG/mat_${ds}_4way_calib_pre.replay.txt" 2> "$ST/replay_${ds}_calib_pre.err" &
  PIDS+=($!)
  python3 scripts/replay_extension.py --record "$rec" --max-rounds 4096 \
    --props oracle \
    > "$RLOG/mat_${ds}_4way_oracle_pre.replay.txt" 2> "$ST/replay_${ds}_oracle_pre.err" &
  PIDS+=($!)
done

log "waiting for ${#PIDS[@]} replay procs"
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
[ $RC -ne 0 ] && { log "REPLAY_ONLY FAILED (see $ST/replay_*_pre.err)"; exit 1; }
log "replays done"

LEG=$FIG/legacy/pre_20260705
mkdir -p "$LEG"
moved=0
for f in "$FIG"/mat/*.png "$FIG"/ladders/*.png; do
  [ -f "$f" ] || continue
  case "$(basename "$f")" in *_4way*) continue;; esac
  mv "$f" "$LEG/"; moved=$((moved+1))
done
log "moved $moved old MAT figures -> $LEG"

python3 scripts/plot_mat_4way.py --log-suffix _pre \
  --note "records: 2026-07-03 capture snapshots (specbench=141-row subset, spider=816-row partial, swe=Jul3 run) — fresh-capture refresh follows" \
  2>&1 | tee -a "$ST/replay_only.log"
log "REPLAY_ONLY COMPLETE"
