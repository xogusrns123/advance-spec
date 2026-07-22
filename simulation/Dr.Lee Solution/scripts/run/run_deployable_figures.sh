#!/usr/bin/env bash
# Deployable-split figure set -> readable_outputs/figures/mat(deployable)/
# (CALIB_effect, CALIB_comparison, MAT_per_workload_4way, MAT_per_segment,
#  MAT_segments_{ds}, MAT_workload_{ds}). Needs run_deployable_split.sh DONE.
#
#   docker exec sglang-bench bash "/workspace/simulation/Dr.Lee Solution/scripts/run/run_deployable_figures.sh"
set -euo pipefail
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""

ST=results/pipeline_deployable
SEGD=/workspace/simulation/results/pipeline_deployable/segments
FIG="readable_outputs/figures/mat(deployable)"
NOTE="deployable split: compose calibrators + hybrid τ* fit on the CALIB half (even within-label convs), MAT measured on the disjoint TEST half (odd)"
mkdir -p "$FIG"

python3 scripts/plot_mat_4way.py --log-suffix _split --sweep-dir "$SEGD" \
  --out-dir "$FIG" --note "$NOTE"

python3 scripts/plot/plot_mat_workload_solo.py --log-suffix _split \
  --sweep-dir "$SEGD" --out-dir "$FIG"

python3 scripts/plot/plot_mat_segments.py --tag _split --seg-dir "$SEGD" \
  --out-dir "$FIG" --taus-json "$ST/taus_split.json" \
  --tau-note "τ* picked on calib half; bars = test half"

python3 scripts/plot/plot_calib_effect.py --log-suffix _split --sweep-dir "$SEGD" \
  --out-dir "$FIG" --note "$NOTE"

python3 scripts/plot/plot_calib_comparison.py --log-suffix _split --sweep-dir "$SEGD" \
  --out-dir "$FIG" --note "$NOTE"

echo "ALL DEPLOYABLE FIGURES DONE -> $FIG"
