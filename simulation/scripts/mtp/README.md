# MTP Draft-Token Collection (H100)

End-to-end pipeline to collect per-step MTP/NEXTN draft trees and
compute the same per-position metrics (sequential / independent /
**conditional**) as the offline simulator.

## Prerequisites (on the H100 host)

* Working SGLang install (commit ≥ the one that ships `NEXTN`).
* Repo checked out at `/path/to/advance-spec`.
* HF cache primed with the target model (e.g. `Qwen/Qwen3.5-9B`).
* Python with `transformers`, `requests`, and the project's runtime deps.
* Disk space for `/tmp/sglang_oracle_vanilla_<wl>.jsonl` (a few GB
  per run with the 80-question default).

## How it works

1. `launch_h100_server.sh` starts SGLang via
   `simulation.oracle.install_hook` with `SGLANG_ORACLE_VANILLA=1`.
   The runtime patch in `simulation/oracle/oracle_patch.py`
    a) force-accepts only the bonus token every verify step
       (`accept_length=0`) and
    b) appends the per-step draft tree (BFS-ordered `token_ids`,
       `parents`) plus the committed bonus token to
       `$SGLANG_ORACLE_LOG` (default
       `/tmp/sglang_oracle_vanilla.jsonl`).
2. `collect_mtp_drafts.py` is the client. For each dataset entry it
   tokenizes a single user prompt with the model's chat template,
   POSTs to `/generate`, then byte-offset-reads the oracle log to
   pick up the entries appended during this request. Output is a
   self-contained JSONL with `(prompt, output_ids, oracle_entries[])`.
3. `compute_mtp_position_accepts.py` reconstructs each request's
   ground-truth-future from its bonus tokens and runs
   `tree_knapsack.position_accept_rates` on every step's draft tree.
   Output JSON has the same `position_accepts.by_proposer.<name>`
   schema as `posacc_<wl>_s8k8.json` so the existing notebook reads
   it via the standard `load_sim` path.

## One-shot run

```bash
# H100 host, repo root
MODEL=Qwen/Qwen3.5-9B \
MTP_STEPS=8 MTP_TOPK=8 MTP_BUDGET=128 \
MAX_QUESTIONS=80 \
bash simulation/scripts/mtp/run_h100_pipeline.sh
```

This sequentially runs all three workloads (`specbench`, `bfcl_v4`,
`swebench_verified`), saving:

```
simulation/results/qwen35_mtp/
  mtp_drafts_<wl>.jsonl   (raw per-request entries — keep for re-analysis)
  posacc_mtp_<wl>.json    (per-position seq/ind/cond stats)
```

## Per-workload only

```bash
WORKLOADS="specbench" \
MODEL=Qwen/Qwen3.5-9B \
bash simulation/scripts/mtp/run_h100_pipeline.sh
```

## Notebook integration

The existing `simulation/notebooks/position_accept_quick.ipynb` reads
`posacc_<wl>_<reslice>.json` from `simulation/results/explorations`.
After the H100 run finishes, copy or symlink the MTP outputs over:

```bash
# Optionally rename to the notebook's expected pattern
for wl in specbench bfcl_v4 swebench_verified; do
  cp simulation/results/qwen35_mtp/posacc_mtp_${wl}.json \
     simulation/results/explorations/posacc_${wl}_mtp.json
done
```

…then change the notebook's `RESLICE = 's8k8'` to `'mtp'` (or load
both for an MTP-vs-EAGLE3 overlay — `load_sim` is parameterised on
`reslice`).

## Tuning notes

* **Memory.** With `--mem-fraction-static 0.85`, the 9B model + an
  MTP head + the budget=128 spec tree fits comfortably in 80 GB.
  If you OOM, drop budget to 64 first (still meaningful). Increasing
  topk × steps grows the tree, not memory.
* **Throughput.** Force-accept means each request decodes 1 token
  per server step, so latency is dominated by `max_new_tokens`. With
  default 512, a single 80-question workload is ~10–20 min on H100.
* **Log size.** Each step entry is ~30–60 KB (vocab-sized verify
  logits *not* logged in this mode — `_oracle_stashed_verify_logits`
  is left to its default; only the draft tree + p_t are kept).
* **Multi-GPU.** Single-server / single-GPU is the simplest path.
  For tensor parallel, set `--tp N` in `launch_h100_server.sh` and
  list multiple `CUDA_VISIBLE_DEVICES`. Note that with TP > 1 only
  rank 0 writes the oracle log (handled by `oracle_patch.py:864`).
