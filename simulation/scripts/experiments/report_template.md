# Extension suffix-hyperparameter sweep — Final Report

(filled in by `_analyze_sfx_sweep.py` after sweeps finish)

## TL;DR

- Best deployable extension_sfx: **TBD** (config TBD, on capture TBD, B=TBD)
- Best vs best hybrid_e3: **±X%** (positive = extension wins)
- Conservative-suffix strictly improves deployable: **YES/NO**
- Cells where extension > hybrid: **N / total** captures × budgets

## Headline numbers

### Best per (capture, budget)
_(per_cell_table.md will be inserted here)_

### Win cells (gap > 0)
_(win_cells.md will be inserted here)_

## Patterns observed

1. **B-effect**: as B grows, extension's win shrinks because target_forward(ext_size) dominates.
   Specifically: TBD
2. **Pool-effect**: smaller (steps, topk) pools favor extension. TBD
3. **Workload-effect**: TBD (specbench vs bfcl_v4 vs swebench)

## Best extension_sfx config

The (F, T, N) triple that wins most often: **TBD**

Reasoning: TBD

## Implications for research goal (50%+ gap vs hybrid)

TBD — based on data, recommend: ...

## Latency-config caveats

- target_forward measured up to B=512; B>512 is linear extrapolation.
- Extension's avg_ext_size at B=128 default sfx ≈ 527 → extrapolation kicks in.
- For conservative sfx (f1.0_t0.2_n16) at B=128, avg_ext_size ≈ 220 → within measured range.
- The interpretation of large-B cells should account for extrapolation noise.

## Next experiments suggested

1. Run `_remeasure_latency.sh` with extended budgets (1024, 2048) once GPU free.
2. Re-run sim with refreshed latency config.
3. Try `extension_nlevel_sfx:N:F:T:M` parametric variant on best (F,T,N) found.
4. Test on rr captures (steps8/topk16/full pool) for swebench_verified, longbench.
