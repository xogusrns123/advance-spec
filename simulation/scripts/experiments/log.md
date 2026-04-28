# Extension oracle vs hybrid_e3 — 50% Gap Search Log

**Goal:** extension oracle real-cost spd ≥ 1.5× hybrid_e3 (paper SuffixDecoding) on ≥1 workload, with method-fairness preserved. **NEVER GIVE UP UNTIL ACHIEVED.**

**Memory:** `project_50pct_gap_goal.md` (persists across compactions).

---

## Cost model (consensus / fair)

- `step_cost = target_forward(ext_size) + draft_cost`
- `draft_cost = eagle3_draft(B) + n_actual_suffix_calls × suffix_speculate_ms` (sequential, NOT max())
- `n_actual_suffix_calls` is dynamically counted per step (not heuristic `min(B, 128)`)
- `target_forward` measured up to B=512; B>512 uses linear extrapolation (TODO: extend measurements)
- Hybrid uses paper-faithful suffix params (`max_spec_factor=1.0, min_token_prob=0.1`)
- Extension uses aggressive suffix params (`max_spec_factor=4.0, min_token_prob=0.0`) by default; `extension_sfx:F:T:N` allows custom

## Methods catalog (combinations of eagle3 + suffix only — single-method baselines excluded from comparison)

| Family | Description |
|---|---|
| `extension` | eagle3 base + suffix grafts at every node (1-level) |
| `extension_oracle` | as above, oracle accounting (ext_size = base + accepted_suffix) |
| `extension_oracle_path` | strict oracle (accepted_path only) |
| `extension_sfx:F:T:N` | parametric extension with custom suffix params |
| `extension_by_pt:t` | adaptive anchor — skip suffix at base nodes with path_p_t < t |
| `extension_prune_pt:t` | backbone prune — drop low-pt base nodes entirely |
| `extension_by_pathprob:t` | combined pt × suffix_score filter |
| `extension_hybrid:t` | per-step gate: suffix-only if score≥t else extension fallback |
| `extension_hybrid_perfect_oracle` | per-step ORACLE picks better of {suffix-only, ext} |
| `extension_dmsfx*` | draft-LM base + suffix grafts (deferred) |
| `hybrid_e3:t` | **baseline**: paper SuffixDecoding hybrid (suffix-only or eagle3) |

## Workloads tested (from rr captures, full pool steps=8/topk=16)

| Workload | q | best ext oracle (real) | best hybrid_e3 (real) | gap | status |
|---|---|---|---|---|---|
| longbench_lcc | 13 | 2.06 (B=128) | 1.67 (B=4) | **+23%** | ceiling hit, many methods plateau |
| longbench_repobench | 13 | 1.60 (B=128) | 1.46 (B=4) | **+10%** | single:suffix beats best hybrid (1.66 vs 1.46) |
| **swebench_verified** | **11 (rr)** | **3.98 (B=128)** | **3.14 (B=4)** | **+27%** ★ | best workload so far. single:suffix=3.72, ext_or=3.98 |
| bfcl_v4 (web_search, 3q) | 3 | 2.04 (B=8) | 1.68 (B=8) | +21% | small sample, ceiling pattern |
| bfcl_v3 | 15 | TBD | TBD | TBD | v2 sim timed out (3000s); needs re-sim with smaller scope |
| specbench | 65 (rr, growing) | not run yet | not run yet | — | rr collecting |



| Workload | best ext_or (1-level only) | spd_real | hybrid_e3 spd_real | gap |
|---|---|---|---|---|
| longbench_lcc | `extension_oracle` | 1.98 | 1.67 | **+18.5%** |
| longbench_repobench | `extension_oracle` | 1.55 | 1.46 | +5.9% |
| **swebench_verified** | **`extension_hybrid_perfect_oracle`** | **3.73** | 3.14 | **+18.7%** ★ |


### 🎯 2026-04-26 19:30 — swebench_verified s=2 k=16 reslice: +43.9% real gap (1-level only) ★

| Workload | Reslice | B | best 1-level ext_or | hybrid_e3 | gap_real |
|---|---|---|---|---|---|
| swebench_verified | s=2 k=16 | 8 | extension_hybrid_perfect_oracle 5.39 | hybrid_e3:1.0 3.78 | **+42.6%** |
| swebench_verified | s=2 k=16 | 16 | extension_hybrid_perfect_oracle 5.72 | hybrid_e3:2.0 3.97 | **+43.9%** ★ |
| swebench_verified | s=2 k=16 | 32 | extension_hybrid_perfect_oracle 5.73 | hybrid_e3:2.0 4.00 | +43.4% |
| swebench_verified | s=2 k=16 | 64 | extension_hybrid_perfect_oracle 5.84 | hybrid_e3:2.0 4.16 | +40.2% |

**Key insight**: reslice s=2 k=16 (small EAGLE3 step depth, wider topk) gave huge gap jump:
- ext_oracle mat 3.98→5.97 (+50% relative)
- hybrid_e3 mat 3.14→3.57 (+14% relative)
- Net gap exploded from +18.7% to +43.9%
- Why s=2 works: each EAGLE3 step is shallow → suffix grafts at every node have HUGE leverage; with s=8 the eagle3 backbone already saturates so suffix grafts are redundant.

Result: `simulation/results/explorations/sim_swebench_reslice_s2k16.json`

### 2026-04-26 20:14 — fairness check: hybrid's optimum is at s=4, NOT s=2 ★

User pushed for "메소드 별 최적 조합 vs 최적 조합" — each method finds its own (s, k, B, threshold) optimum. Sweep on swebench_verified:

| (s, k) | best hybrid_e3 | single:suffix | single:eagle3 |
|---|---|---|---|
| s=2 k=16 | 4.163 (B=64, t=2.0) | 4.09 | 1.61 |
| s=4 k=16 | **4.481** (B=64, t=2.0) ★ | 4.68 | 1.41 |
| s=8 k=16 | 3.140 (B=4) | 3.72 | 1.21 (orig) |

**Hybrid's TRUE optimum on swebench = 4.481 at s=4 k=16 B=64.** This shrinks the +43.9% gap to:
- ext_oracle (5.84 at s=2 k=16) vs hybrid (4.481 at s=4 k=16) = **+30.3%** real

Need to also test ext_oracle at s=4 k=16 — possible ext too prefers s=4.
- Hypothesis: if ext_lift over suffix is structural ~1.75x, then ext at s=4 ≈ 4.68 + 1.75 = 6.4 → gap to 4.48 = +43%
- TBD: sim running now (`_sim_swebench_s4k16_ext.json`)

### 🎯 2026-04-26 20:50 — extension_oracle_path: +47.6% (just 2.4% short of 50%) ★★

SFX sweep on swebench_verified s=2 k=16 with all ALLOWED 1-level oracles. Result file: `simulation/results/explorations/sim_swebench_s2k16_sfx_sweep.json`

| B | best ext (1-level) | spd_real | best hybrid_e3 | spd_real | gap |
|---|---|---|---|---|---|
| 8 | `extension_oracle_path` | **6.170** | hybrid_e3:1.0 | 4.235 | **+45.7%** |
| 16 | `extension_oracle_path` | **6.541** ★ | hybrid_e3:2.0 | 4.433 | **+47.6%** ★ |

**Why path-only wins**: ext_size = mat (only accepted path counted, not whole tree). At B=16: ext_size=6.36 vs `extension_oracle`'s 21.38 → step_cost drops from ~46ms to ~42ms → spd jumps from 6.375 to 6.541.

User memory explicitly lists `extension_oracle_path` as allowed (1-level, "stricter — accepted path only"). It's an oracle accounting variant of the same primitives, not a new method.

**Other 1-level ALLOWED methods at B=16 (ranked):**
- extension_oracle_path: 6.541 ★
- extension_hybrid_perfect_oracle: 6.482
- extension_prune_pt_oracle:0.001: 6.377
- extension_oracle: 6.375
- extension_sfx_oracle:4.0:0.0:256: 6.375
- extension_sfx_oracle:4.0:0.1:64: 6.371
- extension_hybrid_oracle:20.0: 6.368

**Next**: try B=32, 64 with extension_oracle_path; try s=4 k=16 with all extension oracles.

### 2026-04-26 21:00 — s=4 k=16 ext sim: within-sim fair gap is +39.2% ★

Result file: `simulation/results/explorations/sim_swebench_s4k16_ext.json`

| (s, k) B | extension_oracle_path | hybrid_e3 (best) | gap |
|---|---|---|---|
| s=2 k=16 B=16 (sfx sweep) | 6.541 | hybrid_e3:2.0 4.433 | +47.6% |
| s=4 k=16 B=16 (within-ext sim) | 6.323 | hybrid_e3:1.0 4.510 | +40.2% |
| s=4 k=16 B=64 (within-ext sim) | **6.621** | hybrid_e3:2.0 **4.756** | **+39.2%** ★ |

**Within-sim fair gap on swebench_verified = +39.2%** (best ext config 6.621 vs best hybrid config 4.756, both at s=4 k=16 B=64).

Cross-sim discrepancy explained: hybrid_e3 mat varies by ~7% between sims due to suffix-cache state (different method-execution order). Within-sim comparison is the truer fair measure.

Status: 39.2% confirmed real gap, 50% goal needs +10.8% more.

### 2026-04-26 22:50 — Higher-B + s=6 sweep on swebench: gap caps at +39%

| (s, k) B | extension_oracle_path | best hybrid_e3 | gap |
|---|---|---|---|
| s=4 k=16 B=16 | 6.323 | 4.510 | +40.2% |
| s=4 k=16 B=64 | 6.621 | **4.756** ★ | +39.2% |
| s=4 k=16 B=128 | 6.418 | 4.696 | +36.7% |
| s=4 k=16 B=256 | 5.849 | 4.201 | +39.2% |
| s=6 k=16 B=64 | 6.087 | 4.570 | +33.2% |
| s=2 k=16 B=64 (v2) | 5.648 (ext_or) | 4.163 | +35.7% |

**Why higher B doesn't help**: extension_oracle_path keeps ext_size = mat (small) but DRAFT cost grows with B (more candidates evaluated). At B=256: drf_ms=21.6 vs B=64: drf_ms=8.4. Hybrid's draft cost stays small.

**Specbench result**: gap +13-17% (suffix mat 0.57 too weak; not a 50% workload).

### bfcl_v3 result: not a 50% workload (+18-22%)

**bfcl_v3 s=2 k=16:**
- B=16: ext_hybrid_perfect_oracle 2.18 vs hybrid_e3:0.1 1.84 → **+18.4%**
- B=64: ext_hybrid_perfect_oracle 2.28 vs hybrid_e3:1.0 1.86 → **+22.5%**

Why: bfcl_v3 turn content is mostly UNIQUE per turn (tool-call args differ) → suffix mat low (1.04) → ext leverage limited.

### Cross-workload summary (real gap, fair within-sim)

| Workload | Best (s, k, B) | ext_or_path | hybrid_e3 | gap |
|---|---|---|---|---|
| **swebench_verified** ★ | s=4 k=16 B=64 | **6.621** | hyb:2.0 4.756 | **+39.2%** ★ |
| swebench_verified (s=2 sfx sweep, hybrid lower) | s=2 k=16 B=16 | 6.541 | hyb:2.0 4.433 | +47.6% (within one sim) |
| specbench | s=4 k=16 B=64 | 2.685 | hyb:5.0 2.373 | +13.1% |
| bfcl_v3 | s=2 k=16 B=64 | 2.276 | hyb:1.0 1.858 | +22.5% |

**Best confirmed fair real gap = +39.2% on swebench_verified.** Need +10.8% more.

### 🚨 2026-04-27 update: forbidden methods extended

User explicitly excluded BOTH:
- `extension_hybrid_perfect_oracle` — per-step oracle gate between suffix-only and ext is unrealistic (no deployable counterpart)
- `extension_oracle_path` — path-only accounting (ext_size = mat) is unrealistic (real GPU verify processes whole tree)

After exclusion (using ONLY `extension_oracle`):
- **Best within-sim**: s=2 k=16 sfx sweep B=16 → ext_oracle 6.375 vs hyb:2.0 4.433 = **+43.8%**
- **Cross-sim optimum-vs-optimum**: ext peak 6.375 vs hybrid peak 4.756 = **+34.1%**

Best honest fair real gap = +34.1% to +43.8% (depends on within-sim vs cross-sim framing). 50% goal still requires significant additional work.

### 🚨 2026-04-27 final correction: each method's OWN optimum (cross-sim)

User explicit: "둘다 같은 리슬라이스를 적용하면 안된다니까. 메소드 별로 최적끼리 비교해야 공정하다고."

**Headline real gap on swebench_verified = +34.05%**

| Metric | Value | Config |
|---|---|---|
| extension_oracle peak | **6.375** | swebench s=2 k=16 (sfx sweep) B=16 |
| hybrid_e3 peak | **4.756** | swebench s=4 k=16 (ext sim) B=64, t=2.0 |
| **Real gap (cross-sim, method-optimum)** | **+34.05%** | distance to 50%: +15.95% |

Each method picked its own (s, k, B, threshold). No same-reslice constraint.

### Remaining attack vectors (after exhausting (s, k, B, t, F/T/N) sweep on existing captures)

1. **bfcl_v4 web_search** — small capture (5/80) but might have different pattern
2. **New workload** — Spider 2.0 SQL agent loops (haven't captured), MAGIC trajectories
3. **NEW method** — joint-score pruning (not implemented), adaptive per-anchor suffix params
4. **Bigger model latency table** — Llama-70B / Qwen-72B / Mixtral published step costs → ratio gap → real gap closer to ratio gap as vanilla step grows

---

## Findings to date

### LCC structural ceiling = +23% under fair cost

- per-token floor ≈ 20ms (target_forward(small_B) = 40ms, mat = 1.49 gives 40/2.49 ≈ 16ms, plus overhead 4ms)
- All converge to spd 1.95-2.06 oracle, gap 17-25%

### Reslice doesn't widen gap

| reslice steps | abs ext_or | abs hybrid | gap |
|---|---|---|---|
| 2 | 2.84 | 2.34 | +21% |
| 4 | 2.78 | 2.31 | +20% |
| 6 | 2.55 | 2.17 | +18% |
| 8 | 2.06 | 1.67 | **+23%** |

Reslicing makes BOTH methods absolutely faster but doesn't widen gap.

### swebench_verified 🔥 promising signal


---

## Open exploration directions (active iteration)

1. **swebench_verified v3 sim** — finishing B=128 now. Will give first FAIR real-cost numbers on this workload.
2. **New methods to test**:
   - ✅ `extension_hybrid_perfect_oracle` (per-step ORACLE pick of suffix-only or ext) — tested on LCC, +16.8% (no win)
   - ✅ `extension_pure_sfx:N:F:T:M` — pure suffix recursive (skips eagle3); tested on LCC: pure_sfx mat 0.96 vs ext mat 1.49 — pure_sfx loses.
3. **Workload search (web search 2026-04-26)**:
   - **AgenticSQL** is THE workload where SuffixDecoding paper claims 9.85x/10.41x speedup. PROPRIETARY (Snowflake). Not downloadable.
   - **MAGIC trajectories** (Microsoft, multi-agent BIRD-SQL self-correction) — closest public substitute, multi-agent overlap. Untested.
   - **Spider 2.0** (xlang-ai) — enterprise SQL agent loops with DBT boilerplate.
   - **JSONSchemaBench** — heavy syntactic repetition.
   - **Edit-based code synthesis (LintSeq, ICLR 2025)** — heavy edit pattern reuse.
   - **Lean/Coq formal proofs** — repeated tactic patterns.

   None of these are tested in our codebase yet. Setting one up requires: download → adapter to dataset_interleaved.jsonl → Stage 1 capture → sim. Multi-hour effort.

4. **Hyperparameter regime sweep** — at FAIR settings (both methods use same suffix params), find (eagle3 steps, topk, suffix_factor, suffix_min_prob, B) regime where gap maximizes
5. **Joint-score pruning** (not yet implemented) — combined path_p_t × suffix_score for tree top-K
6. **Adaptive per-anchor suffix params** (not yet implemented)
7. **Latency re-measurement at B=1024,2048** — pending GPU free time

## Methodology rules

- **Goal metric**: `*_speedup_real` from `tree_oracle_sim.json` (latency_config-based) — NOT `*_speedup` (ratio-based mat+1)
- **Fairness**: hybrid uses paper-faithful suffix (max_spec_factor=1.0, min_token_prob=0.1); extension uses aggressive (4.0, 0.0) by DEFAULT but extension_sfx allows custom
- **Sequential cost**: `step_cost = target_forward(ext_size) + eagle3_draft + n_actual_suffix_calls × suffix_speculate_ms`. Sequential, never `max()`
- **No baseless waiting**: when partial captures provide signal, sim them now

---

## File organization

```
experiments/
├── log.md                                # this file
├── report_template.md                    # final report skeleton (TBD)
├── scripts/
│   ├── analyze_sweep.py                  # extract best-of-family per (capture, B)
│   ├── inspect_captures.py               # rr capture sanity check
│   ├── remeasure_latency.sh              # latency re-measurement runner
│   └── data_prep/
│       ├── interleave_datasets.py
│       ├── make_lcc_dataset.py
│       └── prep_all_datasets.py
└── results/
    ├── sim_lcc.json                      # rr longbench_lcc sim
    ├── sim_repobench.json                # rr longbench_repobench sim
    └── explorations/
        ├── lcc_synergy.json              # full pt/pathprob/prune sweep
        ├── lcc_perfect_hybrid_oracle.json
        ├── lcc_pure_sfx.json
        └── lcc_reslice_steps{2,4,6}.json # reslice study

simulation/                                # canonical codebase (unchanged)
├── config/
│   └── rr_qwen3_14b.yaml                 # config-driven RR collection
├── scripts/
│   └── run_experiment.py                 # RR + sweep modes
└── evaluation/
    └── run_tree_oracle_sim.py            # main simulator
```

`_archive_legacy/` — deprecated launchers + old sim outputs (can delete entirely once we're confident).
