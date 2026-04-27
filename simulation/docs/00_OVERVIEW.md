# Oracle 시뮬레이션 파이프라인 — 전체 개요

`simulation/` 의 oracle 시뮬레이션 파이프라인은 **3-stage 구조** 이며 Stage 1
은 round-robin (RR) collection 모드만 지원한다 (commit `d1e8247` 이후
legacy Stage 1-6 sweep 모드 + `run_pipeline.sh` 체인은 모두 제거됨).

```
data/<workload>/dataset_*.jsonl
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 1 — RR EAGLE3 Oracle Vanilla Collection                  │
│   • simulation/scripts/run_experiment.py <config.yaml>         │
│     ├─ python -m simulation.oracle.install_hook  (1회 idempotent)│
│     ├─ shard 별 SGLang(EAGLE3) 서버 boot                       │
│     │    + SGLANG_ORACLE_VANILLA=1 → accept_length=0 강제      │
│     │    + SGLANG_CAPTURE_FULL_POOL=1 → eagle3_pool_full 누적  │
│     └─ workload 회전: 각 shard 가 자기 workload 들에 대해      │
│        agent 를 batch 단위 (--num-requests batch --resume)로   │
│        반복 호출                                                │
│   → agent_results_eagle3.json (per workload)                   │
└────────────────────────────────────────────────────────────────┘
    │
    ▼  (DRAFT_LM 별도 측정용 — 현재 latency 측정 외에는 옵션)
┌────────────────────────────────────────────────────────────────┐
│ Stage 2 — Draft LM Per-step Proposals                          │
│   • simulation.pipeline.collect_draft_model                     │
│     매 oracle step prefix → SGLang(draft LM) /generate          │
│     {req_id, call_idx, step_idx, token_ids, parents}로 기록     │
│   → draft_model_drafts.jsonl                                   │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 3 — Oracle Tree Simulation                               │
│   • simulation.evaluation.run_tree_oracle_sim                   │
│     • per question × per LLM call × per step × per budget       │
│     • method: single:* / hybrid_e3 / extension(_oracle/...)     │
│     • SuffixDecodingCache (CPU) 매 method×budget마다 fresh,     │
│       per-(req,call) start_request 으로 LOCAL tree reset        │
│     • greedy_tree_walk 으로 ground-truth 와 longest matched     │
│       prefix 비교 → accept length                               │
│     • latency_config.json 으로 step cost 계산 → speedup         │
│     • optional: --reslice-steps/--reslice-topk 로 capture 한    │
│       full pool 을 (s', k') sub-tree 로 잘라 시뮬레이션          │
│   → tree_oracle_sim.json (+ 콘솔 summary)                      │
└────────────────────────────────────────────────────────────────┘
```

## Artifact 흐름

```
data/<workload>/dataset_*.jsonl
    └─▶ Stage 1 (RR) → agent_results_eagle3.json  (per-step entries에 eagle3_pool_full 포함)
                          ├─▶ Stage 2 → draft_model_drafts.jsonl  (선택)
                          └─▶ Stage 3 → tree_oracle_sim.json
```

`agent_results_eagle3.json` 옆에는 항상 `_response.json` (oracle entries 제거된
사람용 사본) 도 함께 떨어진다. `simulation/pipeline/save_results.py:save_agent_results`
가 같은 호출에서 두 파일을 atomic write 한다.

---

## 단계별 문서 인덱스

| # | 파일 | 다루는 범위 |
|---|---|---|
| 01 | [`01_stage1_rr_collection.md`](01_stage1_rr_collection.md) | RR mode (`run_experiment.py`), shard 분할, full pool capture, resume 동작 |
| 02 | [`02_stage1_agents.md`](02_stage1_agents.md) | benchmark 별 agent (`bfcl_v4_agent`, `specbench_agent`, `swebench_agent`) 의 입력 포맷, 프롬프트 구성, HTTP 호출, tool-call 파싱, iteration loop |
| 03 | [`03_stage1_tools_and_io.md`](03_stage1_tools_and_io.md) | BFCL DDG monkey-patch, SWE-Bench tool 팩토리, `save_results.py` atomic write + checkpoint, `_agent_io.py` extraction, oracle 로그 JSON 스키마 |
| 04 | [`04_stage2_draft_model.md`](04_stage2_draft_model.md) | Stage 2 본체 (`collect_draft_model.py`), per-step prefix 재구성, SGLang `/generate` 호출, JSONL 스키마 |
| 05 | [`05_stage3_simulator_core.md`](05_stage3_simulator_core.md) | Stage 3 시뮬레이터 메인 루프, CLI flag, method dispatch, `greedy_tree_walk`, latency-config 스키마 + topk-aware lookup, output 컬럼 |
| 06 | [`06_tree_knapsack.md`](06_tree_knapsack.md) | `tree_knapsack.py` greedy walk 구현, "knapsack" 명명 misnomer |
| 07 | [`07_side_suffix_trajectory.md`](07_side_suffix_trajectory.md) | `run_side_suffix_trajectory.py` 사이드 도구 |
| 08 | [`08_sglang_patches.md`](08_sglang_patches.md) | `simulation.oracle` 의 SGLang 디스크/런타임 패치 |
| 09 | [`09_pool_reslicer.md`](09_pool_reslicer.md) | `pool_reslicer.py` — captured full pool → (s', k') sub-tree 알고리즘 |

---

## 환경 변수 통합표

| 변수 | 설정 위치 | 읽는 위치 | 역할 |
|---|---|---|---|
| `SGLANG_ORACLE_VANILLA=1` | `run_experiment.py:_run_rr_shard` | `eagle_worker.py.__init__` (Tier-1 디스크 패치 후), `oracle_patch.py` | EAGLE3 worker 에 oracle hook 부착 — `accept_length=0` 강제 + draft tree·verify logits·p_t 로깅 |
| `SGLANG_CAPTURE_FULL_POOL=1` | `run_experiment.py:_run_rr_shard` (`capture_full_pool: true` 일 때) | `oracle_patch.py:341` | EAGLE3 가 만든 full draft pool (`K + (S-1)·K²` 노드) 의 `(parent_list, draft_tokens, path_probs)` 를 oracle entry 의 `eagle3_pool_full` 필드로 stash. Stage 3 의 `pool_reslicer` 가 이걸 받아 (s', k') sub-tree 로 잘라낸다. 두 번째 `organize_draft_results` 호출은 cloned 입력 → generation 영향 없음 |
| `SGLANG_LATENCY_ONLY=1` | `measure_eagle3_cost.py` | `oracle_patch.py` | Vanilla 활성화 상태에서 force-accept·logit clone 비활성 → real speculative + timing 만 |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | `run_experiment.py:_run_rr_shard` | SGLang 서버 | 긴 context 허용 |
| `TORCHINDUCTOR_COMPILE_THREADS=1` | `run_experiment.py:_run_rr_shard`, `install_hook.py` | torch | torch.compile fork-bomb 방지 |
| `CUDA_VISIBLE_DEVICES` | `_run_rr_shard` 가 shard 별 설정 | SGLang | shard 별 GPU 격리 |

---

## 결과 정확성 위험 지점 — Hot List

### 1. Tool-calling / Agent (Stage 1)

- **`bfcl_v4_agent.py`** — efficiency-instruction system-prompt suffix 가 `process_request` 와 `replay_request` 양쪽에 동일하게 들어 있어야 함.
- **`--num-workers 1` 강제** (RR `_run_agent_once`) — oracle log 가 `req_id` 별 인터리빙되면 `_agent_io._extract_entries` 의 most-common-req_id filter 가 일부 entries 를 버림. 동시 workers > 1 사용 금지.
- **모든 agent `temperature=0.0`** — Stage 3 의 force-accept 패치가 greedy 분기만 잡으므로 sampling 을 켜면 `accept_length=0` 보장이 깨짐.

### 2. SGLang oracle patches

- **`oracle_patch.py.verify_tree_greedy_func`** 패치는 SGLang `eagle_info.py` 의 **greedy 분기만** 가로챔 — 모든 agent 가 `temperature=0.0` 로 호출되어야 함.
- **worker `__init__` sentinel** — `self.extend_lens = torch.empty(...)` 라인을 sentinel 로 패치 위치 결정. SGLang 업그레이드 시 sentinel 라인이 사라지면 silently warn 후 return → oracle 비활성. 부팅 로그 확인 필수.
- **logit `.cpu().clone()`** — vanilla 모드에서 verify logits stash 가 `target_forward_ms` 측정값을 오염시킴 → latency 측정 전용 `SGLANG_LATENCY_ONLY=1`.

### 3. Stage 2 ↔ Stage 3 결합

- **step-skip 동기화** — `collect_draft_model.py:64` 의 `n - pos <= 1` 과 `assemble_records.py:231` 의 `len(future) <= 1` 이 동일해야 `(req, call, step) → drafts` lookup 키가 정렬됨.
- **Tokenizer family 일치 가정** — preset 의 (target, draft_lm) 이 같은 토크나이저 패밀리를 공유해야 함.
- **`MAX_DRAFT_MODEL_N=16` ↔ `--max-draft-tokens` 16** — `run_tree_oracle_sim.py` 가 latency 회계와 verify_tokens override 양쪽에 16 을 사용.

### 4. Stage 3 시뮬레이터 내부

- **Suffix 는 라이브 생성** — `_live_suffix_draft` 가 매 step 호출. 사전 생성된 suffix 파일은 없음.
- **Cache state 격리** (eb21043 fix) — `simulate_decoding` 호출마다 `SuffixDecodingCache` fresh; per-(req, call) `start_request` 으로 LOCAL tree 리셋. 누설 시 mat 가 조용히 부풀려짐.
- **Hybrid suffix kwarg** (af92e9f fix) — `_hybrid_step` 에 `"suffix_cache": _SUFFIX_ENABLED` 누락 시 영구 eagle3 fallback.
- **Latency `_interp` extrapolation** — measured budgets 를 넘는 budget 에 linear extrapolation. extension method 가 큰 budget 만들어내므로 `target_forward_ms` 에 256/512 까지 측정 권장.

### 5. Pool reslicer

- **subset assumption silent truncation** (`pool_reslicer.py:151-155`) — k' < K 일 때 resliced alive 가 original alive 의 부분집합이 아닌 경우 발생 가능. 현재는 `continue` 로 silent drop. (자세한 건 `09_pool_reslicer.md`)

---

## 입출력 데이터 스키마 요약

### 입력 — `data/<workload>/dataset_*.jsonl`

| Workload | 한 줄 entry 핵심 필드 | 출처/포맷 |
|---|---|---|
| `bfcl_v4` | `bfcl_id`, `question`, `function`, `missed_function`, agentic 메타 | BFCL v4 (WebSearch + Memory) |
| `specbench` | `question_id`, `category`, `turns: [str, str, ...]` | SpecBench / MT-Bench |
| `swebench_verified` | `instance_id`, `repo`, `problem_statement`, `gold_patch`, `image_name` | SWE-Bench Verified |
| `longbench_lcc` / `longbench_repobench` | `_id`, `context`, `input`, `answers` | LongBench code subsets (specbench_agent 가 처리) |

### Stage 1 → `agent_results_eagle3.json`

```jsonc
{
  "metadata": {
    "model": "...", "num_requests": N,
    "total_tokens": M, "total_oracle_entries": K,
    "oracle_enabled": true
  },
  "questions": [
    {
      "bfcl_id" or "instance_id" or "question_id": "...",
      "category": "...",
      "agent_metrics": {
        "steps": [
          {
            "type": "llm",
            "messages": [...],
            "spec_decode": {
              "oracle_vanilla_entries": [
                {
                  "req_id": "...",
                  "tokens": [[...]],          // committed token (force-accept=0이라 step당 1개)
                  "eagle3": [[...]],          // flat draft chain (depth-first 첫 path)
                  "eagle3_tree": {            // BFS tree
                    "token_ids": [...],
                    "parents": [...]
                  },
                  "eagle3_tree_p_t": [...],
                  "eagle3_tree_path_draft_p_t": [...],
                  "eagle3_pool_full": {       // SGLANG_CAPTURE_FULL_POOL=1
                    "parent_list": [...],
                    "draft_tokens": [...],
                    "path_probs": [...],
                    "pool_size": int
                  }
                },
                ...
              ]
            }
          },
          { "type": "tool", "tool_calls": [...], "results": [...] },
          ...
        ]
      },
      "turns": [...]                  // SWE-Bench: per-iteration messages
    }
  ]
}
```

`pipeline/_agent_io.py:_extract_entries` 가 위 entries 를 읽어 다음으로 변환:
- `per_call_tokens`        — committed token IDs (per LLM call, flat)
- `per_call_eagle3s`       — flat draft chain (per step)
- `per_call_eagle3_trees`  — BFS trees (per step)
- `per_call_eagle3_tree_p_ts`
- `per_call_eagle3_tree_path_draft_p_ts`
- `per_call_eagle3_pool_fulls`  — full pool (reslicer 입력)
- `per_call_prompt_ids`

### Stage 2 → `draft_model_drafts.jsonl`

한 줄 = 한 oracle step.

```jsonc
{
  "request_id": "...", "call_idx": 0, "step_idx": 5,
  "token_ids": [t1, t2, ..., t16],   // ≤ MAX_DRAFT_MODEL_N
  "parents": [-1, 0, 1, ..., 14]     // flat chain
}
```

### Stage 3 → `tree_oracle_sim.json`

per-question × per-method × per-budget 결과 + summary. 자세한 컬럼은
`05_stage3_simulator_core.md`. 주요 metric:
- `mat` — 평균 accept length
- `accept_rate`
- `verify_tokens_mean` (실제 트리 크기)
- `step_real_ms` (latency-config 기반)
- `speedup_real` = `vanilla_step_ms × n_target_tokens / total_step_real_ms`

---

## Latency Model

`simulation/config/latency/<preset>.json` 가 Stage 3 의 `--latency-config` 로 들어감.
스키마 (qwen3_8b 예시):

```jsonc
{
  "vanilla_step_ms": 26.16,
  "target_forward_ms": { "4": 26.16, "8": 26.19, ..., "512": 131.07 },
  "eagle3_draft_ms":   { "4":  6.47, "8":  6.58, ..., "512":   6.34 },
  "eagle3_draft_ms_by_steps":   { "<steps>": { "<budget>": ms } },
  "target_forward_ms_by_topk":  { "<topk>":  { "<budget>": ms } },
  "eagle3_draft_ms_by_topk_steps": { ... },
  "draft_lm_tpot_ms": ...
}
```

생성 책임 분담:
- `measure_eagle3_cost.py` — target_forward / eagle3_draft (real-mode + `SGLANG_LATENCY_ONLY=1`)
- `measure_draft_model_cost.py` — draft LM TPOT
- `measure_suffix_cost.py` — suffix CPU cost (참고용)
- `compile_latency_config.py` — 위 셋을 합쳐 `latency_config.json` 생성

`simulation/scripts/experiments/remeasure_latency.sh` 가 위 4개를 한 번에 호출하는 wrapper.

Stage 3 의 `_interp` 는 measured key 사이는 linear interp, 측정 범위를 벗어나면 linear
**extrapolation**. extension 메소드가 큰 트리를 만들 수 있으니 256/512 까지 측정 권장.

---

## SGLang 패치 요약 (자세한 건 `08_sglang_patches.md`)

두 단계로 나뉘어 적용:

**Tier 1 — `python -m simulation.oracle.install_hook` (디스크 패치, idempotent):**
1. `srt/speculative/spec_info.py` — `SpeculativeAlgorithm` enum 에 `SUFFIX` 추가
2. `srt/server_args.py` — argparse choices 에 `"SUFFIX"` 추가
3. `srt/managers/scheduler.py` — `init_disaggregation` 에서 SUFFIX draft KV pool 스킵
4. `srt/speculative/eagle_worker.py`, `multi_layer_eagle_worker.py` —
   `__init__` 끝에 `patch_eagle_worker_full(self)` 호출 인젝션
   (`SGLANG_ORACLE_VANILLA=1` 일 때만 활성)

**Tier 2 — `oracle_patch.patch_eagle_worker_full(self)` (런타임):**
- `verify_tree_greedy_func` → `accept_length=0` + verify logits stash
- draft 생성 메소드 → tree (token_ids, parents) + path_draft_p_t 캡처
- `SGLANG_CAPTURE_FULL_POOL=1` 이면 `organize_draft_results` 두 번 호출
  (cloned 입력으로 full pool 캡처) → `eagle3_pool_full` stash
- per-request 누적 → `flush_oracle()` 가 step 종료 시 `oracle_vanilla_entries` 로 옮김

---

## 실제 실행 순서 (RR 기준)

1. **Latency 측정** (sweep 시작 전 1회)
   ```bash
   bash simulation/scripts/experiments/remeasure_latency.sh
   # → simulation/config/latency/<preset>.json
   ```

2. **Stage 1 — RR EAGLE3 capture**
   ```bash
   docker exec -u root -d sglang-bench bash -c \
     "cd /workspace && python3 simulation/scripts/run_experiment.py \
       simulation/config/<config>.yaml > /tmp/rr_stage1.log 2>&1"
   ```
   - 단일 GPU (1 shard): `simulation/config/rr_qwen3_14b.yaml`
   - 4-shard 병렬: `simulation/config/mango1.yaml`
   - 진행/재개: yaml 의 `round_robin.resume: true` (default) — 매 iter 마다 workload 별로 batch 만큼 진행, 중단/재시작 안전.

3. **Stage 2 — Draft LM (선택)**
   필요시 `python -m simulation.pipeline.collect_draft_model ...` 직접 호출.
   현재 RR 진행 시 `latency_config` 측정 외에는 draft_lm 사용 안 함.

4. **Stage 3 — Reslice sweep**
   ```bash
   python3 simulation/scripts/experiments/run_reslice_sweep.py \
     --workload swebench_verified \
     --reslices s2k16,s4k16,s6k16 \
     --budgets 8,16,32,64 \
     --in-docker
   ```
   `simulation/scripts/experiments/run_reslice_sweep.py` 가 `(s', k')` 별로
   `run_tree_oracle_sim` 을 호출. captured full pool → reslice → 시뮬레이션.

5. **분석**: `simulation/notebooks/compare_methods.ipynb` 등에서 결과 시각화.

---

## 검증 체크리스트

### A. RR collection 정상 동작
- [ ] **shard 별 SGLang `Server ready`** 가 `_rr_sglang_server_shard{N}.log` 에 찍히는지
- [ ] **`oracle_enabled: true`** in `agent_results_eagle3.json.metadata`
- [ ] **`oracle_vanilla_entries` 가 모든 step 에 존재** — `accept_length=0` 강제 안 된 step 은 entries 가 step 보다 적게 쌓임
- [ ] **모든 agent `temperature=0.0`** — HTTP request 캡처가 가장 확실
- [ ] **`eagle3_pool_full` 필드 존재** (`SGLANG_CAPTURE_FULL_POOL=1` 일 때)
- [ ] **resume 무결성** — 중단 후 재시작 시 done count 가 올라가는지

### B. Stage 2 ↔ Stage 3 결합 (draft_lm 사용 시)
- [ ] **JSONL 라인 수** = 모든 question 의 `len(future) > 1` step 합계
- [ ] **draft tokenizer family** = target tokenizer family
- [ ] **`--max-draft-tokens` ≤ 16**

### C. Stage 3 시뮬레이션
- [ ] **method 별 MAT 단조성** — `extension_oracle >= single:suffix`
- [ ] **budget=1 의 MAT** ≈ 0~1 (1 토큰 제안)
- [ ] **큰 budget 에서 MAT 포화** — budget=256/512 의 MAT 가 budget=128 과 비슷하면 OK
- [ ] **summary 출력에 method 누락 없는지** — hybrid suffix kwarg 누락 회귀 시 hybrid_e3 ≈ single:eagle3 로 무너짐

### D. SGLang 패치 회귀 (SGLang 업그레이드 후)
- [ ] **`install_hook.py` idempotency** — `if "SUFFIX" in text` 가 true 인지
- [ ] **`extend_lens` sentinel 존재** — `grep "self.extend_lens = torch.empty" $(python -c 'import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent / "srt/speculative/eagle_worker.py")')`

---

## 디렉토리 맵

```
simulation/
├── scripts/
│   ├── run_experiment.py              ← RR Stage 1 진입점
│   ├── measure_eagle3_cost.py         ← latency 측정 (사전)
│   ├── measure_draft_model_cost.py
│   ├── measure_suffix_cost.py
│   ├── compile_latency_config.py
│   └── experiments/
│       ├── remeasure_latency.sh       ← 위 4개 wrapper
│       ├── run_reslice_sweep.py       ← Stage 3 reslice sweep
│       ├── analyze_sweep.py
│       ├── inspect_captures.py
│       └── data_prep/
│           ├── prep_all_datasets.py
│           ├── interleave_datasets.py
│           └── make_lcc_dataset.py
│
├── oracle/
│   ├── install_hook.py                ← Tier-1 디스크 패치
│   └── oracle_patch.py                ← Tier-2 런타임 monkey-patch
│
├── agents/
│   ├── bfcl_agent.py                  ← BFCL v3 (deprecated workload)
│   ├── bfcl_v4_agent.py               ← BFCL v4
│   ├── specbench_agent.py             ← SpecBench / MT-Bench / LongBench
│   ├── swebench_agent.py              ← SWE-Bench (LangChain or mini-swe-agent)
│   └── tools/
│       ├── bfcl.py
│       └── swebench.py
│
├── pipeline/
│   ├── save_results.py
│   ├── _agent_io.py
│   ├── collect_draft_model.py
│   ├── assemble_records.py
│   └── pool_reslicer.py
│
├── evaluation/
│   ├── run_tree_oracle_sim.py
│   ├── tree_knapsack.py
│   └── run_side_suffix_trajectory.py
│
├── config/
│   ├── rr_qwen3_14b.yaml              ← 단일 GPU RR (mango3)
│   ├── mango1.yaml                    ← 4-shard RR (mango1)
│   └── latency/
│       ├── qwen3_8b.json
│       └── qwen3_14b.json
│
├── results/                           ← 출력 (gitignored)
└── docs/                              ← 본 문서들
```

`bfcl_v3` 는 active workload 에서 제외됐으나 (`feedback_drop_bfcl_v3` 메모리),
agent 코드 (`bfcl_agent.py`) 와 기존 capture 디렉토리는 참조용으로 남아있다.
