# Oracle 시뮬레이션 파이프라인 — 전체 개요

이 문서는 `advance-spec/simulation/` 의 oracle 시뮬레이션 파이프라인 전체를 코드 단위로 검증하기 위한 인덱스 + 종합 문서다. 각 stage의 깊이 있는 문서는 동일 디렉토리의 `01_*` ~ `08_*` 파일에 분리돼 있다. 이 문서는 그것들을 묶어서 (1) 데이터 흐름, (2) 환경변수, (3) 결과 정확성에 영향을 미치는 위험 지점, (4) 검증 체크리스트를 제공한다.

> **버전 주의** — 최상위 `README.md` 는 6-stage + 3-substage (3a/3b/3c, union trie, p_t collection 등) 구조를 설명하지만, commit `b4f279a` ("Simulation pipeline cleanup: 6 stages → 3 stages") 이후 실제 `simulation/scripts/run_pipeline.sh` 는 **3-stage 구조**로 정리됐다. union trie / EU oracle / Stage 3a~3c 분리는 더 이상 사용되지 않으며, suffix 는 Stage 3 시뮬레이터 안에서 라이브로 그려진다. 본 문서는 현행 구현 기준이다.

---

## 1. 파이프라인 한눈에 보기

```
data/<benchmark>/dataset.jsonl
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 1 — EAGLE3 Oracle Vanilla Trajectory Collection         │
│   • simulation/scripts/run_parallel_stage1.sh                 │
│     ├─ python -m simulation.oracle.install_hook  (디스크 패치)│
│     ├─ N개 GPU 각각 SGLang(EAGLE3) 서버 기동                  │
│     │    + SGLANG_ORACLE_VANILLA=1 → accept_length=0 강제     │
│     │    + draft tree·verify p_t 매 step JSON 누적            │
│     └─ python -m <agent_module>  (HTTP /v1/chat/completions)  │
│   → agent_results_eagle3.json                                 │
└───────────────────────────────────────────────────────────────┘
    │
    ▼  (DRAFT_LM 이 설정된 preset만 — 현재 Qwen / Llama 모두 해당)
┌───────────────────────────────────────────────────────────────┐
│ Stage 2 — Draft LM Per-step Proposals                         │
│   • simulation/scripts/run_parallel_draft_model.sh            │
│     N개 GPU 각각 SGLang(draft LM, prefix-cache) 서버 기동     │
│   • simulation.pipeline.collect_draft_model                    │
│     매 oracle step 의 prefix → /generate(max_new=16) 호출,    │
│     {req_id, call_idx, step_idx, token_ids, parents}로 기록    │
│   → draft_model_drafts.jsonl                                  │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 3 — Oracle Tree Simulation                              │
│   • simulation.evaluation.run_tree_oracle_sim                  │
│     • per question × per LLM call × per step × per budget      │
│     • method 풀: single:* / hybrid_e3 / hybrid_dm /            │
│       extension(_oracle/_oracle_path/_prune_pt/_nlevel*) /     │
│       extension_dmsfx / extension_hybrid                       │
│     • SuffixDecodingCache (CPU) 매 method×budget마다 fresh,    │
│       per-(req,call) start_request 으로 LOCAL tree reset       │
│     • greedy_tree_walk 으로 ground-truth 와 longest matched    │
│       prefix 비교 → accept length                              │
│     • latency_config.json 으로 step cost 계산 → speedup        │
│   → tree_oracle_sim.json (+ 콘솔 summary)                     │
└───────────────────────────────────────────────────────────────┘
```

### Artifact 흐름

```
input_slice.jsonl (REQ_START/REQ_END 슬라이스)
    └─▶ Stage 1 → agent_results_eagle3.json
                    └─▶ Stage 2 → draft_model_drafts.jsonl
                                     └─▶ Stage 3 → tree_oracle_sim.json
                                                   tree_oracle_sim_response.json
```

`agent_results_eagle3.json` 옆에는 항상 `_response.json` (oracle entries 제거된 사람용 사본) 도 함께 떨어진다. `simulation/pipeline/save_results.py:save_agent_results` 가 같은 호출에서 두 파일을 atomic write 한다.

---

## 2. 단계별 문서 인덱스

| # | 파일 | 다루는 범위 |
|---|---|---|
| 01 | [`01_stage1_overview.md`](01_stage1_overview.md) | Stage 1 진입 셸 스크립트, GPU 샤딩, 포트, 환경변수, SGLang 런치 인자, 샤드 머지, 데이터 prepare 스크립트, preset/benchmark 별 구체 CLI 예시 |
| 02 | [`02_stage1_agents.md`](02_stage1_agents.md) | benchmark 별 agent (`bfcl_agent`, `bfcl_v4_agent`, `specbench_agent`, `swebench_agent`) 의 입력 포맷, 프롬프트 구성, HTTP 호출, tool-call 파싱, iteration loop, swallowed 실패 모드 |
| 03 | [`03_stage1_tools_and_io.md`](03_stage1_tools_and_io.md) | BFCL DDG monkey-patch, SWE-Bench tool 팩토리 (timeout / blocked cmd / output truncation), `save_results.py` atomic write + checkpoint, `_agent_io.py` extraction & deinterleave, oracle 로그 JSON 스키마 |
| 04 | [`04_stage2_draft_model.md`](04_stage2_draft_model.md) | Stage 2 셸 스크립트, per-step prefix 재구성 (BFCL / SWE / SpecBench 분기), SGLang `/generate` 호출, JSONL 스키마, idempotency 미지원, step-skip 조건 동기화 |
| 05 | [`05_stage3_simulator_core.md`](05_stage3_simulator_core.md) | Stage 3 시뮬레이터 메인 루프, CLI flag 전부, method dispatch, `greedy_tree_walk` 의미, 캐시 버그 수정 내역, latency-config 스키마 + topk-aware lookup, output 컬럼 |
| 06 | [`06_tree_knapsack.md`](06_tree_knapsack.md) | `tree_knapsack.py` (28줄) 가 실제로 무엇인지 — greedy walk 구현, 정확성 논증, 호출 사이트, "knapsack" 명명 잘못 |
| 07 | [`07_side_suffix_trajectory.md`](07_side_suffix_trajectory.md) | `run_side_suffix_trajectory.py` (사이드 실험 도구), per-step JSONL + meta.json 스키마, `_extension_step._last_*` side-channel 의존성 |
| 08 | [`08_sglang_patches.md`](08_sglang_patches.md) | `simulation.oracle` 의 두 단계 패치 — (1) `install_hook.py` 가 디스크에 SUFFIX 알고리즘 enum/scheduler shim/worker `__init__` 인젝션, (2) `oracle_patch.py` 가 런타임에 EAGLE3 worker 의 draft/verify 메소드를 교체. env 변수, dead code, 위험 패치 감사 |
| 09 | [`09_pool_reslicer.md`](09_pool_reslicer.md) | `pool_reslicer.py` — `SGLANG_CAPTURE_FULL_POOL=1` 로 캡처한 full EAGLE3 score pool 을 시뮬레이션 시점에 (s', k') sub-tree 로 잘라내는 알고리즘. 한 번 capture 로 여러 (S, K) 조합을 sweep 할 때 사용 |

---

## 3. 환경 변수 통합표

파이프라인 또는 SGLang 가 읽는 모든 환경 변수. 자세한 의미는 각 stage 문서 참조.

| 변수 | 설정 위치 | 읽는 위치 | 역할 |
|---|---|---|---|
| `SGLANG_ORACLE_VANILLA=1` | `run_parallel_stage1.sh:49` | `eagle_worker.py.__init__` (디스크 패치 후), `oracle_patch.py` | EAGLE3 worker 에 oracle hook 부착 — `accept_length=0` 강제 + draft tree·verify logits·p_t 로깅 |
| `SGLANG_LATENCY_ONLY=1` | `measure_eagle3_cost.py` | `oracle_patch.py` | Vanilla 활성화 상태에서 force-accept·logit clone 비활성 → real speculative + timing 만 |
| `SGLANG_ORACLE_REPLAY=<json>` | (legacy 3c, 현행 미사용) | `oracle_patch.py` | trajectory.json 을 NEXTN 서버가 replay |
| `SGLANG_CAPTURE_FULL_POOL=1` | sweep capture 셸 (예: `_sweep_sk.sh` 가 호출하는 Stage 1) | `oracle_patch.py:341` (`_install_draft_p_t_tracer` 안의 `capture_full_pool`) | EAGLE3 가 만든 full draft pool (`K + (S-1)·K²` 노드) 의 `(parent_list, draft_tokens, path_probs)` 를 oracle entry 의 `eagle3_pool_full` 필드로 stash. Stage 3 의 `pool_reslicer` 가 이걸 받아 (s', k') sub-tree 로 잘라낸다. 두 번째 `organize_draft_results` 호출은 cloned 입력 → generation 영향 없음 |
| `SGLANG_DRAFT_BUDGET=<N>` | (legacy) | worker | `speculative_num_draft_tokens` 영구 override — sweep 중 서버 재기동 필수 |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | `run_parallel_stage1.sh:50`, Docker | SGLang 서버 | 긴 context 허용 (RTX 4090 + 큰 prompt) |
| `TORCHINDUCTOR_COMPILE_THREADS=1` | `run_parallel_stage1.sh:51`, `install_hook.py:25` | torch | torch.compile fork-bomb 방지 |
| `TOOL_CALL_PARSER` | `run_pipeline.sh:67/75` | SGLang `--tool-call-parser` | LangChain agent (SWE-Bench) 가 OpenAI tool_calls 받기 위해 필요. Qwen → `qwen25`, Llama → `llama3` |
| `STAGE1_TOPK / STAGE1_STEPS / STAGE1_NUM_DRAFT_TOKENS` | env (sweep) | `run_parallel_stage1.sh` | EAGLE3 트리 모양 sweep |
| `STAGE2_MAX_TOKENS / STAGE3B_MAX_TOKENS` | env | `run_pipeline.sh:222` | Stage 2 draft LM 최대 토큰 (기본 16) — `MAX_DRAFT_MODEL_N=16` 와 짝 유지 필수 |
| `REQ_START / REQ_END` | env (분산) | `run_pipeline.sh` | 입력 슬라이스, 출력 dir suffix `_req<S>-<E>` |
| `NUM_GPUS / GPU_IDS` | env | `run_parallel_stage1.sh`, `run_parallel_draft_model.sh` | 사용 GPU 인덱스 |
| `BFCL_V4_INPUT` | env | `run_pipeline.sh:90` | bfcl_v4 입력 파일 override (web_search-only 등 사전 필터링) |
| `BFCL_MAX_ITER / SWE_MAX_ITER` | env | `run_pipeline.sh` | agent loop iteration cap |
| `MAX_TOKENS_OVERRIDE` | env (specbench only) | `run_pipeline.sh:195` | specbench `--max-tokens` 오버라이드 |
| `STAGE1_BASE_PORT / STAGE3B_BASE_PORT` | env | 셸 스크립트 | 서버 포트 베이스 |
| `OUTPUT_DIR_SUFFIX` | env (sweep) | `run_pipeline.sh:127` | 출력 dir 접미사 — 동일 benchmark 의 sweep 결과 분리 |
| `HF_TOKEN` | `/workspace/.env` | SGLang 다운로드 | gated 모델 (Llama) |

---

## 4. 결과 정확성에 영향을 줄 수 있는 위험 지점 — Hot List

Stage 별 문서에서 추출한 "결과를 조용히 왜곡할 수 있는" 항목들. 검증 시 이 목록부터 확인하라.

> **이력**: 초기 감사에서 5건의 이슈가 수정됐다. 자세한 내역은 §11 "수정 이력" 참조.

### 4.1 Tool-calling / Agent behaviour (Stage 1)

- **`bfcl_v4_agent.py:200-208 / :341-349`** — efficiency-instruction system-prompt suffix 가 `process_request` 와 `replay_request` 양쪽에 동일하게 들어 있어야 함. 한쪽만 수정하면 replay 가 round-1 과 분기.
- **`--num-workers 1` 강제** (`run_parallel_stage1.sh`) — oracle log 가 `req_id` 별 인터리빙되면 `_agent_io._extract_entries:42-46` 의 most-common-req_id filter 가 일부 entries 를 버림. 동시 workers > 1 사용 금지.
- **모든 agent 가 `temperature=0.0`** (의도된 설정) — Stage 3 의 force-accept 패치가 greedy 분기만 잡으므로 sampling 을 켜면 `accept_length=0` 보장이 깨짐. `bfcl_v4_agent.py:228-233` 가 의도적으로 `temperature` CLI flag 를 무시하고 0.0 을 강제하는 것은 이 의존성을 보호하기 위함.

### 4.2 SGLang oracle patches

- **`oracle_patch.py.verify_tree_greedy_func`** 패치는 SGLang `eagle_info.py:312-325` 의 **greedy 분기만** 가로챔. `eagle_info.py:327-` 의 tree-spec sampling kernel 은 untouched. 따라서 모든 agent 가 `temperature=0.0` 으로 호출되어야 함 (위 §4.1).
- **worker `__init__` sentinel** — `self.extend_lens = torch.empty(...)` 라는 라인을 sentinel 로 패치 위치를 결정. SGLang 업그레이드 시 그 라인이 사라지거나 이름이 바뀌면 패치는 silently warn 후 return → oracle 기능 자체가 비활성화 됨에도 파이프라인은 계속 돌아감. SGLang 버전 변경 시 부팅 로그 확인 필수.
- **logit `.cpu().clone()`** — vanilla 모드에서는 verify logits 를 stash 하느라 `target_forward_ms` 측정값이 오염됨. 그래서 latency 측정 전용 `SGLANG_LATENCY_ONLY=1` 모드가 따로 존재.

### 4.3 Stage 2 ↔ Stage 3 결합

- **step-skip 조건 동기화** — `collect_draft_model.py:64` 의 `n - pos <= 1` 과 `assemble_records.py:231` 의 `len(future) <= 1` 이 동일해야 `(req, call, step) → drafts` lookup 키가 정렬됨. 한쪽 수정 시 silent miss.
- **Tokenizer family 일치 가정** — `run_pipeline.sh` 의 모든 (target, draft_lm) preset 은 같은 토크나이저 패밀리를 공유 (Qwen3 vs Qwen3, Llama3 vs Llama3). cross-family 페어링 시 prompt 가 silent corruption.
- **`--target-model` 누락 시 silent downgrade** — `collect_draft_model.py` 가 system prompt / chat template 적용 없이 raw context 로 호출 → 외형은 그럴듯하지만 분포가 어긋남.
- **`MAX_DRAFT_MODEL_N=16` ↔ `--max-draft-tokens` 16** — `run_tree_oracle_sim.py:1011, 1089, 1205, 1243` 에서 하드코딩된 16 이 latency 회계와 verify_tokens override 양쪽에 사용됨. Stage 2 의 `--max-draft-tokens` 를 16 외 값으로 바꾸면 Stage 3 가 잘못된 가정으로 latency 계산.

### 4.4 Stage 3 시뮬레이터 내부

- **`tree_knapsack.py` 명명 misnomer** — 실제로는 28줄 greedy walk. budget 은 `_proposer_tree_walk:868` / `_extension_step:615` 에서 단순 노드 카운트 cap.
- **Suffix 는 라이브 생성** — `assemble_records_from_artifacts` 로 만든 record 에는 suffix 가 없고, `_live_suffix_draft` (`run_tree_oracle_sim.py:512`) 가 매 step 마다 호출. 즉 어떠한 사전 생성된 suffix 파일도 없으므로, suffix 결과가 이상하면 simulator 내부 cache 상태부터 의심.
- **Cache state 격리 (eb21043 fix)** — `simulate_decoding` 호출마다 `SuffixDecodingCache` fresh 생성 (`:148-156`). per-(req, call) 시작에 `start_request` 호출로 LOCAL tree 리셋. **method × budget 조합마다 격리되지 않으면** 이전 budget 의 LOCAL tree 가 다음 budget 에 누설.
- **Hybrid suffix kwarg (af92e9f fix)** — `_hybrid_step` 에 `"suffix_cache": _SUFFIX_ENABLED` 를 명시적으로 plumbing 하지 않으면 영구 eagle3 fallback 으로 silent degrade.
- **Virtual-root extension** (`_extension_step:722-753`) — per-node anchor 외에 root graft 가 있어야 `extension_mat >= single:suffix_mat` 보장.
- **Latency `_interp` extrapolation** — measured budgets 를 넘는 budget 에 linear extrapolation. extension method 가 측정 범위 초과 budget 을 자주 만들어내므로, latency_config 의 `target_forward_ms` 에 큰 budget (≥256) 까지 측정이 있어야 신뢰성 있음.

---

## 5. 입출력 데이터 스키마 요약

### 5.1 입력 — `data/<benchmark>/dataset.jsonl`

| Benchmark | 한 줄 entry 핵심 필드 | 출처/포맷 |
|---|---|---|
| `bfcl_v3` | `bfcl_id`, `question` (multi-turn), `function`, `missed_function` | BFCL v3 multi-turn |
| `bfcl_v4` | `bfcl_id`, `question`, `function`, `missed_function`, agentic 메타 | BFCL v4 (WebSearch + Memory) |
| `specbench` | `question_id`, `category`, `turns: [str, str, ...]` | SpecBench / MT-Bench |
| `swebench` | `instance_id`, `repo`, `problem_statement`, `gold_patch`, `image_name` | SWE-Bench (LangChain agent) |

### 5.2 Stage 1 → `agent_results_eagle3.json`

```jsonc
{
  "metadata": {
    "model": "...", "num_requests": N,
    "total_tokens": M, "total_oracle_entries": K,
    "oracle_enabled": true
  },
  "questions": [
    // BFCL/SWE-Bench 형식
    {
      "bfcl_id" or "instance_id" or "question_id": "...",
      "category": "...",
      "agent_metrics": {
        "steps": [
          {
            "type": "llm",
            "messages": [...],          // OpenAI 또는 LangChain 형식
            "spec_decode": {
              "oracle_vanilla_entries": [
                {
                  "req_id": "...",
                  "tokens": [[...]],          // committed token (force-accept=0이라 step 당 1개)
                  "eagle3": [[...]],          // flat draft chain (depth-first 첫 path)
                  "eagle3_tree": {            // BFS tree
                    "token_ids": [...],
                    "parents": [...]
                  },
                  "eagle3_tree_p_t": [...],            // verify logits 기반 확률
                  "eagle3_tree_path_draft_p_t": [...], // draft head local prob (옵션)
                  "mtp_tree": null            // 현행 미사용
                },
                ...  // 한 LLM call 안의 여러 decode step
              ]
            }
          },
          { "type": "tool", "tool_calls": [...], "results": [...] },
          ...
        ]
      },
      "turns": [...]                  // SWE-Bench: per-iteration messages
    },
    // SpecBench 형식
    {
      "question_id": "...", "category": "...",
      "turns": [
        {
          "response": "...",
          "spec_decode": { "oracle_vanilla_entries": [...] }
        },
        ...
      ]
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
- `per_call_prompt_ids`    — chat template 적용 후 토크나이저 인코딩

### 5.3 Stage 2 → `draft_model_drafts.jsonl`

한 줄 = 한 oracle step.

```jsonc
{
  "request_id": "...", "call_idx": 0, "step_idx": 5,
  "token_ids": [t1, t2, ..., t16],   // ≤ MAX_DRAFT_MODEL_N
  "parents": [-1, 0, 1, ..., 14]     // flat chain (parent = prev)
}
```

### 5.4 Stage 3 → `tree_oracle_sim.json`

per-question × per-method × per-budget 결과 + summary. 자세한 컬럼은 `05_stage3_simulator_core.md` 참조. 주요 metric:
- `mat` — 평균 accept length (ground-truth 토큰 단위)
- `accept_rate`
- `verify_tokens_mean` (실제 트리 크기, extension 의 경우 budget 초과 가능)
- `step_real_ms` (latency-config 기반)
- `speedup_real` = `vanilla_step_ms × n_target_tokens / total_step_real_ms`

---

## 6. Latency Model

`simulation/config/latency/<preset>.json` (committed) 또는 `simulation/results/<preset>/latency_config.json` (local) 이 Stage 3 의 `--latency-config` 로 들어감. 스키마 (qwen3_8b 예시):

```jsonc
{
  "vanilla_step_ms": 26.16,
  "target_forward_ms": { "4": 26.16, "8": 26.19, ..., "512": 131.07 },
  "eagle3_draft_ms":   { "4":  6.47, "8":  6.58, ..., "512":   6.34 },
  "eagle3_draft_ms_by_steps":   { "<steps>": { "<budget>": ms } },
  "target_forward_ms_by_topk":  { "<topk>":  { "<budget>": ms } },
  "eagle3_draft_ms_by_topk_steps": { ... },
  "draft_lm_tpot_ms": ...      // 별도 측정
}
```

생성 책임 분담:
- `measure_eagle3_cost.py` — target_forward / eagle3_draft (real-mode + `SGLANG_LATENCY_ONLY=1`)
- `measure_draft_model_cost.py` — draft LM TPOT (vanilla 서버, 오라클 패치 없음)
- `measure_suffix_cost.py` — suffix CPU cost (참고용, 시뮬레이터에서는 0 가정)
- `compile_latency_config.py` — 위 셋을 합쳐 `latency_config.json` 생성

Stage 3 의 `_interp` 는 measured key 사이는 linear interp, 측정 범위를 벗어나면 linear **extrapolation**. extension 메소드가 큰 트리를 만들 수 있으니 256/512 까지 측정 권장.

---

## 7. SGLang 패치 요약 (자세한 건 `08_sglang_patches.md`)

두 단계로 나뉘어 적용된다.

**Tier 1 — `python -m simulation.oracle.install_hook` (디스크 패치, idempotent):**
1. `srt/speculative/spec_info.py` — `SpeculativeAlgorithm` enum 에 `SUFFIX` 추가, `is_suffix()` 메소드, `create_worker()` 에 SUFFIX 분기
2. `srt/server_args.py` — argparse choices 에 `"SUFFIX"` 추가, `__post_init__` 검증에서 NGRAM 과 동일 처리
3. `srt/managers/scheduler.py` — `init_disaggregation` 에서 SUFFIX 도 draft KV pool 스킵
4. `srt/speculative/eagle_worker.py`, `multi_layer_eagle_worker.py` — `__init__` 끝에 `patch_eagle_worker_full(self)` 호출 인젝션 (`SGLANG_ORACLE_VANILLA=1` 일 때만 실행)

**Tier 2 — `oracle_patch.patch_eagle_worker_full(self)` (런타임 monkey-patch):**
- `verify_tree_greedy_func` → `accept_length=0` 강제 (greedy 분기만!) + verify logits stash
- draft 생성 메소드 → tree (token_ids, parents) + path_draft_p_t 캡처
- per-request 누적: `_oracle_state[req_id]` 에 entries append
- `flush_oracle()` 가 step 종료 시 `oracle_vanilla_entries` 로 옮김 → agent 가 결과 dict 에 포함

**위험 패치 (Audit):** `08_sglang_patches.md` 참조 — temperature=0.0 가정, `extend_lens` sentinel 의존, logit clone 의 timing 오염.

---

## 8. 실제 실행 순서 (프로덕션 sweep 기준)

1. **Latency 측정** (sweep 시작 전 1회)
   ```bash
   python3 simulation/scripts/measure_eagle3_cost.py     ...
   python3 simulation/scripts/measure_draft_model_cost.py ...
   python3 simulation/scripts/measure_suffix_cost.py      ...
   python3 simulation/scripts/compile_latency_config.py   ...
   # → simulation/config/latency/<preset>.json
   ```

2. **Stage 1 — EAGLE3 trajectory 수집** (per benchmark, per preset)
   ```bash
   bash simulation/scripts/run_pipeline.sh <benchmark> <preset> [num_requests]
   # 또는 분산:
   REQ_START=0 REQ_END=50 bash simulation/scripts/run_pipeline.sh ...   # 머신 A
   REQ_START=50 REQ_END=100 bash simulation/scripts/run_pipeline.sh ... # 머신 B
   bash simulation/scripts/merge_shards.sh simulation/results/<preset>/<benchmark>
   ```
   내부적으로 `run_parallel_stage1.sh` 가 GPU 별 SGLang 서버를 띄우고 agent 를 병렬 실행, 마지막에 샤드 머지.

3. **Stage 2 — Draft LM 수집** (`DRAFT_LM` 가 설정된 preset)
   `run_pipeline.sh` 가 자동으로 `run_parallel_draft_model.sh` 를 호출. SGLang(draft LM, prefix-cache) N대 병렬 + per-step `/generate(max_new=16)`.

4. **Stage 3 — Oracle 시뮬레이션**
   `run_pipeline.sh` 가 `python -m simulation.evaluation.run_tree_oracle_sim --budgets 1,2,4,...,512 --print-summary` 실행. budget × method 카르테시안에서 MAT / speedup 계산.

5. **분석**: `simulation/notebooks/*.ipynb` (`analyze_oracle_sim.ipynb` 등) 에서 결과 시각화.

---

## 9. 검증 체크리스트

이 파이프라인이 "잘 구성되었는지" 확인하기 위한 항목들. 각 항목은 위 위험 지점들에서 도출.

### A. Tool-calling 정확성
- [ ] **각 agent 의 tool-call 성공률** — `agent_metrics.steps` 에서 `type:"tool"` 의 `error` 필드 비율. SWE-Bench 의 경우 tool 실행 실패는 trajectory 다양성에 영향.
- [ ] **`tool-call-parser` 일치** — Qwen 모델은 `qwen25`, Llama 는 `llama3`. 잘못 매칭 시 SWE-Bench agent 의 응답이 빈 tool_calls 로 들어와 매 iteration 마다 nudge 메시지가 누적된다 (현재는 break 안 함). nudge 메시지 비율로 parser 매칭 상태 추적 가능.
- [ ] **SpecBench API 실패 누적 확인** — turn 별 `spec_decode.oracle_vanilla_entries` 가 비어 있는 비율 확인. 5% 넘어가면 server 안정성 점검 (실패해도 컨텍스트는 시프트 안 함 — 빈 assistant 메시지가 채워짐).
- [ ] **샤드 머지 무결성** — `agent_results_eagle3.json.metadata.num_requests` 가 input_slice line 수와 일치하는지. metadata 의 `total_tool_calls` / `url` / `benchmark` 등도 보존되는지 확인 (수정 후 첫 샤드 metadata 가 베이스로 채택됨).

### B. Oracle hook 활성 검증
- [ ] **Stage 1 server.log 에 `Installed oracle vanilla patch into ...` / `Patched ...`** 라인이 있는지 확인. 없으면 sentinel 변경으로 패치 실패한 상태.
- [ ] **`oracle_vanilla_entries` 가 모든 step 에 존재** — `accept_length=0` 강제가 안 된 step 은 entries 가 step 보다 적게 쌓임.
- [ ] **`oracle_enabled: true`** in `agent_results_eagle3.json.metadata`.
- [ ] **temperature** — 모든 agent 가 0.0 으로 호출하는지 (HTTP request 캡처하면 가장 확실).

### C. Stage 2 ↔ Stage 3 결합 무결성
- [ ] **JSONL 라인 수** = 모든 question 의 `len(future) > 1` step 합계. 한쪽에 더 많거나 적으면 step-skip 조건 desync.
- [ ] **draft tokenizer family** = target tokenizer family.
- [ ] **`--target-model` 명시 확인** in shell command.
- [ ] **`--max-draft-tokens` ≤ 16** (Stage 3 의 `MAX_DRAFT_MODEL_N`).
- [ ] **`draft_model_drafts_shard*.jsonl` 파일 부재** (머지 후 정리됐는지).

### D. Stage 3 시뮬레이션 검증
- [ ] **method 별 MAT 단조성** — `extension >= single:suffix`, `c1 >= max(single:*)`.
- [ ] **budget=1 의 MAT** ≈ 0 또는 1 이내 (1 토큰 제안만 가능).
- [ ] **큰 budget 에서 MAT 포화** — budget=256/512 의 MAT 가 budget=128 과 거의 같으면 OK. 계속 증가하면 latency_config extrapolation 영역에 들어간 것.
- [ ] **`speedup_real` 의 분모** = `total_step_real_ms` 가 0 보다 큰지 (extension extrapolation 시 음수/0 가능).
- [ ] **summary 콘솔 출력** 에 method 누락 없는지 — 일부 method 가 hybrid suffix kwarg 누락으로 fallback 만 쓰면 (af92e9f 회귀) MAT 가 hybrid_e3 ≈ single:eagle3 로 무너짐.

### E. SGLang 패치 회귀 확인
- [ ] **SGLang 업그레이드 후 첫 실행** — `install_hook.py` 의 idempotency 체크 (`if "SUFFIX" in text`) 가 true 인지 확인. 새 SGLang 의 `eagle_info.py:312-` 에 greedy 분기가 그대로 있는지 grep.
- [ ] **`extend_lens` sentinel 존재 확인** — `grep "self.extend_lens = torch.empty" $(python -c 'import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent / "srt/speculative/eagle_worker.py")')`.

---

## 10. 디렉토리 맵 (현재 살아있는 코드만)

```
simulation/
├── scripts/
│   ├── run_pipeline.sh                ← 진입점 (3-stage 호출)
│   ├── run_parallel_stage1.sh         ← Stage 1 GPU 분산
│   ├── run_parallel_draft_model.sh    ← Stage 2 GPU 분산
│   ├── prepare_bfcl_data.py           ← 데이터 prepare
│   ├── prepare_specbench_data.py
│   ├── prepare_swebench_data.py
│   ├── measure_eagle3_cost.py         ← latency 측정 (사전)
│   ├── measure_draft_model_cost.py
│   ├── measure_suffix_cost.py
│   ├── compile_latency_config.py
│   └── merge_shards.sh                ← REQ 분산 결과 합치기
│
├── oracle/
│   ├── install_hook.py                ← Tier-1 디스크 패치 + 서버 런처
│   └── oracle_patch.py                ← Tier-2 런타임 monkey-patch
│
├── agents/
│   ├── bfcl_agent.py                  ← BFCL v3
│   ├── bfcl_v4_agent.py               ← BFCL v4
│   ├── specbench_agent.py             ← SpecBench / MT-Bench
│   ├── swebench_agent.py              ← LangChain ReAct
│   └── tools/
│       ├── bfcl.py                    ← DDG WebSearch (monkey-patched)
│       └── swebench.py                ← Docker repo + bash/edit/str_replace
│
├── pipeline/
│   ├── save_results.py                ← atomic write + per-request checkpoint helpers
│   ├── _agent_io.py                   ← oracle log → per-call tensors (eagle3_pool_full 포함)
│   ├── collect_draft_model.py         ← Stage 2 본체
│   ├── assemble_records.py            ← Stage 3 record 조립 (구 collect_union_trie)
│   ├── pool_reslicer.py               ← captured full pool → (s', k') sub-tree
│   └── calibrate_latency.py           ← (legacy)
│
├── evaluation/
│   ├── run_tree_oracle_sim.py         ← Stage 3 본체
│   ├── tree_knapsack.py               ← greedy walk (knapsack 아님)
│   └── run_side_suffix_trajectory.py  ← 사이드 도구
│
├── config/
│   ├── example.yaml                   ← sweep config 템플릿
│   └── latency/
│       ├── qwen3_8b.json
│       └── qwen3_14b.json
│
├── results/                           ← 출력 (gitignored)
└── docs/                              ← 본 문서들
    ├── 00_OVERVIEW.md                 ← 이 파일
    ├── 01_stage1_overview.md
    ├── 02_stage1_agents.md
    ├── 03_stage1_tools_and_io.md
    ├── 04_stage2_draft_model.md
    ├── 05_stage3_simulator_core.md
    ├── 06_tree_knapsack.md
    ├── 07_side_suffix_trajectory.md
    └── 08_sglang_patches.md
```

> 본 디렉토리에 보이는 `_*.py`, `tmp_*.py`, `_*_logs/` 류는 ad-hoc 분석 / 디버깅 스크립트로 파이프라인 진입점이 아니다. README 의 6-stage 설명에 등장하는 `extract_trajectory.py`, `collect_suffix_drafts.py`, `verify_server.py`, `replay_oracle.py`, `run_oracle_sim.py`, `oracle_verify_patch.py`, `measure_latency.py`, `measure_verify_latency.py` 등은 commit `b4f279a` 이후 제거되었으므로 코드 검색 시 참고 (이 문서들에서도 다루지 않음).

---

## 11. 수정 이력 (이 문서가 작성된 시점 이후)

본 문서가 처음 만들어질 때 발견된 위험 항목 중 다음이 후속 commit 으로 해결됐다. 코드를 추가로 검토할 때 이 변경들이 회귀하지 않았는지 확인하라.

| # | 항목 | 위치 | 처리 |
|---|---|---|---|
| 1 | SpecBench round-1 API 실패 시 컨텍스트 시프트 | `specbench_agent.py:run_single_request` | 실패 시 `messages.append({"role":"assistant", "content":""})` 추가하여 user/assistant 정렬 유지 |
| 2 | SWE-Bench tool-call-parser 미스매치 시 1턴만에 break | `swebench_agent.py` agent loop | `if not ai_msg.tool_calls: break` 제거. 빈 tool_calls 시 HumanMessage nudge 를 주입하고 다음 iteration 진행. `submit` 호출과 `max_iterations` 만이 종료 조건 |
| 3 | 샤드 머지가 마지막 metadata 만 채택 | `run_parallel_stage1.sh` 의 머지 python 블록 | 첫 샤드 metadata 를 베이스로, `total_tokens`/`total_oracle_entries`/`total_tool_calls` 만 합산 |
| 4 | `oracle_verify_patch.py` 전체 dead | 파일 자체 | 삭제 |
| 5 | UnionTrie / verify-tries 잔재 (`oracle_patch.py:_inject_union_trie`, `_log_verify_trie_p_t`, `ORACLE_VERIFY_TRIES_PATH`, `merge_shards.sh`/`run_experiment.py`/`rerun_*.sh`/`measure_latency*.py`/`collect_union_trie.py` 의 union 빌드 로직) | 다수 | 전면 제거. 파일 `pipeline/collect_union_trie.py` → `pipeline/assemble_records.py` rename, 함수 `collect_union_tries` → `collect_step_records` rename. 환경변수 `UNION_TRIE`, `EU_ORACLE`, `SGLANG_ORACLE_VERIFY_TRIES` 도 코드에서 제거 |
| 6 | mini-swe-agent 모드의 종료 신호 충돌 (수정 #2 의 `if not ai_msg.tool_calls: nudge` 가 mini-swe-agent 의 "no tool calls = submit" 종료 신호를 차단) | `swebench_agent.py:411-434` | `has_submit_tool = "submit" in tool_map` 분기 추가. mini-swe-agent (submit tool 없음) 에서는 빈 tool_calls 를 정상 종료로 처리. 추가로 official sentinel `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` 를 bash tool 결과로 인식해서 break (`swebench_agent.py:466-473`) |

---

## 12. 후속 추가 기능 (이 문서가 작성된 시점 이후)

§11 의 5 건 fix 이외에도 사용자가 다음 기능을 추가했다. 검증 시 함께 확인하라.

| # | 기능 | 위치 | 의도 |
|---|---|---|---|
| 1 | **Resume / per-request checkpoint** | `save_results.py` 의 `_atomic_write_json`, `checkpoint_path`, `load_checkpoint`, `done_ids`, `append_to_checkpoint`, `finalize_checkpoint`. `bfcl_v4_agent.py`, `specbench_agent.py`, `swebench_agent.py` 가 `--resume` flag + 인라인 done set 으로 사용 | round-robin coordinator 가 `--num-requests 1 --resume` 으로 반복 호출하며 매번 새 request 1 개씩 진행하는 시나리오 지원. partial JSONL 이 atomic 으로 매 request 마다 갱신 → 중단/재개 안전 |
| 2 | **mini-swe-agent tool style** | `tools/swebench.py:create_minisweagent_tools` (bash-only). `swebench_agent.py` 의 `MINISWEAGENT_SYSTEM_PROMPT` (one-liner) + `MINISWEAGENT_INSTANCE_TEMPLATE` (PR description 을 `{task}` placeholder 로 wrap; `str.format` 대신 `.replace()` 사용 — template 안 JSON-like 예제 보호). `parallel_tool_calls=True` 만 mini-swe-agent 에 활성. 종료는 (a) 빈 tool_calls 또는 (b) `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` sentinel 명령. `--tool-style minisweagent` CLI flag | mini-swe-agent 공식 SWE-Bench config 와 호환되는 bash-only ReAct loop. Phase 5 swebench 비교 실험용 |
| 3 | **`SGLANG_CAPTURE_FULL_POOL=1`** | `oracle_patch.py:341` 의 `capture_full_pool` 분기. `organize_draft_results` 를 두 번 호출 (정상 + cloned 입력으로 `pool_size+1` 풀 캡처). `eagle3_pool_full = {parent_list, draft_tokens, path_probs, pool_size}` 가 oracle entry 에 추가됨 | Stage 1 한 번 실행으로 여러 (S, K) 조합의 시뮬레이션 결과를 얻기 위한 "full pool" 보존. Generation 자체는 영향 없음 (cloned 입력) |
| 4 | **Pool reslicer** | `simulation/pipeline/pool_reslicer.py` (195줄). Stage 3 의 `--reslice-steps`, `--reslice-topk`, `--capture-steps`, `--capture-topk` (4-flag 묶음). `assemble_records.collect_step_records` 의 `eagle3_reslice` 인자 → reslice path 가 활성되면 `proposer_trees["eagle3"]` 를 (s', k') sub-tree 로 교체. fallback 시 원본 truncated tree 로 silent 복귀 + 첫 1회 warn | 한 번 captured pool 로 (S=8, K=16) → (s' ≤ 8, k' ≤ 16) sweep. `_sweep_sk.sh` 가 한 capture 위에서 9개 (s, k) 조합을 시뮬레이션 |
| 5 | **새 extension method 4종** | `run_tree_oracle_sim.py`: `extension_oracle_path`, `extension_dmsfx_oracle_path`, `extension_prune_pt:t`, `extension_prune_pt_oracle:t`. `_extension_step` 에 `backbone_pt_threshold` 인자 — `path_p_t < t` 인 base node 자체를 prune. `path_p_t` 가 monotone non-increasing 이라 keep-mask 가 valid subtree 보장 | base tree 자체를 pt threshold 로 잘라 verify cost 줄이는 실험. `*_oracle_path` 는 accepted path 만 cost 청구하는 더 strict 한 lower-bound |
| 6 | **N-level recursive extension + capped 변종** (commit `20fb84d` + 후속 working tree 변경) | `run_tree_oracle_sim.py:_extension_nlevel_step` + 2 method (`extension_nlevel_capped:N:max_size`, `extension_nlevel_capped_oracle:N:max_size`). `compute_latency_speedup` 에서 `n_lvl ∈ {2,3} × max_size ∈ {B*2, B*4, B*8}` 자동 enroll. 자세한 건 `05_stage3_simulator_core.md` §4.4 | `extension_oracle` 가 1-level suffix graft 의 ceiling 인 걸 깨려고 N 레벨로 재귀 graft. capped 변종은 deployable cost 가드. cap 가드는 root / per-node / per-leaf 세 분기 모두에 적용 |
| 7 | **BFCLv3 resume 지원** | `bfcl_agent.py:428-595` 에 `--resume` flag + `load_checkpoint`/`done` set/`pending` filter/`append_to_checkpoint`/`finalize_checkpoint` 패턴 추가 | 네 agent (BFCLv3 / BFCLv4 / SpecBench / SWE-Bench) 가 모두 일관된 resume 인프라를 갖게 됨. BFCLv3 만 #1 에서 빠져 있던 것을 후속으로 채움 |
| 8 | **Hybrid baseline paper-faithful 보정** | `_live_suffix_draft` 에 `paper_faithful` 인자 (`run_tree_oracle_sim.py:653`). `_hybrid_step` (`:736`), `_extension_hybrid_step` (`:1042`), `is_hybrid` 분기의 ext_size 재호출 (`:467`) 셋이 `paper_faithful=True` 사용 — `max_spec_factor=1.0`, `min_token_prob=0.1`, `max_spec_tokens=None` (ArcticInference default). `use_tree_spec=True` 만 우리 tree 기반 method 와 동등 비교 위해 유지. `single:suffix` / `extension_*` 등 우리 ceiling 측정용 호출은 default (aggressive) regime 유지 (`max_spec_factor=4.0, min_token_prob=0.0`) | SuffixDecoding NeurIPS 2025 paper (arxiv 2411.04975) 의 hybrid baseline 정의 ("score ≥ threshold → suffix, else model") 자체는 이전부터 일치했지만 suffix 호출 hyperparameter 가 논문 default 보다 4× aggressive 했음 → hybrid baseline 이 공정해야 우리 extension 과의 비교가 의미 있음 |
