# Stage 1 개요 — EAGLE3 Trajectory 수집

Stage 1 은 GPU 마다 EAGLE3 speculative decoding 을 켠 SGLang 서버를 띄우고, 그에 대해 benchmark agent 를 실행하여 agent 의 chat trace 와 per-step EAGLE3 draft token (이른바 "oracle" 로그) 을 단일 `agent_results_eagle3.json` 에 기록한다. 이후 모든 단계 (Stage 2 draft-LM 수집, Stage 3 oracle 시뮬레이션) 가 이 artifact 를 소비한다.

본 문서는 진입 셸 스크립트, GPU 샤딩, 서버 플래그, 환경변수, 샤드 머지 로직, 그리고 입력 데이터셋을 만드는 prepare 스크립트를 다룬다.

## 1. 진입점 (Entry points)

### 1.1 `simulation/scripts/run_pipeline.sh`

최상위 드라이버. 모델 preset 을 선택하고, agent 모듈을 선택하고, 부분 실행을 위해 입력 데이터셋을 슬라이스한 뒤 `run_parallel_stage1.sh` 로 dispatch 한다. Stage 1 본체는 `simulation/scripts/run_pipeline.sh:184-201` 에 있다.

호출 형태:

```
bash simulation/scripts/run_pipeline.sh <benchmark> <model_preset> [num_requests]
```

`<benchmark>` ∈ {`bfcl_v3`, `bfcl_v4`, `specbench`, `swebench`}, `<model_preset>` ∈ {`glm4_flash`, `qwen3_8b`, `qwen3_14b`, `qwen3_32b`, `llama3_8b`}.

Per-benchmark dispatch 표 (`run_pipeline.sh:78-116` 에서 구성):

| Benchmark   | AGENT_MODULE                            | INPUT_FILE (default)                      | 추가 agent CLI 플래그                                                |
|-------------|-----------------------------------------|-------------------------------------------|----------------------------------------------------------------------|
| `bfcl_v3`   | `simulation.agents.bfcl_agent`          | `data/bfcl_multi_turn/dataset.jsonl`      | `--model $MODEL --max-iterations ${BFCL_MAX_ITER:-5} --temperature 0.0` |
| `bfcl_v4`   | `simulation.agents.bfcl_v4_agent`       | `${BFCL_V4_INPUT:-data/bfcl_agent/dataset.jsonl}` | `--model $MODEL --max-iterations ${BFCL_MAX_ITER:-5}` (temp 플래그 없음 — agent 가 0.0 을 하드코딩) |
| `specbench` | `simulation.agents.specbench_agent`     | `data/specbench/dataset.jsonl`            | `--dataset $INPUT_FILE --model $MODEL --temperature 0.0`             |
| `swebench`  | `simulation.agents.swebench_agent`      | `data/swebench/dataset.jsonl`             | `--model $MODEL --max-iterations ${SWE_MAX_ITER:-30} --repos-dir data/swebench/repos --temperature 0.0` |

Per-preset 모델/draft 매핑 (`run_pipeline.sh:42-73`):

| Preset       | MODEL                              | DRAFT_MODEL                                  | DRAFT_LM (Stage 2)              |
|--------------|------------------------------------|----------------------------------------------|---------------------------------|
| `glm4_flash` | `zai-org/GLM-4.7-Flash`            | `thoughtworks/GLM-4.7-Flash-Eagle3`          | (unset → Stage 2 skip)          |
| `qwen3_8b`   | `Qwen/Qwen3-8B`                    | `AngelSlim/Qwen3-8B_eagle3`                  | `Qwen/Qwen3-0.6B`               |
| `qwen3_14b`  | `Qwen/Qwen3-14B`                   | `AngelSlim/Qwen3-14B_eagle3`                 | `Qwen/Qwen3-0.6B`               |
| `qwen3_32b`  | `Qwen/Qwen3-32B`                   | `Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3`       | `Qwen/Qwen3-0.6B`               |
| `llama3_8b`  | `meta-llama/Llama-3.1-8B-Instruct` | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`        | `meta-llama/Llama-3.2-1B-Instruct` |

### 1.2 `OUTPUT_DIR` 결정

`run_pipeline.sh:118-137`. 레이아웃:

- 베이스: `simulation/results/${MODEL_PRESET}/${BENCHMARK}` (preset 은 소문자).
- `OUTPUT_DIR_SUFFIX` 가 설정되면: `${OUTPUT_DIR}_${OUTPUT_DIR_SUFFIX}`. 동일 sweep 의 서브디렉토리 (`steps_2`, `steps_4`, …) 가 서로 덮어쓰지 않도록 격리하는 용도.
- `REQ_START` 와 `REQ_END` 가 모두 설정되면: `${OUTPUT_DIR}_req${REQ_START}-${REQ_END}`. 스크립트는 이때 `${OUTPUT_DIR}/input_slice.jsonl` (라인 `[REQ_START:REQ_END)`) 을 만들고 `INPUT_FILE` 을 그 슬라이스로 재지정한다 (`run_pipeline.sh:152-165`). 여러 머신에 걸쳐 benchmark 를 돌릴 때 사용하는 per-machine 샤딩 훅.

### 1.3 EAGLE3 tree-shape sweep 노브

Stage 1 서버에 설정된다 (`run_pipeline.sh:124-126` 에서 env 로 `run_parallel_stage1.sh:24-26` 으로 전달되어 `sglang.launch_server` 플래그가 됨):

| Env var                     | 기본값  | SGLang 플래그                     |
|-----------------------------|---------|-----------------------------------|
| `STAGE1_TOPK`               | 8       | `--speculative-eagle-topk`        |
| `STAGE1_STEPS`              | 5       | `--speculative-num-steps`         |
| `STAGE1_NUM_DRAFT_TOKENS`   | 256     | `--speculative-num-draft-tokens`  |

주의: SGLang draft-tree organizer 는 `budget > topk + (steps-1)·topk² + 1` 일 때 크래시한다. `STAGE1_NUM_DRAFT_TOKENS` 는 그 한계 안에서 골라야 한다.

## 2. `simulation/scripts/run_parallel_stage1.sh`

GPU 마다 SGLang 서버 1개 + agent 프로세스 1개를 fork 한다. 모든 샤드는 `${OUTPUT_DIR}/_stage1_shard${SHARD_IDX}/` 에 쓰며, 마지막에 Python 머거가 그것들을 단일 `${OUTPUT_DIR}/agent_results_eagle3.json` 으로 합친다.

### 2.1 GPU 샤딩

`run_parallel_stage1.sh:28-43`. `GPU_IDS` env (예: `"0,2,3"`) 가 기본값 `0..NUM_GPUS-1` 를 override 한다. 입력 데이터셋이 `NUM_GPUS` 보다 짧으면 `NUM_GPUS` 가 라인 수로 줄어든다.

라운드로빈 슬라이싱은 `run_parallel_stage1.sh:64-70` 에 있다: `shard[i] = lines[i] for i in range(len(lines)) if i % NUM_GPUS == SHARD_IDX`. 각 샤드의 슬라이스는 `_stage1_shard${IDX}/input.jsonl` 로 쓰여진다.

포트 할당: `PORT = ${STAGE1_BASE_PORT:-30000} + SHARD_IDX` (`run_parallel_stage1.sh:59`).

### 2.2 필수 환경변수 (`run_parallel_stage1.sh:49-52` 에서 설정)

| Env var                                       | 역할                                                                                            |
|-----------------------------------------------|-------------------------------------------------------------------------------------------------|
| `SGLANG_ORACLE_VANILLA=1`                     | EAGLE worker 패치를 활성화하여 draft tree 를 `/tmp/sglang_oracle_vanilla.jsonl` 에 로깅. `simulation/oracle/oracle_patch.py:50` (`is_oracle_enabled()`) 에서 읽는다. 이게 없으면 agent 는 silently 돌지만 `oracle_vanilla_entries` 가 비어 있게 된다. |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | SGLang 이 선언된 `max_position_embeddings` 보다 긴 context 를 가진 모델을 받도록 허용. agent 가 긴 tool 히스토리를 보내기 때문에 필요. |
| `TORCHINDUCTOR_COMPILE_THREADS=1`             | 여러 샤드가 동시에 컴파일할 때 torch.compile 의 worker pool 이 RAM 을 고갈시키는 것을 방지. `simulation/oracle/install_hook.py:24-26` 에서도 어떤 torch import 전에 중복 설정. |
| `unset SGLANG_ORACLE_REPLAY`                  | 방어적: 이 변수는 Round-2 replay 실행용이며, unset 으로 이전 셸의 누설을 막는다. |

### 2.3 SGLang 패치 설치

`run_parallel_stage1.sh:54` 가 `python3 -m simulation.oracle.install_hook` 를 실행한다. 추가 argv 없이 호출되면 `simulation/oracle/install_hook.py:282-300` 의 디스크 패치만 트리거한다 — 설치된 `sglang` 패키지 안의 `spec_info.py`, `server_args.py`, `scheduler.py`, `eagle_worker.py`, `multi_layer_eagle_worker.py` 를 편집하여:

1. `--speculative-algorithm SUFFIX` 가 인식되는 choice 가 됨 (Stage 1 에선 사용 안 하지만 무조건 설치됨).
2. EAGLE worker `__init__` 가 `SGLANG_ORACLE_VANILLA=1` 일 때 `patch_eagle_worker_full(self)` 를 호출 — 이게 per-step entries 를 oracle 로그에 쓰는 주체.

이 패치들은 idempotent (`install_hook.py:73`, `:110`, `:144`, `:210`); 재실행해도 안전하다.

### 2.4 샤드별 런치 (`run_parallel_stage1.sh:81-107`)

각 샤드마다 백그라운드 서브셸에서:

1. `/workspace/.env` 가 있으면 source (`HF_TOKEN` 이 sglang 다운로드 경로에 도달하도록 — Llama 같은 gated repo 용).
2. 다음 플래그로 SGLang 서버 런치:
   - `CUDA_VISIBLE_DEVICES=$GPU_ID`
   - `--model-path "$MODEL" --tp-size 1`
   - `--speculative-algorithm EAGLE3`
   - `--speculative-draft-model-path "$DRAFT_MODEL"`
   - `--speculative-num-steps $STAGE1_STEPS`
   - `--speculative-eagle-topk $STAGE1_TOPK`
   - `--speculative-num-draft-tokens $STAGE1_NUM_DRAFT_TOKENS`
   - `--tool-call-parser ${TOOL_CALL_PARSER:-qwen25}`
   - `--mem-fraction-static 0.85`
   - `--disable-cuda-graph`
   - `--watchdog-timeout 600`
   - `--host 0.0.0.0 --port $PORT`
   로그는 `_stage1_shard${IDX}/server.log`.
3. `http://localhost:$PORT/health` 를 3 초마다 health-poll, 최대 120 회 (≈ 6 분). backoff escalation 없음; 끝내 안 뜨더라도 에러 없음 — agent 가 첫 요청에서 실패하면 샤드는 결국 종료된다.
4. `python3 -m $AGENT_MODULE --url http://localhost:$PORT/v1 --model "$MODEL" --input-file shard/input.jsonl --output-file shard/agent_results.json --num-workers 1 $EXTRA_ARGS` 실행. 로그는 `_stage1_shard${IDX}/agent.log`.
5. agent 가 리턴한 뒤: `kill $SRV_PID; wait $SRV_PID`.

`--num-workers 1` 은 하드코딩 — concurrency 가 켜지면 oracle 로그에서 entries 가 인터리빙된다. (`simulation/pipeline/_agent_io.py:42-46` 의 `_extract_entries` 헬퍼에 인터리빙 감지 시 most-common `req_id` 만 필터하는 가드가 있긴 하지만, Stage 1 은 그것에 의존하지 않는다.)

### 2.5 `TOOL_CALL_PARSER` 선택

`run_pipeline.sh:74-75` 에서 설정:

```
export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen25}
```

Llama preset 은 그 위 `run_pipeline.sh:67` 에서 미리 override 한다 (`export TOOL_CALL_PARSER=llama3`). 그 외 모든 preset 은 `qwen25` 사용.

이 플래그는 LangChain 기반 agent (`swebench_agent.py`) 에만 의미가 있다. 이 agent 는 `llm.bind_tools(tools)` 를 호출하여 서버가 OpenAI 포맷의 `tool_calls` 를 반환할 것을 기대한다. Qwen3 의 기본 출력은 XML `<tool_call>…</tool_call>` 이며, `bind_tools()` 가 이를 파싱하지 못한다; `qwen25` 없이는 agent 가 텍스트 한 개짜리 응답을 받고, tool call 을 못 보고, 한 iteration 만에 종료한다. `run_parallel_stage1.sh:73-78` 의 경고 참조.

`bfcl_agent.py` 와 `bfcl_v4_agent.py` 는 자체 native-syntax 디코딩 (`default_decode_execute_prompting`) 을 하며 파싱된 `tool_calls` 필드를 쓰지 않으므로, 이 플래그는 무해하지만 그들에게는 무관하다. `specbench_agent.py` 는 tool 자체를 안 쓴다.

### 2.6 실패 동작

`run_parallel_stage1.sh:111-118`. 스크립트는 모든 샤드 PID 를 `wait` 하고 non-zero exit 를 센다. 실패한 샤드는 `server.log` / `agent.log` 가 보존되며 (`:159-167` 에서 rmtree 안 함), 스크립트는 non-zero 로 종료하므로 둘러싼 파이프라인이 Stage 2/3 전에 중단된다.

모든 샤드가 성공하면 per-shard 디렉토리는 삭제된다 (`run_parallel_stage1.sh:160-162`) — `server.log` 와 per-shard `agent_results.json` 이 사라진다. 머지된 `agent_results_eagle3.json` 만 남는다.

### 2.7 샤드 머지 (`run_parallel_stage1.sh:120-155`)

인라인 Python 스크립트. 각 샤드마다:

1. `_stage1_shard${IDX}/agent_results.json` 을 읽는다.
2. `data["questions"]` 를 `merged_questions` 에 concat 한다.
3. `metadata["total_tokens"]` 와 `metadata["total_oracle_entries"]` 를 합산한다.
4. 누락/손상된 샤드는 `WARN: shard {idx} failed: {e}` 를 출력하지만 abort 하지는 않는다 — 머지된 파일이 단지 question 이 적어질 뿐. 부모 스크립트가 어떤 샤드의 프로세스가 실패하면 이미 종료했기 때문에, 이 경로는 agent 가 exit 0 했으면서도 malformed JSON 을 쓴 경우에만 도달한다.

최종 write 는 `:142-153`:

```json
{
  "metadata": {
    "model": "<MODEL>",
    "num_requests": <int>,
    "total_tokens": <int>,
    "total_oracle_entries": <int>,
    "oracle_enabled": true
  },
  "questions": [ ... merged ... ]
}
```

주의: 이 머지 metadata 는 각 샤드의 agent 가 쓴 것을 대체한다. `total_tool_calls` (BFCLv3) 또는 `benchmark` (BFCLv4 / SpecBench / SWE-Bench) 같은 per-benchmark 필드는 머지 시점에 drop 된다. `metadata.url` 필드도 사라진다. per-question payload 만 verbatim 으로 보존된다.

## 3. 데이터셋 prepare

세 prepare 스크립트 모두 `data/<benchmark>/dataset.jsonl` (한 줄당 한 JSON 객체) 을 쓰며, 이를 `run_parallel_stage1.sh` 가 슬라이스한다.

### 3.1 BFCL — `simulation/scripts/prepare_bfcl_data.py`

`--benchmark {all,v3,v4}` 로 두 sub-benchmark 중 선택.

**`prepare_bfcl_v3`** (`prepare_bfcl_data.py:106-132`): 설치된 `bfcl_eval` 패키지의 `data/` 디렉토리에서 네 파일 (`BFCL_v4_multi_turn_{base,miss_func,miss_param,long_context}.json`) 을 읽어, `question_id`, `category` (`bfcl_v3/` prefix), `bfcl_id`, `id`, `question`, `function`, `initial_config`, `involved_classes`, optional `missed_function`, `scenario` 필드를 가진 단일 JSONL 로 flatten 한다. `possible_answer/<filename>` 으로부터 `ground_truth.jsonl` 도 함께 쓴다.

**`prepare_bfcl_v4`** (`prepare_bfcl_data.py:135-212`): `AGENTIC_CATEGORY` 를 순회하며 BFCL 의 `load_dataset_entry(cat, include_prereq=True)` 를 사용. 이 경로가 prereq conversation 을 포함한다. 레코드는 같은 모양에 `depends_on` 만 추가됨. 마지막에 `id` 로 dedup.

### 3.2 SpecBench — `simulation/scripts/prepare_specbench_data.py`

`HuggingFaceH4/mt_bench_prompts` (split=train) 를 다운로드하고 row 마다 한 레코드를 emit:

```json
{"question_id": "<prompt_id>", "category": "<category>", "turns": [<user_msgs>]}
```

`function`/`tools` 필드 없음 — SpecBench 는 plain Q&A.

### 3.3 SWE-Bench — `simulation/scripts/prepare_swebench_data.py`

설정된 HF 데이터셋 (기본 `princeton-nlp/SWE-bench_Verified [test]`) 을 다운로드. Per-row 스키마:

```json
{
  "instance_id": "...",
  "repo": "<org>/<name>",
  "base_commit": "<sha>",
  "problem_statement": "...",
  "turns": ["<problem_statement>"],
  "category": "<version>",
  "hints_text": "...",            // optional
  "test_patch": "...",
  "patch": "...",
  "FAIL_TO_PASS": [...],
  "PASS_TO_PASS": [...],
  "environment_setup_commit": "...",
  "created_at": "..."
}
```

`turns[0]` = `problem_statement` 이 `swebench_agent.py:213-215` 가 초기 `HumanMessage` 로 넘기는 값이다. agent 의 `--repos-dir` 플래그는 스크립트가 인스턴스마다 `git clone` / `git reset --hard` 할 디렉토리를 가리킨다 (`swebench_agent.py:98-134`).

## 4. 구체 실행 예시

단일 머신, GPU 4개, BFCLv4 전체:

```
bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b
```

두 머신 샤드 실행, BFCLv4 를 request 50 에서 split:

```
# Box A
REQ_START=0  REQ_END=50  bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b
# Box B
REQ_START=50 REQ_END=100 bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b
# 그 다음 머지:
simulation/scripts/merge_shards.sh simulation/results/qwen3_8b/bfcl_v4
```

depth 에 대한 EAGLE3 sweep, 단일 머신, output dir 분리:

```
for s in 2 4 6 8; do
  STAGE1_STEPS=$s OUTPUT_DIR_SUFFIX=steps_$s \
    bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b
done
```

특정 GPU 만 사용 (GPU 1 skip, 0/2/3 에서 실행):

```
GPU_IDS=0,2,3 bash simulation/scripts/run_pipeline.sh swebench qwen3_14b
```

Llama preset (`TOOL_CALL_PARSER=llama3` 강제, `/workspace/.env` 에 `HF_TOKEN` 필요):

```
bash simulation/scripts/run_pipeline.sh swebench llama3_8b 30
```
