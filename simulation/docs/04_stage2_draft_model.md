# Stage 2 — Draft Model 수집

## 1. 목적

Stage 1 이 EAGLE3 oracle trajectory 와 ground-truth token 시퀀스를 만든 뒤,
Stage 3 oracle simulator 가 step 마다 여러 종류의 proposer (eagle3 / suffix /
draft_model) 중 선택할 수 있도록 union trie 를 구성한다. 이 가운데
`draft_model` proposer 는 별도의 small LM (e.g. `Qwen/Qwen3-0.6B`) 이 동일한
prefix 를 보고 어떤 토큰을 생성했을지를 기록한 결과이다. Stage 2 는 그
small-LM proposal 을 step 별로 미리 수집해두는 단계이고, 산출물은
`draft_model_drafts.jsonl` 이다.

기록이 `(request_id, call_idx, step_idx)` 단위로 만들어지지 않으면 Stage 3 의
`dm_by_key.get((rid, ci, si))` 조회 (`assemble_records.py:267-283`) 가 빈
결과를 돌려주고 `single:draft_model` / `hybrid_dm` / `extension_dmsfx` 계열
방법이 전부 noop 이 된다.

> **Note**: 현재 RR 기반 연구에서는 draft_model proposer 를 active method
> 셋에서 제외한 상태 (`feedback_drop_bfcl_v3`, `project_50pct_gap_progress`
> 메모리 참조). Stage 2 는 latency 측정 (draft LM TPOT) 용 또는 추가 비교
> 실험 시에만 호출. 정상 RR 파이프라인에서는 skip 가능.

## 2. 진입점 — `collect_draft_model.py`

호출 형태:

```bash
python -m simulation.pipeline.collect_draft_model \
    --agent-results <agent_results_eagle3.json> \
    --output <draft_model_drafts.jsonl> \
    --model <draft_lm_path> \
    --server-url http://localhost:31000 \
    --max-draft-tokens 16 \
    [--target-model <target_lm_path>] \
    [--shard <I>/<N>] \
    [--exclude <ids.txt>] \
    [--responses <responses.jsonl>] \
    [--dataset <dataset.jsonl>]
```

- `--server-url` 이 있으면 SGLang draft 서버에 HTTP 호출 (production 경로).
  없으면 in-process HuggingFace fallback (`_generate_hf`).
- `--shard I/N`: greedy bin-packing (step 수 균형) 으로 자기 몫 request 만
  처리 → 출력 `<output>_shard<I>.jsonl`.
- `--target-model`: prompt 재구성용 tokenizer 로드. **빼먹으면 chat template
  / system prompt 없이 raw assistant 토큰만 보고 추론 → distribution 어긋남.**

여러 shard 로 병렬 실행 시 각 shard 가 자기 SGLang(draft LM, prefix-cache)
서버를 다른 GPU/port 에 띄워야 한다. legacy `run_parallel_draft_model.sh`
는 제거됐으므로 직접 launch + collect_draft_model 호출 (또는 새 wrapper
작성) 필요.

## 3. SGLang draft 서버 launch (참고용)

```bash
CUDA_VISIBLE_DEVICES=$gpu \
python -m sglang.launch_server \
    --model-path $DRAFT_LM \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph \
    --watchdog-timeout 600 \
    --host 0.0.0.0 --port $port
```

- `--mem-fraction-static 0.85`: KV-cache pool 크게 잡아 prefix-cache hit
  ratio 확보. SGLang 의 `RadixAttention` 기반 prefix-cache 가 default 활성.
- `--disable-cuda-graph`: input prefix 길이가 step 마다 다르므로 capture
  overhead 만 누적.
- 별도 `--enable-prefix-caching` flag 없음 — default.

## 4. 질문별 prefix 재구성

Stage 2 의 정확성을 좌우하는 가장 위험한 부분이다. Stage 3 는 `prompt +
tokens[0:step_idx]` 를 "현재까지 본 context" 로 정의하는데
(`assemble_records.py:227-309`), Stage 2 도 이 정의를 정확히 일치시켜야 한다.

### 4.1 `_iter_steps` — step 정의 (`collect_draft_model.py:45-69`)

```python
prompt = list(prompt_ids_list[call_idx]) if ... else []
decoded = []
for pos in range(n):
    if n - pos <= 1:
        decoded.append(tokens[pos])
        continue
    context = prompt + decoded if prompt else list(decoded)
    yield call_idx, pos, context
    decoded.append(tokens[pos])
```

- step `pos` 에서 context = `prompt_ids ++ tokens[0:pos]`. draft LM 은 다음
  토큰 `tokens[pos]` 를 예측해야 한다.
- `n - pos <= 1` 인 step (남은 미래가 1토큰 이하) 은 verify 의미가 없어
  skip — Stage 3 의 `if len(future) <= 1: continue`
  (`assemble_records.py:230-233`) 와 정확히 같은 조건.
- `prompt_ids_list` 가 `None` 이면 prompt 없이 `decoded` 만으로 context.
  이 경우 chat template / system prompt 가 빠지므로 small LM 의 출력 분포가
  실제 inference 와 달라진다 → `--target-model` flag 누락 시 silent corruption.

### 4.2 Prompt token id 재구성

`_extract_bfcl` (`_agent_io.py:243-269`) 에서 `tokenizer` 가 주어지면 prompt
복원. 우선순위:

1. `agent_metrics.steps[i].messages` 가 그대로 저장돼 있으면 (Stage 1 dump
   fast path) `apply_chat_template(messages, add_generation_prompt=True)` 로
   토큰화.
2. messages 가 없고 `turns[i]` 가 dict 며 `messages` 키를 가지면 SWE-bench
   path → `_reconstruct_swebench_prompts`. LangChain `type/content` →
   OpenAI `role/content` 변환 후 tool_calls/tool_call_id 보존.
3. 위 둘 다 없고 `bfcl_dataset` + `resp_by_id` 가 있으면 BFCL path →
   `_reconstruct_bfcl_prompts`. `system_prompt_pre_processing_chat_model` 로
   BFCL system prompt 첨부.

SpecBench 는 `_extract_online` 내부 `_reconstruct_specbench_prompts` 가
user/assistant turn 누적하며 매 user turn 마다 prompt id 생성.

### 4.3 Tokenizer 정합성

`collect_draft_model.py` 의 tokenizer 는 **target 모델** 기준 (`--target-model`)
으로 로드. 즉 reconstruction 된 `prompt_ids` 는 target tokenizer 토큰. SGLang
draft 서버에 그대로 `input_ids` 로 전달.

현재 사용하는 모든 (target, draft_lm) 조합 — `Qwen3-8B/14B`+`Qwen3-0.6B` —
은 같은 family vocabulary 공유. cross-family pair 시 `input_ids` 잘못 해석
→ 자동 검출 안 됨, 주의.

## 5. SGLang HTTP request 형태

`collect_draft_model.py:103-115`:

```python
resp = http_requests.post(
    f"{server_url}/generate",
    json={
        "input_ids": context,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0,
        },
    },
    timeout=120,
)
draft_tokens = resp.json().get("output_ids", []) or []
```

- Endpoint: `/generate` (SGLang native, OpenAI-compatible API 미사용 — text
  round-trip 없음).
- Input: token id list 직접. chat template 재적용 / detokenization round-trip
  없으므로 token-level 정합성 깨질 위지가 없음.
- Sampling: greedy (`temperature=0`).
- `max_new_tokens = --max-draft-tokens` (default 16). Stage 3 의
  `MAX_DRAFT_MODEL_N = 16` (`run_tree_oracle_sim.py`) 와 짝. 16 외 값으로
  바꾸면 latency 모델 어긋남.
- timeout 120s. step request timeout 시 빈 chain 기록 → Stage 3 에서 해당
  step 의 draft_model proposer attach 안 됨.

## 6. 출력 JSONL 스키마

레코드 단위 = per (request, call, step):

```json
{"request_id": "<bfcl_id|instance_id|question_id>",
 "call_idx": <int>,
 "step_idx": <int>,
 "token_ids": [tok0, tok1, ...],
 "parents": [-1, 0, 1, ...]}
```

- `parents` 는 `[-1, 0, 1, ..., n-2]` 형태의 선형 chain (draft LM 은 tree
  안 만들고 greedy 한 줄만 뽑음).
- `token_ids[0]` 은 `tokens[step_idx]` 자리 — root 는 next-token prediction.
- 빈 record (timeout 등) 는 Stage 3 가 무시.

Stage 3 consumer (`assemble_records.py:155-170`) 가 `(request_id, call_idx,
step_idx)` 키로 dict 화 → `proposer_trees["draft_model"] = (token_ids,
parents)` 로 합침. dict 키 mismatch 시 모든 lookup miss.

## 7. Idempotency / resume

지원 안 함. `collect_draft_model.py` 의 `open(output_path, "w")` 가 매 실행
마다 truncate. shard merge 도 단순 concat 이므로 부분 재실행 → merge 시
stale shard 와 새 shard 섞일 수 있음. 부분 실패 시 출력 jsonl 삭제 후 전체
재실행이 안전.

## 8. 정확성 체크리스트

1. **`--target-model` 누락 검사** — 누락 시 prompt 복원 skip → distribution 어긋남
2. **step 정의 일치** — `n - pos <= 1` 과 `len(future) <= 1` 둘 다 현재 일치
3. **Tokenizer family 일치** — preset 조합 모두 안전
4. **`request_id` 표현형 일치** — BFCL=`bfcl_id`, SWE=`instance_id`, SpecBench=`str(question_id)`
5. **`MAX_DRAFT_MODEL_N=16` ↔ `--max-draft-tokens 16`** — 둘 다 16 으로 정렬
