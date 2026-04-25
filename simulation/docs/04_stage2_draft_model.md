# Stage 2 — Draft Model 수집

## 1. 목적 (왜 필요한가)

Stage 1이 EAGLE3 oracle trajectory와 ground-truth token 시퀀스를 만든 뒤, Stage 3 oracle simulator는 step마다 여러 종류의 proposer (eagle3 / suffix / draft_model / mtp) 가운데 하나를 선택할 수 있도록 union trie를 구성한다. 이 가운데 `draft_model` proposer는 별도의 small LM (e.g. `Qwen/Qwen3-0.6B`)이 동일한 prefix를 보고 어떤 토큰을 생성했을지를 기록한 결과이다. Stage 2는 그 small-LM proposal을 step별로 미리 수집해두는 단계이고, 산출물은 `draft_model_drafts.jsonl`이다.

기록이 `(request_id, call_idx, step_idx)` 단위로 만들어지지 않으면 Stage 3의 `dm_by_key.get((rid, ci, si))` 조회 (`simulation/pipeline/assemble_records.py:267-283`)가 빈 결과를 돌려주고 `single:draft_model` / `hybrid_dm` / `extension_dmsfx` 계열 방법이 전부 noop이 된다. 따라서 Stage 2의 정확성은 oracle pipeline 결과 해석에 직접 영향을 준다.

## 2. 셸 entry point — `run_parallel_draft_model.sh`

호출 형태 (`simulation/scripts/run_parallel_draft_model.sh:6-15`):

```
bash simulation/scripts/run_parallel_draft_model.sh \
    <agent_results_eagle3.json> <draft_model_drafts.jsonl> \
    <draft_model> [num_gpus=4] [max_draft_tokens=16] [extra_args...]
```

`run_pipeline.sh`에서 wiring은 `simulation/scripts/run_pipeline.sh:213-224`에 있다. `DRAFT_LM`이 비어 있으면 Stage 2 자체가 skip되고 Stage 3는 `--draft-model-drafts` 없이 돌아간다 (`run_pipeline.sh:225-228`).

### 2.1 GPU/포트 할당

- `GPU_IDS`가 환경변수로 주어지면 그대로 사용 (예: `GPU_IDS=0,2,3`), 아니면 `0..NUM_GPUS-1` (`run_parallel_draft_model.sh:27-33`).
- `N_REQS`는 `agent_results.questions` 길이로 측정해서 (`run_parallel_draft_model.sh:36-45`) 요청 수보다 GPU가 많으면 GPU 수를 줄인다. shard가 빈 채로 도는 사고를 막는 안전장치.
- 포트는 `STAGE3B_BASE_PORT` (default `31000`)에서 shard별로 +1씩 할당 (`run_parallel_draft_model.sh:58-69`). Stage 1 SGLang 포트 (`PORT=30000`)와 충돌하지 않도록 의도적으로 다른 base를 쓴다.

### 2.2 SGLang 서버 플래그

`run_parallel_draft_model.sh:62-66`:

```
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m sglang.launch_server \
  --model-path "$MODEL" --tp-size 1 \
  --mem-fraction-static 0.85 --disable-cuda-graph \
  --host 0.0.0.0 --port $PORT
```

핵심 포인트:
- `--tp-size 1`: draft LM은 0.6B/1B 정도라 single-GPU.
- `--mem-fraction-static 0.85`: KV-cache pool을 크게 잡아 prefix-cache hit ratio 확보. SGLang은 `RadixAttention` 기반 prefix-cache가 기본 활성화이므로 별도 flag 없이도 같은 prefix가 반복 들어오면 KV가 공유된다. 이 단계의 throughput은 prefix-cache hit에 거의 전적으로 의존한다.
- `--disable-cuda-graph`: input prefix 길이가 step마다 다르므로 CUDA graph capture가 무효함. capture overhead만 누적되므로 끈다.
- 별도의 `--enable-prefix-caching` flag는 없음 — SGLang은 prefix caching이 default.

서버 ready 체크는 `curl /health`로 최대 120 × 3s = 360s까지 polling (`run_parallel_draft_model.sh:73-79`). 모델 로드가 그 이상 걸리는 환경에서는 timeout을 키워야 한다.

### 2.3 샤딩과 머지

각 shard 프로세스는 `--shard $SHARD_IDX/$NUM_GPUS` 인자를 받아 `collect_draft_model.py` 내부에서 greedy bin-packing (`collect_draft_model.py:211-229`)으로 step 수가 균형 잡히도록 자기 몫의 request만 처리한다. 단순 modulo가 아니라 step count 기준 bin-packing이므로 long-tail request 1개가 한 shard를 늦게 끝내는 사고를 줄인다.

shard 출력은 `${OUTPUT%.jsonl}_shard${SHARD_IDX}.jsonl`. 모든 shard가 끝나면 단순 `cat ... >> $OUTPUT` concat 후 shard 파일을 삭제한다 (`run_parallel_draft_model.sh:114-120`). JSONL 포맷이라 순서 보장은 없으며 record 단위로 lookup되므로 무관하다.

서버는 collect 프로세스 종료 직후 `kill $PID`로 정리 (`run_parallel_draft_model.sh:106-109`). 한 shard라도 실패하면 전체 종료 코드 1 (`run_parallel_draft_model.sh:101-111`).

## 3. 질문별 prefix 재구성

Stage 2의 정확성을 좌우하는 가장 위험한 부분이다. Stage 3는 `prompt + tokens[0:step_idx]`를 "현재까지 본 context"로 정의하는데 (`assemble_records.py:227-309`), Stage 2도 이 정의를 정확히 일치시켜야 한다.

### 3.1 `_iter_steps` — step 정의 (`collect_draft_model.py:45-69`)

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

- step `pos`에서 context = `prompt_ids ++ tokens[0:pos]`. 즉 마지막에 본 토큰은 `tokens[pos-1]`이고 draft LM은 그 다음 `tokens[pos]`를 예측해야 한다.
- `n - pos <= 1`인 step (남은 미래가 1토큰 이하)은 verify 의미가 없어 skip — Stage 3의 `if len(future) <= 1: continue` (`assemble_records.py:230-233`)와 정확히 같은 조건.
- `prompt_ids_list`가 `None`이면 prompt 없이 `decoded`만으로 context를 만든다. 이 경우 chat template / system prompt가 빠지므로 small LM의 출력 분포가 실제 inference와 달라진다 — `--target-model` flag를 빼먹으면 조용히 이런 상태가 된다.

### 3.2 Prompt token id 재구성

`_extract_bfcl` (`_agent_io.py:243-269`)에서 `tokenizer`가 주어지면 prompt를 복원한다. 우선순위:

1. `agent_metrics.steps[i].messages`가 그대로 저장돼 있으면 (Stage 1이 메시지 dump를 남긴 fast path) 그걸 `apply_chat_template(messages, add_generation_prompt=True)`로 토큰화 (`_agent_io.py:248-256`). 결과가 list면 그대로, dict면 `input_ids` 추출. `add_generation_prompt=True`라서 `<|im_start|>assistant\n` 같은 generation prefix까지 포함된다.
2. messages가 없고 `turns[i]`가 dict이며 `messages` 키를 가지면 SWE-bench path → `_reconstruct_swebench_prompts` (`_agent_io.py:97-114`). LangChain `type/content` → OpenAI `role/content` 변환을 거쳐 (`_agent_io.py:71-94`) tool_calls/tool_call_id를 보존하면서 chat template 적용.
3. 위 둘 다 없고 `bfcl_dataset` + `resp_by_id`가 있으면 BFCL path → `_reconstruct_bfcl_prompts` (`_agent_io.py:117-164`). `system_prompt_pre_processing_chat_model`로 BFCL system prompt를 첫 turn에 끼워넣고, `holdout_function`이 있는 turn은 `DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING` 템플릿으로 user 메시지를 추가한 뒤 turn별 step마다 `apply_chat_template(messages, add_generation_prompt=True)`.

SpecBench는 `_extract_online` (`_agent_io.py:347-352`) 내부에서 `_reconstruct_specbench_prompts` (`_agent_io.py:167-182`)로 user/assistant turn을 누적하며 매 user turn마다 prompt id를 만든다.

### 3.3 Tokenizer 정합성 (target vs draft)

`collect_draft_model.py:264-268`의 tokenizer는 **target 모델**(`--target-model`, e.g. `Qwen/Qwen3-8B`) 기준으로 로드된다. 즉 reconstruction된 `prompt_ids`는 target tokenizer 토큰. SGLang draft 서버에 보낼 때는 그대로 `input_ids`로 전달한다 (`collect_draft_model.py:103-113`).

이 코드 베이스에서 사용하는 모든 (target, draft_lm) 조합 — `Qwen3-8B/14B/32B`+`Qwen3-0.6B`, `Llama-3.1-8B-Instruct`+`Llama-3.2-1B-Instruct`, `GLM-4.7-Flash` 단독 — 은 같은 family 내 vocabulary 공유 모델이다. 따라서 target tokenizer 출력 token id를 draft LM에 그대로 넣어도 의미가 보존된다. Family 다른 draft LM을 끼워 넣으면 `input_ids`가 잘못 해석된다 — 이 경우 코드에서 자동 검출되지 않으므로 주의.

특수 토큰(`<|im_start|>`, `<|im_end|>`, BOS/EOS)은 chat template에 의해 `apply_chat_template(...tokenize=False)` → `tokenizer.encode(text)` 경로로 자연스럽게 포함된다. 이 경로는 BFCL/SpecBench/SWE-bench reconstruction 모두에서 사용 (`_agent_io.py:109-111, 156-158, 176-178`).

### 3.4 응답 token id 누적

`_iter_steps`는 prompt를 제외한 `decoded`에 `tokens[pos]`를 append만 한다 (`collect_draft_model.py:69`). `tokens`는 `_extract_entries`에서 `e["tokens"][0]`을 모든 step에서 모은 것 (`_agent_io.py:54-57`)으로, Stage 1에서 EAGLE3 oracle vanilla가 실제로 emit한 ground-truth token이다. 즉 step `pos`의 prefix는 그때까지 실제 모델이 뱉어낸 토큰을 그대로 이어붙인 것이고, draft LM의 이전 잘못된 추측이 prefix를 오염시키는 일은 없다 (oracle 정의 그 자체).

## 4. SGLang HTTP request 형태

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

- Endpoint: `/generate` (SGLang native, OpenAI-compatible API 미사용 — text round-trip 없음).
- Input: token id list 직접. chat template 재적용 / detokenization round-trip 없으므로 token-level 정합성이 깨질 위지가 없다.
- Sampling: greedy (`temperature=0`). `top_p` / stop sequence 미지정. `top_p`는 SGLang default가 적용되지만 greedy면 무관.
- `max_new_tokens = --max-draft-tokens` (default 16, `run_pipeline.sh:222`에서 `STAGE2_MAX_TOKENS` / 구버전 `STAGE3B_MAX_TOKENS` 환경변수로 override 가능). 16의 의미는 Stage 3에서 `MAX_DRAFT_MODEL_N = 16`으로 hard-cap돼 있고 (`run_tree_oracle_sim.py:1011`) draft chain 길이가 그보다 길면 어차피 잘리기 때문. Stage 2에서 16보다 더 만들어도 latency 모델이 16 forward 비용만 청구하므로 현실과 어긋난다.
- Stop token도 없음. EOS가 나오면 SGLang이 자체적으로 stop하지만 코드에서 명시적으로 강제하지는 않는다. greedy + `max_new_tokens=16`이라 endless gen 위험은 없다.
- timeout 120s. 한 step request가 timeout되면 `except`로 잡혀 (`collect_draft_model.py:116-120`) 빈 chain (`token_ids=[]`)이 기록된다 — 해당 step에서는 `dm_rec.get("token_ids")`가 falsy라 Stage 3가 draft_model proposer를 attach하지 않는다 (`assemble_records.py:278-283`). 즉 실패는 silent하지만 결과를 손상시키지는 않는다.

`max_tokens` enforcement는 전적으로 SGLang `max_new_tokens` 인자에 위임. 코드에서 길이 검사 / truncation 없이 그대로 `_flat_chain`에 넣는다 (`collect_draft_model.py:122`).

## 5. 출력 JSONL 스키마

레코드 단위는 **per (request, call, step)** — request 1개당 step 수만큼 line이 나온다. `collect_draft_model.py:123-129`:

```json
{"request_id": "<bfcl_id|instance_id|question_id>",
 "call_idx": <int>,
 "step_idx": <int>,
 "token_ids": [tok0, tok1, ...],
 "parents": [-1, 0, 1, ...]}
```

- `parents`는 `_flat_chain` (`collect_draft_model.py:72-75`)으로 `[-1, 0, 1, ..., n-2]` 형태의 선형 chain. draft LM은 tree 구조를 만들지 않고 greedy 한 줄짜리 sequence만 뽑으므로 정의상 chain.
- `token_ids[0]`은 `tokens[step_idx]`(ground truth) 자리 — 즉 root는 next-token prediction. `parents[0] = -1`로 root 표시.
- 빈 record (`token_ids=[]`, `parents=[]`)는 SGLang 호출이 실패한 step. Stage 3에서 무시된다.

Stage 3 consumer (`assemble_records.py:155-170`)는 `(request_id, call_idx, step_idx)` 키로 dict화한 뒤 `proposer_trees["draft_model"] = (token_ids, parents)`로 union trie에 합친다 (`assemble_records.py:278-283`). 이 dict 키 일치가 깨지면 (e.g. `request_id`가 string vs int로 mismatch) 모든 lookup이 miss.

> 주의: SpecBench의 `request_id`는 `qid = str(q.get("question_id", qi))` (`_agent_io.py:316-317`) — 문자열. BFCL은 `bfcl_id` 그대로 (이미 문자열). SWE-bench는 `instance_id`. Stage 3 loader도 `r["request_id"]`를 그대로 dict 키로 쓰므로 (`assemble_records.py:168`) 양쪽이 같은 표현을 쓰는 한 정상.

## 6. Idempotency / resume

지원 안 함. `collect_draft_model.py:313`은 `open(output_path, "w")` — shard 파일을 매 실행마다 truncate하고 처음부터 다시 쓴다. shard 파일 merge도 단순 concat이므로 부분 재실행 → merge 시 stale shard와 새 shard가 섞일 수 있다. 부분 실패 시에는 `OUTPUT/draft_model_drafts.jsonl`을 지우고 전체 재실행하는 게 안전.

`--exclude` flag (`collect_draft_model.py:249`)로 특정 `bfcl_id`를 빼는 것은 가능 (`load_exclude_ids` `_agent_io.py:24-34`) — 한 줄당 id 하나, `#` comment 허용. 이건 resume용이 아니라 알려진 broken sample을 제거할 때 쓰는 용도.

체크포인트 flush는 200 step마다 (SGLang) 또는 `--checkpoint-every` request마다 (HF) 일어난다 (`collect_draft_model.py:131-137, 190-198`) — partial output이라도 디스크에 떨어진다는 의미일 뿐 resume 용도는 아님.

## 7. 대체 backend — HuggingFace in-process

`--server-url`을 빼면 `_generate_hf` (`collect_draft_model.py:142-204`)가 직접 모델을 메모리에 올려 `model.generate(..., do_sample=False, max_new_tokens=...)`로 그리디 디코딩. fp16, `use_cache=True`. prefix caching이 없어서 step마다 full prefill이 새로 돈다 — production용이 아닌 fallback / 소량 sanity용.

`run_parallel_draft_model.sh`는 항상 `--server-url`을 넘기므로 (`run_parallel_draft_model.sh:93`) 정상 파이프라인은 SGLang backend만 사용.

## 8. 정확성 체크리스트 (파이프라인 검증 시 확인할 것)

1. `--target-model`이 빠지지 않았는지. 빠지면 `tokenizer=None`이 되고 (`collect_draft_model.py:264-268`) `per_call_prompt_ids=None`이 되어 (`_agent_io.py:243-269`) draft LM이 system prompt / chat template 없이 raw assistant 토큰만 보고 추론하게 된다. 결과는 의미 있어 보일 수 있으나 실제 serving 분포와 다름.
2. SpecBench인데 `--dataset`을 빼먹으면 `specbench_dataset=None` → prompt 복원 skip. `run_pipeline.sh:208-211`에서 SpecBench는 `--dataset`을 강제로 추가하므로 standard path는 안전. BFCL은 `--responses`까지 같이 있어야 reconstruction 성공.
3. step 정의 일치: `_iter_steps`의 `n - pos <= 1` skip 조건이 `assemble_records.py`의 `len(future) <= 1` skip과 동일해야 한다. 둘 다 현재 코드에서 일치.
4. Tokenizer family 일치: target과 draft LM의 tokenizer vocabulary가 같은 family여야 token id가 의미 보존. 현재 preset 조합은 모두 안전.
5. Shard merge 후 record 수 = $\sum_{q,c} \max(0, |\text{tokens}_{q,c}| - 1)$ 와 일치하는지. 누락은 SGLang 호출 timeout이나 shard 프로세스 도중 종료를 시사한다. shard 실행 마지막 줄 `Done: $(wc -l < "$OUTPUT") records`에서 확인.
6. `request_id` 표현형: BFCL=`bfcl_id`, SWE-bench=`instance_id`, SpecBench=`str(question_id)`. Stage 3 lookup 키와 정확히 일치.
7. `MAX_DRAFT_MODEL_N=16` (`run_tree_oracle_sim.py:1011`)와 `--max-draft-tokens`(default 16)가 정렬돼 있어야 latency 모델과 실제 chain 길이가 일치. 이 둘을 따로 변경하면 reported speedup이 잘못된다.
