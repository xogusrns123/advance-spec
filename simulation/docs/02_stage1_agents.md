# Stage 1 Agent 별 상세

벤치마크별 agent 레퍼런스. 각 섹션은 입력 포맷, 프롬프트 구성, HTTP 호출 형태, tool-call 파싱 경로, agent loop, 실패 모드, 출력 스키마를 담는다. trajectory step 을 떨어뜨리거나 출력을 silent 하게 자를 수 있는 모든 지점은 명시적으로 호출한다.

네 가지 agent:

| Module                                    | Benchmark    | Tool API                     | Loop 구조                   |
|-------------------------------------------|--------------|------------------------------|-----------------------------|
| `simulation.agents.bfcl_agent`            | BFCLv3       | Native syntax (text decode)  | Multi-turn × multi-step     |
| `simulation.agents.bfcl_v4_agent`         | BFCLv4       | Native syntax (text decode)  | Single-turn × multi-step    |
| `simulation.agents.specbench_agent`       | SpecBench    | 없음                          | Multi-turn, tool 없음        |
| `simulation.agents.swebench_agent`        | SWE-Bench    | LangChain `bind_tools`       | Single-turn × multi-step    |

모든 agent 는 `is_oracle_enabled()` → `os.environ.get(
"SGLANG_ORACLE_VANILLA", "0") == "1"` 를 호출한다
(`simulation/oracle/oracle_patch.py:49-50`). 이 값이 꺼져 있으면 agent 는 여전히 동작하지만 각 step 의 `spec_decode` 필드가 부재 — 결과 파일은 Stage 3 입장에서 무용지물이 된다.

per-step oracle 수집 패턴은 모든 agent 가 동일하다.

```python
oracle_pos = get_oracle_log_position()        # byte offset before request
response = client.chat.completions.create(...)
oracle_entries = read_oracle_log(oracle_pos)  # entries appended since
if oracle_entries:
    step_data["spec_decode"] = {"oracle_vanilla_entries": oracle_entries}
```

`get_oracle_log_position` 은 `/tmp/sglang_oracle_vanilla.jsonl` 의 현재 `stat().st_size` 를 반환하고
(`simulation/oracle/oracle_patch.py:61-69`), `read_oracle_log` 는 그 offset 으로 seek 해서 그 이후 작성된 모든 라인을 파싱한다
(`simulation/oracle/oracle_patch.py:71-94`). 이 oracle 파일은 같은 서버의 동시 요청들 간에 공유되므로 Stage 1 에서 agent 는 `--num-workers 1` 로 실행된다.

---

## 1. `bfcl_agent.py` (BFCLv3 multi-turn)

### 1.1 입력 데이터셋 포맷

`data/bfcl_multi_turn/dataset_stratified_interleaved.jsonl`, `simulation/scripts/experiments/data_prep/` 의 prep 스크립트로 생성. 한 question 당 한 record.

> **Note**: `bfcl_v3` 는 active workload 에서 제외됨 (`feedback_drop_bfcl_v3` 메모리). agent 코드는 참조용으로 유지.

```json
{
  "question_id": 17,
  "category": "bfcl_v3/multi_turn_base",
  "bfcl_id": "multi_turn_base_17",
  "id": "multi_turn_base_17",
  "question": [[{"role": "user", "content": "..."}], [{"role": "user", "content": "..."}]],
  "function": [],                          // empty here, populated at runtime
  "initial_config": {"GorillaFileSystem": {...}},
  "involved_classes": ["GorillaFileSystem"],
  "missed_function": {"1": ["delete_file"]} // optional, miss_func category only
}
```

`question` 은 **turn 의 list** 이며 각 turn 자체가 chat 메시지의 list 다. `preprocess_bfcl_requests`
(`bfcl_agent.py:72-104`) 는 각 entry 의 `function` list 를 `involved_classes` 의 모든 클래스에 대해 `MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name]` 을 로드하여 채운다. 그런 다음 `missed_function` 의 각 turn 키에 대해 명시된 doc 들을 `function` 에서 per-turn list 로 옮긴다 — 이 doc 들은 지정된 turn 에서 `system_prompt_pre_processing_chat_model` 에 의해 system prompt 에 다시 주입된다.

### 1.2 프롬프트 구성

`run_single_request` 는 `bfcl_agent.py:146-305` 에 있다. 모든 turn 안의 모든 step 에 대해.

```python
formatted_messages = system_prompt_pre_processing_chat_model(
    messages, functions, bfcl_id
)
```

이는 `bfcl_eval.model_handler.utils` 의 BFCL 공식 helper 다. function doc 를 BFCL 의 `[func(param=val)]` 텍스트 형식으로 embed 한 system 메시지를 prepend 한다. `bfcl_agent.py:107-127` 의 `_format_prompt` helper 는 dead code (사용되지 않음 — runtime 에는 `system_prompt_pre_processing_chat_model` 만 호출됨).

### 1.3 HTTP 호출

```python
response = client.chat.completions.create(
    model=model,
    messages=formatted_messages,
    temperature=temperature,    # CLI default 0.0
    max_tokens=4096,
)
```

`bfcl_agent.py:204-210`. `max_tokens=4096` 하드코딩. `tools` 필드 없음 — BFCLv3 는 native-syntax decoding 을 사용하지 OpenAI tool_calls 를 사용하지 않는다.

### 1.4 Tool-call 파싱 경로

Native only.

```python
text_to_decode = _strip_thinking_tags(content)
decoded_calls = default_decode_execute_prompting(text_to_decode)
```

`bfcl_agent.py:243-244`. `_strip_thinking_tags`
(`bfcl_agent.py:130-143`) 는 세 가지 패턴을 다룬다: 표준 `<think>…</think>`, GLM-4.7-Flash 의 bare-prefix `…</think>` (열린 tag 없음), 그리고 다중 closing. `re.sub(r"<think>.*?</think>\s*",
"", text, flags=re.DOTALL)` 다음에, `</think>` 가 남아 있으면 `text.rsplit("</think>", 1)[-1].strip()` 를 적용한다.

`default_decode_execute_prompting` 은 BFCL 의 `[func1(p=v), func2(...)]` syntax 를 실행 가능한 문자열 list 로 파싱한다. 파싱 실패는 raise 되어 `bfcl_agent.py:245-248` 에서 catch 됨 → `step_data["decode_error"]` 기록, 이 turn 의 agent loop 를 break (turn 이 조기 종료되며, break 가 outer turn loop 가 아닌 `for step in range(max_iterations)` 안에 있으므로 남은 turn 들은 계속 실행됨).

`is_empty_execute_response(decoded_calls)` (역시 `bfcl_eval` 출신) 은 decode 된 call 이 없으면 True 반환 → `action = "end_of_turn"`, break.

### 1.5 Tool 실행

`bfcl_eval` 의 `execute_multi_turn_func_call(decoded_calls, initial_config,
involved_classes, model_name_safe, bfcl_id, long_context=long_context,
is_evaL_run=False)`
(`bfcl_agent.py:262-276`). `model_name_safe = model.replace("/", "_")`.
예외 발생 시: `step_data["exec_error"] = str(e)`, break.

성공적인 실행 직후마다 `patch_websearch_in_globals(bfcl_id)` 가 호출되어 (`bfcl_agent.py:272`) 새로 인스턴스화된 `WebSearchAPI` 가 있다면 DuckDuckGo 로 monkey-patch 한다 (`simulation/agents/tools/bfcl.py:70-77` 참조).

Tool 결과는 `{"role": "tool", "content": str(r)}` 로 append (`bfcl_agent.py:288-292`).

### 1.6 Iteration loop

Outer: `for turn_idx, turn_messages in enumerate(question)`
(`bfcl_agent.py:177`).
Inner: `for step in range(max_iterations)`
(`bfcl_agent.py:187`).
`max_iterations` 는 함수 시그니처상 default 20. RR config (`workload_overrides.bfcl_v3.max_iterations`) 로 override. turn 내부의 stop 조건:

- API call 예외 → `error` 기록, break.
- Decode 예외 → `decode_error` 기록, break.
- `is_empty_execute_response` → `action="end_of_turn"` 기록, break.
- Exec 예외 → `exec_error` 기록, break.
- end-of-turn 없이 `step` 이 `max_iterations` 도달 → silent truncation: turn 이 그냥 멈추며 마커 없음.

### 1.7 실패 모드 (모두 swallowed)

| 위치                                  | 기록 형태                  | trajectory 영향                                |
|---------------------------------------|----------------------------|------------------------------------------------|
| OpenAI API call (`bfcl_agent.py:211`) | `step_data["error"]`       | turn 종료, 이 step 에 oracle 없음              |
| Decode (`:245`)                       | `step_data["decode_error"]`| turn 종료                                      |
| Execute (`:273`)                      | `step_data["exec_error"]`  | turn 종료                                      |
| `max_iterations` 도달                 | (없음)                     | silent truncation; downstream 은 N step 만 본다 |

### 1.8 출력 스키마

per-question dict.

```json
{
  "bfcl_id": "...",
  "category": "bfcl_v3/multi_turn_base",
  "agent_metrics": {
    "steps": [...per-step dicts...],
    "total_turns": <int>,
    "total_steps": <int>
  }
}
```

per-step dict (항상 다음을 가진다).

```json
{
  "type": "llm",
  "turn": <int>,
  "step": <int>,
  "latency_s": <float>,
  "prompt_tokens": <int>,
  "completion_tokens": <int>,
  "content": "<raw assistant text incl. <think>…</think>>",
  "finish_reason": "...",
  "messages": [...formatted_messages...]    // copy.deepcopy
}
```

step 별 옵션 필드:

- `spec_decode.oracle_vanilla_entries` — EAGLE3 log entry 의 list.
- `decoded_calls`, `has_tool_calls`, `exec_results`, `exec_latency_s`
  — tool call 이 decode 및 실행됐을 때만 존재.
- `action="end_of_turn"` — `is_empty_execute_response` 일 때만 존재.
- `error` / `decode_error` / `exec_error` — error 마커.

### 1.9 Top-level metadata

`bfcl_agent.py:528-539` 의 `_meta()` closure 가 작성. shard merger 의 metadata 보존은 `01_stage1_overview.md` §2.7 참조 — 첫 샤드 metadata 가 베이스로 채택되고 합산 가능 필드 (`total_tokens`/`total_oracle_entries`/`total_tool_calls`) 만 합쳐진다.

### 1.10 Resume / per-request checkpoint

`--resume` flag (`bfcl_agent.py:581-582`). `bfcl_v4_agent.py:573-574` 와 동일 패턴. 활성화되면 `simulation.pipeline.save_results.load_checkpoint(output_file)` 가 `<output>.partial` (없으면 finalized `<output>`) 을 읽어 done set 을 만들고 입력 dataset 에서 제외 (`:464-484`). `--num-requests` 는 resume 필터 **이후** 적용. 매 result 후 `append_to_checkpoint(...)` (`:553`), 종료 시 `finalize_checkpoint(...)` (`:557`) 로 partial → final rename. **Replay 경로는 resume 미지원** (`:483-487` 의 코멘트) — Round 1 trajectory 와 1:1 짝짓기가 필요해서.

---

## 2. `bfcl_v4_agent.py` (BFCLv4 agentic)

### 2.1 입력 데이터셋 포맷

`data/bfcl_agent/dataset_stratified_interleaved.jsonl`, `simulation/scripts/experiments/data_prep/` 의 prep 스크립트로 생성. BFCLv3 record 와 같지만 옵션으로 `depends_on` 추가. 카테고리는 `AGENTIC_CATEGORY` 아래 (web_search, memory, …) 위치. Loader: `load_bfcl_v4_dataset`
(`bfcl_v4_agent.py:99-136`). `--num-requests N` 이 설정되면 loader 가 카테고리별로 stratify 하여 카테고리당 `N//len(cats)` 개를 취하고 N 으로 cap 한 뒤, 누락된 prereq dependency 를 다시 추가한다. `bfcl_eval.utils.sort_key` 로 정렬.

RR config 의 `workload_overrides.bfcl_v4.include_category` (예: `web_search`) 로 카테고리 필터링 가능 — prereq 확장이 요청 수를 부풀리지 않게 한다.

### 2.2 프롬프트 구성

`process_request` (`bfcl_v4_agent.py:139-313`). 한 conversation (per-turn loop 없음).

1. `cleanup_globals(entry_id)` 후 involved class 들을 인스턴스화하기 위해 `execute_multi_turn_func_call([], …)` 의 no-op 호출 (`bfcl_v4_agent.py:173-178`).
2. `is_memory(test_category)` 이고 helper 가 존재하면 `add_memory_instruction_system_prompt` 를 통해 memory-specific system prompt 를 prepend
   (`bfcl_v4_agent.py:182-186`).
3. `is_web_search(test_category)` 이면 기존 `WebSearchAPI` 인스턴스를 DuckDuckGo 로 monkey-patch
   (`bfcl_v4_agent.py:189-190`).
4. `system_prompt_pre_processing_chat_model(all_turn_messages[0],
   functions, entry_id)` 는 **첫 번째 turn 의 메시지에만** 적용된다 (`bfcl_v4_agent.py:193-195`). 이후의 모든 turn 은 그대로 append 된다.
5. 첫 system 메시지에 추가 문단 하나가 append 된다 (`bfcl_v4_agent.py:200-208`).

   > IMPORTANT: Be efficient. Use the minimum number of function
   > calls needed. Once you have enough information to answer,
   > respond with the final answer immediately instead of making
   > additional searches. Do NOT over-verify or repeat similar
   > queries.
   > BUT: if the question references any entity, event, or fact that
   > may be outside your training data (including anything dated
   > after your knowledge cutoff), you MUST attempt at least one
   > search_engine_query before responding. Do NOT answer 'I do not
   > know' or 'I cannot answer this question' without first trying a
   > search.

   두 번째 문단은 Qwen3-14B 가 cutoff 이후 주제를 거부하는 습관을 카운터하기 위함. replay code path 에서도 두 문단 모두 **동일하게** append 된다 (`bfcl_v4_agent.py:341-349`); 한쪽만 변경하면 replay 가 Round 1 과 분기한다.

### 2.3 HTTP 호출

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.0,             # hard-coded
    max_tokens=4096,
)
```

`bfcl_v4_agent.py:228-233`. `--temperature` CLI flag 는 **무시된다** — 0.0 하드코딩. `tools` 필드 없음.

`:225-227` 에 dead local 이 있다.

```python
formatted_prompt = system_prompt_pre_processing_chat_model(
    messages, functions, entry_id
) if not messages else messages
```

`formatted_prompt` 는 한 번도 사용되지 않음 — 실제로 보내지는 것은 `messages`.

### 2.4 Tool-call 파싱 경로

BFCLv3 와 동일 (native): `_strip_thinking(content)` 다음 `default_decode_execute_prompting`
(`bfcl_v4_agent.py:262-264`). decode 예외 발생 시 `:265-268` 의 bare `except Exception:` 이 응답을 final answer 로 간주하고 loop 를 break.

`is_empty_execute_response(decoded_calls)` → break, 역시 final answer 로 간주 (`:270-272`).

### 2.5 Tool 실행

`execute_multi_turn_func_call(decoded_calls, initial_config,
involved_classes, model_name_safe, entry_id, long_context=False,
is_evaL_run=False)` (`bfcl_v4_agent.py:281-285`).
`model_name_safe = f"bench_v4_{entry_id}"`. 성공적인 실행 직후마다 web_search 카테고리에서는 `WebSearchAPI` 를 다시 patch. 예외 시: `exec_error` 기록, break.

### 2.6 Iteration loop

단일 loop: `for step in range(max_iterations)`
(`bfcl_v4_agent.py:218`). `max_iterations` 는 함수 시그니처상 default 10. RR config 의 `workload_overrides.bfcl_v4.max_iterations` 로 override. cap 도달은 silent (마커 없음).

### 2.7 Resume / checkpoint 동작

`bfcl_v4_agent.py:421-475`. v4 (그리고 SpecBench / SWE-Bench) 에 고유 — BFCLv3 는 resume 이 없다.

- `--resume` 은 `simulation.pipeline.save_results.load_checkpoint` 를 통해 `<output_file>.partial` (또는 partial 이 없으면 finalized 파일) 을 읽는다. 이미 완료된 ID (`bfcl_id` 다음 `id` 로 매칭) 는 skip.
- `--num-requests` 는 resume 필터 **이후에** 적용된다. 따라서 coordinator 가 `--num-requests 1 --resume` 를 반복 호출하면 호출 한 번에 새 request 한 개씩 진행된다.
- 매 request 완료 후 `append_to_checkpoint` 가 `<output>.partial` 을 atomic 하게 다시 쓴다.
- 실행 완료 시 `save_agent_results` 가 final 파일을 쓰고 partial 은 unlink.

### 2.8 실패 모드

| 위치                                   | 기록 형태                          | 영향                                |
|----------------------------------------|------------------------------------|-------------------------------------|
| API call (`:234`)                      | `{"type":"llm","step":i,"error"}`  | loop break, request 종료            |
| Decode (`:265`)                        | (없음 — final answer 로 간주)      | loop 정상 break                     |
| Exec (`:288`)                          | `step_data["exec_error"]`          | loop break                          |
| Empty `conversation_turns` (`:165`)    | top-level `error` 필드             | `agent_metrics.steps` 없음          |
| `max_iterations` 도달                  | (없음)                             | silent truncation                   |

### 2.9 출력 스키마

```json
{
  "bfcl_id": "...",
  "category": "bfcl_v4/web_search",
  "agent_metrics": {"steps": [...], "total_steps": <int>}
}
```

step dict 는 BFCLv3 에서 `turn` 과 `finish_reason` 을 뺀 것과 같다. 옵션 필드도 동일 (`spec_decode`, `decoded_calls`, `has_tool_calls`,
`exec_results`, `exec_latency_s`, `error`, `exec_error`).

---

## 3. `specbench_agent.py` (SpecBench / MT-Bench)

### 3.1 입력 데이터셋 포맷

`data/specbench/dataset.jsonl`. 각 record.

```json
{"question_id": "<id>", "category": "<cat>", "turns": ["user msg 1", "user msg 2"]}
```

Loader: `load_specbench_dataset` (`specbench_agent.py:44-57`).

### 3.2 프롬프트 구성

system prompt 없음, tool 없음. 각 turn 마다.

```python
messages.append({"role": "user", "content": user_msg})
```

(`specbench_agent.py:85`). 모델이 각 turn 에 답하면서 conversation 이 누적된다.

### 3.3 HTTP 호출

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,    # default 0.0
    max_tokens=max_tokens,      # default 2048; CLI --max-tokens override
)
```

`specbench_agent.py:88-94`. `--max-tokens` CLI flag 를 노출하는 **유일한** agent. RR config 의 `workload_overrides.specbench.max_tokens_override` (또는 longbench_* 의 `max_tokens`) 가 `--max-tokens` 로 변환되어 전달.

### 3.4 Tool 파싱

없음. 응답은 messages 에 그대로 append 된다 (`specbench_agent.py:105`).

### 3.5 Iteration loop

`turns` 위로 단일 loop (`specbench_agent.py:81`). turn 당 LLM call 한 번, agent loop 없음, tool 없음, 조기 stop 없음.

### 3.6 실패 모드

API 예외 (`specbench_agent.py:95-100`) 는 `{"response": "", "error": str(e)}` 를 기록하고 다음 turn 으로 `continue`. conversation 은 빈 assistant 메시지로 업데이트되지 **않으므로**, 다음 turn 은 이전 user 메시지 직후에 새 user 메시지가 따라오는 형태를 본다. clean run 대비 이후 turn 이 어긋난다.

### 3.7 출력 스키마

```json
{
  "question_id": "...",
  "category": "...",
  "turns": [
    {
      "response": "...",
      "latency_s": <float>,
      "prompt_tokens": <int>,
      "completion_tokens": <int>,
      "spec_decode": {"oracle_vanilla_entries": [...]}   // optional
    },
    ...
  ],
  "total_oracle_entries": <int>,
  "total_tokens": <int>
}
```

### 3.8 Resume / checkpoint

BFCLv4 와 동일 패턴 (`specbench_agent.py:246-265, 304-316`). Done ID 는 `question_id` 로 매칭.

---

## 4. `swebench_agent.py` (SWE-Bench, LangChain)

LangChain 의 `ChatOpenAI.bind_tools` 를 통해 OpenAI `tool_calls` 필드를 사용하는 유일한 agent. SGLang 서버에 `--tool-call-parser qwen25` 를 설정하는 모든 이유는 바로 이 agent 때문이다 — 없으면 Qwen3 가 LangChain 이 파싱하지 못하는 text-only `<tool_call>…</tool_call>` 을 반환하고, agent 는 iteration 0 에서 `ai_msg.tool_calls = []` 를 보게 되어 trajectory step 한 개로 loop 가 종료된다.

### 4.1 입력 데이터셋 포맷

`data/swebench/dataset.jsonl`, 스키마는 `docs/01_stage1_overview.md` §3.3 에 문서화. agent 는 `instance_id`, `repo`, `base_commit`, `turns[0]` 만 읽는다.

### 4.2 Repository setup

`_setup_repo(workdir, repo, base_commit)`
(`swebench_agent.py:98-134`). 각 instance 마다.

1. `<repos_dir>/<instance_id>/.git` 가 존재하면:
   - `git reset --hard <base_commit>` (timeout 30s, check=True)
   - `git clean -fd` (timeout 30s)
   - `CalledProcessError` 발생 시 `shutil.rmtree(workdir)` 후 clone fall through.
2. 없으면: `git clone https://github.com/<repo>.git <workdir>` (timeout 300s) 다음 `git checkout <base_commit>` (timeout 30s).
3. 모든 clone 실패 시: error 문자열 반환. instance 결과는 error stub
   `{"instance_id":..., "error":..., "turns":[], "agent_metrics": {"steps":[]}}`
   가 된다 (`swebench_agent.py:194-201`) — oracle data 없음.

`_cleanup_repos(repos_dir, base_commits)`
(`swebench_agent.py:137-159`) 는 batch 시작 전과 종료 후 각각 한 번씩 호출된다 (`swebench_agent.py:514-516, 561-563`). `repos_dir` 의 모든 하위 디렉토리를 돌며 `git reset --hard <base_commit>` (실패 시 `HEAD` 로 fallback) + `git clean -fd` 를 둘 다 10s timeout 으로 실행. 실패 시 `WARN: cleanup failed for <name>: <e>` 출력 후 계속 진행.

### 4.3 Tool 생성

```python
if tool_style == "sweagent":
    tools = create_sweagent_tools(workdir, repo=repo)
else:
    tools = create_swebench_tools(workdir, repo=repo)
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
```

`swebench_agent.py:204-209`. `tool_style` default 는 `"full"`
(`create_swebench_tools` — 6 개 tool). `parallel_tool_calls=False` 가 LLM step 당 tool call 한 개를 강제한다.

전체 tool semantics 는 `03_stage1_tools_and_io.md` 참조.

### 4.4 프롬프트 구성

`swebench_agent.py:55-76` 의 하드코딩된 `SYSTEM_PROMPT` (전문: "You are an expert software engineer…", TOOLS / STRATEGY / RULES 섹션 포함). 초기 메시지.

```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=request["turns"][0]),  # = problem_statement
]
```

`swebench_agent.py:212-215`.

### 4.5 HTTP 호출

`ChatOpenAI` (`swebench_agent.py:498-504`):

```python
llm = ChatOpenAI(
    base_url=url,
    model=model,
    api_key="dummy",
    temperature=temperature,    # default 0.0
    max_tokens=4096,
)
```

step 마다: `ai_msg = llm_with_tools.invoke(messages)`
(`swebench_agent.py:228`). LangChain 은 tool list 를 OpenAI `tools` 필드로 직렬화하고, SGLang 서버의 `--tool-call-parser qwen25` 가 Qwen3 의 native `<tool_call>` 출력을 LangChain 이 기대하는 OpenAI `tool_calls` 배열로 다시 변환한다.

### 4.6 Tool-call 파싱 경로

OpenAI `tool_calls`: 각 entry 는 LangChain 파싱 후 `{"name": str, "args": dict, "id": str}` 형태. step 당 single-call 은 `parallel_tool_calls=False` 와 `swebench_agent.py:276-281` 의 명시적 가드 양쪽으로 강제된다.

```python
if len(ai_msg.tool_calls) > 1:
    ai_msg = AIMessage(content=ai_msg.content,
                       tool_calls=[ai_msg.tool_calls[0]])
    messages[-1] = ai_msg
```

(Llama 3.1 호환성 — Llama 는 multi-call 을 emit 하지만 agent 는 첫 번째만 실행.)

### 4.7 Tool 실행

```python
for tc in ai_msg.tool_calls:
    if tool_name in tool_map:
        result = tool_map[tool_name].invoke(tc["args"])
    else:
        result = f"[ERROR] Unknown tool: {tool_name}"
    messages.append(ToolMessage(content=str(result),
                                tool_call_id=tc["id"]))
```

`swebench_agent.py:284-297`. per-tool 예외는 catch 되어 `[ERROR] {e}` 문자열로 변환된다 — agent loop 를 break 하지 않는다.

### 4.8 Iteration loop

`for iteration in range(max_iterations)`
(`swebench_agent.py:222`). `max_iterations` 는 함수상 default 15. RR config 의 `workload_overrides.swebench_verified.max_iterations` 로 override (현재 mango3/mango1 RR 에서는 250).

Stop 조건:
- `not ai_msg.tool_calls` (`:270-271`) — 모델이 text-only 응답을 생성.
- `any(tc["name"] == "submit" for tc in ai_msg.tool_calls)`
  (`:272-273`) — `tool_style="sweagent"` 일 때만 의미가 있음 (`submit` tool 은 거기에만 존재).
- Iteration cap 도달 — silent truncation.

전체 loop 본체는 `try: … except Exception as e:` 로 감싸진다
(`:221, 299-306`). uncaught 예외 발생 시 agent 는 그때까지 수집된 step 들과 함께 error stub 을 반환하며, `turns_with_messages` 가 `turns` 로 들어간다.

### 4.9 Step 기록

각 LLM call 후 agent 는 type→role map `{"human": "user", "ai": "assistant", "system": "system",
"tool": "tool"}` 를 사용하여 **들어오는** LangChain 메시지 list 를 OpenAI 포맷으로 다시 직렬화한다 (`swebench_agent.py:232-238`). 이것이 downstream 프롬프트 재구성을 위해 `step_data["messages"]` 에 deep-copy 되는 내용이다. 결정적으로, 이 LLM call 이 생성한 assistant `ai_msg` 는 이 snapshot 에 **없다** — `messages.append(ai_msg)` 는 `:259` 의 snapshot 이후에 일어난다.

직렬화된 `tool_calls` 는 `turns_with_messages[i]["tool_calls"]` 에 별도로 저장된다 — 이들은 출력에 살아남고, `_agent_io._reconstruct_swebench_prompts` 는 prompt-id 재구성을 위해 `pipeline/_agent_io.py:71-94` 의 `_langchain_to_openai_messages` 를 통해 OpenAI tool_calls 로 다시 변환한다.

### 4.10 실패 모드

| 위치                                | 기록 형태                                            | 영향                                |
|-------------------------------------|------------------------------------------------------|-------------------------------------|
| Repo setup (`:193`)                 | top-level `error` 필드                               | `agent_metrics.steps` 가 비어 있음  |
| LangChain `invoke` (모든 iteration) | top-level `error`, partial `turns`/`steps` 반환      | loop 종료                           |
| per-tool 실행                       | `ToolMessage` content `[ERROR] {e}`                  | loop 계속                           |
| 알 수 없는 tool name                | `ToolMessage` content `[ERROR] Unknown tool: ...`    | loop 계속                           |
| `max_iterations` 도달               | (없음)                                               | silent truncation                   |

### 4.11 출력 스키마

```json
{
  "instance_id": "...",
  "category": "<version>",
  "num_turns": <int>,
  "total_latency": <float>,
  "output": "<final assistant content>",
  "turns": [
    {
      "messages": [...serialized langchain messages with tool_calls/tool_call_id...],
      "tool_calls": [...],
      "latency": <float>,
      "response": "..."
    },
    ...
  ],
  "agent_metrics": {
    "steps": [
      {
        "type": "llm",
        "turn": 0,
        "step": <iteration>,
        "latency_s": <float>,
        "content": "...",
        "has_tool_calls": <bool>,
        "messages": [...openai format snapshot before this step...],
        "spec_decode": {"oracle_vanilla_entries": [...]}   // optional
      },
      ...
    ],
    "total_steps": <int>
  }
}
```

### 4.12 Patch 추출 (post-loop)

`swebench_agent.py:578-601`. 모든 question 처리 후 각 instance repo 에서 `git diff HEAD` (timeout 10s) 를 실행하고 비어 있지 않은 diff 를 출력 파일 옆 `patches.json` 에 저장한다. SWE-Bench scoring 용이며 Stage 2/3 는 소비하지 않는다.

### 4.13 Resume / checkpoint

동일한 `<output>.partial` 패턴 (`swebench_agent.py:469-487, 558`).
Done ID 는 `instance_id` 로 매칭. `--resume` 과 partial 이 있으면 `_setup_repo` 가 이미 완료된 instance 를 완전히 skip 하므로 그들의 repo 상태는 perturb 되지 않는다.

### 4.14 Replay 경로 (Stage 1 범위 밖)

네 agent 모두 Round 1 의 trajectory 를 재구성하고 각 prompt 를 다른 speculative backend (보통 MTP) 로 재발행하여 그 draft tree 만 캡처하는 `--replay` 모드를 가지고 있다. canonical pipeline 의 Stage 1 은 replay 를 호출하지 **않는다**; 이 모드는 ad-hoc round-2 수집용이다. 비교 도구를 연구한다면 agent 들의 `replay_*` 함수 참조.

### 4.15 mini-swe-agent tool style (`--tool-style minisweagent`)

세 가지 tool style 중 하나 (`full` / `sweagent` / `minisweagent`). 기본 `full` 은 `swebench.py:create_swebench_tools` (6 개 tool: bash + file_view/read/write/str_replace + search). `sweagent` 는 `create_sweagent_tools` (3 개: bash + str_replace_editor + submit). `minisweagent` 는 `create_minisweagent_tools` (bash 1 개만; mini-swe-agent 의 공식 SWE-Bench config 와 호환).

mini-swe-agent 는 **submit tool 이 없다.** 대신:

- **System prompt** 는 1 줄 (`MINISWEAGENT_SYSTEM_PROMPT`, `swebench_agent.py:88-91`).
- **Instance template** (`MINISWEAGENT_INSTANCE_TEMPLATE`, `:93-199`) 가 PR description 을 wrap 한다. `{task}` placeholder 를 `str.format` 대신 `.replace()` 로 치환 — template 에 `{"command": "ls"}` 같은 JSON 예제가 있어서 format 으로 파싱하면 KeyError.
- `parallel_tool_calls=True` (`:337`) — mini-swe-agent 는 multi-tool-call response 를 허용. 다른 style 들은 single-call truncation (`:439`).
- **공식 submission 프로토콜**: `bash` tool 안에서 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` 를 실행. agent loop (`:466-473`) 가 그 sentinel 문자열을 감지해 tool 결과를 messages 에 append 한 뒤 `break`. LLM 한 턴 더 안 부르고 깔끔히 종료.
- **Fallback 종료**: 빈 `tool_calls` 도 종료 신호로 처리 (`:419-422` 의 `has_submit_tool = "submit" in tool_map` 분기). `full`/`sweagent` 와 달리 nudge HumanMessage 를 안 보냄.

### 4.16 Resume / per-request checkpoint

`--resume` flag (`:573-574`). 활성화되면 `simulation.pipeline.save_results.load_checkpoint(output_file)` 로 `<output>.partial` (없으면 finalized `<output>`) 을 읽어 처리 완료된 `instance_id` 의 `done` set 을 만들고 입력 dataset 에서 제외 (`:507-540`). `--num-requests` 는 resume 필터 **이후** 적용되므로 round-robin coordinator 가 `--num-requests 1 --resume` 으로 반복 호출하면 매번 새 instance 1 개씩 진행한다.

매 result 후 `append_to_checkpoint(output_file, result, _meta())` (`:559`) 가 partial JSONL 을 atomic write (tmpfile + `os.replace`) 로 다시 쓴다 — 중단/재개 안전하지만 N 개 request 처리 시 매 step 마다 전체 partial 을 rewrite 하므로 IO 가 O(N²) 누적 (`save_results.py:88` 주석에서 "수십 MB 까진 OK").

`bfcl_v4_agent.py:417-579`, `specbench_agent.py:215-360`, `bfcl_agent.py:428-595` 모두 동일 패턴으로 resume 지원. 네 agent 모두 일관됨.

---

## 5. Cross-agent invariant

- 네 agent 모두 `simulation.pipeline.save_results.save_agent_results` 를 통해 최종 저장 JSON 을 구성한다. 이 함수는 full 파일과 `_response.json` light 파일 (oracle entry 가 카운트로 대체된 사본) 을 둘 다 쓴다. `03_stage1_tools_and_io.md` §3 참조.
- 네 agent 모두 per-question 출력을 `bfcl_id`, `instance_id`, `question_id` 중 하나로 키잉한다. `pipeline/_agent_io.py:215-217` 의 `_extract_bfcl` 은 읽을 때 이 세 키를 그 순서로 fall back 한다.
- `messages` snapshot 은 LLM step 별 (BFCL agent) 또는 turn 별 (SWE-Bench, `turns_with_messages` 안) 로 저장된다. Stage 3 는 suffix-cache trie 용으로 prompt 를 re-tokenize 하기 위해 이들이 필요하다. SpecBench 는 per-step `messages` 필드가 없고 대신 `dataset.turns[]` 에서 재구성된다 (`pipeline/_agent_io.py:167-182`).
- 하드코딩된 `max_tokens` (BFCL v3/v4 는 4096, SWE-Bench 는 4096, SpecBench 는 2048) 가 유일한 출력 길이 cap. `</think>` 등에 기반한 동적 stopping 은 없으며, thinking text 는 `completion_tokens` 에 빌링되고 `content` 에 기록된다.
