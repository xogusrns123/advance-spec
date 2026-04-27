# Stage 1 Tool 및 I/O

BFCL 및 SWE-Bench agent 가 사용하는 tool 구현, 그리고 모든 Stage 1 agent 가 쓰는 read/write/checkpoint helper 와 Stage 2/3 가 소비하는 `_agent_io` consumer 에 대한 레퍼런스.

## 1. BFCL tool 패치 — `simulation/agents/tools/bfcl.py`

BFCL agent (`bfcl_agent.py`, `bfcl_v4_agent.py`) 는 `bfcl_eval.eval_checker.multi_turn_eval.execute_multi_turn_func_call` 을 호출하고, 이 함수는 per-test class 객체 (예: `WebSearchAPI`, `MemoryAPI`, `GorillaFileSystem`) 를 `multi_turn_utils` 모듈 globals 에 인스턴스화한다. 이 파일은 유료 API 를 피하기 위해 그 클래스들을 monkey-patch 하고, cleanup helper 를 제공한다.

### 1.1 `_ddg_search_engine_query` (`bfcl.py:16-51`)

`WebSearchAPI.search_engine_query` 의 대체. `ddgs.DDGS` 를 import (legacy `duckduckgo_search.DDGS` 로 fallback) 한 뒤 `DDGS().text(keywords, region=region, max_results=max_results)` 를 실행하고, 각 결과를 `{title, href, body}` 로 사영한다. 인스턴스가 `show_snippet=False` 이면 `body` 를 떨군다.

에러는 swallow 된다: `except Exception as e: return {"error": str(e)}`. 즉 네트워크 실패나 DDG rate limit 이 `{"error": "..."}` content 를 가진 성공한 tool 결과처럼 보인다 — agent 입장에서는 tool 실패가 아니라 tool reply 로 받아들여지고, BFCL evaluator 는 해당 question 이 tool output 을 가졌다고 마킹한다. **정확성 함의**: 실패한 search 도 step 을 소비하고 trajectory token 을 추가한다; 모델은 보통 retry 하지 않고 에러 메시지 텍스트에 응답한다.

### 1.2 `patch_websearch_class` (`bfcl.py:54-67`)

모듈 레벨 idempotent 클래스 패치. `bfcl_agent.py` import 시점에 1회 호출됨 (`bfcl_agent.py:55`) — 첫 번째 인스턴스가 패치되도록 반드시 어떤 `execute_multi_turn_func_call` 보다 **먼저** 실행돼야 한다.

`bfcl_v4_agent.py` 는 이를 호출하지 **않는다** — `patch_websearch_in_globals` (instance-level patching, 클래스 인스턴스 생성 후 호출) 에만 의존한다.

### 1.3 `patch_websearch_in_globals(entry_id)` (`bfcl.py:70-77`)

globals key 가 sanitized `entry_id` 를 포함하는 모든 `WebSearchAPI` 인스턴스를 다시 패치한다 (BFCL 은 per-test 인스턴스를 `multi_turn_utils` 모듈 globals 에 `WebSearchAPI_<entry>_instance` 같은 이름으로 박아 둔다). BFCL 프레임워크가 trajectory 중간에 인스턴스를 재생성하는 경우를 방어하기 위해 매 tool 실행 직후 호출된다. Sanitization 은 `:73`: `re.sub(r"[-./:]", "_", entry_id)`.

### 1.4 `cleanup_globals(entry_id)` (`bfcl.py:80-88`)

per-test 인스턴스 객체를 `multi_turn_utils` globals 에서 삭제한다. 각 request 종료 시 (`bfcl_agent.py:295`, `bfcl_v4_agent.py:304`) 와 `bfcl_v4` request 시작 시 (`bfcl_v4_agent.py:173`) 호출되어 같은 test class 를 공유하는 테스트 사이의 상태 누설을 막는다.

## 2. SWE-Bench tool — `simulation/agents/tools/swebench.py`

두 개의 tool 팩토리. 둘 다 file 작업을 `workdir` 로 스코프하고, `repo` 를 agent 가 제공하는 path 에서 자동 strip 되는 prefix 리스트로 resolve 한다 (LLM 은 `foo.py` 대신 `astropy/astropy/foo.py` 를 자주 쓴다).

### 2.1 `create_swebench_tools(workdir, repo)` (`swebench.py:19-329`)

기본값 (`tool_style="full"`) 으로 사용되는 6개 tool 을 반환:

#### `_safe_path(path)` (`swebench.py:43-65`)

`path` 를 resolve (relative → `workdir` 기준 absolute) 하고 escape 를 거부 (`raise ValueError("Path escapes working directory: …")`) 한다. resolved path 가 존재하지 않으면 알려진 repo prefix (예: `astropy/astropy/`, `astropy/`) 를 하나씩 strip 해 보며 복구를 시도한다.

#### `bash(command)` (`swebench.py:67-110`)

- python 을 `shutil.which("python3") or shutil.which("python") or "python3"` 로 탐지. `python …` → `<python_exe> …` 로 rewrite.
- Env: 모든 환경변수 상속 + `GIT_TERMINAL_PROMPT="0"` 추가 (git prompt 가 hang 되지 않도록).
- `subprocess.run(command, shell=True, cwd=workdir, capture_output=True, text=True, timeout=60, env=env)`.
- Output: `stdout + stderr + "\n[exit code N]"` (둘 다 비어 있으면 `[exit code N]` 만), **4000 자로 truncate**.
- `TimeoutExpired` → `"[ERROR] Command timed out after 60 seconds."`.
- 그 외 exception → `"[ERROR] {e}"`.

#### `file_view(path, view_range)` (`swebench.py:112-169`)

디렉토리: 알파벳순 listing, 디렉토리는 `/` 접미. 파일: 번호 매긴 라인 `f"{i:>6}\t{line}"`. `view_range` 가 없으면 default range = 첫 100 라인. Output 은 8000 자 truncate. `view_range=(start_line, end_line)` 은 1-based 이며 `start = max(0, start_line - 1)`, `end = min(len(lines), end_line)`.

#### `file_read(path, start_line, end_line)` (`swebench.py:171-203`)

호환성 wrapper. 같은 numbering 포맷, default range = 파일 전체. 8000 자 truncate.

#### `file_write(path, content)` (`swebench.py:205-238`)

파일 전체 덮어쓰기. `mkdirs(dirname)` 자동 수행. `"Successfully wrote N bytes to <path>"` 반환.

기존 파일을 원본 크기의 50% 미만 content 로 덮어쓰면 안전 경고:
`" [WARNING] Original file was X bytes but new content is only Y bytes (Z% of original). You may have accidentally deleted important code. Consider using file_str_replace for targeted edits instead."`.

#### `file_str_replace(path, old_str, new_str)` (`swebench.py:240-292`)

- 파일 read, `old_str` 출현 횟수 카운트.
- 0회 → `[ERROR] Pattern not found in <path>. Read the file first with file_read to see its current content.`
- 1회 초과 → `[ERROR] Pattern appears N times … Include more surrounding context in old_str to make it unique. Occurrences found at: <up to 5 lines with line numbers>`
- 정확히 1회 → 치환 후 write; `Successfully replaced in <path> (X lines removed, Y lines added)` 보고.

#### `search(pattern, path, file_pattern)` (`swebench.py:294-327`)

`grep -rn --include=<file_pattern or '*'> <pattern> <safe_path>` 를 `timeout=30, cwd=workdir` 로 wrap. 출력 path 에서 workdir prefix 제거. 8000 자 truncate. 매치 없음 → `"No matches found for pattern: <pattern>"`.

### 2.2 `create_sweagent_tools(workdir, repo)` (`swebench.py:335-620`)

SWE-agent-LM training schema (SWE-smith) 와 호환되는 3개 tool (`bash`, `str_replace_editor`, `submit`) 반환.

6-tool 변종과의 차이점:

- `bash` 가 interactive command (`python`, `python3`, `ipython`, `bash`, `sh`, `vim`, `vi`, `emacs`, `nano`, `nohup`) 와 prefix-block (`vim `, `vi `, `emacs `, `nano `, `gdb `, `less `, `tail -f `, `nohup `) 을 차단. `[ERROR] Interactive command '...' is not allowed.` 반환.
- `bash` timeout 120s (60s 대비).
- `bash` output truncation: 16000 자; 초과 시 head/tail split 후 `... (truncated) ...` separator.
- `str_replace_editor(command, …)` 는 다섯 개 subcommand 를 가지는 multiplexed tool: `view` (파일 또는 2-deep dir tree), `create` (파일이 이미 있으면 에러), `str_replace`, `insert`, `undo_edit` (per-file in-memory undo stack via `_file_history`). Output snippet 은 edit point 주변 ±3 라인 표시. 16000 자 truncate.
- `submit()` 은 `"Submission successful."` 반환. agent loop 가 이를 체크하고 break (`swebench_agent.py:432-434`).

### 2.3 `create_minisweagent_tools(workdir, repo)` (`swebench.py:622-678`)

Mini-swe-agent 의 공식 SWE-Bench config (`benchmarks/swebench.yaml`) 에 매칭되는 **bash 1 개만** 반환. submit tool 없음.

- `bash` 구현은 `create_swebench_tools` 의 6-tool 변종과 사실상 동일 (`shutil.which("python3")` 으로 python rewrite, `GIT_TERMINAL_PROMPT="0"`, `subprocess.run(timeout=60)`, output `[:4000]` truncate).
- 종료 신호는 두 가지: (a) 빈 `tool_calls` (mini-swe-agent 스타일은 "no further tool calls" 가 곧 submit), (b) bash command 안에 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` 가 들어 있으면 agent loop (`swebench_agent.py:466-473`) 가 그 sentinel 을 잡아서 break. 두 경로 모두 system prompt + instance template 에 명시.
- `parallel_tool_calls=True` 만 mini-swe-agent 에 활성 (`swebench_agent.py:337`). 다른 style 은 single-call truncation.

### 2.4 `create_swebench_tools` vs `create_sweagent_tools` vs `create_minisweagent_tools` — 언제

`swebench_agent.py:794-795` 의 `--tool-style {full,sweagent,minisweagent}` 로 선택. Default `full`.

- `full` (6-tool): 일반 LLM (Qwen3-8B/14B/32B, Llama3-8B 등) 의 baseline 옵션. RR config 의 `workload_overrides.swebench_verified.tool_style` 미지정 시 default.
- `sweagent` (3-tool): SWE-agent / SWE-smith fine-tune 모델 전용.
- `minisweagent` (1-tool): mini-swe-agent 의 공식 reference setup 과 동일한 bash-only ReAct loop. Phase 5 swebench 비교 실험용.

## 3. 결과 쓰기 — `simulation/pipeline/save_results.py`

네 개의 agent 모두 `run_benchmark` 종료 시점에 `save_agent_results(output, output_path)` 를 정확히 한 번 호출한다. BFCLv4, SpecBench, SWE-Bench 는 추가로 매 request 후 `append_to_checkpoint` 를 호출한다.

### 3.1 `_atomic_write_json(data, path)` (`save_results.py:14-28`)

`path` 와 같은 디렉토리에 `tempfile.mkstemp` 으로 write 한 뒤 `os.replace(tmp, path)`. 예외 발생 시 tmp 파일 삭제. write 도중 crash 가 일어나도 visible 파일이 절대 corrupt 되지 않는다 — reader 는 항상 이전 버전이거나 완전한 새 버전 중 하나를 본다.

### 3.2 `checkpoint_path(output_path)` (`save_results.py:31-34`)

`<output>.partial` 반환. `*_agent.py` 의 resume code path 가 사용한다.

### 3.3 `load_checkpoint(output_path)` (`save_results.py:37-62`)

읽기 순서: `<output>.partial` 우선, 다음 `<output>` 본체. partial 의 corrupt JSON → silent 하게 폐기하고 checkpoint 없음으로 처리; finalized JSON 의 corrupt → `None` 반환. finalized 파일을 반환한다는 것은, coordinator 가 `--num-requests N --resume` 을 반복 호출할 때 partial 이 unlink 된 뒤에도 invocation 사이에 진행이 이어진다는 의미다.

### 3.4 `done_ids(checkpoint, id_keys)` (`save_results.py:65-77`)

`checkpoint["questions"]` 를 loop 하면서 question 별로 `bfcl_id`, `instance_id`, `question_id` 중 처음 존재하는 값을 yield. agent 에서는 사용하지 않지만 (agent 들은 자체 per-id resume logic 을 구현) caller 에게 노출된다.

### 3.5 `append_to_checkpoint(output_path, question, metadata)` (`save_results.py:80-93`)

checkpoint 를 load (없으면 `{metadata: {}, questions: []}`) 하고 `question` 을 append, `metadata` 가 주어지면 set 한 뒤 atomic 하게 다시 write. per-request rewrite 비용은 oracle entries 가 dominant 하지만 partial 은 수십 MB 안에 머무르므로 acceptable.

### 3.6 `save_agent_results(data, output_path)` (`save_results.py:115-157`)

**두 개의 파일** 을 쓴다:

1. `<output>.json` — 전체 데이터 그대로 (파이프라인 artifact). Atomic.
2. `<output>_response.json` — oracle 제거 (`oracle_entries_count` 로 대체) 된 사람용 사본. `copy.deepcopy(data)` 후 in-place mutation:
   - BFCL/SWE-Bench 포맷: `q["agent_metrics"]["steps"][i]` 를 walk 하며 `pop("spec_decode")` 후 `oracle_entries_count` 로 치환.
   - SpecBench 포맷: `q["turns"][i]` 를 walk 하며 동일 처리.
3. 사이즈 출력 (`Full: <name> (X KB)` / `Response: <name> (Y KB)`).

Stage 2/3 은 full 파일을 읽는다. response 파일은 사람 검수 전용.

### 3.7 `finalize_checkpoint(output_path, metadata)` (`save_results.py:96-112`)

partial load → `save_agent_results` 호출 → partial unlink. agent 들이 run 종료 시 방어적 cleanup 으로 호출하지만, 직접 `save_agent_results` + `unlink` 도 같이 호출한다.

## 4. agent 결과 읽기 — `simulation/pipeline/_agent_io.py`

이 모듈은 Stage 2 (`collect_draft_model.py`) 와 Stage 3 (`assemble_records.py`) 에서 소비된다; Stage 1 자체는 사용하지 않는다. 여기서 문서화하는 이유는, Stage 1 이 보존해야 하는 on-disk schema invariant 를 이 모듈이 canonicalize 하기 때문이다.

### 4.1 `extract_requests(data, exclude_ids, dm_by_id, tokenizer, …)` (`_agent_io.py:185-208`)

per-question 포맷 감지:

- `agent_metrics` 존재 → BFCL / SWE-Bench 경로 (`_extract_bfcl`).
- 그 외 → SpecBench 경로 (`_extract_online`).

혼합 포맷 question 리스트 지원 — 각 question 이 독립적으로 분류된다.

### 4.2 `_extract_entries(entries)` (`_agent_io.py:38-71`)

> 반환값은 7-tuple: `(call_tokens, call_eagle3s, call_eagle3_trees, call_eagle3_tree_p_ts, call_eagle3_tree_path_draft_p_ts, call_mtp_trees, call_eagle3_pool_fulls)`. 마지막 `call_eagle3_pool_fulls` 가 capture-full-pool 모드용으로 추가됨 — 다른 모드에선 모든 항목이 `None` 이라 `_extract_bfcl` / `_extract_online` 의 `has_pool_fulls` 가드가 false 가 되어 req dict 에 attach 되지 않는다.

`oracle_vanilla_entries` 를 walk 하면서 per-LLM-call 데이터를 concat. 먼저, concurrent request 의 entry 들을 가장 빈번한 `req_id` 만 남기는 방식으로 **deinterleave** 한다:

```python
if entries and len(set(e.get("req_id","") for e in entries)) > 1:
    primary_rid = Counter(...).most_common(1)[0][0]
    entries = [e for e in entries if e.get("req_id") == primary_rid]
```

Stage 1 은 항상 `--num-workers 1` 로 도므로 이 guard 는 보통 no-op 이지만, misconfigured run 에 대비한 보호장치다.

살아남은 각 entry 에서 다음을 추출:

| 필드                         | 출처                                   | 의미                                                      |
|------------------------------|----------------------------------------|-----------------------------------------------------------|
| `tokens`                     | `e["tokens"][0]` (non-empty 시)        | Verified next-token (oracle vanilla 가 step 당 1개 강제)  |
| `eagle3`                     | `e["eagle3"][0]` (non-empty 시)        | 그 step 에서 emit 된 flat draft-token list                |
| `eagle3_tree`                | `e["eagle3_tree"]`                     | `{"token_ids":[...], "parents":[...]}` 트리               |
| `eagle3_tree_p_t`            | `e["eagle3_tree_p_t"]`                 | 트리 노드별 parent acceptance probability                  |
| `eagle3_tree_path_draft_p_t` | `e["eagle3_tree_path_draft_p_t"]`      | Draft-LM acceptance probability path                      |
| `mtp_tree`                   | `e["mtp_tree"]`                        | MTP draft tree (MTP-replay 모드에서만 set)                |
| `eagle3_pool_full`           | `e["eagle3_pool_full"]`                | `SGLANG_CAPTURE_FULL_POOL=1` 일 때만 존재. `{parent_list, draft_tokens, path_probs, pool_size}` — Stage 3 의 `pool_reslicer` 가 (s', k') sub-tree 를 만들 때 입력. 자세한 건 `09_pool_reslicer.md` |

### 4.3 BFCL/SWE-Bench extractor (`_agent_io.py:211-299`)

per question:

- `bfcl_id` 를 `q["bfcl_id"]` → `q["instance_id"]` → `str(q["question_id"])` 순으로 resolve.
- `q["agent_metrics"]["steps"]` 를 loop 하며 `spec_decode` 블록이 있는 step 에 대해서만 per-call data 추출.
- id 가 `exclude_ids` 에 있는 question 은 통째로 skip.

prompt-id 재구성 (tokenizer 가 주어진 경우):

1. 각 step 에 저장된 `s["messages"]` 스냅샷을 우선 사용 (BFCL agent 와 SWE-Bench 의 `step_data["messages"]` 스냅샷 — `bfcl_agent.py:228`, `bfcl_v4_agent.py:248`, `swebench_agent.py:247`). chat template 적용, encode, append.
2. Fallback A: SWE-Bench `turns[i]["messages"]` — `_reconstruct_swebench_prompts` (`_agent_io.py:97-114`). LangChain 포맷 (`messages[0]["type"]` 존재, `role` 부재) 을 감지하고 `_langchain_to_openai_messages` 로 변환.
3. Fallback B: BFCL — `bfcl_dataset` 와 별도의 `resp_by_id` 매핑 필요; `system_prompt_pre_processing_chat_model` 을 통해 user turn + missed_function injection 을 replay 하여 재구성 (`_reconstruct_bfcl_prompts`, `_agent_io.py:117-164`).

### 4.4 `_langchain_to_openai_messages(messages)` (`_agent_io.py:71-94`)

`{"system","human","ai","tool"}` → `{"system","user","assistant","tool"}`. `tool_calls` 를 가진 assistant 메시지는 OpenAI 형태로 변환:

```json
{"id": "...", "type": "function",
 "function": {"name": "...", "arguments": "<JSON-encoded args dict>"}}
```

`tool` 메시지는 `tool_call_id` 를 보존. 이것이 `swebench_agent.py:83-95` 에서 직렬화된 SWE-Bench `turns[].messages` payload 를 LangChain client 가 원래 framing 했던 방식 그대로 재토큰화할 수 있게 해 주는 round-trip glue 다.

### 4.5 SpecBench extractor (`_agent_io.py:302-418`)

두 경로:

- Modern (`questions[].turns[]`): per-turn dict 를 walk 하며 oracle entry 추출, 옵션으로 `_reconstruct_specbench_prompts(dataset_entry, result_question, tokenizer)` 가 user 메시지와 assistant 응답을 interleave 하여 prompt id 재구성 (SpecBench 는 system prompt 없음).
- Legacy `per_request[]` flat fallback: 거의 사용되지 않음, LLM call 당 1 entry.

## 5. Oracle metadata — 각 entry 가 실제로 담는 것

`SGLANG_ORACLE_VANILLA=1` 이 `simulation/oracle/install_hook.py:157-251` 가 설치한 EAGLE worker 패치를 트리거한다. 매 step 패치된 worker 는 `/tmp/sglang_oracle_vanilla.jsonl` (`SGLANG_ORACLE_LOG` 로 override 가능) 에 한 줄의 JSON 을 쓴다. Schema (`oracle_patch.py:13-17` 출처, `_extract_entries` 가 소비):

```json
{
  "req_id": "<string>",
  "proposer": "eagle3" | "mtp",
  "tokens": [[<verified_next_token_int>]],
  "eagle3": [[<draft_token_int>, ...]],
  "eagle3_tree": {
    "token_ids": [<int>, ...],
    "parents":   [<int>, ...]
  },
  "eagle3_tree_p_t":             [<float>, ...],   // optional
  "eagle3_tree_path_draft_p_t":  [<float>, ...],   // optional
  "mtp_tree":                    {...}             // present only in MTP runs
}
```

oracle-vanilla 모드에서는 verifier 가 accept length 0 으로 강제되므로, `tokens` 는 항상 target model 이 고른 단일 greedy next token 이고, `eagle3` / `eagle3_tree` 는 그 step 에서 draft proposer 가 제안했을 내용을 정확히 캡처한다. 이것이 trajectory 를 downstream tree-oracle simulation 에서 replayable 하게 만드는 핵심이다.

## 6. Stage 2/3 가 의존하는 Stage 1 invariant

Stage 1 출력이 ("나머지 파이프라인이 소비할 수 있다" 는 의미에서) "올바르다" 는 것은 다음과 동치:

1. `agent_results_eagle3.json` 이 `metadata.oracle_enabled = true` 와 최소 하나의 question 을 가진 채 존재.
2. `spec_decode` 블록이 있는 모든 per-question step 이 non-empty `eagle3[0]` 와 길이 1 의 `tokens[0]` 를 가진 `oracle_vanilla_entries[i]` 를 최소 하나 가짐.
3. BFCL/SWE-Bench agent 의 경우 per-step `messages` 스냅샷이 존재 (Stage 3 에서 prompt 재토큰화에 사용) — 또는 SWE-Bench 의 경우 `turns[i].messages` 가 존재하고 `_langchain_to_openai_messages` 로 변환 가능.
4. SpecBench question 의 `question_id` 가 `data/specbench/dataset.jsonl` 과 일치 — Stage 3 prompt 재구성에서 dataset 이 파일명으로 keying 된다.
5. BFCL question 의 `bfcl_id` 가 `data/bfcl_{multi_turn,agent}/dataset.jsonl` 과 일치 — prompt 재구성 fallback 과 BFCL ground-truth join 양쪽에 필요.

서버 startup 시점에 `SGLANG_ORACLE_VANILLA=1` 이 빠져 있으면, agent 들은 에러 없이 종료하지만 모든 step 에 `spec_decode` 가 결락된다. 이것이 Stage 1 의 가장 흔한 silent failure 이다; `run_experiment.py:_run_rr_shard` 가 SGLang 서브프로세스에 자동 주입하지만 SGLang 직접 launch 시에는 env 에 들어 있는지 반드시 재확인하라.
