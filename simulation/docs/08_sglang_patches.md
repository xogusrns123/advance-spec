# 08. SGLang 패치 (Oracle 파이프라인)

오라클 파이프라인은 SGLang EAGLE3/MTP 워커를 두 단계로 패치한다.
이 문서는 패치 표면 전체와 부작용을 한 번에 검증할 수 있도록, **무엇이 패치되고 / 어디에 데이터가 적재되며 / 어떤 상태가 변형되는지**를 파일·라인 단위로 기록한다.

---

## 1. 두 단계 패치 구조 (Two-Tier Patching)

오라클은 SGLang을 두 가지 방식으로 변형한다.

| Tier | 파일 | 시점 | 대상 |
|------|------|------|------|
| 1. **on-disk source patch** | `simulation/oracle/install_hook.py` | 서버 부팅 직전 (`python -m simulation.oracle.install_hook ...`) | SGLang의 `srt/speculative/spec_info.py`, `srt/server_args.py`, `srt/managers/scheduler.py`, `srt/speculative/eagle_worker.py`, `srt/speculative/multi_layer_eagle_worker.py` |
| 2. **runtime monkey-patch** | `simulation/oracle/oracle_patch.py` | 각 EAGLE/MTP 워커 인스턴스의 `__init__` 끝부분에서 자동 호출 | `EAGLEWorker.draft / verify / forward_batch_generation`, `eagle_info.verify_tree_greedy_func`, `eagle_worker.organize_draft_results`, `target_worker.forward_batch_generation` |

Tier 1은 모든 자식 프로세스(scheduler, TP rank들)가 공유해야 하는 enum/argparse 변경, 그리고 `__init__` 마지막 줄에 hook을 심는 정적 변경에만 사용된다. Tier 2는 인스턴스별 메서드 교체로, 한 워커 안에서만 수명을 가진다.

### 활성화 경로 (Activation Path)

1. RR launcher (`run_experiment.py:_run_rr_shard`) 또는 latency 측정 스크립트가 `SGLANG_ORACLE_VANILLA=1` (그리고 필요시 `SGLANG_LATENCY_ONLY=1` 등) 을 SGLang 서브프로세스에 export.
2. `python3 -m simulation.oracle.install_hook` 또는 `sglang.launch_server` 실행.
   - `__main__` 블록(`install_hook.py:282-300`)이
     `_start_process_watchdog()` → `install_suffix_algorithm()` →
     `install_oracle_patch()` → `sglang.launch_server.run_server(...)` 순으로 진입.
3. `install_oracle_patch()` (`install_hook.py:157-182`)는
   `SGLANG_ORACLE_VANILLA=1` 일 때만 `eagle_worker.py` / `multi_layer_eagle_worker.py` 의 `__init__` 마지막 줄(`self.extend_lens = torch.empty(...)`) 뒤에 다음 코드를 디스크에 삽입한다:

   ```python
   # Oracle vanilla patch: log draft tokens per step (EAGLEWorker)
   import os as _os
   if _os.environ.get('SGLANG_ORACLE_VANILLA', '0') == '1':
       from simulation.oracle.oracle_patch import patch_eagle_worker_full
       patch_eagle_worker_full(self)
   ```
4. 서버가 부팅하면서 `EAGLEWorker.__init__` 끝에서 위 코드가 실행되어
   `patch_eagle_worker_full(self)` 가 인스턴스에 monkey-patch들을 설치 →
   매 step마다 draft tree / 수락 token / verify logits가
   `/tmp/sglang_oracle_vanilla.jsonl` 에 기록된다.
5. 클라이언트(agents)는 매 LLM 호출 직전 `get_oracle_log_position()` 으로 byte offset을 잡고, 호출 이후 `read_oracle_log(pos)` 로 그 호출에서 추가된 줄만 가져와 step record의 `spec_decode.oracle_vanilla_entries` 로 적재한다 (`simulation/agents/bfcl_v4_agent.py:218-257`, `simulation/agents/swebench_agent.py:222-256`).

> 참고: Tier 1 패치는 **idempotent** 하다 (각 `_patch_*` 함수의 첫 두 줄이 sentinel 검사). Docker 빌드 시점(`Dockerfile:50-54`)에 같은 sentinel-기반 `sed` 가 한 번 박히고, 컨테이너 안에서 `install_hook.py` 가 다시 호출돼도 두 번째 적용은 스킵된다.

---

## 2. `install_hook.py` — 디스크 소스 패치

전부 `_get_sglang_root()` (`install_hook.py:39-42`)가 반환하는 `Path(sglang.__file__).parent` 아래 파일을 직접 `read_text()` / `write_text()` 한다.
프로세스 부팅 전에 `_start_process_watchdog(max_children=50)` (`install_hook.py:254-279`)도 띄워두는데, fork bomb 발생 시 `os.killpg(SIGKILL)` 로 서버를 통째로 죽인다.

### 2.1 `_patch_spec_info` — `spec_info.py:15-105`

**파일:** `srt/speculative/spec_info.py`
**Idempotency:** `if "SUFFIX" in text: return` (`install_hook.py:73-74`).
**변경 내용 (3건):**

1. `SpeculativeAlgorithm` enum에 `SUFFIX = auto()` 를 `NGRAM` 과 `NONE` 사이에 삽입 (`install_hook.py:76-80`).
2. `is_suffix(self) -> bool` 메서드를 `is_ngram` 직후에 추가 (`install_hook.py:82-87`).
3. `create_worker()` 의 마지막 `raise ValueError("Unreachable …")` 직전(`spec_info.py:105`)에
   ```python
   elif self.is_suffix():
       if enable_overlap: raise ValueError(...)
       from hybrid_spec_decoding.sglang_integration.suffix_worker import SuffixWorker
       return SuffixWorker
   ```
   분기를 추가 (`install_hook.py:89-100`).

**SGLang 동작 변화:** `--speculative-algorithm SUFFIX` 가 enum 변환·worker dispatch 양쪽에서 받아들여지며, NGRAM 처럼 draft model 없이 떠올릴 수 있게 된다. **Generation 자체에는 영향 없음** — 새 분기는 SUFFIX 알고리즘 한정.

> SUFFIX 코드패스는 본 문서의 EAGLE3 오라클과 별개이지만(같은 패치 스크립트가 같이 들어 있을 뿐), 같은 디스크 패치가 EAGLE3 오라클 모드에서도 그대로 적용된다. EAGLE3 오라클은 SUFFIX를 쓰지 않으므로 위 분기는 dead code 로 머문다.

### 2.2 `_patch_server_args` — `server_args.py`

**파일:** `srt/server_args.py`
**Idempotency:** `if '"SUFFIX"' in text: return` (`install_hook.py:110-111`).
**변경 내용 (3건):**

1. argparse choices(`server_args.py:4727`)에서
   `["EAGLE","EAGLE3","NEXTN","STANDALONE","NGRAM"]` →
   `[..., "NGRAM", "SUFFIX"]` (`install_hook.py:114-117`).
2. `if self.speculative_algorithm == "NGRAM":` (`server_args.py:3080`) →
   `in ("NGRAM", "SUFFIX")` (`install_hook.py:124-127`).
3. `self.speculative_algorithm != "NGRAM"` (`server_args.py:1304`) →
   `not in ("NGRAM", "SUFFIX")` (`install_hook.py:131-134`).

**SGLang 동작 변화:** 메모리 예약(`reserved_mem`)과 NGRAM-스러운 서버 셋업 분기가 SUFFIX에도 활성화. 다시 말하지만 EAGLE3 오라클 모드에서는 트리거되지 않는다.

### 2.3 `_patch_scheduler` — `scheduler.py:956`

**파일:** `srt/managers/scheduler.py`
**Idempotency:** `if "is_suffix()" in text: return` (`install_hook.py:144-145`).
**변경 내용:**

```python
# 변경 전 (scheduler.py:956)
if self.draft_worker is None or self.spec_algorithm.is_ngram():
    draft_token_to_kv_pool = None
# 변경 후
if (self.draft_worker is None
    or self.spec_algorithm.is_ngram()
    or self.spec_algorithm.is_suffix()):
    draft_token_to_kv_pool = None
```

**SGLang 동작 변화:** Disaggregation 모드에서 SUFFIX 또한 draft KV pool 을 만들지 않게 한다. EAGLE3 모드에서는 무효 (`is_suffix()` 가 False).

### 2.4 `install_oracle_patch` + `_inject_oracle_into_worker` — `eagle_worker.py:213`, `multi_layer_eagle_worker.py:197`

**파일:** `srt/speculative/eagle_worker.py`, `srt/speculative/multi_layer_eagle_worker.py`
**Sentinel:** `self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)` (각 워커 `__init__` 의 마지막 텐서 할당; 구조가 다르면 정규식 fallback으로 `self.extend_lens = …` 검색).
**Idempotency:** 세 단계 검사. (a) 옛 import path `hybrid_spec_decoding.sglang_integration.oracle_patch` 가 보이면 새 path `simulation.oracle.oracle_patch` 로 교체 후 종료(`install_hook.py:204-208`). (b) 새 path 가 이미 있으면 종료(`install_hook.py:210-211`). (c) `from __future__ import annotations` 가 없으면 license 헤더 직후에 삽입(`install_hook.py:215-224`).
**삽입 코드 (`install_hook.py:241-249`):**

```python
self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        # Oracle vanilla patch: log draft tokens per step (EAGLEWorker)
        import os as _os
        if _os.environ.get('SGLANG_ORACLE_VANILLA', '0') == '1':
            from simulation.oracle.oracle_patch import patch_eagle_worker_full
            patch_eagle_worker_full(self)
```

**SGLang 동작 변화:** `SGLANG_ORACLE_VANILLA != "1"` 일 때는 완전 무동작(`if` 문 false → 패치 미설치 → 평소 SGLang 동작). 따라서 디스크 패치 자체는 안전하다.

### 2.5 `_start_process_watchdog` — Bash watchdog

`install_hook.py:254-279`. `pgrep -c -P <pid>` 로 자식 프로세스 수를 5초 간격으로 폴링하다 50개 초과 시 `SIGKILL`. 패치 자체는 아니지만 디스크 패치 후 fork bomb 위험을 차단하는 안전망.

---

## 3. `oracle_patch.py` — 런타임 monkey-patch

워커 인스턴스에 attach된 `EAGLEWorker` (혹은 `MultiLayerEagleWorker`) 객체에 대해 `patch_eagle_worker_full(self)` 가 호출된다. 모든 stash는 `self._oracle_*` 속성으로 워커 인스턴스에 둔다 (모듈 레벨이 아니라 인스턴스 레벨이라 TP-rank 별로 격리).

### 3.1 모듈 상수와 환경변수 (`oracle_patch.py:35-43`)

| 변수 | 기본값 | 용도 |
|------|--------|------|
| `ORACLE_LOG_PATH` | `/tmp/sglang_oracle_vanilla.jsonl` | per-step draft+token JSONL append |
| `ORACLE_TIMING_PATH` | `/tmp/sglang_oracle_timing.jsonl` | per-step latency JSONL append |
| `ORACLE_REPLAY_PATH` | `""` | replay mode 활성화 (path 가 truthy 이면 trajectory load) |
| `ORACLE_DRAFT_BUDGET` | `""` | `speculative_num_draft_tokens` runtime override |
| `SGLANG_CAPTURE_FULL_POOL` | `"0"` | `_install_draft_p_t_tracer` 안의 `capture_full_pool` 분기 (`oracle_patch.py:341`). `=1` 이면 매 draft step 에서 `organize_draft_results` 를 cloned 입력으로 한 번 더 호출해 truncate 안 된 full pool (`pool_size = K + (S-1)·K²`) 의 `(parent_list, draft_tokens, path_probs)` 를 stash → `eagle3_pool_full` 필드로 entry 에 추가. 두 번째 호출은 `sl_clone/tl_clone/pl_clone` 으로 입력 격리되어 generation 영향 없음. 자세한 건 `09_pool_reslicer.md` |

### 3.2 Public read API

#### `is_oracle_enabled()` / `is_replay_mode()` / `clear_oracle_log()` — `oracle_patch.py:49-59`

상태 질의 + 로그 truncate. `clear_oracle_log()` 는 `OSError` 를 silent swallow.

#### `get_oracle_log_position() -> int` — `oracle_patch.py:61-69`

현재 로그 파일의 `st_size` 반환 (없으면 `0`). LLM 호출 직전 호출.

#### `read_oracle_log(start_position=None) -> list[dict]` — `oracle_patch.py:71-94`

`start_position` 부터 EOF까지 읽어 JSONL 디코드. 동시 요청들이 같은 로그 파일을 안전하게 공유할 수 있게 byte offset 기반 슬라이싱을 사용 — 호출자별 entry만 분리.

**소비자:**
- `simulation/agents/bfcl_v4_agent.py:220, 253` — agent step 의 LLM 호출 전후로 잡음. 결과를 `step_data["spec_decode"]["oracle_vanilla_entries"]` 로 저장.
- `simulation/agents/bfcl_v4_agent.py:369, 392` — function-call 응답 step에서 동일.
- `simulation/agents/swebench_agent.py:224, 252, 383, 397` — SWE-Bench agent에서 동일 패턴.
- `simulation/agents/specbench_agent.py:228` — 단지 enable 메시지 출력.

`save_results.py:140-149` 가 full JSON엔 그대로 두고, `*_response.json` 에서는 `oracle_vanilla_entries` 를 빼고 `oracle_entries_count` 만 남기는 정리를 한다.

### 3.3 `patch_eagle_worker_full(eagle_worker)` — `oracle_patch.py:402-446`

진입점. 흐름:

1. `_oracle_proposer_type` ← `"mtp"` 또는 `"eagle3"` (클래스 이름으로 판별, `_detect_proposer_type`).
2. `ORACLE_REPLAY_PATH` 가 있으면 trajectory JSON load → `TrajectoryState` 인스턴스.
3. `ORACLE_DRAFT_BUDGET` 가 있으면 `eagle_worker.speculative_num_draft_tokens` 와 `eagle_worker.server_args.speculative_num_draft_tokens` 를 함께 덮어씀 (latency 측정에서 budget sweep용). **주의: 이 변경은 영구적이며 같은 서버 인스턴스가 사는 동안 유지된다.**
4. `SGLANG_LATENCY_ONLY=1` 이면 `_setup_latency_only()` 만 호출하고 **early return** — accept-force / tree extraction / p_t logging 모두 비활성. 즉 진짜 speculative decoding이 돌아가고, timing 측정만 한다.
5. 그 외엔 4 단계 패치를 모두 설치:
   - `_patch_verify_greedy_func()` — accept_length 강제 0
   - `_patch_draft_stash(...)` — draft tree 캡처
   - `_patch_verify_logits(...)` — verify timing + accept-length + verify logits stash
   - `_patch_forward_log(...)` — per-step JSONL append + replay 토큰 강제

### 3.4 `_patch_verify_greedy_func()` — `oracle_patch.py:557-595`

**대상:** `sglang.srt.speculative.eagle_info.verify_tree_greedy_func` (모듈 어트리뷰트).
- 정의 자체는 `eagle_utils.py:161` 이지만 `eagle_info.py:28` 에서
  `from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func` 로 re-import 되며,
  실제 호출 사이트는 `eagle_info.py:315` 에서 module-level name 으로 호출 → `eagle_info.verify_tree_greedy_func = ...` 로 덮어쓰면 호출 시 패치본이 잡힌다.

**Idempotency:** `eagle_info._oracle_patched_greedy` 플래그 (`oracle_patch.py:566`).

**핵심 로직:**

```python
# Run original to get correct target_predict (bonus token)
predicts, accept_index, accept_token_num = original_func(...)

# Force accept_length=0: only keep the first accepted token (bonus)
bs = accept_index.shape[0]
first_col = accept_index[:, 0].clone()
accept_index.fill_(-1)
accept_index[:, 0] = first_col
accept_token_num.fill_(0)
return predicts, accept_index, accept_token_num
```

**효과:**
- Bonus token (target argmax at root candidate position) 은 그대로 두고, 모든 draft 수락을 무효화 → 매 step **정확히 1 토큰** 만 commit.
- `accept_index` / `accept_token_num` 가 mutable 텐서이므로 SGLang 내부에서 batch state(`output_ids`, `kv_committed_len`, `hidden_states`, `accept_length_per_req_cpu`) 가 **자연스럽게 1-token 진행으로 갱신**된다 → roll-back 불필요.
- Sampling rule 자체는 건드리지 않는다 — `target_predict` 는 원본 함수의 argmax 결과 그대로. (단, 원본은 greedy path 한정. Tree-spec sampling kernel 이 잡히는 non-greedy path 는 `eagle_info.py:327-` 에서 `verify_tree_greedy_func` 를 거치지 않고 sampling kernel 을 직접 호출 → 이 패치가 적용되지 않는다. **오라클은 `is_all_greedy=True` 또는 AMD/HIP 빌드를 가정한다.**)

**Risk note:** Agents 가 모두 `temperature=0.0` 으로 호출하므로 `is_all_greedy=True` 가 되어 패치된 분기로 들어간다 (e.g. `bfcl_v4_agent.py:231`). Temperature > 0 요청은 패치를 우회한다 — 이는 알려진 가정이다.

### 3.5 `_patch_draft_stash(eagle_worker)` — `oracle_patch.py`

**감싸는 메서드:** `eagle_worker.draft(batch) -> EagleVerifyInput`.
**부수 효과:**
- 처음 호출 시 `_install_draft_p_t_tracer()` 가 module-level `eagle_worker.organize_draft_results` 를 traced 버전으로 교체 (`oracle_patch.py:329-387`). traced 버전은 원본의 `score_list / token_list / parents_list` cat 결과에서 `top_scores_index` 위치의 path probability 를 `torch.gather` 로 뽑아 `ew_module._oracle_last_path_probs` 에 stash. 모듈 어트리뷰트지만 매 호출 직후 `patched_draft` 가 즉시 소비/`None` 으로 리셋하므로 leak 없음.

**stash 되는 인스턴스 속성:**
| 속성 | 형태 | 용도 |
|------|------|------|
| `_oracle_last_draft_ms` | float | draft 단계 wall time |
| `_oracle_stashed_path_probs` | `(bs, num_draft)` CPU tensor | per-node draft-side cumulative path prob |
| `_oracle_stashed_draft` | `(bs * num_draft,)` CPU tensor | flat draft token IDs (root + draft) |
| `_oracle_stashed_num_draft` | int | `spec_info.draft_token_num` |
| `_oracle_stashed_topk` | int | `spec_info.topk` |
| `_oracle_stashed_retrive_next_token` | `(bs, num_draft)` CPU tensor | tree first-child pointers |
| `_oracle_stashed_retrive_next_sibling` | `(bs, num_draft)` CPU tensor | tree next-sibling pointers |

### 3.6 `_patch_verify_logits(eagle_worker, stash_verify_logits=True)` — `oracle_patch.py`

**감싸는 메서드 (둘):**
1. `eagle_worker.verify(batch, spec_info)` — wall time 측정 + 실제 accept length per request stash.
2. `eagle_worker.target_worker.forward_batch_generation(model_worker_batch, is_verify=False)` — `is_verify=True` 인 호출에 대해서만:
   - `_oracle_last_target_forward_ms` 측정.
   - `stash_verify_logits=True` 일 때 `result.logits_output.next_token_logits.cpu().clone()` 을 `_oracle_stashed_verify_logits` 로 저장.

**stash 되는 속성:**
| 속성 | 용도 |
|------|------|
| `_oracle_last_verify_total_ms` | float, full verify (target forward + greedy) 시간 |
| `_oracle_last_target_forward_ms` | float, target forward 만 |
| `_oracle_last_accept_lengths` | `accept_length_per_req_cpu` 의 list 복사본 |
| `_oracle_stashed_verify_logits` | `(bs * num_draft, vocab_size)` CPU tensor |

**Risk note:** Logit 텐서 `.cpu().clone()` 은 매 step 마다 vocab × num_draft × bs × 4byte (e.g. 152k × 16 × 1 × 4 = ~10 MB) 를 GPU→CPU 이동. **Throughput 측정용 latency-only 모드는 일부러 `stash_verify_logits=False` 로 켜서 (`oracle_patch.py:474`) sync overhead 가 target_forward_ms 에 묻어들어가지 않게 한다.** Vanilla 오라클 모드에선 logit이 필요해서 켠다 (Risk: 측정된 timing 이 실제 production 보다 느리게 나옴 — 그래서 vanilla 모드는 timing 신뢰도가 낮고, 별도의 latency-only 패스가 존재).

### 3.7 `_patch_forward_log(eagle_worker, replay_state)` — `oracle_patch.py`

**감싸는 메서드:** `eagle_worker.forward_batch_generation(batch) -> GenerationBatchResult`.

**Step 1 — replay token override (`oracle_patch.py:852-861`):**

`SGLANG_ORACLE_REPLAY=path` 가 켜져 있으면 모든 TP rank 에서 (`tp_rank != 0` 검사 전에) `req.output_ids[-1]` 을 trajectory 의 다음 토큰으로 덮어쓴다. 이렇게 해야 KV cache 와 다음 step 의 draft input 이 trajectory 의 토큰을 보고 진행한다.

`TrajectoryState` (`oracle_patch.py:115-142`)는 `req_id → trajectory key` 매핑을 lazy 로 만든다. 처음 보는 rid 가 들어오면 정렬된 미할당 trajectory key queue 에서 pop해 매핑을 만들고 (deterministic FIFO), 매 호출마다 `positions[orig_rid]` 를 1씩 진행한다. Trajectory exhaustion 시 `None` 반환 → override 스킵.

**Step 2 — TP rank 0 에서만 logging (`oracle_patch.py:864`):** rank 0 외에는 즉시 return → 같은 로그 파일에 중복 기입을 막는다.

**Step 3 — per-request 로그 entry 조립 (`oracle_patch.py:867-955`):**

각 request 별로:
1. `verified_cpu` 에서 `accept_length+1` 토큰을 슬라이스 (force-accept 패치로 인해 `accept_length=0` → 항상 1 토큰).
2. `vanilla_token = req_accepted[0]` (replay 모드에선 `req.output_ids[-1]` 로 갱신 — 위에서 override 됐으므로 동일).
3. `req_draft = draft_cpu[i*D + 1 : (i+1)*D]` — root 슬롯 제외한 flat draft.
4. `_extract_eagle3_tree(...)` 로 BFS-ordered (`token_ids`, `parents`, `candidates_positions`) 를 추출. parent index 가 child index 보다 작아야 한다는 건전성 검사 포함 (`oracle_patch.py:228-231`).
5. `_extract_tree_p_t(...)` 로 verify logits 의 부모 위치에서 softmax → 자식 토큰의 prob 를 모은 list 생성 (target-side per-edge prob).
6. `_extract_tree_path_draft_p_t(...)` 로 draft-side 누적 path prob (organize tracer 가 stash한 텐서에서 lookup).
7. (`SGLANG_CAPTURE_FULL_POOL=1` 일 때) `_oracle_stashed_full_pool` 에서 텐서를 꺼내 per-request 로 슬라이스해 `entry["eagle3_pool_full"]` 에 추가 (`oracle_patch.py:884-921`). batched (2D) vs flat (1D) layout 모두 처리.
8. `_log_entry(entry)` 로 JSONL 한 줄 append:

   ```json
   {"eagle3": [[draft_ids...]], "tokens": [[next_token]],
    "req_id": "...", "proposer": "eagle3"|"mtp",
    "eagle3_tree": {"token_ids": [...], "parents": [...]},
    "eagle3_tree_p_t": [...],
    "eagle3_tree_path_draft_p_t": [...],
    "eagle3_pool_full": {"draft_tokens": [...], "parent_list": [...],
                         "path_probs": [...], "pool_size": int}}
   ```
   `eagle3_pool_full` 은 `SGLANG_CAPTURE_FULL_POOL=1` 일 때만 존재.
9. 마지막에 stash 들을 `None` 으로 reset → 다음 step 이 stale 텐서를 보지 않도록.

**Step 4 — timing JSONL append (`oracle_patch.py:959-970`):**

`{"eagle3_draft_ms":..., "target_forward_ms":..., "num_draft":..., "num_reqs":...}` 한 줄을 `ORACLE_TIMING_PATH` 에 append. Vanilla 모드에서도 켜져 있다는 점에 주의.

**Risk note (try/except masking):** 전체 try/except (`oracle_patch.py:867-973`) 가 모든 예외를 swallow 하고 warning 한 줄만 찍는다. 즉 oracle 로깅 중 어떤 코드 경로가 깨져도 generation 자체는 멈추지 않는다 — 이는 의도된 안전망이지만, **로그 entry 가 누락된 step 은 나중에 reconstruct 단계에서 step count mismatch 로 잡혀야 한다.**

### 3.8 `_setup_latency_only(eagle_worker)` — `oracle_patch.py:449-554`

`SGLANG_LATENCY_ONLY=1` 일 때만 진입. 기존 두 메서드(`draft`, `forward_batch_generation`) 와 `_patch_verify_logits(stash_verify_logits=False)` 만 설치하고, **force-accept 패치 / draft tree 추출 / p_t 추출 / oracle JSONL 은 전부 비활성화**.

차이점:
- `accept_length` 가 forced-0 이 아니라 진짜 값. `_oracle_last_accept_lengths` 가 실제 수락 길이.
- `_log_timing(...)` 가 `phase / step_total_ms / num_tokens / eagle3_draft_ms / target_forward_ms / verify_total_ms / verify_overhead_ms / post_verify_ms / accept_lengths / committed_tokens / vd_*` 같은 풍부한 필드를 ORACLE_TIMING_PATH 에 append.
- `SGLANG_VERIFY_DETAILED=1` 옵션이 있으면 `simulation.oracle.verify_detail_patch` 모듈을 import 시도하지만 본 브랜치에는 없어서 silent skip (`oracle_patch.py:478-485`).

**Risk note:** Real speculative decoding 이 돌아가므로 generation 결과가 force-accept 모드와 달라진다 — 이 모드는 generation 결과를 쓰지 않는 latency 측정 전용. 결과 정확성은 별도 vanilla 패스에서 검증한다.

### 3.9 `_install_draft_p_t_tracer()` — `oracle_patch.py:329-457`

`organize_draft_results` 를 module 어트리뷰트로 교체. 원본을 호출해 동일한 `(parent_list, top_scores_index, draft_tokens)` 를 반환하되, side-effect 로 `flat_scores.gather(top_scores_index)` 를 계산해 `_oracle_last_path_probs` 에 stash. **계산만 추가하고 결과 텐서는 원본 그대로 반환** → SGLang의 후속 단계는 영향 없음.

**Full-pool capture 분기 (`oracle_patch.py:341-396`)** — `SGLANG_CAPTURE_FULL_POOL=1` 이면 `original(...)` 을 한 번 더 호출. 이때 `num_draft_token` 자리에 `pool_size + 1` 을 넣어 truncate 없이 전체 pool 을 받는다. 입력은 `[s.clone() for s in score_list]` 등 cloned 텐서로 격리 (defensive — `organize_draft_results` 가 in-place mutation 안 하리라 기대하지만 cheap insurance). 결과 `(full_parent, full_top_idx, full_tokens)` 와 root prob 1.0 을 prepend 한 `full_path_probs` 를 `ew_module._oracle_last_full_pool` 에 dict 로 stash. `_patch_draft_stash` 가 그걸 `eagle_worker._oracle_stashed_full_pool` 로 옮기고, `_patch_forward_log` 가 entry 의 `eagle3_pool_full` 로 dump.

> 안전성: 두 번째 호출은 cloned 입력으로만 동작하고, **반환 텐서를 SGLang 의 verify path 에 넣지 않는다** (verify 는 첫 번째 호출 결과로 진행). 즉 generation 출력은 capture 여부와 무관하게 동일. `_test_isolate.sh` / `_test_skip_verify.sh` 가 이 invariant 를 byte-identical token 비교로 검증.

Idempotency: `ew_module._oracle_organize_traced` 플래그.

---

---

## 4. 환경변수 요약

| 변수 | 설정자 | 사용자 | 의미 / 부작용 |
|------|--------|--------|---------------|
| `SGLANG_ORACLE_VANILLA` | `run_experiment.py:_run_rr_shard`, `measure_eagle3_cost.py` | `install_hook.install_oracle_patch`, 워커 `__init__` (Tier 1로 박힌 코드), `oracle_patch.is_oracle_enabled` | `=1` 일 때만 oracle hook 이 실제로 install. 미설정/`=0` 이면 SGLang 정상 동작. |
| `SGLANG_ORACLE_REPLAY` | (legacy, 현행 미사용) | `oracle_patch.py:39` 에서 path 로 load → `TrajectoryState` 활성 | path 가 truthy 면 매 step `req.output_ids[-1]` 을 trajectory 토큰으로 강제. RR 에서는 `pop` 으로 제거됨. |
| `SGLANG_ORACLE_LOG` | (선택적, latency 측정 시) | `oracle_patch.py:36` | per-step JSONL 경로 (default `/tmp/sglang_oracle_vanilla.jsonl`). 동시 서버 인스턴스를 띄울 때 충돌 방지용. |
| `SGLANG_ORACLE_TIMING_LOG` | 위와 같은 스크립트 | `oracle_patch.py:38`, `simulation/scripts/measure_eagle3_cost.py:45` | per-step timing JSONL 경로 (default `/tmp/sglang_oracle_timing.jsonl`). |
| `SGLANG_DRAFT_BUDGET` | (legacy, 일부 sweep) | `oracle_patch.py:40, 420-425` | `speculative_num_draft_tokens` runtime override. 워커 + server_args 양쪽을 덮어쓰며 영구 (서버 lifetime 동안). |
| `SGLANG_CAPTURE_FULL_POOL` | `run_experiment.py:_run_rr_shard` (yaml `capture_full_pool: true` 일 때) | `oracle_patch.py:341` (`_install_draft_p_t_tracer` 안의 `capture_full_pool`) | `=1` 이면 매 draft step 에서 `organize_draft_results` 를 cloned 입력으로 한 번 더 호출 (`pool_size+1` 로) → full pool 을 stash. `eagle3_pool_full` 필드가 oracle entry 에 추가됨. Stage 3 의 `pool_reslicer` 가 (s', k') sub-tree 로 reslice 할 때 입력. **Generation 영향 없음** (cloned 입력). 자세한 건 `09_pool_reslicer.md`. |
| `SGLANG_LATENCY_ONLY` | `simulation/scripts/measure_eagle3_cost.py` | `oracle_patch.py:436` | `=1` 이면 force-accept 비활성, real speculative decoding + timing instrumentation. README: "실제 speculative decoding 으로 동작". |
| `SGLANG_VERIFY_DETAILED` | (선택적) | `oracle_patch.py:478` | `=1` + `verify_detail_patch` 모듈 존재 시 추가 verify breakdown. 본 브랜치엔 모듈 없음 → silent skip. |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `run_experiment.py:_run_rr_shard`, `docker-compose.yml`, `Dockerfile` | SGLang `srt/configs/model_config.py:380` | EAGLE3 draft model 의 max_position_embeddings 가 target context_length 보다 짧을 때 overwrite 허용 (없으면 ValueError). |
| `TORCHINDUCTOR_COMPILE_THREADS` | `install_hook.py:25-26`, `run_experiment.py:_run_rr_shard`, `hybrid_spec_decoding/sglang_integration/suffix_worker.py:23-24` | torch.compile 내부 | torch import 전에 `=1` 을 강제해 inductor compile worker fork 폭증을 방지. Pipeline에서 fork bomb watchdog 과 더불어 자식 프로세스 수를 안정화. |

---

## 5. 위험 audit

이 섹션은 "오라클 패치가 정말 generation 출력의 정확성을 해치지 않는가" 를 점검한다.

### 5.1 Sampling / 결정성

- **Temperature·top-p·top-k 미변경.** 패치는 `verify_tree_greedy_func` 의 결과 텐서만 mutate (accept_index, accept_token_num) 하고 sampling distribution 은 건드리지 않는다.
- **Force-accept 의 의미.** `target_predict` 는 원본 함수가 계산한 그대로 — 이는 root candidate position 의 argmax (greedy) 또는 sampled token. 매 step 정확히 그 토큰 하나만 commit. **즉 force-accept 모드의 출력 = greedy 베이스라인의 1-step-at-a-time 출력 = vanilla 모델 greedy decoding 과 동일.** 따라서 결과 정확성은 보존된다.
- **Non-greedy path.** Tree spec sampling kernel 분기 (`eagle_info.py:327-`) 는 패치를 우회한다. Pipeline 의 모든 agent 가 `temperature=0.0` 으로 부르므로 greedy path 만 사용 — 가정이 깨지지 않는 한 안전하다.

### 5.2 KV cache / batch state

- `accept_index` / `accept_token_num` 는 mutable 텐서라 SGLang 내부의 후속 처리 (`spec_info.verify` → `output_ids` 갱신, `kv_committed_len`, `hidden_states` 슬라이싱) 가 전부 1-token 진행으로 자동 정합. 별도 rollback 없음.
- Replay 모드는 한 단계 더 나아가 `req.output_ids[-1]` 자체를 trajectory token 으로 덮어쓴다 (`oracle_patch.py:859-861`) → KV cache 에 들어간 토큰과 다음 draft 의 input token 이 trajectory 와 일치.

### 5.3 Try/except 가리는 에러

다음 코드 경로가 예외를 swallow 한다 — generation 은 멈추지 않지만 오라클 데이터가 누락될 수 있다:

| 위치 | swallow 대상 | 영향 |
|------|-------------|------|
| `oracle_patch.py:55-59` | `clear_oracle_log` 의 `OSError` | 로그 truncate 실패 → 다음 step 의 entry 가 이전 entry 뒤에 append 되어 mismatch 가능. |
| `oracle_patch.py:96-101` / `:103-108` | 로그 write `OSError` | warning 만 찍고 entry 누락. |
| `oracle_patch.py:238-239` | `_extract_eagle3_tree` 의 임의 Exception | 해당 entry 의 `eagle3_tree` 필드 누락 (다른 필드는 살아남음). |
| `oracle_patch.py:291-292` / `:325-326` | p_t 추출 실패 | 해당 필드 누락. |
| `oracle_patch.py:382-383` | gather 실패 | path_probs stash 가 None → 그 step 의 draft p_t 누락. |
| `oracle_patch.py:683-687` | draft stash 자체 실패 | 모든 stash 가 None → 그 step 의 entry 자체가 skip (`_patch_forward_log` 의 `if stashed_draft is None: return result`). |
| `oracle_patch.py:793-795` | accept length stash 실패 | accept length empty list. |
| `oracle_patch.py:817-819` | logits stash 실패 | verify p_t 누락. |
| `oracle_patch.py:867-973` 의 outer try | 로깅 전체의 임의 실패 | warning 만 찍고 generation 은 정상 진행. |

**검증 권장 사항:** 파이프라인의 reconstruct 단계가 step count mismatch / 누락 entry 를 surface 하는지 확인. 특히 long sequence 에서 `_oracle_stashed_draft is None` (`oracle_patch.py:869`) 분기로 빠지는 횟수가 0 인지 확인하는 sanity check 를 추가하면 안전.

### 5.4 영구 상태 변경

- `SGLANG_DRAFT_BUDGET` (`oracle_patch.py:420-425`) 은 `eagle_worker.speculative_num_draft_tokens` 와 `server_args.speculative_num_draft_tokens` 를 둘 다 덮어쓴다. 이는 서버 인스턴스 lifetime 동안 유지되므로 sweep 시 서버를 재기동해야 깨끗한 baseline 을 얻는다.
- `_install_draft_p_t_tracer()` 는 module-level `eagle_worker.organize_draft_results` 자체를 교체한다 → 같은 프로세스의 모든 `EAGLEWorker` 인스턴스에 영향. 단일 프로세스 내 멀티-워커가 없으므로 실무엔 무해.

### 5.5 디스크 패치의 멱등성

모든 `_patch_*` 함수가 sentinel 검사를 먼저 한다. 다만 sentinel 이 변경된 SGLang 버전에 대해서는 silent skip 하므로 (e.g. `extend_lens` 라인이 사라지면 `_inject_oracle_into_worker` 가 warning 후 return), **SGLang 업그레이드 시 sentinel 매칭이 깨졌는지 부팅 로그에서 확인 필요**.

---

## 6. End-to-End 데이터 흐름 정리

```
Pipeline shell                Server process
─────────────────             ─────────────────────────────────────────────
SGLANG_ORACLE_VANILLA=1   ──> install_hook __main__:
                                _start_process_watchdog()
                                install_suffix_algorithm()         # disk patch
                                install_oracle_patch()             # disk patch
                                run_server(...)
                                  EAGLEWorker.__init__:
                                    ... self.extend_lens = ...
                                    patch_eagle_worker_full(self)  # runtime patch
                                      _patch_verify_greedy_func()
                                      _patch_draft_stash()
                                      _patch_verify_logits()
                                      _patch_forward_log()

Agent step                    Server step
─────────────────             ─────────────────────────────────────────────
oracle_pos = get_oracle_log_position()
chat.completions.create(...) ─>  forward_batch_generation:
                                   draft()  ── stash draft tree, path_probs
                                   verify() ── stash logits, accept_lengths
                                   verify_tree_greedy_func:
                                     accept_token_num.fill_(0)     # force 1-token
                                   ─── per-request entry append to
                                       /tmp/sglang_oracle_vanilla.jsonl
                                   ─── per-step timing append to
                                       /tmp/sglang_oracle_timing.jsonl
oracle_entries = read_oracle_log(oracle_pos)
step_data["spec_decode"] = {"oracle_vanilla_entries": oracle_entries}

save_agent_results(...) ─> save_results.py:
                              full JSON  : keep oracle_vanilla_entries
                              *_response : strip → oracle_entries_count int
```

후속 reconstruct / scoring 단계는 위 JSONL entries 를 ground-truth 로 사용해 다양한 가짜 speculator 정책의 hit-rate 와 latency 를 시뮬레이션한다 — 그쪽 문서는 별개.
