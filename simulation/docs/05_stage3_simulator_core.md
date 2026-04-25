# Stage 3 — Oracle 시뮬레이터 본체 (`run_tree_oracle_sim.py`)

Stage 3는 Stage 1 (`agent_results_eagle3.json`) 과 Stage 2 (`draft_model_drafts.jsonl`) 산출물을 step 단위로 재생하면서 EAGLE3 / draft-LM tree와 라이브 SuffixDecodingCache draft를 합쳐 다양한 method (single, hybrid, extension)에 대한 MAT, accept rate, latency-aware speedup을 계산한다. 본 문서의 line 참조는 모두 `simulation/evaluation/run_tree_oracle_sim.py` 기준.

## 1. CLI 인자와 입력

`main()` (lines 1540–1677) 의 모든 flag.

| Flag | 의미 |
|---|---|
| `--agent-results` (required) | Stage 1 EAGLE3 oracle vanilla JSON. `assemble_records_from_artifacts()` 의 1차 입력 |
| `--draft-model-drafts` | Stage 2 draft-LM JSONL. 없으면 `single:draft_model`, `hybrid_dm`, `extension_dmsfx*` 가 자동 비활성 |
| `--dataset` | BFCL/SpecBench dataset.jsonl, prompt 재구성용 |
| `--responses` | BFCL용 `agent_results_responses.json` (대화 turn 재구성에 필요) |
| `--model` | tokenizer 로드 — chat template 기반 prompt 재구성 |
| `--exclude` | exclude id 파일 (`load_exclude_ids` in `_agent_io.py:24`) |
| `--output` | per-budget 결과를 적을 JSON path |
| `--budgets` (default `1,2,4,8,16,32,64`) | budget B sweep |
| `--latency-config` | `simulation/config/latency/<preset>.json`. 없으면 stub config (vanilla=1ms) |
| `--topk`, `--steps` | EAGLE3 hyperparam — latency table에서 per-(K,S) 행을 고를 때 사용 |
| `--reslice-steps`, `--reslice-topk`, `--capture-steps`, `--capture-topk` | 4 개 묶음 — Stage 1 에서 `SGLANG_CAPTURE_FULL_POOL=1` 로 캡처한 full pool 을 (s', k') sub-tree 로 reslice. `(capture_S, capture_K)` 가 캡처 당시 config, `(reslice_S', reslice_K')` 가 재구성 target. 4 개 중 하나라도 빠지면 `parser.error`. 자세한 건 `09_pool_reslicer.md` |
| `--print-summary` | stderr summary 출력 |

`--output` 또는 `--print-summary` 중 하나는 반드시 있어야 한다 (line 1580).

`assemble_records_from_artifacts()` (`simulation/pipeline/assemble_records.py`) 가 호출돼 per-proposer dict (`per_proposer["eagle3"|"draft_model"|"mtp"]`) 만 담은 step record list 를 돌려준다. Suffix 는 file 에서 로드되지 않고 simulator 내부에서 라이브로 그려진다 (모듈 docstring 참고). union trie 는 더 이상 만들지 않는다.

## 2. Per-record 스키마 (시뮬레이터 입력)

`collect_step_records()` (`assemble_records.py`) 가 step 마다 record를 만든다. Simulator는 이 dict들을 list로 받는다.

```python
{
  "request_id": str,
  "call_idx": int,
  "step_idx": int,
  "context_token_ids": list[int],   # prompt + decoded prefix; suffix speculate에 입력
  "ground_truth_future": list[int], # 이 step부터 시퀀스 끝까지 GT token
  "per_proposer": {
      "eagle3":      {"token_ids": [...], "parents": [...], "size": N,
                      "p_t": ..., "path_draft_p_t": [...]?},
      "draft_model": {"token_ids": [...], "parents": [...], ...},
      "mtp":         {"token_ids": [...], "parents": [...], ...},
  },
}
```

`parents[i] == -1` 이면 root 직속 child. `parents[i] < i` 가 BFS 보장 invariant (`assemble_records.py:145`).

## 3. 외부 루프 — Question / Call / Step 단위

핵심 함수는 `simulate_decoding(records, budget, method, ...)` (line 57). 한 번 호출 = 한 method × 한 budget.

1. `record_index[(req_id, call_idx, step_idx)] = rec`, `sequences[(req_id, call_idx)] = sorted(step_idx list)` 로 인덱싱 (lines 92–102).
2. **Per-method local SuffixDecodingCache**: `suffix_cache` 인자가 sentinel 객체 (`_SUFFIX_ENABLED`, line 1169)면 simulator 내부에서 `_FreshCache(max_tree_depth=64, max_cached_requests=100000)` 를 method 단위로 새로 만든다 (lines 148–156). 이 fix가 `eb21043` 의 핵심 — 이전에는 모든 method가 cache를 공유해서 oracle/realistic 같은 정의상 동일 trajectory가 다른 cache state를 봤다.
3. **Per-(req,call) 처리** (line 158): seq_len 추정 (`last_gt_len <= 1` 이면 마지막 step 가 sentinel 이므로 빼고, 아니면 first step+future len 그대로 — lines 168–173), `cache_req_id = f"{req_id}_{call_idx}"`, `local_cache.start_request(cache_req_id, prompt)` 로 LOCAL tree만 reset (global tree는 누적; 같은 method 내 이전 request의 패턴이 후속 request에 효력).
4. **Step skip-ahead loop** (line 186): `pos` 가 `step_indices` 에 있는 동안 method dispatch → `accepted` → `advance = accepted + 1` (line 360) → `pos += advance` (line 448). `accepted+1` 인 이유: target verify 시 마지막 위치에서 항상 1 token을 확정 commit (verify의 next-token sample) 하므로.
5. record가 비어있는 위치는 vanilla 1 token 진행 (lines 188–201).
6. 끝나면 `seq_len - pos` 만큼의 잔여 token도 vanilla cost로 더한다 (lines 450–462).
7. `local_cache.add_active_response(cache_req_id, gt[:advance])` 로 매 step accept 후 GT prefix를 cache에 feed (lines 442–446) — 다음 step의 suffix speculate가 이 prefix까지 본다.
8. `stop_request` (line 465) 후 다음 (req,call).

## 4. Method dispatch (lines 204-358)

Method 종류는 docstring (lines 9–17) 에 정리. Dispatch 결과는 항상 `accepted: int` (+ 일부 method는 `ext_size`, `used_suffix` 같이 latency 계산용 부산물).

### 4.1 `single:{name}` — `_single_proposer_step` (line 876)

```python
if proposer_name == "suffix":
    tids, pids, _ = _live_suffix_draft(suffix_cache, cache_req_id, base_context)
    return greedy_tree_walk(tids, pids, gt)   # 절대 budget truncate 안 함
return _proposer_tree_walk(rec.per_proposer, proposer_name, gt, budget)
```

`_proposer_tree_walk` (line 852): suffix 가 아닌 경우 `tids[:budget]`, `pids[:budget]` 로 BFS 잘라내고 `parents[i] >= budget` 는 -1 로 강제한다 (root 승격). 즉 budget B 는 **수직 (depth) 가 아니라 노드 수**를 자른다는 뜻.

### 4.2 `hybrid_e3:{t}` / `hybrid_dm:{t}` — `_hybrid_step` (line 537)

```python
sfx_tids, sfx_pids, sfx_score = _live_suffix_draft(...)
use_suffix = (sfx_tids is not None and sfx_score >= threshold)
if use_suffix:
    return greedy_tree_walk(sfx_tids, sfx_pids, gt), True
return _proposer_tree_walk(per_proposer, fallback, gt, budget), False
```

`af92e9f` 의 fix — `compute_latency_speedup` 호출부에서 `suffix_cache=_SUFFIX_ENABLED` 를 hybrid kwargs 에 명시적으로 설정 (lines 1226–1253). 누락되면 `_live_suffix_draft` 가 `(None, None, 0.0)` 을 반환해서 영원히 fallback 만 선택되고, 그 결과 `hybrid_e3_t*_mat == eagle3_mat` 가 되는 silent bug 였다.

### 4.3 `extension*` — `_extension_step` (line 566)

핵심: base proposer (eagle3 또는 draft_model) 의 budget-truncated tree에 **모든 노드 (virtual root 포함)** 를 anchor 로 suffix speculate 결과를 graft.

알고리즘 단계:

1. base tree를 `n = min(budget, len(tids))` 까지 자르고 out-of-range parent를 -1 로 (lines 615–618).
2. 모든 node `i` 에 대해 root→i path 를 미리 계산 (lines 629–636) — `paths[i]` 는 token id 들의 sequence.
3. `path_draft_p_t` (Stage 1 oracle_patch에서 잡힌 누적 draft 확률; `assemble_records.py:243` 에서 record에 attached) 가 있으면 per-node `node_p_t[i] = path_p_t[i] / path_p_t[parent]` 로 분해 (lines 645–664). 없으면 `pt`/`pathprob`/`prune_pt` filter는 silent no-op.
4. **Backbone prune** (`backbone_pt_threshold`, lines 671–699): `path_p_t[i] >= t` 인 노드만 유지, mapping 으로 parents 재배선. `path_p_t` 는 path를 따라 monotone non-increasing 이므로 keep mask가 valid subtree 를 만든다 (orphan 처리 불필요).
5. **Children dedup index** (lines 707–711): `children[parent_idx][token_id] = node_idx` — 같은 (parent, token) pair가 두 번 등장하지 않게 한다.
6. **Virtual-root suffix graft** (lines 722–753): `base_context` 만으로 `suffix_cache.speculate()` 호출 → 반환된 chain을 root child (`tree_parent=-1`) 로 attach. 이 step이 없었다면 EAGLE3 가 position 0 에서 miss 할 때 extension의 walk 도 즉시 종료되어 `extension < single:suffix` 가 되었다 (`af92e9f` 에서 추가).
7. **Per-node suffix graft** (lines 755–816): 각 base node `i` 마다 `ext_context = base_context + paths[i]` 로 speculate → score / pathprob / pt filter 통과하면 chain attach. dedup은 `children` map으로. `local_to_tree[j]` 가 draft index → 확장된 tree index 변환.
8. **Greedy walk + base/suffix split** (lines 818–840): `_children` adjacency rebuild → root 부터 GT를 따라가며 매치되는 child를 잡고 `_picked < n` 이면 `_acc_base += 1`. 결과를 함수 attribute (`_extension_step._last_base_size`, `_last_accepted_base`, `_last_accepted_suffix`, `_last_ext_size_full`) 에 저장 — return signature를 안 깨려는 side-channel.

`*_oracle` variant (lines 210, 226) 는 MAT 는 base extension 과 동일하지만 latency 계산 시 `ext_size = base_size + accepted_in_suffix` 로 줄여서 "useful한 suffix chain만 verify" 시나리오를 재현한다. `*_oracle_path` 는 base 도 accepted path 만 남겨서 더 tight 한 lower bound (lines 232–246).

`*_by_count:r` 는 `cap = max(1, round(B*r))` 로 전체 ext tree 크기를 제한, `*_by_score:t` / `*_by_pathprob:t` / `*_by_pt:t` 는 anchor 필터 (suffix anchor 만 skip, base node 는 유지). `*_prune_pt:t` / `*_prune_pt_oracle:t` 는 base tree 자체를 prune — `_extension_step` 의 `backbone_pt_threshold` 인자 (lines 633-668) 가 `path_p_t < t` 인 base node 와 그 subtree 를 제거. `path_p_t` 가 path 따라 monotone non-increasing 이므로 keep-mask 가 valid subtree 를 만든다 (orphan fixup 불필요). 두 변종의 차이는 oracle cost: `*_oracle` 은 accepted suffix 만 verify cost 로 청구.

`extension_oracle_path` / `extension_dmsfx_oracle_path` (lines 232-244) 는 가장 strict 한 lower bound — base node 도 accepted path 만 cost 로 친다 (`ext_size = base_accepted + suffix_accepted`).

## 5. Acceptance 계산

핵심 primitive는 `greedy_tree_walk(token_ids, parents, ground_truth)` (`tree_knapsack.py:6`):

```python
node = -1
for gt_token in ground_truth:
    matched = False
    for i in range(len(parents)):
        if parents[i] == node and token_ids[i] == gt_token:
            accepted += 1; node = i; matched = True; break
    if not matched: break
return accepted
```

Virtual root (-1) 부터 시작해서 매 GT token마다 현재 node 의 children 중 first match를 잡고 accept count를 증가, miss 시 즉시 종료. 즉 **한 path 만** 따라가며 matching prefix length를 센다. `extension_step` 은 재구성된 children index로 인라인 walk하지만 의미는 동일 (lines 818–840).

GT 길이 == 1 이면 step 이 trivial (마지막 token 이고 verify 할 future가 없음) — `simulate_decoding` 의 seq_len 보정 (line 170) 이 이를 처리한다.

## 6. Hybrid live-speculate 와 cache state 버그 수정 (eb21043 / af92e9f)

설명:

1. **Cache instantiation scope** (`eb21043`): `simulate_decoding` 진입 시 `_FreshCache` 생성 (lines 148–154). Method × budget 호출마다 새 cache. Global tree 누적은 method 내부 request 사이에서만 일어나며 `per-(req,call)` 는 `start_request` 로 LOCAL tree reset (line 180).
   - **잘못된 옛 동작**: 단일 cache를 sweep 전체 공유 → method A 의 trajectory가 method B의 speculate 결과를 오염. Oracle MAT < realistic MAT 라는 정의상 불가능한 상황 발생.
   - **현재 올바른 동작**: 같은 (method, budget) 안에서 request 들이 순서대로 처리되며 cache에 trajectory가 누적, 다음 method 호출 시 cache는 비어있는 상태에서 다시 시작.
2. **Hybrid suffix kwarg plumb** (`af92e9f`): `compute_latency_speedup` 가 hybrid 호출 시 `"suffix_cache": _SUFFIX_ENABLED` 를 kwargs 에 포함 (lines 1229, 1248). 빠지면 `local_cache=None` → `_live_suffix_draft` 가 항상 fail → suffix 분기 영원히 미선택.
3. **Virtual-root extension** (`af92e9f`): `_extension_step` lines 722–753. EAGLE3 root miss 상황에서도 suffix가 sibling 으로 합쳐져서 `extension MAT >= single:suffix MAT` (cache state 동일 가정) invariant 가 성립.

검증 방법: `extension_oracle_mat >= extension_mat`, `extension_mat >= single:eagle3_mat`, `extension_mat >= single:suffix_mat` 가 같은 budget/method scope에서 성립해야 한다.

## 7. Budget 별 집계

`simulate_decoding` 의 return dict (lines 482–509):

```python
{
  "total_generated": ...,   # accepted + commit token 누적
  "total_accepted":  ...,   # accepted 만
  "total_steps":     ...,   # iteration count
  "total_time_ms":   ...,   # ratio-free flat time (verify_latency_ms 또는 v_ms 사용)
  "vanilla_time_ms": total_generated * vanilla_latency_ms,
  "speedup":         vanilla_time_ms / total_time_ms,
  "mat":             total_accepted / total_steps,
  "speedup_per_ratio":        {0.05: ..., 0.10: ..., 0.20: ..., 0.30: ..., 0.50: ...},
  "speedup_per_ratio_always": {...},   # hybrid 만 — draft cost 매 step
  "speedup_real":             vanilla_time_ms / total_time_real_ms,
  "speedup_real_always":      vanilla_time_ms / total_time_real_always_ms,  # hybrid only
  "total_time_real_ms": ..., "total_target_ms": ..., "total_draft_ms": ...,
  "total_target_tokens": ..., "total_target_tokens_sq": ...,
  "total_target_tokens_min": ..., "total_target_tokens_max": ...,
}
```

**Ratio-based cost** (lines 366–379): `step_cost = vanilla_ms * (1 + ratio)` for methods with draft cost; `vanilla_ms` 만 for `no_draft = method == "single:suffix"`. Hybrid 의 `time_per_ratio` 는 conditional (suffix branch 시 draft 무료), `time_per_ratio_always` 는 every step에 draft 비용.

**Real-cost path** (lines 381–439): `real_step_target_fn` (보통 `_target_forward`) 이 주어지고 dispatch가 `ext_size` 를 채우면

```
step_real = real_step_target_fn(ext_size) + real_step_draft_only_ms
```

로 계산. ext_size 가 None 이거나 fn 미설정이면 hybrid 는 `real_step_cost_ms` (또는 suffix branch의 `real_step_cost_suffix_ms`) flat 으로 fallback (lines 405–425), single:eagle3/draft_model 은 flat `real_step_cost_ms` (lines 426–439).

각 step 마다 target_tokens (= verified tree size) 를 누적 합/제곱합/min/max 로 추적하므로 notebook 에서 분포 분석이 가능하다.

## 8. Latency 모델 조회 (`compute_latency_speedup`, line 913)

### 8.1 Latency config 스키마

`simulation/config/latency/qwen3_8b.json` 예시 — `compile_latency_config.py` 가 측정 산출물에서 합성한다:

```jsonc
{
  "vanilla_step_ms": 26.16,                // baseline TPOT, 측정 또는 --vanilla-tpot-ms
  "target_forward_ms": {"4": ..., "8": ..., ..., "512": ...},  // legacy flat (cross-topk median)
  "eagle3_draft_ms":   {"4": ..., ..., "512": ...},            // legacy flat (canonical topk/steps)
  "eagle3_draft_ms_by_steps":      {"S": {"B": ms}},           // legacy flat (canonical topk)
  "target_forward_ms_by_topk":     {"K": {"B": ms}},           // 새 schema
  "eagle3_draft_ms_by_topk_steps": {"K": {"S": {"B": ms}}},
  "draft_lm_tpot_ms": 31.68,                                   // per-token draft model fwd
  "suffix_speculate_ms": 0.013,                                // per call
  "_metadata": {"target_model", "draft_model", "draft_lm",
                "canonical_topk", "canonical_steps", "available_topks", ...},
}
```

측정 파이프라인 요약 — 자세한 설명은 본 문서 범위 밖, 어떤 필드를 채우는지만 정리:

| script | 출력 필드 |
|---|---|
| `measure_eagle3_cost.py` | `target_forward_ms_by_topk[K][B]`, `eagle3_draft_ms_by_topk_steps[K][S][B]` (실제 SGLang EAGLE3 oracle latency 모드) |
| `measure_draft_model_cost.py` | `draft_lm_tpot_ms_by_n` → `draft_lm_tpot_ms` (canonical n=3) |
| `measure_suffix_cost.py` | `suffix_speculate_ms_by_workload` → median |
| `measure_step_latency.py` / `measure_verify_latency.py` / `measure_latency.py` | 옛 prototype 측정. 현재는 `compile_latency_config.py` 가 위 3 개를 결합 |
| `bench_eagle3_configs.py` | end-to-end speedup 검증 (config compile 의 결과 검증용) |
| `compile_latency_config.py` | 위 3 개 cost JSON → 단일 `latency_config.json` 합성 |
| `calibrate_latency.py` | ad-hoc TPOT 측정 (vanilla baseline 만) — production pipeline 과는 별도 |

### 8.2 Topk-aware 조회 (lines 949–993)

`_pick_topk_table(table_by_k, label)` (line 949): `--topk` 정수에 해당하는 key 가 있으면 그 행, 없으면 nearest topk 로 fallback + WARN 출력. EAGLE3 draft 도 동일하게 (K, S) 검색.

선택된 표가 비어있으면 legacy flat (`target_forward_ms`, `eagle3_draft_ms`) 로 fallback (lines 991–1002). flat 도 비어있는데 legacy verify_latencies 만 있다면 `target_fwd[B] = vanilla_ms`, `eagle3_draft[B] = max(step - vanilla, 0)` 로 분리 (lines 998–1002).

### 8.3 보간 (`_interp`, line 1013)

- key 직접 매치 → 그대로
- B ≤ 최소 key → vanilla_ms 와 그 key 사이 선형 보간 (B==1 이면 vanilla_ms 그대로)
- B ≥ 최대 key → 마지막 두 점 기반 선형 외삽 (extension에서 ext_size 가 측정 범위 초과 가능)
- 그 사이 → piecewise linear

### 8.4 Step cost 구성 (lines 1056–1132)

```python
def _target_forward(B):  return _interp(target_fwd, B, vanilla_ms)
def _eagle3_draft(B):    return _interp(eagle3_draft, B, 0.0)

def _proposer_draft_cost(name, B, suffix_matches=1):
    if name == "eagle3":      return _eagle3_draft(B)
    elif name == "draft_model": return min(B, MAX_DRAFT_MODEL_N) * draft_lm_tpot
    elif name == "suffix":      return suffix_matches * suffix_speculate_ms
    elif name == "mtp":         return 0.0
```

`MAX_DRAFT_MODEL_N=16` (line 1011) — Stage 2 의 `--max-draft-tokens` hard cap. 그 이상 budget 에서 draft model 은 더 이상 forward 하지 않는다.

`_step_cost(active_proposers, B) = target_forward(B) + max(draft_costs)` (line 1097): proposer 들은 GPU 에서 parallel 이므로 합이 아니라 max. Suffix 는 CPU side 라 GPU forward 와 overlap, 그러나 보수적으로 max 안에 포함.

`_real_cost(active, B, suffix_matches=1, verify_tokens=None)` (line 1113): `verify_tokens` 로 target_forward 의 인자를 별도 지정 가능 — `single:draft_model` 의 경우 chain 길이가 `min(B, 16)` 이므로 target도 그만큼만 verify.

### 8.5 Budget 별 method wiring (lines 1188–1454)

Budget B 한 개에 대해:

- **single:** dispatcher 마다 `draft_only` 계산 → `real_step_cost_ms = _real_cost([pname], B, verify_tokens=...)`, `real_step_target_fn = _target_forward`, `real_step_draft_only_ms = draft_only`. simulator 가 step 마다 `_target_forward(actual_tree_size) + draft_only` 로 dynamic 계산.
- **hybrid_e3 / hybrid_dm:** `real_step_cost_ms` (fallback 분기 cost), `real_step_cost_suffix_ms = _target_forward(B) + suffix_speculate_ms` (suffix 분기 cost), `real_step_target_fn` + `real_step_draft_only_ms = suffix_speculate_ms` 로 suffix branch 가 dynamic ext_size.
- **extension / extension_oracle / extension_by_*:** `e3_nodes = min(B, 16*8)` 로 EAGLE3 node 수 추정 (topk×steps default), `ext_draft_only = max(_eagle3_draft(B), e3_nodes * suffix_speculate_ms)`, `ext_cost_fallback = _target_forward(B) + ext_draft_only`. `real_step_target_fn` 으로 dynamic ext_size verify.
- **extension_dmsfx*:** `dm_nodes = min(B, MAX_DRAFT_MODEL_N)` 로 draft-model linear chain, draft cost = `max(dm_nodes*draft_lm_tpot, dm_nodes*suffix_speculate_ms)`.

`thresholds = [0.1, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]` (line 1221) hybrid threshold; extension_by_score 는 `+ [25, 30, 35]` (line 1330). count caps `r ∈ {2,4,8}`, count×score combos `r ∈ {2,4} × t ∈ {1,3,10}`. `pathprob` / `pt` / `prune_pt` 는 `path_draft_p_t` 가 record에 있을 때만 동작 (lines 1343–1393).

### 8.6 최종 speedup 공식

```
speedup       = (total_generated * vanilla_step_ms) / total_time_ms          # ratio-free flat
speedup_real  = (total_generated * vanilla_step_ms) / total_time_real_ms     # measured latency
speedup_per_ratio[r] = vanilla_time_ms / time_per_ratio[r]                   # vanilla * (1+r) per step
```

`total_generated = Σ (accepted + 1)` 가 vanilla 로 만들었다면 걸렸을 step 수 (각 step 1 token).

## 9. 출력 JSON (line 1635–1676)

```jsonc
{
  "metadata": {"input_source": <agent_results path>,
               "n_steps": len(records),
               "budgets": [...]},
  "latency": {
    "vanilla_step_ms": ...,
    "proposers": ["draft_model", "eagle3", "mtp"?, "suffix"],
    "pairs":     ["draft_model+eagle3", ...],   // 보고용 메타
    "has_latency_config": bool,
    "budget_sweep": [
      {
        "budget": B,
        "target_forward_ms": _target_forward(B),
        "eagle3_draft_ms":   _eagle3_draft(B),
        "draft_lm_tpot_ms":  draft_lm_tpot,
        // 메소드별 컬럼 (compute_latency_speedup의 _store_sim, line 1134 참고):
        "<prefix>_mat":          ...,
        "<prefix>_steps":        ...,
        "<prefix>_speedup_r0.05": ...,  // ... r0.10, r0.20, r0.30, r0.50
        "<prefix>_always_speedup_r*": ..., // hybrid only
        "<prefix>_speedup_real":          ...,
        "<prefix>_always_speedup_real":   ..., // hybrid only
        "<prefix>_total_time_real_ms":    ...,
        "<prefix>_total_target_ms":       ...,
        "<prefix>_total_draft_ms":        ...,
        "<prefix>_total_target_tokens":   ...,
        "<prefix>_total_target_tokens_sq":  ...,
        "<prefix>_total_target_tokens_min": ...,
        "<prefix>_total_target_tokens_max": ...,
      }
    ],
    "note": "latency_config not provided; ..."?    // stub config 시
  }
}
```

`<prefix>` 는 `_run(..., prefix)` 두 번째 위치 인자 — `eagle3`, `draft_model`, `suffix`, `hybrid_e3_t1.0`, `extension`, `extension_oracle`, `extension_by_count_r4`, `extension_by_score_t10.0`, `extension_dmsfx_by_count_score_r4_t3.0` 등 (lines 1217, 1237, 1253, 1282, 1292, 1313, 1326, 1337, 1393, 1413, 1420, 1427, 1435, 1446, 1454).

`print_latency_summary` (line 1465) 는 `--print-summary` 시 stderr 에 method × budget table 을 그리고 prefix 별 best speedup 을 요약한다.

## 10. 정확성 체크리스트

다음 invariants 가 같은 (workload, budget, topk, steps) scope 에서 성립해야 한다 — 논리상 자명하지만 cache state bug 가 있을 때 깨졌었다:

1. `extension_oracle_mat == extension_mat` (oracle 은 cost 만 줄이고 MAT 은 동일 정의).
2. `extension_mat >= single:eagle3_mat` (extension은 EAGLE3 backbone에 노드 추가만 함).
3. `extension_mat >= single:suffix_mat` (virtual-root graft 덕분; cache state 동일 가정 — `af92e9f`).
4. `hybrid_e3:t_mat >= single:eagle3_mat` (suffix score < t 면 그대로 EAGLE3, 아니면 더 높은 confidence suffix 선택).
5. `hybrid_e3:0.0_mat == single:suffix_mat` 에 거의 가까움 (모든 step에서 suffix가 threshold 통과).
6. `extension_oracle_path_mat == extension_mat` 이지만 `speedup_real >= extension_oracle_speedup_real >= extension_speedup_real`.
7. budget 증가 시 모든 method 의 MAT 는 monotonic non-decreasing.
