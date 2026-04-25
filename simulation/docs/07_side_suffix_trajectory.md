# Side Suffix Trajectory 도구 (`run_side_suffix_trajectory.py`)

파일: `simulation/evaluation/run_side_suffix_trajectory.py` (340 줄). Stage 3 simulator 의 main output (`tree_oracle_sim.json`) 에는 영향이 없는 **side experiment** — Stage 1 EAGLE3 trajectory 를 suffix accept 로 driving 하면서 매 step 의 counterfactual accept 를 별도 JSONL 로 dump 한다. Notebook `simulation/notebooks/side_suffix_trajectory.ipynb` 가 결과를 소비.

## 1. 목적

EAGLE3 가 자연스럽게 만든 step trajectory 는 EAGLE3 자체 accept 에 의해 결정되어 있다. 이걸 "만약 suffix 가 운전했다면 trajectory 가 어떻게 흘렀을까?" 로 바꿔서, 같은 step_idx context 에서 EAGLE3 / suffix / extension 셋이 만드는 accept length 분포를 비교하기 위함. Stage 3 main loop 와 다르게 budget 도 단일이고 method 들이 모두 같은 cache state 를 본다 — 셋의 차이가 순수히 proposer 차이에서 오도록.

`af92e9f` commit 에서 함께 추가됨 (root extension fix 와 한 commit).

## 2. CLI 인자

| Flag | 의미 |
|---|---|
| `--agent-results` (required) | Stage 1 EAGLE3 oracle vanilla JSON |
| `--model` (required) | tokenizer 로딩 (BFCL/SpecBench prompt 재구성) |
| `--budget` (required) | EAGLE3 base budget B (단일 값; default 없음) |
| `--output` (required) | per-step rows JSONL path. `meta.json` 도 같은 디렉토리에 쓴다 |
| `--dataset`, `--responses`, `--exclude` | Stage 3 와 동일 — record assembly 입력 |
| `--req-start`, `--req-end` | request slice (record 단위 아님; `_slice_records_by_request` line 209) |
| `--verify` | smoke mode: 1 request × 1 call × 5 steps 만 처리 후 extension growth assert (line 195, 299) |

`assemble_records_from_artifacts(...)` (line 272) 로 records 로딩 — Stage 3 와 동일 입력.

출력 디렉토리는 `simulation/results/side_suffix_trajectory/` 가 권장 (docstring line 22).

## 3. Trajectory 드라이버

`_collect_per_step(records, budget, verify=False)` (line 106) 의 알고리즘:

```python
cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)

by_req = _group_by_request(records)   # {req_id: {call_idx: {step_idx: rec}}}
seq_req_id = 0

for bfcl_id, calls in by_req.items():
    for call_idx, steps in calls.items():
        prompt = _extract_prompt(steps)        # steps[0]["context_token_ids"] 또는 빈 array
        cache.start_request(seq_req_id, prompt)

        step = min(steps.keys())
        while step in steps:
            rec = steps[step]
            gt = rec["ground_truth_future"]
            per_proposer = rec["per_proposer"]

            # 3 method 모두 같은 cache state 에서 evaluate
            e_acc  = _proposer_tree_walk(per_proposer, "eagle3", gt, budget)
            e_size = _walk_tree_size(per_proposer, "eagle3", budget)
            s_acc, s_size = _live_suffix_walk(cache, seq_req_id, ctx, gt)
            _, ext_size_total = _extension_step(rec, budget, cache, seq_req_id, base_proposer="eagle3")
            ext_base = _extension_step._last_accepted_base
            ext_sfx  = _extension_step._last_accepted_suffix
            ext_base_size = _extension_step._last_base_size

            commit = min(s_acc + 1, len(gt))   # SUFFIX accept 가 trajectory 운전
            per_step_rows.append({...})

            for t in gt[:commit]:               # cache warming: 한 token 씩 feed
                cache.add_active_response(seq_req_id, [int(t)])
            step += commit

        cache.stop_request(seq_req_id)
        seq_req_id += 1
```

핵심 디자인 결정:

1. **단일 글로벌 cache** (line 114): 모든 request 가 같은 cache 를 공유. Stage 3 의 per-method fresh cache 와 다름 — 여기는 method 비교가 목적이 아니라 trajectory 내 measurement 이므로 cache state 일관성이 더 중요.
2. **Suffix-driven advance** (line 170): `commit = min(s_acc + 1, len(gt))`. `+1` 은 verify 의 commit token. eagle3 / extension 의 accept 는 기록만 되고 step pointer 를 움직이지 않는다.
3. **Same-step counterfactual**: 같은 step_idx 의 같은 context 에서 세 method 모두 evaluate, 그 중 suffix accept 만으로 다음 step 위치 결정.
4. **Per-token cache feed** (line 190): `add_active_response(req_id, [int(t)])` 를 한 token 씩 호출. Stage 3 simulator (`run_tree_oracle_sim.py:445`) 도 동일 패턴 (`gt[:advance]` 한 번에) — 둘이 동등한 cache state 를 만드는지는 SuffixDecodingCache 내부 구현에 따라 다르지만 docstring (line 188) 에 의도가 mirror 라고 명시.
5. **Sequential `seq_req_id`** (line 126, 199): cache 가 정수 id 를 받기 때문에 매 (request, call) 마다 0,1,2,... 로 증가. bfcl_id 와는 무관.

## 4. 출력 스키마

### 4.1 Step 별 JSONL

각 줄 (line 172–185):

```jsonc
{
  "request_id": str,         // bfcl_id
  "call_idx": int,
  "step_idx": int,
  "eagle3_acc": int,          // budget-truncated EAGLE3 tree 의 greedy walk 결과
  "eagle3_tree_size": int,    // min(budget, len(tree))
  "suffix_acc": int,          // 라이브 suffix tree (truncate 안 됨) 의 walk 결과 — trajectory 운전
  "suffix_tree_size": int,    // 실제 draw 된 tree 의 노드 수
  "ext_acc_base": int,        // extension walk 중 base tree 안에서 accept 된 step 수
  "ext_acc_sfx":  int,        // extension walk 중 suffix graft 부분에서 accept 된 step 수
  "ext_base_size": int,       // extension 의 base tree 노드 수 (= min(budget, ...))
  "ext_tree_size_total": int, // base + suffix graft 후 총 노드 수
  "advance": int              // commit = suffix_acc + 1 (capped to len(gt))
}
```

Notebook 에서:
- `ext_acc_base + ext_acc_sfx` = extension total accept (walk 가 base→suffix 로 transition 가능).
- `ext_tree_size_total - ext_base_size` = suffix graft 가 추가한 노드 수 (= verify 비용 증가분).
- `eagle3_acc` vs `ext_acc_base` 비교 가능 — 같은 base tree 인데 walk 결과가 다를 수 있는 이유는 extension 의 children dedup index 가 base tree 외에도 suffix anchor 를 인식하기 때문 (실제로는 base 부분에서는 동일).

### 4.2 `meta.json`

`output.with_name("meta.json")` (line 333) — 같은 디렉토리에:

```jsonc
{
  "script": "run_side_suffix_trajectory",
  "agent_results_path": <abs path>,
  "model": "...",
  "budget": int,
  "req_start": int|null, "req_end": int|null,
  "dataset_path": str|null, "responses_path": str|null,
  "verify": bool,
  "n_requests": int,
  "n_calls":    int,
  "n_steps_visited": int,
  "n_steps_with_ext_growth": int,   // ext_tree_size_total > ext_base_size 인 step 수
  "created_at": "<ISO 8601 UTC>"
}
```

`n_steps_with_ext_growth == 0` 이면 `--verify` 가 AssertionError raise (line 299) — extension path 가 깨졌거나 cache warming 이 동작 안 한다는 뜻.

## 5. 상태 / Invariant 주의사항

1. **Tokenizer alignment**: `_extract_prompt` (line 60) 는 `steps[0]["context_token_ids"]` 를 prompt 로 가정한다. step_idx=0 의 context 는 prompt 만 포함 (decoded 가 아직 없음). step_idx=0 이 없는 경우 `np.array([], dtype=np.int32)` 빈 prompt 로 fallback 되며 docstring (line 64) 이 안전하다고 주장 — 이유는 `_live_suffix_walk` 가 항상 `ext_context = ctx_full` 을 통째로 speculate 에 전달하므로 prompt 를 cache 가 따로 알 필요가 없어서.
2. **Cache req id 충돌 없음**: `seq_req_id` 가 0 부터 monotonic 증가하고 `start_request` / `stop_request` 로 명확히 lifecycle 관리.
3. **Fixed budget**: Stage 3 와 다르게 sweep 하지 않는다. budget 별 비교를 원하면 별도 run.
4. **`_extension_step._last_*` side-channel** (line 162–164): `run_tree_oracle_sim._extension_step` 의 함수 attribute 에 의존. 두 호출 사이에 다른 코드가 같은 attribute 를 덮어쓸 수 있으니 같은 process 에서 병렬 호출 금지 (현재는 single-thread).
5. **Suffix-driven advance 가 GT 끝을 벗어나지 못함**: `min(s_acc + 1, len(gt))` (line 170). suffix 가 GT 전체와 일치할 때도 commit 이 안전.
6. **`--verify` 경로의 break**: 한 request × 한 call 만 처리하고 break (lines 201–204). main loop 최상위 break 가 아니라 inner break 들이 if 분기로 쌓여있어서 cache 가 정상적으로 stop 된 뒤 나간다.

## 6. 정적 "Prereq" 참조 (line 262)

스크립트는 시작 시 stderr 에:

> `[prereq] _extension_step() grafts suffix at the virtual root and at every base-tree node.`

를 출력한다. 이는 `_extension_step` (`run_tree_oracle_sim.py:722–753` virtual-root + lines 755+ per-node) 가 **이미** root anchor 를 포함한다는 사실에 대한 reminder — `--verify` 의 runtime assertion 만으로는 root 가 동작한다는 보장이 안 되고, 정적 source 참조가 그 보장을 제공한다.

## 7. Stage 3 메인 시뮬레이터와의 관계

본 도구는 Stage 3 main 의 보조 진단용이고, output 이 main 에 feedback 되지 않는다 (`tree_oracle_sim.json` 무관). Notebook 에서 분석할 때 `tree_oracle_sim.json` 의 budget=B 결과와 cross-reference 가능 — 같은 B 에서 single:suffix MAT, single:eagle3 MAT, extension MAT 와 본 도구의 per-step 평균이 (cache state 차이 modulo) 비슷해야 한다.
