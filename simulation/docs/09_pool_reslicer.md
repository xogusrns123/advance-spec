# EAGLE3 Pool Reslicer (`pool_reslicer.py`)

파일: `simulation/pipeline/pool_reslicer.py` (전체 195 줄). Stage 1 에서 한 번 capture 한 EAGLE3 full score pool 을 받아, 더 작은 `(s', k')` sub-config 의 per-step tree 로 다시 잘라낸다 (reslice). 같은 capture artifact 한 벌로 여러 `(S, K)` 조합을 별도 SGLang run 없이 simulator 에서 비교 가능.

## 1. 목적

원래 sweep 방식: 각 `(S, K)` 마다 Stage 1 EAGLE3 run 을 다시 돌려야 했음 (`run_tree_oracle_sim.py` 는 Stage 1 에서 만들어진 truncated tree 를 그대로 소비). GPU 시간이 K · S grid 에 비례.

Reslicer 는 한 번의 RR Stage 1 capture (가장 큰 `(S=S_max, K=K_max)` 로) 만 돌리고, 그 안에 들어 있는 **full pool** (truncate 전 score pool 전체) 로부터 임의의 `(s' ≤ S_max, k' ≤ K_max)` 의 tree 를 simulator-time 에 재구성한다. `simulation/scripts/experiments/run_reslice_sweep.py` 가 한 capture artifact 위에서 여러 `(s, k)` 조합을 직렬로 돌리는 것이 typical use.

핵심 가정: SGLang 의 `organize_draft_results` 가 path-prob 기준 top-k truncation 으로 tree 를 잘라낸다는 점 — 따라서 `(s', k')` 가 `(S, K)` 의 prefix-and-narrow 이면 같은 path_prob 랭킹을 reslicer 가 재현한다.

## 2. 캡처 측 (Stage 1)

Capture 는 `oracle_patch.py:341–396` 의 `_install_draft_p_t_tracer` 안 `capture_full_pool` 분기에서 일어난다. 환경변수 `SGLANG_CAPTURE_FULL_POOL=1` 일 때 활성.

동작:

1. SGLang 의 `organize_draft_results` 를 한 step 당 **두 번** 호출.
2. 첫 호출 (line 366): `num_draft_token = pool_size + 1` 로 호출 → truncate 없이 full pool 노드를 모두 BFS-list 로 받음. 입력 텐서들은 `clone()` 한 사본 (line 363–365) — SGLang 이 functional 이지만 방어적.
3. 두 번째 호출 (line 399): 원래 인자로 호출 → 평소대로 verify-side truncated tree. **Generation 결과는 이쪽만 본다**. cloned 입력으로 첫 호출이 두 번째에 영향 없음.
4. 첫 호출 결과는 `ew_module._oracle_last_full_pool` 에 stash → Stage 1 record dump 시 per-step `eagle3_pool_full` 필드로 직렬화.

스키마 (`pool_reslicer.py:5–12`, capture site `oracle_patch.py:377–382`):

| 필드 | shape | 의미 |
|---|---|---|
| `draft_tokens`  | `[pool_size]`        | pool position p 의 token id |
| `parent_list`   | `[(S-1)·K + 1]`      | step 별 alive-parent 의 pool position (concat) |
| `path_probs`    | `[pool_size + 1]`    | cumulative path prob, idx 0 은 root sentinel = 1.0 |
| `pool_size`     | scalar               | `K + (S-1)·K²` |

`path_probs` 가 cumulative product (= ∏ child-conditional probs along root→node path) 라는 점은 SGLang 의 `score_list` 가 step 마다 parent-conditional softmax 를 누적해서 들어오는 구조에 의존한다 — `oracle_patch.py:368` 의 `torch.gather(flat_scores, 1, full_top_idx)` 가 그 누적값을 읽고 root column 1.0 을 prepend.

## 3. Pool layout

`(S, K)` 캡처에서 pool 의 BFS 구조:

- step 0 (root 의 child): pool index `[0, K)` — K 개
- step i ≥ 1: pool index `[K + (i-1)·K², K + i·K²)` — K² 개 (= K alive parents × K children per parent)

총 `pool_size = K + (S-1)·K²`.

`parent_list` (= SGLang `select_top_k_tokens` 의 `cat(parents_list[:-1])`) 의 layout (`pool_reslicer.py:20–28`):

```
idx 0:                     -1                              (root sentinel)
idx 1..K:                  step 0 의 K alive going INTO step 1   (= [0..K-1])
idx K+1+i·K..K+1+(i+1)·K:  step i+1 의 K alive going INTO step i+2  (i ∈ [0, S-2))
```

총 길이 `(S-1)·K + 1`.

`_build_pool_parents` (`pool_reslicer.py:49`) 가 각 pool position p 의 부모 pool position 을 계산해 `pool_parents[p]` 로 반환 — root child (p < K) 는 -1.

## 4. Reslice 알고리즘

`reslice_eagle3_pool(draft_tokens, parent_list, path_probs, pool_size, S, K, s_prime, k_prime)` (`pool_reslicer.py:93`).

```
step 0:
  step0_candidates = [0, 1, ..., K-1]                   # root 의 K children
  kept_step_0      = top-k' of step0_candidates by path_probs

for i in 1..s'-1:
  for parent_pp in alive_pool_pos[i-1]:                  # 직전 step 의 k' alive
    s_orig = orig_alive_slot(parent_pp)                  # 0..K-1
    block  = pool index [K + (i-1)·K² + s_orig·K, ... + K)
    children_sorted = top-k' by parent-conditional score
                    = path_probs[c+1] / path_probs[parent_pp+1]
    kept_per_step[i] += children_sorted[:k']
  alive_pool_pos[i] = top-k' of kept_per_step[i] by path_probs   # 다음 step alive
```

핵심:

1. **Step 0 ranking** (line 135): root children K 개 중 path_prob top-k'. Cumulative product 정의상 path_probs[c+1] = child-conditional prob 와 동일 (parent path_prob = 1.0).
2. **Step i ≥ 1 의 per-parent ranking** (line 158–166): parent 별 K 형제 안에서만 top-k' 를 뽑는다. 정렬 키는 child-conditional prob `path_probs[c+1] / path_probs[parent+1]`. 이는 SGLang 의 `select_top_k_tokens` 가 step 마다 parent 별 k 를 뽑는 동작과 매칭. `parent_pp_prob <= 0` (degenerate) 시 cumulative path_prob fallback (line 159–162).
3. **Next-step alive (line 172)**: 이번 step 에 keep 된 `k'·k'` 개 중 cumulative path_prob 기준 top-k' 가 다음 step 의 alive parent. 이게 `parent_list` 의 next chunk 역할을 reslice space 에서 재현.

`_orig_alive_step` (`pool_reslicer.py:81`) 이 원본 capture 의 step별 alive 를 꺼내고, reslicer 는 `orig_alive` 안에서의 `slot_of[parent_pp]` 로 pool block 을 인덱싱한다 (line 156). Resliced alive 가 항상 original alive 의 subset 이라는 invariant — 매 step 에서 K 개 alive 중 k' ≤ K 개를 뽑기 때문.

Resliced tree 크기: layer 1 = k', layer 2..s' = 각 k'² → 총 `k' + (s' - 1)·k'²` (root 제외).

`(s', k') == (S, K)` 일 때 출력은 원본 full pool tree 와 동일.

## 5. 입력 검증

`reslice_eagle3_pool` 진입부 (line 116–129) 의 dimension check:

| 검증 | 식 |
|---|---|
| range  | `1 ≤ s_prime ≤ S`, `1 ≤ k_prime ≤ K` |
| pool   | `len(draft_tokens) == pool_size` |
| probs  | `len(path_probs) == pool_size + 1` |
| parent | `len(parent_list) == (S - 1)·K + 1` |
| layout | `pool_size == K + (S - 1)·K²` (`_build_pool_parents` line 57–61) |

위반 시 `ValueError`. `assemble_records.py` 가 이걸 잡아서 fallback (§7).

## 6. 출력 포맷

Return: `(sub_token_ids: List[int], sub_parents: List[int], sub_path_probs: List[float])` (`pool_reslicer.py:100`).

- `sub_token_ids[i]` — kept child 의 token id (root 미포함)
- `sub_parents[i]`   — resliced list 안 부모 인덱스, root 직속 child 면 -1
- `sub_path_probs[i]`— cumulative path prob

format 은 Stage 1 의 per-step `eagle3_tree` 와 동일 → `assemble_records.py:153` 가 `proposer_trees["eagle3"] = (sub_ids, sub_par)` 로 그대로 attach 하고, simulator 의 `greedy_tree_walk` (`tree_knapsack.py`) 와 budget truncation (`run_tree_oracle_sim.py:618`) 이 수정 없이 동작한다.

`sub_path_probs` 는 `eagle3_path_draft_p_t` 로 따로 저장 (`assemble_records.py:154`) — knapsack DP 가 아닌 latency report 부 분석용.

## 7. CLI / 호출 경로

### CLI flags (`run_tree_oracle_sim.py:1578–1604`)

4 개를 묶음으로 사용:

| Flag | 역할 |
|---|---|
| `--reslice-steps`  | 새 `s'` (resliced depth) |
| `--reslice-topk`   | 새 `k'` (resliced per-parent topk) |
| `--capture-steps`  | 원본 capture `S` (e.g. 8) |
| `--capture-topk`   | 원본 capture `K` (e.g. 16) |

네 개 중 하나라도 빠지면 `parser.error` (line 1599–1602). 모두 set 시 `eagle3_reslice = (S, K, s', k')` 가 `assemble_records_from_artifacts(...)` 로 전달.

### Reslice 분기 (`assemble_records.py:139–164`)

```
if reslice_args is not None and e3_pool_fulls and call_idx < len(e3_pool_fulls):
    fp = call_pools[pos]
    if fp is not None:
        try:
            sub_ids, sub_par, sub_pp = reslicer_fn(fp["draft_tokens"], fp["parent_list"],
                                                   fp["path_probs"], fp["pool_size"],
                                                   S_orig, K_orig, s_p, k_p)
            proposer_trees["eagle3"] = (sub_ids, sub_par)
            e3_attached = True
        except Exception as e:
            # silent fallback to original truncated tree
            ...

if not e3_attached and e3_trees and ...:
    # use Stage 1's truncated tree as-is
```

Fallback 동작:

1. Pool data 가 없는 step (capture 가 부분적 / step 누락) → 자동으로 원본 truncated tree 사용. silent.
2. Reslicer 가 raise (dimension mismatch / corrupt entry) → 첫 1 회만 stderr warn (`_reslice_warned` flag, line 159–164), 이후는 silent. 그 step 도 원본 tree 로 fallback.
3. 둘 다 없으면 flat chain (assemble_records.py 의 일반 fallback, 본 모듈 외).

## 8. 위험 / 주의사항

1. **`eagle3_p_t` 손실**: Stage 1 의 truncated tree 와 함께 dump 되는 target-side verify prob (`per_call_eagle3_tree_p_ts`) 는 reslice path 에서 재구성되지 **않는다**. `assemble_records.py:130` 의 `eagle3_p_t = None` 이 그 자리. Verify-side prob 를 쓰는 분석 (e.g. reject-sampling 시뮬, threshold based accept) 은 reslice 모드에서 부정확 — 현재 simulator 는 greedy walk (target prob 무시) 만 쓰므로 영향 없음.
2. **Cumulative product 가정**: §2 의 `path_probs` 가 root→node cumulative product 라는 가정은 SGLang `score_list` 누적 동작에 의존. SGLang 업그레이드 시 누적 방식이 바뀌면 reslicer 의 ranking 이 silent 하게 틀릴 수 있다 — capture 측 (`oracle_patch.py:368`) 에 detection 없음.
3. **Sanity unit test 부재**: `reslice_eagle3_pool(..., S, K, S, K)` 가 원본 truncated tree 와 노드 set 으로 동일해야 한다는 invariant 의 자동 검증이 없다. 손으로 한 번 확인했으나 regression guard 없음. 추가하려면 한 step 의 full pool + truncated tree 두 개를 fixture 로 두고 keep set 비교.
4. **`(s', k')` 가 `(S, K)` 의 strict sub** 만 가능. 더 큰 `(s'', k'')` 로의 확장 (extrapolation) 은 정의 자체가 불가 — pool 에 없는 후보를 만들 수 없다. CLI 검증은 `1 ≤ s' ≤ S` (line 116) 으로 강제.
5. **`(s', k')` 별 parent_list 길이 invariant**: reslicer 는 새 `parent_list` 를 따로 만들지 않고 알고리즘 내부에서만 alive 추적. 출력은 `(token_ids, parents, path_probs)` 셋 뿐 — 다른 코드가 reslice 결과 위에서 또 reslice 하려 하면 (현재 호출자 없음) full pool 을 다시 받아야 한다.
6. **Reslice 와 base capture 의 budget 대소**: simulator 가 budget B 로 truncate 할 때 (`run_tree_oracle_sim.py:618`) resliced tree 크기 `1 + k' + (s'-1)·k'²` 가 B 보다 클 수도 작을 수도 있다 — 둘 다 정상. B 가 큰 쪽이면 BFS 끝까지 사용, 작으면 BFS 앞쪽만.

## 9. 호출 위치 요약

| File:Line | 역할 |
|---|---|
| `simulation/pipeline/pool_reslicer.py:93`              | `reslice_eagle3_pool` — 알고리즘 본체 |
| `simulation/pipeline/pool_reslicer.py:49`              | `_build_pool_parents` — pool position → parent pool position |
| `simulation/pipeline/pool_reslicer.py:81`              | `_orig_alive_step` — 원본 capture 의 step 별 alive |
| `simulation/pipeline/assemble_records.py:89–93`        | reslicer import 및 `reslice_args` 셋업 |
| `simulation/pipeline/assemble_records.py:139–164`      | per-step reslice 분기 + fallback warn |
| `simulation/oracle/oracle_patch.py:341–396`            | full pool capture (`SGLANG_CAPTURE_FULL_POOL=1`) |
| `simulation/evaluation/run_tree_oracle_sim.py:1578–1604` | `--reslice-steps/topk + --capture-steps/topk` flags |
| `simulation/scripts/experiments/run_reslice_sweep.py`   | 한 capture artifact 위에서 여러 (s, k) 조합 sweep orchestrator |
