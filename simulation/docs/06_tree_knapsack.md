# `tree_knapsack.py` — Greedy Tree Walk 알고리즘

파일: `simulation/evaluation/tree_knapsack.py` (전체 28 줄).

이름이 `tree_knapsack` 이지만 현재 단계에서는 **knapsack 알고리즘이 아니다**. 본 모듈은 한 함수만 export 한다 — `greedy_tree_walk(token_ids, parents, ground_truth) -> int`. 이름은 옛 단계 (DP-기반 budget allocation 실험) 의 잔재이고, Stage 3 simulator 가 실제로 사용하는 의미는 "draft tree 와 ground-truth token sequence 의 longest matched prefix length" 이다.

## 1. 입력

| 인자 | 타입 | 의미 |
|---|---|---|
| `token_ids`  | `list[int]` length N | 트리 각 노드의 token id, BFS 순으로 직렬화 |
| `parents`    | `list[int]` length N | `parents[i]` 는 노드 i 의 부모 인덱스. -1 이면 virtual root 의 직속 child |
| `ground_truth` | `list[int]` | 이 step 에서 시퀀스 끝까지의 GT token sequence (`record["ground_truth_future"]`) |

Invariant: `parents[i] < i` (BFS 순서 보장 — `oracle_patch.py` 가 EAGLE3 트리를 BFS 순으로 dump 하고, draft model chain 은 정의상 BFS). Stage 3 simulator는 base tree 를 truncate 한 뒤 `pids = [p if p < n else -1 for p in pids]` 로 root 승격을 해서 invariant 를 보존한다 (`run_tree_oracle_sim.py:618`).

## 2. 알고리즘

```python
def greedy_tree_walk(token_ids, parents, ground_truth) -> int:
    accepted = 0
    node = -1   # virtual root
    for gt_token in ground_truth:
        matched = False
        for i in range(len(parents)):
            if parents[i] == node and token_ids[i] == gt_token:
                accepted += 1
                node = i
                matched = True
                break
        if not matched:
            break
    return accepted
```

- Virtual root (-1) 에서 시작.
- 매 GT token 마다 현재 노드의 children 을 linear scan (O(N) per step) 해서 `token_id == gt_token` 인 첫 child 를 잡는다.
- 매치되면 `accepted += 1`, 현재 노드를 그 child 로 이동하고 다음 GT token 으로.
- 미스 시 즉시 종료. 즉 한 번이라도 어긋나면 그 뒤는 관심 없음.

복잡도: `O(|ground_truth| * N)` — 각 step 마다 children scan. `_extension_step` 의 인라인 walk (`run_tree_oracle_sim.py:822–840`) 는 `defaultdict(list)` 로 children adjacency 를 미리 만들어서 `O(|gt| * deg(node))` 로 줄인다.

## 3. 정확성 논증

알고리즘이 반환하는 `accepted` 는 트리 안 root→leaf path 중 GT prefix 와 일치하는 가장 긴 prefix 의 길이를 반환한다. 증명 스케치:

1. 트리는 trie-invariant — same (parent, token_id) pair 가 두 번 나오지 않는다 (`run_tree_oracle_sim.py:706–711` 의 `children` dedup; assemble_records 가 만드는 per-proposer 트리도 BFS-trie 구조라 같은 invariant 보유).
2. 따라서 root 에서 GT prefix `gt[:k]` 와 일치하는 path 가 존재할 경우 unique 하고, 매 단계 `node` 의 children 중 `token_id == gt[i]` 인 child 도 unique 하다 → first match 가 곧 the match.
3. 첫 mismatch 가 발생한 GT 위치 k 에서 트리 안에 `gt[:k+1]` 와 일치하는 path 는 (1) 에 의해 존재하지 않으므로 더 긴 매치가 가능할 수 없다.

따라서 `accepted = longest_matched_prefix_length(tree, gt)` 가 invariant. Speculative decoding 의 표준 acceptance 정의 — verifier가 한 path 를 따라가다 첫 미스에서 끊는 — 와 정확히 일치한다.

## 4. 엣지 케이스

- `len(parents) == 0` 또는 `len(ground_truth) == 0` → 즉시 0 반환 (loop 미진입).
- `parents[i] == -1` 이지만 동일 token 을 가진 root child 가 여러 개 → first 가 선택. dedup 로 인해 이런 case 는 정상 input 에서 나오지 않지만 만약 상위 layer 에서 dedup 을 깼다면 first wins, deterministic.
- Tree 가 chain (linear) 이면 path matching 은 자동.
- GT 가 트리보다 길어서 트리 안 leaf 를 지나는 경우 → 마지막 leaf 에서 children 없으니 `matched=False` 로 break.

## 5. "Knapsack" / Budget allocation 부분이 없는 이유

Budget B 에 의한 트리 truncation 은 본 함수 진입 전에 처리된다:

- `_proposer_tree_walk` (`run_tree_oracle_sim.py:852`):
  ```python
  if name != "suffix" and budget < len(tids):
      tids = tids[:budget]; pids = pids[:budget]
      pids = [p if p < budget else -1 for p in pids]
  return greedy_tree_walk(tids, pids, gt)
  ```
  BFS 첫 B 개 노드만 남기고 잘라낸 뒤 호출. Suffix 는 truncate 하지 않음 (CPU draft 라 free).
- `_extension_step` (`run_tree_oracle_sim.py:614–618`): base tree 를 `min(B, len)` 으로 자르고 (`pids` 참조 정정), 그 위에 suffix graft 후 인라인 walk.
- `extension_by_count:r` 같은 cap 변형은 `max_count = max(1, round(B*r))` 로 graft 도중 stop (`_extension_step` lines 747, 810).

따라서 본 모듈은 "budget-aware tree" 가 이미 만들어진 상태에서 acceptance length 만 계산한다. DP 기반 knapsack 코드는 현재 pipeline 에 없다 — "tree knapsack" 이라는 이름은 옛 prototyping 잔재.

## 6. 호출 위치

`greedy_tree_walk` 는 다음 곳에서 호출된다:

| File:Line | 용도 |
|---|---|
| `simulation/evaluation/run_tree_oracle_sim.py:559` | `_hybrid_step` — suffix 분기 시 raw suffix tree walk |
| `simulation/evaluation/run_tree_oracle_sim.py:626` | `_extension_step` — `suffix_cache` 가 None 인 fallback 경로 |
| `simulation/evaluation/run_tree_oracle_sim.py:873` | `_proposer_tree_walk` — single proposer truncated tree |
| `simulation/evaluation/run_tree_oracle_sim.py:895` | `_single_proposer_step` — `single:suffix` 의 라이브 draft walk |
| `simulation/evaluation/run_side_suffix_trajectory.py:102` | `_live_suffix_walk` — side experiment |

`_extension_step` 의 main path (suffix_cache 가 살아있는 정상 경로) 는 `greedy_tree_walk` 를 직접 호출하지 않고 lines 822–840 의 인라인 변형을 쓴다 — `_acc_base` (base tree 안에서 accept 된 token 수) 를 동시에 추적해야 oracle cost 분리가 가능하기 때문이다.
