# DFlash + Suffix Composition — Controller & Reproduction Guide

이 문서는 **DFlash(Predictor) + Suffix(Memorizer) 조합**의 컨트롤러 동작 방식과 모든 변수/조건을
정리한다. 목표: 다른 사람이 그대로 재현할 수 있게. (측정 스크립트는 CUDA/GB10 기준.)

---

## 0. 한눈에

매 디코딩 라운드마다:
1. **DFlash**가 한 블록(최대 15토큰)을 병렬로 draft → 위치별 로짓 + 위치별 confidence.
2. **Suffix**(Arctic SuffixDecoding)가 warm corpus에서 꼬리를 복사 → 꼬리 + warmth 점수 T.
3. **컨트롤러(compCONF)** 가 confidence와 T로 **DFlash head 길이 n_head\*** 를 정함.
4. head(DFlash) + tail(Suffix)을 **chain**(gated) 또는 **tree**(ungated)로 합쳐 draft 구성.
5. **hybrid tree-verify**로 target이 검증 → 채택 경로만 확정 → 컨텍스트 전진.
6. 채택 토큰으로 Suffix를 **eval-time warming**.

핵심 수식: `n_head* = argmax_k [ 1 + G_k + S_k · T ]`

---

## 1. 모델

| 역할 | 모델 | 비고 |
|---|---|---|
| Target (verify) | `Qwen/Qwen3.5-27B` | 하이브리드 (48 GatedDeltaNet + 16 attention). 4B/2B도 가능 |
| Draft (Predictor) | `z-lab/Qwen3.5-27B-DFlash` | DFlashDraftModel, 5 full-attention layers, `block_size=16` |
| Memorizer | Arctic **SuffixDecoding** | 모델 프리, count-ordered suffix tree |

- 차원 (27B): `linear_num_key_heads=16`, `linear_num_value_heads=48`, `Dk=Dv=128`,
  `linear_conv_kernel_dim=4`. (4B: Hv=32, 2B: Hv=16.)

---

## 2. DFlash draft (Predictor)

- **block_size = 16** → **draft_horizon W = block_size − 1 = 15** (한 블록당 최대 15 draft 토큰).
- 블록 = `[root, MASK, MASK, …]` (root = target이 ctx 뒤에 낸 greedy argmax, 나머지는 mask_token).
- DFlash가 이 블록을 **한 번의 non-causal forward**로 병렬 draft (`parallel_drafting`, AR 루프 없음).
- 필요한 draft-config 값 (DFlash config에서 읽음, **하드코딩 아님**):
  - `block_size` (16)
  - `mask_token_id` (예: 4B는 248070)
  - `target_layer_ids` (예: 4B는 `[1,8,15,22,29]`) — draft가 cross-attend할 target hidden layer들.
- 출력: `draft_logits[1, W, vocab]`. `logp = log_softmax(draft_logits)`.
  - **위치별 confidence** `conf_j = exp(max_j logp) = softmax argmax 확률` (컨트롤러 입력).
  - **head 토큰** `block[j] = argmax_j logp`.

> 상세 캐시/포지션 계약은 `scripts/dflash_candidates.py` 상단 주석 참고 (prefill → 1블록 draft → crop).

---

## 3. Suffix (Memorizer) — Arctic SuffixDecoding

`scripts/measure_k_fusion.py::ArcticSuffix`가 vLLM 실험과 **동일한 사용법**으로 감쌈.

### 하이퍼파라미터 (기본값)
| 변수 | 기본 | 의미 |
|---|---|---|
| `max_spec_factor` (**msf**) | **4.0** | 꼬리 길이 ≤ msf × match_length |
| `min_token_prob` (**mtp**) | **0.1** | 이 확률 미만 후보 컷 |
| `max_tree_depth` | 64 | suffix 트리 최대 깊이 |

> arctic 기본 msf=1.0 인데 우리 Extension은 **msf=4.0** 사용 (긴 꼬리 허용). vLLM `ext_suffix.py`와 동일.

### 동작
- **Warming** (global tree): warm corpus의 각 시퀀스를 `start_request → add_active_response(seq[1:]) →
  stop_request`로 넣음 (첫 토큰 중복 방지). 요청이 쌓일수록 트리가 데워짐(super-linear K).
- **Probe (warmth T)**: `speculate(ctx)` → `(tokens, score)`. **T = score** = ctx만으로 기대되는
  채택 꼬리 길이. 컨트롤러 입력.
- **Tail**: `speculate(ctx + head[:k])` → head 뒤에 붙일 복사 꼬리.
- **Eval-time warming**: 매 라운드 채택 토큰을 `add_active_response`로 다시 넣음 (vLLM과 동일).
- **주의(측정 함정)**: warm/eval corpus는 **target greedy 트레이스**여야 함. 순차 글로벌 warming
  (여러 태스크 누적)이 K를 크게 좌우 → 비교 시 warm 프로토콜을 반드시 맞출 것.

---

## 4. 컨트롤러 (compCONF) — n_head 결정

**단일 출처**: `scripts/ext_suffix.py::_adaptive_nhead_conf` (vLLM 실험과 문자 그대로 동일 함수).
`scripts/fusion_tree.py::adaptive_nhead`가 이걸 그대로 호출한다 (복사본 drift 방지).

### 수식
```
a_k = clip(0.69 · conf_k + 0.29, 0, 1)     # DFlash 위치별 확신도 → 채택확률 (r=0.99 fit)
S = [1.0] + cumprod(a)                       # S_k = head가 위치 k까지 살아남을 확률
G = [0.0] + cumsum(S)                         # G_k = head 기대 채택 길이 (누적 생존)
n_head* = argmax_k [ 1 + G_k + S_k · T ]     # T = suffix warmth (probe.score)
```
- **T 큼 (suffix 따뜻함)** → `S_k·T` 항이 작은 k를 선호 → **짧은 head**, 꼬리에 맡김.
- **T 작음 (suffix 차가움)** → `G_k` 항 지배 → **긴 head**, DFlash에 의존.
- `n_head=0` 허용(순수 suffix), `n_head ≥ W`면 순수 DFlash(꼬리 없음).
- **conf 캡처 실패 시 fallback**: `_adaptive_nhead(T, num_spec)` (T만, conf 없음) → head를 길게 뽑는
  경향. conf 기반(compconf)이 정식. conf는 DFlash softmax에서 캡처 (`ext_proposer_patch`의 hook).

### conf 공식 확인
우리 측정 경로: `conf = logp.max(-1).exp()`. vLLM: `softmax(logits).gather(argmax)`. **동일 값**.

---

## 5. Composition — chain vs tree

`scripts/fusion_tree.py`, `scripts/run_partialwarm_tree.py::build_draft`.

- **num_spec (총 draft 예산)** = 32 (vLLM과 동일). `tail_budget = num_spec − n_head`.
- **chain (gated, linear)** — `build_extension_chain(block[:k], tail)`:
  head 뒤에 꼬리 하나. head가 전부 채택돼야 꼬리 도달(= gate).
  ```
  H1 → H2 → … → Hk → t1 → t2 → …
  ```
- **tree (ungated)** — `build_extension_tree(block[:k], tails)`:
  prefix j=0..k **모든 지점**에 suffix 꼬리를 매닮. head가 중간에 깨져도 그 앞 꼬리는 살아있음(gate 제거).
  ```
  root ─┬ H1 ─┬ H2 … ─ Hk ─ (tail_k)
        │     └ (tail at j=1)
        └ (tail at j=0)
  ```
  `tails[j] = suffix.speculate(ctx + block[:j])`.

### tree 예산 분배 정책 (`ALLOC`)
tree만 해당. 남은 예산 `B−k`를 k+1개 prefix 꼬리에 어떻게 나눌지:
| 정책 | 방식 |
|---|---|
| `even` (기본) | `(B−k)//(k+1)` 토큰씩 균등 |
| `bestfirst` | 분기확률 `p_j = S_j·(1−a_j)` (j<k), `p_k=S_k`에 비례 배분 (자연 꼬리 길이로 캡, 남으면 높은 p_j부터 그리디) |
| `none` | 캡 없음 (각 꼬리 자연 길이 → 노드 폭증) |

---

## 6. 매 라운드 알고리즘 (의사코드)

```python
ctx = tokenize(chat_template(prompt))            # enable_thinking=False
suffix.new_eval(ctx)                             # fresh Arctic eval request
for round in range(max_rounds):
    root, logp = dflash_block_logp(target, draft, ctx, cfg_d)   # W=15 위치 로짓
    W        = len(logp)
    block    = [argmax(logp[j]) for j in range(W)]
    conf     = [exp(max(logp[j])) for j in range(W)]
    suf, T   = suffix.probe(ctx + [root], num_spec)             # T = probe.score
    k        = adaptive_nhead(conf, T, num_spec=W)              # compCONF
    tails    = [suffix.speculate(ctx+[root]+block[:j], num_spec) for j in range(k+1)]
    draft_tree = build_draft(mode, block, suf, tails, k, num_spec, budget, alloc, conf)
    acc, bonus = tree_verify(target, ctx+[root], draft_tree, cfg)   # hybrid tree-verify
    nxt = [root] + [draft_tree.tokens[i] for i in acc] + [bonus]
    suffix.add_response(nxt)                     # eval-time warming
    ctx += nxt
    if eos in nxt: break
K = mean(len(acc) per round)                     # 채택 토큰/라운드
```

---

## 7. 전체 변수 요약 (env / 기본값)

| 변수 | 기본 | 위치 | 설명 |
|---|---|---|---|
| `TGT` | Qwen/Qwen3.5-27B | env | target |
| `DFT` | z-lab/Qwen3.5-27B-DFlash | env | draft |
| `NUM_SPEC` | 32 | env | 총 draft 예산 (head+tail) |
| `block_size` | 16 | DFlash config | → draft_horizon W=15 |
| `msf` (max_spec_factor) | 4.0 | ArcticSuffix | 꼬리 ≤ msf×match |
| `mtp` (min_token_prob) | 0.1 | ArcticSuffix | 후보 컷 |
| `max_tree_depth` | 64 | ArcticSuffix | suffix 트리 깊이 |
| a_k 계수 | 0.69, 0.29 | `_SG_from_conf` | conf→accept 선형 fit |
| `ALLOC` | even | env (sweep) | tree 예산 분배 (even/bestfirst/none) |
| `budget` (B) | =num_spec | build_draft | 총 노드 상한 (sweep 변수) |
| `ROUNDS` | 8 | env | 라운드 수 |
| `NWARM` / `NEVAL` | 12 / 8 | env | warm/eval 프롬프트 수 |
| `MAXTOK` | 96 | env | greedy 트레이스 길이 |
| `K` (k_slots) | 4 | env | multislot 워크로드 novel slot 수 |
| `GLOBAL_WARM` | (off) | env | 1이면 KS 누적 순차 warming (vLLM 프로토콜) |

---

## 8. 워크로드 & warm/eval 실행

### 파일 위치
```
scripts/bench_prompts_multislot_k0.jsonl
scripts/bench_prompts_multislot_k1.jsonl
scripts/bench_prompts_multislot_k2.jsonl
scripts/bench_prompts_multislot_k4.jsonl
scripts/bench_prompts_multislot_k8.jsonl
```
각 파일 = 공유 Python-fn skeleton + **k개의 varying slot**(novel 값). 파일당 **warm 12개 + eval 8개**.
slot 값은 warm에서 held-out(suffix가 못 외움) 이지만 프롬프트엔 명시(DFlash는 예측 가능). k↑ →
head break 잦음.

**jsonl 한 줄 포맷** (JSON):
```json
{"prompt": "Write a Python function named `process` ...",
 "role": "warm",                     // "warm" | "eval"
 "slots": {"fname":"process", "filter_val":"active", "agg":"sum", ...}}
```

### warming set 실행 (suffix 트리 데우기)
warm 프롬프트(`role=="warm"`, 앞 `NWARM=12`개) 각각을 **target greedy**로 디코딩해 트레이스를 만들고,
그 트레이스를 Arctic suffix 트리에 넣는다:
```python
warm = [r for r in rows if r["role"]=="warm"][:NWARM]
warm_toks = [greedy(target, tok, r["prompt"], MAXTOK)[1] for r in warm]  # target greedy, enable_thinking=False
suffix.fit(warm_toks)     # 각 트레이스 start_request→add_active_response(seq[1:])→stop_request
```
- **트레이스 = target greedy** (spec-decode는 lossless라 AR greedy와 동일). `MAXTOK=96`.
- suffix는 이 warm 트레이스들의 skeleton을 복사할 수 있게 됨. novel slot 값은 없음(held-out).

### evaluation set 실행 (K 측정)
eval 프롬프트(`role=="eval"`, 앞 `NEVAL=8`개) 각각에 대해 **조합 디코딩 루프**(§6)를 돌려 라운드당
채택 토큰 K를 측정한다:
```python
ev = [r for r in rows if r["role"]=="eval"][:NEVAL]
for r in ev:
    suffix.new_eval(prompt_ids)               # fresh Arctic eval request
    # §6 루프: DFlash블록 → 컨트롤러 n_head → chain/tree → tree_verify → 전진
    #          매 라운드 채택 토큰으로 suffix.add_response(nxt)  # eval-time warming
K_eval = mean(라운드별 len(accepted))
```
- **coverage** = eval 출력 n-gram 중 warm corpus에 있는 비율 (K를 좌우하는 마스터 변수).

### warming 두 모드
| 모드 | env | 트리 구성 |
|---|---|---|
| **isolated** (기본) | — | proposer마다 fresh 트리, **해당 k의 warm만** fit |
| **sequential global** | `GLOBAL_WARM=1` | proposer당 트리 하나, KS 순서로 **누적**(msw0→mse0→msw4…). eval-time warming이 태스크 넘어 유지 = vLLM 프로토콜 |

> ⚠️ warm 프로토콜이 K를 크게 좌우한다. 순차 글로벌 warming은 트리가 더 데워져(특히 k=0 exact-repeat가
> skeleton 강화) K↑. 다른 구현과 비교할 땐 반드시 같은 warm 모드로.

---

## 9. 실험 — Partial-warm K 크로스오버

**무엇을 보나**: 공유 skeleton은 warm이지만 slot이 novel한 partial-warm 워크로드에서,
**조합(chain)이 두 standalone(dflash, suffix)을 이기는지**. novel slot 수 k가 커질수록:

- **suffix-only**: skeleton은 복사하지만 novel slot마다 멈춤 → k↑ 하면 K 하락
- **dflash-only**: 긴 skeleton을 복사 못함(warming 없음) → 대체로 낮고 평탄
- **chain (조합)**: suffix가 skeleton을 복사 + DFlash head가 각 novel slot을 bridge → 높게 유지

→ 어떤 coverage / slot 수 k에서 **chain이 두 standalone을 추월(크로스오버)** 하는지를 측정한다.

```bash
KS="0 1 2 4 8" NWARM=12 NEVAL=8 ROUNDS=8 NUM_SPEC=32 \
PROPS="dflash suffix chain" \
  python scripts/run_partialwarm_tree.py
```
출력: 각 k에서 `k=4 cov=.. | dflash=K.. suffix=K.. chain=K..`. 조합(chain)이 크로스오버 이후
두 standalone 이상으로 유지되고, k가 클수록 best-standalone 대비 격차가 벌어지는 것이 논지.
