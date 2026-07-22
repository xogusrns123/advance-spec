# DFlash + Suffix Composition — Controller 실제 구현 (슬라이드 내용물)

> 슬라이드 각 장에 들어갈 텍스트 내용물. 나중에 PPT로 변환.
> 범위: 우리 구현사항인 **Controller(compCONF) + 꼬리 부착**의 실제 동작만. 파이프라인 순서대로.
> DFlash/Suffix 내부 동작, tree 구조, 노드 예산, verify, warming, 동작 특성/이론은 제외.
> 비유 없이 기술 용어로만.
>
> 용어 통일: **probability** (위치별 확률), **prob_k** (위치 k의 확률).

---

## Slide 1 — 파이프라인 (한 라운드 순서)

**제목:** Controller가 head 길이 k를 정하고, 그 k로 head/tail을 붙인다

한 디코딩 라운드는 다음 순서로 진행된다.

1. **DFlash draft** → 위치별 후보 토큰 `block[0..W-1]` 과 위치별 probability `prob[0..W-1]` 를 낸다 (W=15). 내부 동작은 범위 밖, Controller는 여기서 **probability만** 받는다.
2. **Suffix probe** → 현재 문맥만으로 warmth 값 `T` 를 낸다. 내부 동작은 범위 밖, Controller는 **T만** 받는다.
3. **Controller(compCONF)** → probability와 T로 **head 길이 `k`를 정한다.** (Slide 2)
4. **꼬리 부착** → head는 DFlash 앞 k개, tail은 그 뒤에서 Suffix가 뽑은 복사본. `head → tail` 선형 chain으로 합친다. (Slide 3)
5. 이 chain을 target이 검증하고 문맥을 전진시킨다 (범위 밖).

진입점: `run_partialwarm_tree.py::eval_K` 라운드 루프.

---

## Slide 2 — Controller compCONF: head 길이 k 계산  ★핵심

**제목:** 각 head 길이 후보의 라운드 가치 K(k)를 계산해 최댓값을 고른다

Controller는 `ext_suffix.py::_adaptive_nhead_conf` 한 곳에만 있다 (`fusion_tree.adaptive_nhead`는 이 함수를 그대로 호출만 함).

입력은 DFlash의 위치별 probability `prob[0..W-1]` 와 Suffix의 warmth `T`. 가능한 head 길이 `k = 0, 1, …, W` 각각에 대해 `K(k)`를 계산하고 최댓값을 주는 k를 고른다. 계산 순서는:

1. **survival `S_k` 계산.** `prob_k`를 위치별 채택확률로 보고 누적 곱한다. `S_k = prob_1 · prob_2 · … · prob_k` ("앞 k개가 전부 맞을 확률"). `S_0 = 1`.

2. **`G_k` 계산.** survival을 누적 합한다. `G_k = S_1 + S_2 + … + S_k`. `G_0 = 0`. (S_k는 곱, G_k는 그 곱들의 합 — 서로 다른 값.)

3. **라운드 가치 `K(k) = 1 + G_k + S_k · T` 계산.**

4. **argmax.** `k = 0…W` 중 `K(k)`가 최대인 k를 고른다. head 길이 상한은 draft horizon W(=15)이며 그 이상은 될 수 없다. 부등호가 strict(`>`)라 동점이면 **가장 짧은 head**를 택한다.

5. **fallback.** DFlash probability를 못 받은 경우엔 `full head`(k = W)를 반환한다. probability가 있을 때만 위 1~4가 작동한다.

---

## Slide 3 — 꼬리 부착 (라운드 루프에서의 실제 호출)

**제목:** Controller가 정한 k로 head/tail을 붙인다

라운드 루프(`run_partialwarm_tree.py::eval_K`)에서의 순서:

1. DFlash 로짓에서 위치별 probability를 뽑는다.
2. probability와 warmth T를 Controller에 넘겨 **head 길이 k를 받는다.**
3. head는 DFlash 후보의 앞 k개(`block[:k]`), tail은 그 head 뒤 문맥에서 Suffix가 뽑은 복사본(`suffix.speculate(ctx + block[:k])`)이다. 꼬리는 항상 "head를 붙인 다음 지점"에서 이어진다.
4. 둘을 `head → tail` 선형 chain으로 합친다 (`fusion_tree.py::build_extension_chain`).

**k에 따른 부착 결과** (k는 0~W, W=15가 상한)
- `k = 0` : head 없이 **순수 Suffix**.
- `0 < k ≤ W` : **DFlash가 앞(head), Suffix가 뒤(tail)**.

**chain 구조**
- `H1 → H2 → … → Hk → t1 → t2 → …` 로 이어지는 하나의 선형 경로 (각 노드의 부모는 바로 앞 노드).
</content>
