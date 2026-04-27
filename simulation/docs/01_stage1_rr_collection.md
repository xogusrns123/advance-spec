# Stage 1 — RR EAGLE3 Trajectory Collection

Stage 1 은 EAGLE3 speculative decoding 을 켠 SGLang 서버를 띄우고, 그에 대해
benchmark agent 를 실행하여 agent 의 chat trace + per-step EAGLE3 oracle
로그를 `agent_results_eagle3.json` 에 누적한다. 이후 Stage 3 (시뮬레이션) 가
이 artifact 를 소비한다.

본 문서는 round-robin (RR) 진입 스크립트, shard 분할, full pool capture,
resume 동작을 다룬다.

## 1. 진입점 — `run_experiment.py`

```bash
python3 simulation/scripts/run_experiment.py <config.yaml> [--dry-run]
```

config 는 RR mode 만 지원한다 (legacy Stage 1-6 sweep 은 commit `d1e8247`
에서 제거됨). config 에 `round_robin.enabled: true` 가 있어야 한다.

### 1.1 핵심 동작

`execute_round_robin` (`run_experiment.py:141`):

1. `simulation.oracle.install_hook` 호출 — SGLang 디스크 패치 (idempotent)
2. shard 별로 `_run_rr_shard` 를 띄움 (단일 shard 면 직접 호출, 여러 shard
   는 `ThreadPoolExecutor` 로 병렬)
3. 각 shard:
   - 자신의 GPU/port 로 SGLang(EAGLE3) 서버 boot
   - workload 회전 루프: `iter` 마다 자신이 담당한 workload 들에 대해
     `_run_agent_once` 호출 (한 workload 당 `--num-requests batch --resume`)
   - 모든 workload 가 exhausted 되면 종료
4. 모든 shard 종료 시 합산 종료 코드 반환

### 1.2 핵심 환경변수

`_run_rr_shard` 가 SGLang 서브프로세스에 주입:

| 변수 | 값 | 효과 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | shard 의 `gpu_ids` | shard 별 GPU 격리 |
| `SGLANG_ORACLE_VANILLA` | `1` | oracle hook 활성, `accept_length=0` 강제 |
| `SGLANG_CAPTURE_FULL_POOL` | `1` (yaml 의 `capture_full_pool: true` 일 때) | full draft pool 캡처 → reslice 가능 |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `1` | 긴 context 허용 |
| `TORCHINDUCTOR_COMPILE_THREADS` | `1` | torch.compile fork-bomb 방지 |

## 2. Config 스키마

```yaml
name: rr_qwen3_14b           # experiment 식별자
model_preset: qwen3_14b      # MODEL_PRESETS 의 key 중 하나

# (선택) preset default 를 override
models:
  draft_model: AngelSlim/Qwen3-14B_eagle3

# 단일 (steps, topk, num_draft_tokens) — RR 은 sweep 안 함
stage1_steps: 8
stage1_configs:
  - topk: 16
    num_draft_tokens: 2     # full-pool capture 모드에선 작은 값으로 OK

round_robin:
  enabled: true
  capture_full_pool: true   # SGLANG_CAPTURE_FULL_POOL=1
  batch: 1                  # --num-requests per agent invocation
  resume: true              # --resume

# workload 목록 (회전 순서 = shard 안에서 처리 순서)
workloads:
  - specbench
  - bfcl_v4
  - swebench_verified
  - longbench_lcc
  - longbench_repobench

# per-workload 런타임 override (CLI flag 로 변환)
workload_overrides:
  specbench:
    max_tokens_override: 1024     # → --max-tokens 1024
  bfcl_v4:
    max_iterations: 20            # → --max-iterations 20
    include_category: web_search  # → --include-category web_search
  swebench_verified:
    max_iterations: 250
    tool_style: minisweagent      # → --tool-style minisweagent
    repos_dir: data/swebench_verified/repos
  longbench_lcc:
    max_tokens: 256
  longbench_repobench:
    max_tokens: 256

# (옵션) shard 분할 — 4-GPU 병렬용
shards:
  - id: 0
    gpu_ids: [0]
    port: 30000
    workloads: [swebench_verified]
  - id: 1
    gpu_ids: [1]
    port: 30001
    workloads: [longbench_lcc]
  - id: 2
    gpu_ids: [2]
    port: 30002
    workloads: [longbench_repobench]
  - id: 3
    gpu_ids: [3]
    port: 30003
    workloads: [bfcl_v4, specbench]

infra:
  port: 30000               # `shards:` 없을 때만 쓰임
  gpu_ids: [0]              # 위와 같음

output:
  root: simulation/results  # 출력 루트
```

`shards:` 가 있으면 그 정보가 우선이고, `infra.port` / `infra.gpu_ids` 는
fallback 으로만 쓰인다.

## 3. Shard 분할 동작

`_run_rr_shard` 의 RR 루프는 shard 별로 완전히 독립이다:

- 각 shard 는 자기 SGLang 서버를 자기 GPU/port 에서 단독 운영
- 자기 workload 만 회전 — workload partition 검증 (config 단계) 으로 한
  workload 가 두 shard 에 분배되는 일은 없음 → 같은 출력 디렉토리에 동시
  쓰기가 발생하지 않음
- 각 shard 가 동일한 `_count_progress` / `_run_agent_once` 헬퍼를 사용 →
  resume 도 shard 별로 독립

출력 디렉토리는 shard 와 무관하게 workload 명만 쓴다:
```
simulation/results/<preset>/<workload>_steps<S>_topk<K>_capture/
    ├── agent_results_eagle3.json
    ├── agent_results_eagle3.json.partial         (in-flight checkpoint)
    └── _rr_agent.log
```

`_rr_sglang_server_shard{id}.log` 는 `out_base` 에 쓰여서 shard 별 로그
가 분리된다.

## 4. SGLang 서버 launch 인자

`_run_rr_shard` 가 SGLang 을 launch 할 때:

```python
[
    sys.executable, "-m", "sglang.launch_server",
    "--model-path", target_model,
    "--tp-size", str(len(gpu_ids)),       # shard 의 GPU 수
    "--speculative-algorithm", "EAGLE3",
    "--speculative-draft-model-path", draft_model,
    "--speculative-num-steps", str(steps),
    "--speculative-eagle-topk", str(topk),
    "--speculative-num-draft-tokens", str(ndt),
    "--tool-call-parser", tool_call_parser,
    "--mem-fraction-static", "0.85",
    "--disable-cuda-graph",
    "--watchdog-timeout", "600",
    "--host", "0.0.0.0", "--port", str(port),
]
```

- `--tp-size`: shard 의 `gpu_ids` 길이. 즉 single-GPU shard 는 TP=1, 4-GPU
  shard 는 TP=4. 일반적으로 작은 모델 (qwen3_8b) 은 GPU 1대씩 4 shard 로
  나누는 게 TP=4 한대보다 효율적임.
- `--speculative-num-draft-tokens`: yaml 의 `stage1_configs[0].num_draft_tokens`.
  `capture_full_pool: true` 면 verify 단계에선 이 값만큼만 쓰지만, 두 번째
  `organize_draft_results` 호출이 `pool_size+1` 로 cloned 입력에 대해
  돌아 full pool 을 stash → reslice 시 (s', k') sub-tree 로 자유롭게
  잘라쓸 수 있음 (`09_pool_reslicer.md` 참조).
- `--mem-fraction-static 0.85`: 적당한 KV 캐시 할당
- `--disable-cuda-graph`: oracle 패치가 동적 instrumentation 을 추가하므로
  CUDA graph 비활성

## 5. Agent 호출

`_run_agent_once` (`run_experiment.py:458`):

```bash
python -m <agent_module> \
  --url http://localhost:<port>/v1 \
  --model <target_model> \
  --input-file <dataset> \
  --output-file <out_dir>/agent_results_eagle3.json \
  --num-requests <batch> \
  --num-workers 1 \
  [--resume] \
  [<extra_flags from workload_overrides>]
```

`<extra_flags>` 는 `_build_agent_extra_flags` 가 workload override 로부터
변환:

| Workload | Override key | CLI flag |
|---|---|---|
| `specbench` / `longbench_*` | `max_tokens_override` (specbench) / `max_tokens` (longbench) | `--max-tokens` |
| `bfcl_v4` | `max_iterations` | `--max-iterations` |
| `bfcl_v4` | `include_category` | `--include-category` |
| `swebench_verified` | `max_iterations` | `--max-iterations` |
| `swebench_verified` | `tool_style` | `--tool-style` |
| `swebench_verified` | `repos_dir` (default `data/swebench_verified/repos`) | `--repos-dir` |

`--num-workers 1` 은 절대 변경 금지 — 자세한 건 `00_OVERVIEW.md` Hot List
§1 참조.

## 6. Resume 동작

각 agent 가 자체 checkpoint 패턴을 갖는다 (`save_results.py`:
`save_agent_results`, `load_checkpoint`, `done_ids`, `append_to_checkpoint`,
`finalize_checkpoint`):

- partial 산출 = `agent_results_eagle3.json.partial` 에 append
- 매 request 종료 시 atomic rename
- `--resume` 시 partial 의 done set 을 로드, 데이터셋의 다음 미완료 request
  부터 진행

RR 의 `_count_progress` (`run_experiment.py:436`) 가 매 iter 시작 시
`done_n / total_n` 을 출력해서 진행률 모니터링이 가능하다. 모든 shard 의
모든 workload 가 exhausted (`done_n >= total_n`) 되면 RR 루프 종료.

## 7. 흔한 운영 패턴

### 7.1 단일 GPU (mango3)

```bash
docker exec -u root -d sglang-bench bash -c \
  "cd /workspace && python3 simulation/scripts/run_experiment.py \
    simulation/config/rr_qwen3_14b.yaml > /tmp/rr_stage1.log 2>&1"
```

### 7.2 4-shard 병렬 (mango1)

```bash
docker exec -u root -d sglang-bench bash -c \
  "cd /workspace && python3 simulation/scripts/run_experiment.py \
    simulation/config/mango1.yaml > /tmp/rr_stage1.log 2>&1"
```

### 7.3 RR 모니터링

```bash
tail -F /tmp/rr_stage1.log | grep -E 'iter [0-9]+\]|ok  done=|ERROR|FAIL'
```

### 7.4 RR 중단 (다른 작업 위해)

`feedback_pause_rr_for_big_sims` 메모리 참조 — 큰 sim 돌리기 전 RR 종료
필요. 절차:
```bash
docker exec sglang-bench bash -c "kill <RR_PID>"      # SIGTERM
sleep 10                                              # finally 절 실행 대기
docker exec sglang-bench bash -c "kill <SGLANG_PID>"  # SGLang 도 죽일 것
```
재개는 같은 yaml 로 재시작 (`resume: true` 라 자동으로 이어감).

## 8. 데이터 prep (사전)

dataset jsonl 들은 `simulation/scripts/experiments/data_prep/` 의 prep
스크립트가 생성:

| 스크립트 | 산출물 |
|---|---|
| `prep_all_datasets.py` | 전체 dataset 일괄 prep 진입점 |
| `interleave_datasets.py` | category 균형 잡힌 stratified interleave |
| `make_lcc_dataset.py` | LongBench LCC 변환 |

각 dataset 은 한 줄=한 request 인 jsonl. RR config 의
`workload_overrides` 가 dataset 경로를 직접 지정하지 않고 `WORKLOAD_REGISTRY`
(`run_experiment.py:78`) 가 workload 명 → dataset 경로 매핑을 제공.

`feedback_no_dataset_slicing` 메모리 — dataset 을 jsonl 로 미리 자르지 말
것. 모두 한 캐노니컬 full dataset 만 디스크에 두고, request 수는 RR 의
`batch` + agent 의 `--num-requests` 로 런타임 결정.
