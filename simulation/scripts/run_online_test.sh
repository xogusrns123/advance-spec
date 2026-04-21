#!/bin/bash
# End-to-end test: SGLang with --speculative-algorithm SUFFIX
#
# Run inside the Docker container (sglang-bench):
#   cd /workspace && bash simulation/scripts/run_online_test.sh
#
# Prerequisites:
#   - GPU available (4x RTX 4090)
#   - Model downloaded: zai-org/GLM-4.7-Flash

set -euo pipefail

PORT=30000
SERVER_URL="http://localhost:${PORT}"

echo "=== Step 1: Patch deepseek_v2.py (enable_a2a_moe fix) ==="
sed -i \
    's/if self.enable_a2a_moe and i > self.first_k_dense_replace:/if getattr(self, "enable_a2a_moe", False) and i > self.first_k_dense_replace:/' \
    /opt/venv/lib/python3.11/site-packages/sglang/srt/models/deepseek_v2.py
echo "Patched."

echo ""
echo "=== Step 2: Launch SGLang server with SUFFIX algorithm ==="
echo "Starting server in background..."
cd /workspace
python3 -m simulation.oracle.install_hook -- \
    --model-path zai-org/GLM-4.7-Flash \
    --tp-size 4 \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.8 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port ${PORT} &
SERVER_PID=$!

echo "Waiting for server to be ready (PID=${SERVER_PID})..."
for i in $(seq 1 300); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s."
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    if [ "$i" -eq 300 ]; then
        echo "ERROR: Server did not start within 300s"
        kill ${SERVER_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Step 3: Run unit tests (no GPU) ==="
python3 -m pytest /workspace/simulation/tests/test_online_integration.py \
    -v --tb=short -k "not e2e"

echo ""
echo "=== Step 4: Run E2E tests ==="
python3 -m pytest /workspace/simulation/tests/test_online_integration.py \
    -v --tb=short -m "e2e"

echo ""
echo "=== Step 5: Quick benchmark (5 prompts) ==="
python3 -c "
import requests, time

SERVER = 'http://localhost:${PORT}'
prompts = [
    'Write a Python function that sorts a list using quicksort:\n\`\`\`python\n',
    'Explain the difference between TCP and UDP in networking:\n',
    'Write a SQL query to find duplicate emails in a users table:\n\`\`\`sql\n',
    'Implement a binary search tree in Python:\n\`\`\`python\n',
    'What is the time complexity of merge sort and why?\n',
]

total_tokens = 0
total_time = 0.0

for i, prompt in enumerate(prompts):
    start = time.time()
    resp = requests.post(f'{SERVER}/generate', json={
        'text': prompt,
        'sampling_params': {'max_new_tokens': 128, 'temperature': 0.0},
    })
    elapsed = time.time() - start
    data = resp.json()
    n_tokens = len(data.get('meta_info', {}).get('output_token_ids', []))
    total_tokens += n_tokens
    total_time += elapsed
    print(f'  [{i+1}/5] {n_tokens:3d} tokens in {elapsed:.2f}s ({n_tokens/elapsed:.1f} tok/s)')

print(f'\n  Total: {total_tokens} tokens in {total_time:.2f}s')
print(f'  Avg throughput: {total_tokens/total_time:.1f} tok/s')
"

echo ""
echo "=== Done ==="
echo "Server is still running at ${SERVER_URL} (PID=${SERVER_PID})"
echo "To stop: kill ${SERVER_PID}"
