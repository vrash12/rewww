#!/usr/bin/env bash
set -e

export PORT="${PORT:-8080}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Optional: support either HF_TOKEN or HUGGING_FACE_HUB_TOKEN.
if [ -n "${HF_TOKEN}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype auto \
  --trust-remote-code
