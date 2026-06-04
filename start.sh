#!/usr/bin/env bash
set -euo pipefail

echo "Starting AmoraCare Qwen vLLM service..."
echo "PORT=${PORT:-8080}"
echo "MODEL_ID=${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-4b}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}"

export PORT="${PORT:-8080}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

vllm serve "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype auto \
  --trust-remote-code
