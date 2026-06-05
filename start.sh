#!/usr/bin/env bash
set -euo pipefail

export PORT="${PORT:-8080}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"

echo "Starting vLLM in background..."
echo "MODEL_ID=${MODEL_ID}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "Cloud Run public PORT=${PORT}"
echo "Internal vLLM PORT=${VLLM_PORT}"

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

vllm serve "${MODEL_ID}" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype auto \
  --trust-remote-code &

echo "Starting FastAPI proxy on Cloud Run PORT=${PORT}..."

exec uvicorn proxy_server:app \
  --host 0.0.0.0 \
  --port "${PORT}"
