FROM vllm/vllm-openai:latest

WORKDIR /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
ENV SERVED_MODEL_NAME="qwen3-4b"
ENV MAX_MODEL_LEN="8192"
ENV GPU_MEMORY_UTILIZATION="0.90"
ENV PORT="8080"

ENTRYPOINT ["/app/start.sh"]
