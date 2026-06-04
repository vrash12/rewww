FROM vllm/vllm-openai:latest

WORKDIR /app

COPY start.sh /app/start.sh

# Fix possible Windows CRLF line endings and make script executable.
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

ENV MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
ENV SERVED_MODEL_NAME="qwen3-4b"
ENV MAX_MODEL_LEN="4096"
ENV GPU_MEMORY_UTILIZATION="0.80"
ENV PORT="8080"

ENTRYPOINT ["/bin/bash", "/app/start.sh"]
