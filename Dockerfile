FROM vllm/vllm-openai:latest

WORKDIR /app

COPY start.sh /app/start.sh
COPY proxy_server.py /app/proxy_server.py

RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

RUN pip install --no-cache-dir fastapi uvicorn httpx

ENV MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
ENV SERVED_MODEL_NAME="qwen3-4b"
ENV MAX_MODEL_LEN="2048"
ENV GPU_MEMORY_UTILIZATION="0.70"
ENV PORT="8080"
ENV VLLM_PORT="8000"
ENV VLLM_NO_USAGE_STATS="1"

ENTRYPOINT ["/bin/bash", "/app/start.sh"]
