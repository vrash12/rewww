FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
ENV SERVED_MODEL_NAME="qwen3-4b"
ENV HF_HOME="/tmp/huggingface"
ENV TRANSFORMERS_CACHE="/tmp/huggingface"
ENV TORCH_HOME="/tmp/torch"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    huggingface_hub

RUN pip install --no-cache-dir --upgrade \
    git+https://github.com/huggingface/transformers.git

COPY server.py /app/server.py

CMD ["python", "server.py"]
