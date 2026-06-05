import os
import time
import threading
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "qwen3-4b")
PORT = int(os.getenv("PORT", "8080"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

tokenizer = None
model = None
model_ready = False
model_error = None
model_started_at = None

app = FastAPI(title="AmoraCare Qwen Self-Hosted Server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 700


def load_model():
    global tokenizer, model, model_ready, model_error, model_started_at

    model_started_at = time.time()

    try:
        print("Loading model:", MODEL_ID, flush=True)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available. Check Cloud Run GPU settings.")

        print("CUDA available:", torch.cuda.is_available(), flush=True)
        print("CUDA device:", torch.cuda.get_device_name(0), flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        model.eval()

        model_ready = True
        print("Model loaded successfully.", flush=True)

    except Exception as exc:
        model_error = str(exc)
        model_ready = False
        print("MODEL LOAD ERROR:", model_error, flush=True)


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()


@app.get("/")
def root():
    return {
        "service": "AmoraCare Qwen Self-Hosted Server",
        "model": SERVED_MODEL_NAME,
        "status": "ready" if model_ready else "loading_or_error",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "server": "listening",
        "model_ready": model_ready,
    }


@app.get("/status")
def status():
    elapsed = None

    if model_started_at:
        elapsed = round(time.time() - model_started_at, 2)

    return {
        "model_id": MODEL_ID,
        "served_model_name": SERVED_MODEL_NAME,
        "model_ready": model_ready,
        "model_error": model_error,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "load_elapsed_seconds": elapsed,
    }


@app.get("/v1/models")
def models():
    if not model_ready:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "Model is still loading or failed. Check /status.",
                "model_error": model_error,
            },
        )

    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "owned_by": "amoracare",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if not model_ready:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "Model is still loading or failed. Check /status.",
                "model_error": model_error,
            },
        )

    try:
        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        temperature = float(request.temperature or 0.2)
        max_tokens = int(request.max_tokens or 700)

        do_sample = temperature > 0

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=float(request.top_p or 0.9) if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs.input_ids.shape[-1]:]
        answer = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVED_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Generation failed.",
                "error": str(exc),
            },
        )


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
