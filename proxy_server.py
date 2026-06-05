import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI()

VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}"


@app.get("/")
async def root():
    return {
        "status": "starting_or_ready",
        "service": "AmoraCare Qwen vLLM Proxy",
        "model": os.getenv("SERVED_MODEL_NAME", "qwen3-4b"),
        "message": "Proxy is running. vLLM may still be loading.",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "message": "Proxy is listening. vLLM may still be loading.",
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    target_url = f"{VLLM_BASE_URL}/{path}"

    try:
        body = await request.body()

        headers = dict(request.headers)
        headers.pop("host", None)

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=request.query_params,
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={
                "content-type": response.headers.get("content-type", "application/json")
            },
        )

    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "The Qwen model is still loading. Please try again in a few minutes.",
            },
        )

    except Exception as error:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Proxy error while contacting vLLM.",
                "error": str(error),
            },
        )
