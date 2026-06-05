import os
import subprocess
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, PlainTextResponse

app = FastAPI()

VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}"


@app.get("/")
async def root():
    return {
        "status": "starting_or_ready",
        "service": "AmoraCare Qwen vLLM Proxy",
        "model": os.getenv("SERVED_MODEL_NAME", "qwen3-4b"),
        "message": "Proxy is running. Check /vllm-status and /vllm-log.",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "message": "Proxy is listening.",
    }


@app.get("/vllm-status")
async def vllm_status():
    pid_path = "/tmp/vllm.pid"

    if not os.path.exists(pid_path):
        return {
            "vllm_running": False,
            "message": "No vLLM PID file found.",
        }

    with open(pid_path, "r") as file:
        pid = file.read().strip()

    result = subprocess.run(
        ["sh", "-c", f"ps -p {pid} -o pid,cmd"],
        capture_output=True,
        text=True,
    )

    return {
        "vllm_running": result.returncode == 0,
        "pid": pid,
        "process": result.stdout,
        "error": result.stderr,
    }


@app.get("/vllm-log")
async def vllm_log():
    log_path = "/tmp/vllm.log"

    if not os.path.exists(log_path):
        return PlainTextResponse("No /tmp/vllm.log file found yet.")

    with open(log_path, "r", errors="ignore") as file:
        content = file.read()

    return PlainTextResponse(content[-12000:])


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
                "message": "The Qwen model is still loading or vLLM crashed. Check /vllm-status and /vllm-log.",
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
