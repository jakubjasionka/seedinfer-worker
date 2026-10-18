"""SeedInfer Worker — FastAPI Priority Proxy, Telemetry Daemon, and CLI Entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from seedinfer_worker.adapters import EngineAdapter, create_engine_adapter
from seedinfer_worker.backoff import HeartbeatClient
from seedinfer_worker.scheduler import PriorityScheduler
from seedinfer_worker.telemetry import (
    collect_gpu_info,
    collect_host_info,
    sanitize_worker_id,
    set_mock_gpu_mode,
    shutdown_nvml,
)

log = logging.getLogger("seedinfer-worker")


def build_heartbeat_payload(
    *,
    worker_id: str,
    model_id: str,
    region: str,
    scheduler: PriorityScheduler,
    requests_served: int,
    tokens_generated: int,
    start_ts: float,
    engine_health: dict[str, Any],
) -> dict[str, Any]:
    """Build a sanitized worker heartbeat payload ready for router ingestion."""
    gi = collect_gpu_info()
    hi = collect_host_info(worker_id)
    devices = gi.get("devices", [])
    dev_name = devices[0]["name"] if devices else "NVIDIA GPU"
    vram_free_mb = int(devices[0].get("memory_free_mb", 16384)) if devices else 16384
    mem_gb = float(gi.get("total_memory_gb") or hi.get("memory_gb") or 24.0)
    status = "serving" if engine_health.get("status") == "ok" else "draining"
    port = int(os.getenv("WORKER_PORT", "3001"))
    agent_url = os.getenv("WORKER_PUBLIC_URL", f"http://worker-node.example:{port}")

    return {
        "id": sanitize_worker_id(worker_id),
        "chip": dev_name,
        "memory_gb": mem_gb,
        "vram_free_mb": vram_free_mb,
        "current_model": model_id,
        "models": [model_id],
        "status": status,
        "verification": {"status": "verified"},
        "vllm_health": engine_health,
        "gpu": gi,
        "host": hi,
        "region": region,
        "agent_url": agent_url,
        "max_concurrency": scheduler.max_concurrent_requests,
        "max_kv_tokens": scheduler.max_kv_cache_tokens,
        "active_requests": scheduler.active_requests,
        "active_tokens": scheduler.active_tokens,
        "requests_served": requests_served,
        "tokens_generated": tokens_generated,
        "uptime_s": int(max(0.0, time.time() - start_ts)),
    }


def create_worker_app(
    *,
    worker_id: str | None = None,
    model_id: str | None = None,
    adapter: EngineAdapter | None = None,
    scheduler: PriorityScheduler | None = None,
    enable_background_heartbeat: bool = False,
) -> FastAPI:
    """Create and configure the SeedInfer Worker FastAPI application."""
    wid = sanitize_worker_id(
        str(worker_id or os.getenv("WORKER_ID") or "worker-node-placeholder-01")
    )
    model = str(model_id or os.getenv("MODEL") or "qwen-2.5-7b")
    region = str(os.getenv("WORKER_REGION") or "eu-central")
    engine = adapter or create_engine_adapter(model_id=model)
    sched = scheduler or PriorityScheduler(
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "32")),
        max_kv_cache_tokens=int(os.getenv("MAX_KV_CACHE_TOKENS", "1500000")),
    )
    hb_client = HeartbeatClient(
        gateway_url=os.getenv("SEEDINFER_GATEWAY_URL", "https://seedinfer.com"),
        node_token=os.getenv("SEEDINFER_NODE_TOKEN", ""),
    )
    start_ts = time.time()
    stats = {"requests_served": 0, "tokens_generated": 0}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        stop_event = asyncio.Event()
        hb_task: asyncio.Task[None] | None = None
        if enable_background_heartbeat:
            interval = float(os.getenv("HEARTBEAT_INTERVAL", "30"))

            async def _loop() -> None:
                while not stop_event.is_set():
                    eh = await engine.health()
                    payload = build_heartbeat_payload(
                        worker_id=wid,
                        model_id=model,
                        region=region,
                        scheduler=sched,
                        requests_served=stats["requests_served"],
                        tokens_generated=stats["tokens_generated"],
                        start_ts=start_ts,
                        engine_health=eh,
                    )
                    ok, retry_wait = await hb_client.send_heartbeat(payload)
                    wait_sec = interval if ok else retry_wait
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=wait_sec)
                    except TimeoutError:
                        continue

            hb_task = asyncio.create_task(_loop())
        yield
        stop_event.set()
        if hb_task is not None:
            hb_task.cancel()
        shutdown_nvml()

    app = FastAPI(
        title="SeedInfer Worker Daemon",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.worker_id = wid
    app.state.model = model
    app.state.engine = engine
    app.state.scheduler = sched
    app.state.hb_client = hb_client

    @app.get("/health")
    async def health() -> dict[str, Any]:
        gi = collect_gpu_info()
        eh = await engine.health()
        return {
            "status": "ok",
            "worker_id": wid,
            "model": model,
            "engine_backend": engine.backend_name,
            "engine_health": eh,
            "gpu": gi,
            "active_requests": sched.active_requests,
            "max_concurrent_requests": sched.max_concurrent_requests,
            "requests_served": stats["requests_served"],
            "uptime_s": int(max(0.0, time.time() - start_ts)),
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        gi = collect_gpu_info()
        hi = collect_host_info(wid)
        eh = await engine.health()
        return {
            "worker_id": wid,
            "model": model,
            "gpu": gi,
            "host": hi,
            "engine_health": eh,
            "active_requests": sched.active_requests,
            "active_tokens": sched.active_tokens,
            "median_prompt_tokens": sched.get_median_prompt_tokens(),
            "requests_served": stats["requests_served"],
            "tokens_generated": stats["tokens_generated"],
            "uptime_s": int(max(0.0, time.time() - start_ts)),
        }

    @app.get("/v1/models")
    @app.get("/api/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 1735689600,
                    "owned_by": "seedinfer",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    @app.post("/api/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid JSON payload",
                        "type": "invalid_request_error",
                    }
                },
            )

        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Expected JSON object",
                        "type": "invalid_request_error",
                    }
                },
            )

        lease = await sched.try_acquire_slot(body, request.headers)
        if lease is None:
            return Response(
                status_code=429,
                headers={"Retry-After": "1", "Access-Control-Allow-Origin": "*"},
                content=json.dumps(
                    {
                        "error": {
                            "message": (
                                f"Worker node capacity saturated ({sched.active_requests}/"
                                f"{sched.safe_max_concurrency} slots, 10% safety buffer engaged)."
                            ),
                            "type": "tokens_exceeded",
                            "param": None,
                            "code": "rate_limit_exceeded",
                            "retry_after": 1,
                        }
                    }
                ),
                media_type="application/json",
            )

        stats["requests_served"] += 1
        body["priority"] = lease.priority
        fwd_headers = {
            "X-Vllm-Priority": str(lease.priority),
            "Content-Type": "application/json",
        }

        is_stream = bool(body.get("stream", False))
        if not is_stream:
            try:
                result = await engine.chat_completions(body, fwd_headers)
                usage = result.get("usage") if isinstance(result, dict) else {}
                if isinstance(usage, dict):
                    stats["tokens_generated"] += int(usage.get("completion_tokens", 0))
                return JSONResponse(
                    status_code=200,
                    headers={"X-Worker-Priority": str(lease.priority)},
                    content=result,
                )
            finally:
                await sched.release_slot(lease)

        async def _stream_gen() -> AsyncGenerator[bytes, None]:
            try:
                async for chunk in engine.stream_chat_completions(body, fwd_headers):
                    stats["tokens_generated"] += 1
                    yield chunk
            finally:
                await sched.release_slot(lease)

        return StreamingResponse(
            _stream_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Worker-Priority": str(lease.priority),
            },
        )

    return app


app = create_worker_app()


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for `seedinfer-worker`."""
    parser = argparse.ArgumentParser(description="SeedInfer Edge GPU Worker Daemon")
    parser.add_argument(
        "--mock-gpu",
        action="store_true",
        help="Enable simulated GPU telemetry when running without NVIDIA hardware/drivers.",
    )
    parser.add_argument(
        "--engine",
        choices=["mock", "vllm", "llamacpp"],
        default=os.getenv("ENGINE_BACKEND", "mock"),
        help="Downstream inference engine adapter (default: mock).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("WORKER_HOST", "0.0.0.0"),
        help="Bind host address.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("WORKER_PORT", "3001")),
        help="Bind port (default: 3001).",
    )
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> None:
    """CLI entrypoint supporting `--mock-gpu` and `--engine`."""
    args = parse_cli_args(argv)
    if args.mock_gpu:
        set_mock_gpu_mode(True)
    os.environ["ENGINE_BACKEND"] = args.engine
    worker_app = create_worker_app(enable_background_heartbeat=True)
    uvicorn.run(worker_app, host=args.host, port=args.port)


if __name__ == "__main__":
    cli_main()
