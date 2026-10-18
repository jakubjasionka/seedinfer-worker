"""Pluggable Inference Engine Adapters (vLLM, llama.cpp, and Mock CPU Engine)."""

from __future__ import annotations

import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import httpx


class EngineAdapter(ABC):
    """Abstract adapter bridging the SeedInfer Worker proxy to a local LLM inference engine."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Identifier of the engine backend (`vllm`, `llamacpp`, `mock`)."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return health status dictionary `{'status': 'ok' | 'degraded' | 'down'}`."""

    @abstractmethod
    async def detect_kv_cache_tokens(self, default_tokens: int = 1_500_000) -> int:
        """Auto-detect total KV-cache token pool from engine metrics or return fallback."""

    @abstractmethod
    async def chat_completions(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        """Execute a non-streaming chat completion request."""

    @abstractmethod
    def stream_chat_completions(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncGenerator[bytes, None]:
        """Execute a streaming SSE chat completion request."""


class MockEngineAdapter(EngineAdapter):
    """Deterministic CPU-only mock adapter for testing, CI, and hardware-less quickstarts."""

    def __init__(self, model_id: str = "qwen-2.5-7b") -> None:
        self.model_id = model_id

    @property
    def backend_name(self) -> str:
        return "mock"

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": "mock", "code": 200}

    async def detect_kv_cache_tokens(self, default_tokens: int = 1_500_000) -> int:
        return default_tokens

    async def chat_completions(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        _ = headers
        model = str(payload.get("model") or self.model_id)
        prio = payload.get("priority", 0)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Simulated worker response (engine=mock, priority={prio}).",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 24,
                "completion_tokens": 12,
                "total_tokens": 36,
            },
        }

    async def stream_chat_completions(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncGenerator[bytes, None]:
        _ = headers
        model = str(payload.get("model") or self.model_id)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        tokens = ["Simulated ", "streaming ", "response ", "from ", "seedinfer-worker."]
        delim = chr(10) * 2
        for idx, tok in enumerate(tokens):
            frame: dict[str, Any] = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": tok},
                        "finish_reason": None if idx < len(tokens) - 1 else "stop",
                    }
                ],
            }
            if idx == len(tokens) - 1:
                frame["usage"] = {
                    "prompt_tokens": 24,
                    "completion_tokens": len(tokens),
                    "total_tokens": 24 + len(tokens),
                }
            yield f"data: {json.dumps(frame)}{delim}".encode()
        yield f"data: [DONE]{delim}".encode()


class HttpOpenAIEngineAdapter(EngineAdapter):
    """Base HTTP OpenAI-compatible engine adapter shared by vLLM and llama.cpp."""

    def __init__(self, base_url: str, backend: str = "vllm") -> None:
        self.base_url = base_url.rstrip("/")
        self._backend = backend

    @property
    def backend_name(self) -> str:
        return self._backend

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    return {"status": "ok", "backend": self._backend, "code": 200}
                r2 = await client.get(f"{self.base_url}/v1/models")
                status = "ok" if r2.status_code == 200 else "degraded"
                return {"status": status, "backend": self._backend, "code": r2.status_code}
        except Exception as exc:
            return {"status": "down", "backend": self._backend, "error": str(exc)}

    async def detect_kv_cache_tokens(self, default_tokens: int = 1_500_000) -> int:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base_url}/metrics")
                if r.status_code != 200:
                    return default_tokens
                num_blocks: float | None = None
                block_size = 16
                for line in r.text.splitlines():
                    if line.startswith("#"):
                        continue
                    if "num_total_gpu_blocks" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            num_blocks = float(parts[-1])
                    elif "block_size" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            block_size = int(float(parts[-1]))
                if num_blocks and num_blocks > 0:
                    return int(num_blocks * block_size)
        except Exception:
            pass
        return default_tokens

    async def chat_completions(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return data

    async def stream_chat_completions(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncGenerator[bytes, None]:
        delim = chr(10) * 2
        lf = chr(10)
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload, headers=headers
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        yield f"{line}{delim}".encode()
                    else:
                        yield f"{line}{lf}".encode()


class VllmEngineAdapter(HttpOpenAIEngineAdapter):
    """Adapter for vLLM OpenAI-compatible server with priority scheduling & KV metrics."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        super().__init__(base_url=base_url, backend="vllm")


class LlamaCppEngineAdapter(HttpOpenAIEngineAdapter):
    """Adapter for llama.cpp (`llama-server`) OpenAI-compatible HTTP endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080") -> None:
        super().__init__(base_url=base_url, backend="llamacpp")


def create_engine_adapter(
    backend: str | None = None,
    engine_url: str | None = None,
    model_id: str = "qwen-2.5-7b",
) -> EngineAdapter:
    """Factory that constructs the configured `EngineAdapter` instance."""
    resolved = str(backend or os.getenv("ENGINE_BACKEND") or "mock").strip().lower()
    url = str(engine_url or os.getenv("ENGINE_URL") or "http://127.0.0.1:8000").strip()
    if resolved == "vllm":
        return VllmEngineAdapter(base_url=url)
    if resolved in ("llamacpp", "llama.cpp", "llama_cpp"):
        return LlamaCppEngineAdapter(base_url=url)
    return MockEngineAdapter(model_id=model_id)
