"""Unit and integration tests for SeedInfer Worker daemon (runs 100% on CPU)."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from seedinfer_worker.adapters import (
    LlamaCppEngineAdapter,
    MockEngineAdapter,
    VllmEngineAdapter,
    create_engine_adapter,
)
from seedinfer_worker.backoff import HeartbeatClient, compute_backoff_delay
from seedinfer_worker.main import build_heartbeat_payload, create_worker_app, parse_cli_args
from seedinfer_worker.scheduler import PriorityScheduler
from seedinfer_worker.telemetry import (
    collect_gpu_info,
    collect_host_info,
    is_mock_gpu_mode,
    sanitize_worker_id,
    set_mock_gpu_mode,
)


def test_mock_gpu_telemetry_and_cli_args() -> None:
    args = parse_cli_args(["--mock-gpu", "--engine", "mock", "--port", "3005"])
    assert args.mock_gpu is True
    assert args.engine == "mock"
    assert args.port == 3005

    set_mock_gpu_mode(True)
    assert is_mock_gpu_mode() is True
    gi = collect_gpu_info()
    assert gi["count"] == 1
    assert gi["mock"] is True
    assert gi["devices"][0]["memory_free_mb"] > 0

    hi = collect_host_info("worker-node-test!@#")
    assert hi["node_id"] == "worker-node-test"
    assert sanitize_worker_id("  node__01  ") == "node-01"


def test_exponential_backoff_and_retry_after() -> None:
    d0 = compute_backoff_delay(0, base_sec=1.0, max_sec=60.0, jitter_ratio=0.0)
    d2 = compute_backoff_delay(2, base_sec=1.0, max_sec=60.0, jitter_ratio=0.0)
    d10 = compute_backoff_delay(10, base_sec=1.0, max_sec=60.0, jitter_ratio=0.0)
    assert d0 == 1.0
    assert d2 == 4.0
    assert d10 == 60.0

    hb = HeartbeatClient(gateway_url="https://seedinfer.com", node_token="sipn_test_token")
    assert hb.build_headers()["Authorization"] == "Bearer sipn_test_token"
    assert hb.next_retry_delay("15") == 15.0


def test_priority_scheduler_and_dual_90_percent_guard() -> None:
    sched = PriorityScheduler(max_concurrent_requests=10, max_kv_cache_tokens=100_000)
    assert sched.safe_max_concurrency == 9
    assert sched.safe_max_kv_tokens == 90_000

    # Short interactive prompt -> Priority 0 (Express Lane)
    prio_short, tok_short = sched.calculate_priority(
        {"messages": [{"role": "user", "content": "Hello!"}], "stream": True},
        {},
    )
    assert prio_short == 0
    assert tok_short >= 1

    # Explicit background priority mapping (1000 -> 15)
    prio_bg, _ = sched.calculate_priority(
        {"messages": [{"role": "user", "content": "Batch job"}], "priority": 1000},
        {},
    )
    assert prio_bg == 15

    async def _run_leases() -> None:
        leases = []
        for _ in range(9):
            lease = await sched.try_acquire_slot(
                {"messages": [{"role": "user", "content": "hi"}]}, {}
            )
            assert lease is not None
            leases.append(lease)
        # 10th slot exceeds 90% guard (9/10) -> returns None
        overflow = await sched.try_acquire_slot(
            {"messages": [{"role": "user", "content": "hi"}]}, {}
        )
        assert overflow is None
        for lease in leases:
            await sched.release_slot(lease)
        assert sched.active_requests == 0

    asyncio.run(_run_leases())


def test_engine_adapters_factory() -> None:
    mock_ad = create_engine_adapter("mock")
    vllm_ad = create_engine_adapter("vllm", "http://127.0.0.1:8000")
    llama_ad = create_engine_adapter("llamacpp", "http://127.0.0.1:8080")
    assert isinstance(mock_ad, MockEngineAdapter)
    assert isinstance(vllm_ad, VllmEngineAdapter)
    assert isinstance(llama_ad, LlamaCppEngineAdapter)


def test_worker_fastapi_endpoints_and_saturation() -> None:
    set_mock_gpu_mode(True)
    sched = PriorityScheduler(max_concurrent_requests=10, max_kv_cache_tokens=200_000)
    app = create_worker_app(
        worker_id="worker-unit-01",
        model_id="qwen-2.5-7b",
        adapter=MockEngineAdapter("qwen-2.5-7b"),
        scheduler=sched,
    )
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["worker_id"] == "worker-unit-01"
    assert r_health.json()["gpu"]["mock"] is True

    r_models = client.get("/v1/models")
    assert r_models.status_code == 200
    assert r_models.json()["data"][0]["id"] == "qwen-2.5-7b"

    r_chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-2.5-7b",
            "messages": [{"role": "user", "content": "Run mock inference"}],
            "stream": False,
        },
    )
    assert r_chat.status_code == 200
    assert r_chat.headers.get("x-worker-priority") == "0"
    assert "Simulated worker response" in r_chat.json()["choices"][0]["message"]["content"]

    r_stream = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-2.5-7b",
            "messages": [{"role": "user", "content": "Stream tokens"}],
            "stream": True,
        },
    )
    assert r_stream.status_code == 200
    assert "data: [DONE]" in r_stream.text

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert r_metrics.json()["requests_served"] == 2

    hb_payload = build_heartbeat_payload(
        worker_id="worker-unit-01",
        model_id="qwen-2.5-7b",
        region="eu-central",
        scheduler=sched,
        requests_served=2,
        tokens_generated=17,
        start_ts=1000.0,
        engine_health={"status": "ok"},
    )
    assert hb_payload["id"] == "worker-unit-01"
    assert hb_payload["status"] == "serving"
