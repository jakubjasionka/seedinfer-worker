"""GPU & Host Telemetry collector with `--mock-gpu` hardware simulation mode.

Supports real NVIDIA GPUs via `pynvml` (`nvidia-ml-py`) and gracefully falls back
to deterministic simulated telemetry when `--mock-gpu` or `SEEDINFER_MOCK_GPU=1` is active.
"""

from __future__ import annotations

import logging
import os
import platform
import re
from typing import Any

log = logging.getLogger("seedinfer-worker.telemetry")

try:
    import pynvml  # type: ignore

    _HAS_NVML = True
except ImportError:
    pynvml = None
    _HAS_NVML = False

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False

_NVML_INITIALIZED = False
_FORCE_MOCK_GPU = False


def set_mock_gpu_mode(enabled: bool) -> None:
    """Explicitly enable or disable `--mock-gpu` mode at runtime."""
    global _FORCE_MOCK_GPU
    _FORCE_MOCK_GPU = bool(enabled)


def is_mock_gpu_mode() -> bool:
    """Return True if mock GPU telemetry is enabled via CLI flag or environment variable."""
    if _FORCE_MOCK_GPU:
        return True
    return os.getenv("SEEDINFER_MOCK_GPU", "0").strip() in ("1", "true", "TRUE", "yes")


def _ensure_nvml() -> bool:
    global _NVML_INITIALIZED
    if is_mock_gpu_mode() or not _HAS_NVML or pynvml is None:
        return False
    if _NVML_INITIALIZED:
        return True
    try:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True
        return True
    except Exception as exc:
        log.warning("pynvml initialization failed (falling back to mock telemetry): %s", exc)
        return False


def shutdown_nvml() -> None:
    """Cleanly shut down NVML if initialized."""
    global _NVML_INITIALIZED
    if _NVML_INITIALIZED and pynvml is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        _NVML_INITIALIZED = False


def _simulated_gpu_info() -> dict[str, Any]:
    name = os.getenv("MOCK_GPU_NAME", "NVIDIA GeForce RTX 4090 (Simulated)")
    total_mb = int(os.getenv("MOCK_GPU_VRAM_TOTAL_MB", "24576"))
    free_mb = int(os.getenv("MOCK_GPU_VRAM_FREE_MB", "18432"))
    used_mb = max(0, total_mb - free_mb)
    gpu_cores = 21760 if "5090" in name else 16384
    return {
        "count": 1,
        "mock": True,
        "devices": [
            {
                "index": 0,
                "name": name,
                "memory_total_mb": total_mb,
                "memory_used_mb": used_mb,
                "memory_free_mb": free_mb,
                "util_gpu": 22,
                "util_memory": int((used_mb / max(1, total_mb)) * 100),
                "temperature_c": 54,
            }
        ],
        "total_memory_gb": round(total_mb / 1024.0, 1),
        "gpu_cores": gpu_cores,
    }


def collect_gpu_info() -> dict[str, Any]:
    """Collect GPU VRAM, utilization, and thermal metrics via NVML or `--mock-gpu`."""
    if is_mock_gpu_mode() or not _ensure_nvml():
        return _simulated_gpu_info()

    try:
        count = pynvml.nvmlDeviceGetCount()
        devices: list[dict[str, Any]] = []
        total_mem_bytes = 0
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem_bytes += int(mem.total)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = int(util.gpu)
                mem_util = int(util.memory)
            except Exception:
                gpu_util = 0
                mem_util = 0
            try:
                temp = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                temp = 0
            devices.append(
                {
                    "index": idx,
                    "name": name,
                    "memory_total_mb": int(mem.total // (1024 * 1024)),
                    "memory_used_mb": int(mem.used // (1024 * 1024)),
                    "memory_free_mb": int(mem.free // (1024 * 1024)),
                    "util_gpu": gpu_util,
                    "util_memory": mem_util,
                    "temperature_c": temp,
                }
            )
        gpu_cores = 21760 if any("5090" in d["name"] for d in devices) else count * 2048
        return {
            "count": count,
            "mock": False,
            "devices": devices,
            "total_memory_gb": round(total_mem_bytes / (1024**3), 1),
            "gpu_cores": gpu_cores,
        }
    except Exception as exc:
        log.warning("NVML telemetry read failed, returning simulated metrics: %s", exc)
        return _simulated_gpu_info()


def sanitize_worker_id(raw: str) -> str:
    """Sanitize a node identifier to safe RFC-1123 / alphanumeric token characters."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", (raw or "").strip())
    cleaned = re.sub(r"[-_.]{2,}", "-", cleaned).strip("-_.")
    return (cleaned or "worker-node-01")[:64]


def collect_host_info(worker_id: str) -> dict[str, Any]:
    """Collect sanitized host system metrics without exposing private machine hostnames."""
    cpu_total = os.cpu_count() or 4
    mem_gb = 32.0
    if _HAS_PSUTIL and psutil is not None:
        try:
            mem_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            pass
    return {
        "node_id": sanitize_worker_id(worker_id),
        "platform": platform.system().lower(),
        "cpu_total": cpu_total,
        "memory_gb": mem_gb,
    }
