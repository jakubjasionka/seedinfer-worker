"""Adaptive Priority Scheduler and Dual 90% Capacity Guard.

Computes request priority tiers (0 = Express, 5 = Standard, 10-15 = Background)
using rolling p50/p90 prompt token length percentiles, interactive streaming boost,
and active slot saturation. Enforces a 10% safety buffer across both concurrent
request slots and KV-cache token capacity.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class CapacityLease:
    approx_tokens: int
    priority: int
    _released: bool = False


class PriorityScheduler:
    """Rolling-percentile priority scheduler and concurrency/KV-cache capacity guard."""

    def __init__(
        self,
        max_concurrent_requests: int = 32,
        max_kv_cache_tokens: int = 1_500_000,
    ) -> None:
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.max_kv_cache_tokens = max(100_000, int(max_kv_cache_tokens))
        self.active_requests: int = 0
        self.active_tokens: int = 0
        self._lock = asyncio.Lock()
        self._history: deque[int] = deque([1000] * 10, maxlen=100)

    @property
    def safe_max_concurrency(self) -> int:
        """Return 90% slot concurrency limit (10% safety reserve)."""
        return max(1, int(self.max_concurrent_requests * 0.90))

    @property
    def safe_max_kv_tokens(self) -> int:
        """Return 90% KV-cache token limit (10% safety reserve)."""
        return max(90_000, int(self.max_kv_cache_tokens * 0.90))

    def record_prompt_tokens(self, tokens: int) -> None:
        if tokens > 0:
            self._history.append(int(tokens))

    def get_prompt_percentiles(self) -> tuple[float, float]:
        """Return rolling (p50, p90) prompt token lengths."""
        if not self._history:
            return 1000.0, 4000.0
        sorted_hist = sorted(self._history)
        n = len(sorted_hist)
        p50 = float(statistics.median(sorted_hist))
        p90_idx = min(n - 1, int(n * 0.90))
        p90 = float(sorted_hist[p90_idx])
        return p50, max(p50 * 1.5, p90)

    def get_median_prompt_tokens(self) -> float:
        p50, _ = self.get_prompt_percentiles()
        return p50

    def calculate_priority(
        self, body: dict[str, Any], headers: dict[str, str] | Any
    ) -> tuple[int, int]:
        """Calculate (priority, approx_tokens) for an incoming chat completion payload."""
        total_chars = 0
        system_chars = 0
        messages = body.get("messages")
        if isinstance(messages, list):
            for idx, m in enumerate(messages):
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "")
                content = m.get("content", "")
                char_len = 0
                if isinstance(content, str):
                    char_len = len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            char_len += len(str(part.get("text", "")))
                total_chars += char_len
                if role == "system" or idx == 0:
                    system_chars += char_len
            approx_tokens = max(1, int(total_chars // 3.8))
            user_approx_tokens = max(1, int((total_chars - system_chars) // 3.8))
        else:
            approx_tokens = 256
            user_approx_tokens = 256

        p50, p90 = self.get_prompt_percentiles()
        is_stream = bool(body.get("stream", False))

        if user_approx_tokens <= p50 or approx_tokens <= p50:
            priority = 0
        elif approx_tokens <= p90:
            priority = 5 if self.active_requests <= 24 else 10
        else:
            priority = 10 if self.active_requests <= 24 else 15

        if is_stream and priority > 0:
            priority = max(0, priority - 1)

        hdr_prio = headers.get("x-vllm-priority") if hasattr(headers, "get") else None
        if hdr_prio is not None:
            try:
                return int(hdr_prio), approx_tokens
            except (ValueError, TypeError):
                pass

        if "priority" in body:
            try:
                body_prio = int(body["priority"])
                if body_prio >= 1000:
                    return 15, approx_tokens
                if body_prio != 100:
                    return body_prio, approx_tokens
            except (ValueError, TypeError):
                pass

        return priority, approx_tokens

    async def try_acquire_slot(
        self, body: dict[str, Any], headers: dict[str, str] | Any
    ) -> CapacityLease | None:
        """Attempt to acquire an active slot and KV token reservation under the 90% guard."""
        priority, approx_tokens = self.calculate_priority(body, headers)
        async with self._lock:
            if (
                self.active_requests >= self.safe_max_concurrency
                or (self.active_tokens + approx_tokens) >= self.safe_max_kv_tokens
            ):
                return None
            self.active_requests += 1
            self.active_tokens += approx_tokens
            self.record_prompt_tokens(approx_tokens)
            return CapacityLease(approx_tokens=approx_tokens, priority=priority)

    async def release_slot(self, lease: CapacityLease) -> None:
        """Idempotently release an acquired capacity lease."""
        async with self._lock:
            if lease._released:
                return
            lease._released = True
            self.active_requests = max(0, self.active_requests - 1)
            self.active_tokens = max(0, self.active_tokens - lease.approx_tokens)
