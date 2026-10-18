"""Exponential Backoff with Jitter and Resilient Heartbeat Reconnect Manager."""

from __future__ import annotations

import logging
import random
from typing import Any

import httpx

log = logging.getLogger("seedinfer-worker.backoff")


def compute_backoff_delay(
    attempt: int,
    *,
    base_sec: float = 1.0,
    max_sec: float = 60.0,
    jitter_ratio: float = 0.2,
) -> float:
    """Compute exponential backoff delay in seconds with bounded uniform jitter."""
    exp_step = min(max_sec, base_sec * (2 ** max(0, int(attempt))))
    if jitter_ratio <= 0.0:
        return round(exp_step, 3)
    delta = exp_step * jitter_ratio
    jittered = exp_step + random.uniform(-delta, delta)
    return round(max(0.1, min(max_sec, jittered)), 3)


class HeartbeatClient:
    """Manages worker heartbeat transmission with exponential backoff on transient failures."""

    def __init__(
        self,
        gateway_url: str,
        node_token: str = "",
        base_backoff_sec: float = 1.0,
        max_backoff_sec: float = 60.0,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.node_token = node_token.strip()
        self.base_backoff_sec = base_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.consecutive_failures: int = 0

    def build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.node_token:
            headers["Authorization"] = f"Bearer {self.node_token}"
        return headers

    def next_retry_delay(self, retry_after_header: str | None = None) -> float:
        """Compute next wait duration after a failure or HTTP 429 response."""
        if retry_after_header:
            try:
                parsed = float(retry_after_header.strip())
                if parsed > 0:
                    return min(300.0, max(1.0, parsed))
            except ValueError:
                pass
        return compute_backoff_delay(
            self.consecutive_failures,
            base_sec=self.base_backoff_sec,
            max_sec=self.max_backoff_sec,
        )

    async def send_heartbeat(
        self,
        payload: dict[str, Any],
        client: httpx.AsyncClient | None = None,
    ) -> tuple[bool, float]:
        """Send heartbeat payload to gateway. Returns `(success, next_wait_seconds)`."""
        urls = (
            f"{self.gateway_url}/v1/providers/heartbeat",
            f"{self.gateway_url}/api/v1/providers/heartbeat",
        )
        headers = self.build_headers()
        own_client = client is None
        http_client = client or httpx.AsyncClient(timeout=10.0)
        try:
            for url in urls:
                try:
                    resp = await http_client.post(url, json=payload, headers=headers)
                except Exception as exc:
                    log.debug("Heartbeat transport failure against %s: %s", url, exc)
                    continue

                if resp.status_code in (200, 201, 202, 204):
                    self.consecutive_failures = 0
                    return True, 0.0

                if resp.status_code == 429:
                    self.consecutive_failures += 1
                    wait_s = self.next_retry_delay(resp.headers.get("retry-after"))
                    return False, wait_s

                if resp.status_code in (401, 403):
                    self.consecutive_failures += 1
                    return False, self.next_retry_delay()

            self.consecutive_failures += 1
            return False, self.next_retry_delay()
        finally:
            if own_client:
                await http_client.aclose()
