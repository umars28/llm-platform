"""Talking to providers, and deciding what counts as their fault.

The distinction this module exists for: **which failures are worth failing over,
and which are the caller's own.**

A 500, a timeout or a connection refusal is the provider's problem, and trying
the next one is right. A 400 is a malformed request and will be malformed at
every provider, so failing over turns one clear error into three slow ones and
hides the actual cause. A 401 is our credentials, not their health -- failing
over on it would mark a perfectly healthy provider as broken because someone
rotated a key.

Getting that wrong is how a gateway manufactures its own outage: mark everything
as a provider failure, and one bad deploy on the client side opens every breaker
in the fleet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .routing import AllProvidersFailed, Provider, Router

# Status codes that say "this provider is unwell", as opposed to "your request
# is wrong" or "your credentials are wrong".
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


@dataclass
class UpstreamResult:
    provider: str
    status: int
    body: dict[str, Any]
    latency_ms: float
    attempts: list[tuple[str, str]] = field(default_factory=list)

    @property
    def usage(self) -> dict[str, int]:
        raw = self.body.get("usage") or {}
        return {
            "input_tokens": int(raw.get("input_tokens", 0) or 0),
            "output_tokens": int(raw.get("output_tokens", 0) or 0),
            "cache_read_input_tokens": int(raw.get("cache_read_input_tokens", 0) or 0),
            "cache_creation_input_tokens": int(
                raw.get("cache_creation_input_tokens", 0) or 0
            ),
        }


def is_provider_fault(status: int) -> bool:
    """Whether this status justifies failing over and opening a breaker."""
    return status in RETRYABLE_STATUS


class Upstream:
    """Sends a request down the provider chain until one answers."""

    def __init__(
        self,
        router: Router,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 60.0,
        api_keys: dict[str, str] | None = None,
    ) -> None:
        self.router = router
        self._client = client
        self.timeout_s = timeout_s
        self.api_keys = api_keys or {}

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    def _headers(self, provider: Provider) -> dict[str, str]:
        key = self.api_keys.get(provider.name, "")
        headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if key:
            # Gateways in front of the API take a bearer token; the API itself
            # takes x-api-key. Sending both is rejected, so choose by key shape.
            if key.startswith("sk-ant-"):
                headers["x-api-key"] = key
            else:
                headers["authorization"] = f"Bearer {key}"
        return headers

    async def send(
        self, model: str, payload: dict[str, Any], now: float | None = None
    ) -> UpstreamResult:
        attempts: list[tuple[str, str]] = []

        for provider in self.router.chain(model, now):
            body = {**payload, "model": provider.upstream_model(model)}
            started = time.perf_counter()
            try:
                response = await self.client.post(
                    provider.base_url.rstrip("/") + "/v1/messages",
                    json=body,
                    headers=self._headers(provider),
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                provider.observe(ok=False, now=now)
                attempts.append((provider.name, f"{type(exc).__name__}"))
                continue

            latency_ms = (time.perf_counter() - started) * 1000

            if response.status_code < 400:
                provider.observe(ok=True, latency_ms=latency_ms, now=now)
                return UpstreamResult(
                    provider.name, response.status_code, response.json(),
                    latency_ms, attempts,
                )

            if is_provider_fault(response.status_code):
                provider.observe(ok=False, now=now)
                attempts.append((provider.name, f"HTTP {response.status_code}"))
                continue

            # The caller's fault. Return it as-is rather than asking three
            # providers the same malformed question.
            provider.observe(ok=True, latency_ms=latency_ms, now=now)
            return UpstreamResult(
                provider.name, response.status_code, _safe_json(response),
                latency_ms, attempts,
            )

        raise AllProvidersFailed(model, attempts)


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text[:500]}}
