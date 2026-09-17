"""The gateway service.

One request path, and the order of operations in it is the design:

  authenticate -> check the model -> reserve budget -> rate limit
  -> pick a provider -> call it -> settle the real cost -> record

Reserving before calling and settling after is what keeps concurrent requests
from collectively overshooting a budget. Settling in a `finally` is what keeps a
crashed request from leaking its reservation forever -- a leak there looks like a
tenant slowly losing quota for no reason, which is a miserable thing to debug.

Two endpoints exist for Kubernetes rather than for users, and they answer
different questions. `/healthz` asks "is this process alive"; `/readyz` asks "can
it serve traffic", which is false while every provider's breaker is open. Wiring
both to the same check is a common mistake that makes a rollout either hang or
send traffic into a service that cannot serve it.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import AsyncIterator
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .metrics import Metrics
from .observability import (
    Lifecycle,
    configure_logging,
    extract_request_id,
    log,
    observed,
    tenant_var,
)
from .policy import PolicyEngine
from .pricing import estimate_usd, settle_usd
from .routing import AllProvidersFailed, Router
from .store import build_store
from .upstream import Upstream

CONFIG_PATH = Path(os.environ.get("GATEWAY_CONFIG", "config/gateway.yaml"))


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"no gateway config at {path}. Set GATEWAY_CONFIG, or mount one -- "
            "a gateway with no tenants and no providers can only return errors."
        )
    return yaml.safe_load(path.read_text()) or {}


def create_app(config: dict[str, Any] | None = None, upstream: Upstream | None = None) -> FastAPI:
    config = config if config is not None else load_config()
    store = build_store(os.environ.get("GATEWAY_REDIS_URL"))
    policy = _policy_from(config, store)
    router = Router.from_config(config)
    metrics = Metrics()
    sender = upstream or Upstream(
        router,
        api_keys={
            p["name"]: os.environ.get(p.get("api_key_env", ""), "")
            for p in config.get("providers", [])
        },
    )

    logger = configure_logging()
    lifecycle = Lifecycle()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log(logger, "info", "starting", shared_quota=policy.shared_state,
            providers=[p.name for p in router.providers])
        yield
        # Runs on SIGTERM, before the process exits: readiness has already
        # started failing, so this is the window for in-flight work to finish.
        await lifecycle.drain(logger)

    app = FastAPI(title="llm-gateway", version="0.2.0", lifespan=lifespan)
    app.state.policy = policy
    app.state.router = router
    app.state.metrics = metrics
    app.state.upstream = sender
    app.state.lifecycle = lifecycle
    app.state.logger = logger

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process is running. Never depends on an upstream."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Readiness: traffic can actually be served right now."""
        healthy = [p.name for p in router.providers if p.healthy]
        quota_ok = policy.store.healthy()
        # Fail readiness the moment SIGTERM arrives, so the pod leaves the
        # Service's endpoints while it still has time to finish what it has.
        if lifecycle.shutting_down:
            return JSONResponse(
                {"ready": False, "reason": "shutting down",
                 "in_flight": lifecycle.in_flight},
                status_code=503,
            )
        # Quota state is part of readiness. With a shared store unreachable,
        # budgets fail closed, so serving traffic would mean refusing every
        # request -- better to leave the rotation than to answer 402 to everyone.
        ready = bool(healthy) and quota_ok
        return JSONResponse(
            {
                "ready": ready,
                "healthy_providers": healthy,
                "quota_store": {
                    "shared": policy.shared_state,
                    "healthy": quota_ok,
                },
                "providers": router.report(),
            },
            status_code=200 if ready else 503,
        )

    @app.get("/metrics")
    async def prometheus() -> PlainTextResponse:
        return PlainTextResponse(metrics.render(), media_type=metrics.content_type)

    @app.get("/v1/usage")
    async def usage() -> dict[str, Any]:
        """Cost attribution: who spent what. The reason finance asks for a gateway."""
        return {"tenants": policy.report(), "providers": router.report()}

    @app.post("/v1/messages")
    async def messages(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        started = time.perf_counter()
        request_id = extract_request_id(request.headers)
        key = x_api_key or (authorization or "").removeprefix("Bearer ").strip() or None

        # A malformed body is the caller's mistake and must not look like ours.
        # Letting it raise produced a 500, which pages the gateway's owner for
        # somebody else's bad request.
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                {"type": "error", "error": {
                    "type": "invalid_request", "message": "body is not valid JSON"}},
                status_code=400,
            )
        if not isinstance(payload, dict):
            return JSONResponse(
                {"type": "error", "error": {
                    "type": "invalid_request", "message": "body must be a JSON object"}},
                status_code=400,
            )

        model = str(payload.get("model", ""))
        if not model:
            return JSONResponse(
                {"type": "error", "error": {
                    "type": "invalid_request", "message": "model is required"}},
                status_code=400,
            )

        estimated = estimate_usd(model, payload)
        decision, tenant = policy.admit(key, model, estimated)
        tenant_name = tenant.name if tenant else "unknown"
        tenant_var.set(tenant_name)

        if not decision.allowed:
            reason = decision.denial.value if decision.denial else "?"
            metrics.denied(tenant_name, model, reason)
            log(logger, "info", "denied", request_id=request_id, tenant=tenant_name,
                model=model, reason=reason, status=decision.status_code)
            headers = (
                {"retry-after": str(max(1, int(decision.retry_after_s or 1)))}
                if decision.retry_after_s
                else None
            )
            return JSONResponse(
                {"type": "error", "error": {
                    "type": reason, "message": decision.reason}},
                status_code=decision.status_code,
                headers={**(headers or {}), "x-request-id": request_id},
            )

        settled = 0.0
        lifecycle.enter()
        try:
            result = await sender.send(model, payload)
        except AllProvidersFailed as exc:
            metrics.upstream_failed(tenant_name, model)
            log(logger, "error", "upstream unavailable", request_id=request_id,
                tenant=tenant_name, model=model,
                attempts=[{"provider": n, "why": w} for n, w in exc.attempts])
            return JSONResponse(
                {"type": "error", "error": {
                    "type": "upstream_unavailable", "message": str(exc),
                    "attempts": [{"provider": n, "why": w} for n, w in exc.attempts]}},
                status_code=503,
                headers={"x-request-id": request_id},
            )
        else:
            settled = settle_usd(model, result.usage)
            duration = time.perf_counter() - started
            metrics.served(
                tenant_name, model, result.provider, result.status,
                duration, settled, result.usage,
            )
            log(logger, "info", "served", request_id=request_id, tenant=tenant_name,
                model=model, provider=result.provider, status=result.status,
                duration_ms=round(duration * 1000, 1), cost_usd=round(settled, 6),
                **result.usage)
            return JSONResponse(
                result.body, status_code=result.status,
                headers={"x-request-id": request_id},
            )
        finally:
            lifecycle.leave()
            # In a finally so a crash cannot leak the reservation. A leak here
            # looks like a tenant slowly losing quota for no reason.
            if tenant is not None:
                policy.settle(tenant, estimated, settled)

    return app


def _policy_from(config: dict[str, Any], store=None) -> PolicyEngine:
    from .policy import Tenant

    tenants = {}
    for entry in config.get("tenants", []):
        t = Tenant(
            name=entry["name"], api_key=entry["api_key"],
            models=list(entry.get("models", [])),
            requests_per_s=float(entry.get("requests_per_s", 10)),
            burst=int(entry.get("burst", 20)),
            monthly_budget_usd=float(entry.get("monthly_budget_usd", 100)),
        )
        tenants[t.api_key] = t
    return PolicyEngine(tenants, store) if store is not None else PolicyEngine(tenants)


def run() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.environ.get("GATEWAY_PORT", "8080")),
    )
