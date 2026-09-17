"""Estimating a request's cost before it runs, and settling it after.

The estimate exists so budget can be enforced *before* money is committed, and
it is necessarily wrong: output length is unknown until the model has finished.
It errs high on purpose. An estimate that errs low lets a tenant overshoot its
budget on every request, which is the failure that makes a budget decorative.

Settlement then replaces the estimate with the billed usage, so the error does
not accumulate.
"""

from __future__ import annotations

import json
from typing import Any

# USD per million tokens: input, output, cache write, cache read.
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5": (10.00, 50.00, 12.50, 1.00),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5": (3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}
DEFAULT_PRICING = PRICING["claude-opus-5"]

# Characters per token. Deliberately low, so the estimate errs high.
CHARS_PER_TOKEN = 3.5


def pricing_for(model: str) -> tuple[float, float, float, float]:
    name = (model or "").split("/")[-1].lower()
    if name.endswith(":free"):
        return (0.0, 0.0, 0.0, 0.0)
    return PRICING.get(name, DEFAULT_PRICING)


def estimate_usd(model: str, payload: dict[str, Any]) -> float:
    """Upper-bound cost, charging max_tokens as if they will all be produced."""
    price_in, price_out, _, _ = pricing_for(model)
    text = json.dumps(payload.get("messages", [])) + str(payload.get("system", ""))
    input_tokens = len(text) / CHARS_PER_TOKEN
    output_tokens = float(payload.get("max_tokens", 1024))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def settle_usd(model: str, usage: dict[str, int]) -> float:
    """Actual cost from the usage the provider reported."""
    price_in, price_out, price_write, price_read = pricing_for(model)
    return (
        usage.get("input_tokens", 0) * price_in
        + usage.get("output_tokens", 0) * price_out
        + usage.get("cache_creation_input_tokens", 0) * price_write
        + usage.get("cache_read_input_tokens", 0) * price_read
    ) / 1_000_000
