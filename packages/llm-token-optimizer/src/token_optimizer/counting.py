"""Counting tokens, and being honest about what the count is.

Anthropic's tokenizer is not published, so an offline count is necessarily an
approximation. This uses a real BPE tokenizer rather than a characters-divided-by-
four rule, which matters because the two disagree most on exactly the content
this project optimises: JSON, code, repeated structure and long identifiers.

What that approximation is good for and not:

**Good for** comparing two versions of the same prompt. Every saving reported
here is a ratio between two counts produced by the same tokenizer, and a
systematic offset cancels out of a ratio. "38% fewer tokens" survives the
approximation.

**Not good for** predicting a bill to the cent. Absolute dollar figures are
computed from published list prices and a proxy count, so they are estimates
with the tokenizer's error in them. They are labelled as computed, never as
billed, and `count_tokens` against the real API is the way to settle an absolute
number when one is needed.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

# A widely used BPE vocabulary, close enough in behaviour for ratio work.
PROXY_TOKENIZER = "gpt2"


@functools.lru_cache(maxsize=2)
def _tokenizer(name: str = PROXY_TOKENIZER):
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(name)


def count(text: str, tokenizer: str = PROXY_TOKENIZER) -> int:
    if not text:
        return 0
    return len(_tokenizer(tokenizer).encode(text).ids)


def count_messages(messages: Sequence[dict[str, Any]], system: str = "") -> int:
    """Token count for a request's system prompt and messages.

    Per-message framing overhead is not modelled: it is a small constant per
    message that appears on both sides of every comparison this project makes.
    """
    total = count(system)
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += count(content)
        else:
            total += count(json.dumps(content))
    return total


def count_tools(tools: Sequence[dict[str, Any]]) -> int:
    """Tool schemas are part of the prefix and are usually the largest fixed cost."""
    return sum(count(json.dumps(tool, sort_keys=True)) for tool in tools)


# Published list price, USD per million tokens: input, output, cache write, cache read.
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5": (10.00, 50.00, 12.50, 1.00),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-4-6": (3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}
DEFAULT_PRICING = PRICING["claude-opus-5"]


def pricing_for(model: str) -> tuple[float, float, float, float]:
    name = model.split("/")[-1].lower()
    if name.endswith(":free"):
        return (0.0, 0.0, 0.0, 0.0)
    return PRICING.get(name, DEFAULT_PRICING)


@dataclass(frozen=True)
class Spend:
    """A token breakdown and what it costs at list price.

    `cost_usd` is computed, not billed. It carries the proxy tokenizer's error.
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens

    @property
    def cost_usd(self) -> float:
        price_in, price_out, price_write, price_read = pricing_for(self.model)
        return (
            self.input_tokens * price_in
            + self.output_tokens * price_out
            + self.cache_write_tokens * price_write
            + self.cache_read_tokens * price_read
        ) / 1_000_000

    @property
    def uncached_cost_usd(self) -> float:
        """The same tokens with no caching, which is the honest counterfactual."""
        price_in, price_out, _, _ = pricing_for(self.model)
        return (self.total_input * price_in + self.output_tokens * price_out) / 1_000_000

    def __add__(self, other: "Spend") -> "Spend":
        model = self.model or other.model
        return Spend(
            model=model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


def reduction(before: int, after: int) -> float:
    """Percentage reduction, negative when something got bigger."""
    if not before:
        return 0.0
    return (before - after) / before * 100
