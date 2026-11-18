"""Model price table used to turn token counts into dollars.

Prices are USD per 1M tokens: (input, output, cache_read, cache_write).
They change often, so treat this table as a default and override it with
``register_price()`` or the ``AGENTLENS_PRICES`` env var (a JSON file path).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


# Last checked: 2026-11. Anthropic cache reads bill at 0.1x input, writes at 1.25x.
_PRICES: dict[str, Price] = {
    # Anthropic
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),
    "claude-sonnet-4-6": Price(3.00, 15.00, 0.30, 3.75),
    "claude-haiku-4-5": Price(1.00, 5.00, 0.10, 1.25),
    # OpenAI
    "gpt-4.1": Price(2.00, 8.00, 0.50),
    "gpt-4.1-mini": Price(0.40, 1.60, 0.10),
    "gpt-4.1-nano": Price(0.10, 0.40, 0.025),
    "gpt-4o": Price(2.50, 10.00, 1.25),
    "gpt-4o-mini": Price(0.15, 0.60, 0.075),
    "gpt-5": Price(1.25, 10.00, 0.125),
    "gpt-5-mini": Price(0.25, 2.00, 0.025),
    "gpt-5-nano": Price(0.05, 0.40, 0.005),
}


def _load_env_overrides() -> None:
    path = os.getenv("AGENTLENS_PRICES")
    if not path or not os.path.exists(path):
        return
    with open(path) as f:
        for model, p in json.load(f).items():
            _PRICES[model] = Price(**p)


_load_env_overrides()


def register_price(model: str, input: float, output: float,
                   cache_read: float | None = None, cache_write: float | None = None) -> None:
    """Add or replace the price for a model (USD per 1M tokens)."""
    _PRICES[model] = Price(input, output, cache_read, cache_write)


def lookup(model: str | None) -> Price | None:
    if not model:
        return None
    if model in _PRICES:
        return _PRICES[model]
    # Match dated snapshots such as "gpt-4o-mini-2024-07-18" by longest prefix.
    best = max((k for k in _PRICES if model.startswith(k)), key=len, default=None)
    return _PRICES[best] if best else None


def cost_usd(model: str | None, input_tokens: int, output_tokens: int,
             cache_read: int = 0, cache_write: int = 0) -> float | None:
    """Dollar cost of one call, or None when the model has no known price.

    ``input_tokens`` is the total prompt size, including cached tokens
    (LangChain's ``usage_metadata`` convention).
    """
    price = lookup(model)
    if price is None:
        return None
    uncached = max(input_tokens - cache_read - cache_write, 0)
    total = uncached * price.input + output_tokens * price.output
    total += cache_read * (price.cache_read if price.cache_read is not None else price.input)
    total += cache_write * (price.cache_write if price.cache_write is not None else price.input)
    return total / 1_000_000
