"""`Judge`: the LLM access a benchmark's grader gets (priced, retried, model-configurable).

Benchmarks call `await judge.ask(prompt)` and never construct clients themselves, so the grader
model is set per experiment (`grader:` in the YAML) and its cost is always logged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..core.pricing import PriceCatalog
from ..core.types import Message
from ..providers import CallOptions, ProviderError


@dataclass
class JudgeReply:
    text: str
    cost_usd: float
    latency_seconds: float


class JudgeError(RuntimeError):
    pass


class Judge:
    def __init__(self, client, model: str, catalog: PriceCatalog, effort: str | None = "low", max_tokens: int = 4000):
        self.client, self.model, self.catalog = client, model, catalog
        self.effort, self.max_tokens = effort, max_tokens
        catalog.model(model)  # fail fast if the grader model is unpriced

    async def ask(self, prompt: str, system: str = "", retries: int = 3) -> JudgeReply:
        t0 = time.perf_counter()
        last: Exception | None = None
        for _ in range(retries):
            try:
                turn, _ = await self.client.complete(
                    system, [Message("user", text=prompt)], [], CallOptions(max_tokens=self.max_tokens, effort=self.effort)
                )
                return JudgeReply(turn.text, self.catalog.cost(self.model, turn.usage), round(time.perf_counter() - t0, 3))
            except ProviderError as e:
                last = e
                if not e.retryable:
                    break
        raise JudgeError(f"{getattr(last, 'exception_type', 'Error')}: {last}")
