"""Deterministic scripted client for tests and offline harness checks.

`script` is a callable (messages, tools) -> dict describing the next turn:
    {"text": str, "reasoning": str, "tool_calls": [(name, args), ...], "stop": "tool_use"|"end_turn",
     "raise": "ExceptionType"}
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

from ..core.types import AssistantTurn, Message, StopReason, ToolCall, ToolSpec, Usage
from .base import CallOptions, ProviderError


class MockClient:
    provider = "mock"

    def __init__(self, model: str = "mock-model", script: Callable[[list[Message], list[ToolSpec]], dict] | None = None):
        self.model = model
        self.script = script or (lambda m, t: {"text": "no script", "stop": "end_turn"})
        self._ids = itertools.count()

    async def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec], options: CallOptions
    ) -> tuple[AssistantTurn, dict[str, Any]]:
        step = self.script(messages, tools)
        if step.get("raise"):
            raise ProviderError(step["raise"], "scripted failure", retryable=False)
        calls = [
            ToolCall(id=f"call_{next(self._ids)}", name=n, arguments=a) for n, a in step.get("tool_calls", [])
        ]
        prompt_chars = len(system) + sum(
            len(m.text) + sum(len(r.content) for r in m.tool_results) for m in messages
        )
        usage = Usage(
            input_tokens=max(1, prompt_chars // 4) // 2,
            cache_read_tokens=max(1, prompt_chars // 4) - max(1, prompt_chars // 4) // 2,
            output_tokens=20 + 10 * len(calls),
        )
        stop = StopReason.TOOL_USE if calls else StopReason(step.get("stop", "end_turn"))
        native = [{"type": "text", "text": step.get("text", "")}] + [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in calls
        ]
        turn = AssistantTurn(
            text=step.get("text", ""),
            reasoning=step.get("reasoning", ""),
            tool_calls=calls,
            stop_reason=stop,
            usage=usage,
            native=native,
            raw_response={"content": native, "stop_reason": stop.value},
        )
        return turn, {"model": self.model, "n_messages": len(messages)}
