"""Tool interface + per-task tool runtime.

Every tool returns a *raw* output; the runtime then applies the run's DeliveryPolicy to
produce what the model actually sees. Both are logged, which is what makes the
"retrieved vs delivered vs consumed" analysis possible.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..core.types import ToolCall, ToolResultBlock, ToolSpec, Usage


@dataclass
class RawToolOutput:
    text: str
    is_error: bool = False
    error_type: str | None = None
    urls: list[str] = field(default_factory=list)
    docids: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    deliverable: bool = True  # False -> never truncated (e.g. final_answer ack, read_more pages)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    call_id: str
    step: int
    tool: str
    tool_input: dict[str, Any]
    raw_text: str
    delivered_text: str
    delivery_meta: dict[str, Any]
    is_error: bool
    error_type: str | None
    latency_seconds: float
    urls: list[str]
    docids: list[str]
    cost_usd: float  # search fees + summarizer cost
    summarizer_usage: Usage | None = None


SummarizeFn = Callable[[str, str], Awaitable[tuple[str, Usage, float]]]


class Tool:
    spec: ToolSpec

    async def run(self, args: dict[str, Any], ctx: "ToolContext") -> RawToolOutput:  # pragma: no cover
        raise NotImplementedError


@dataclass
class ToolContext:
    prompt: str  # the task prompt (used by question-focused delivery policies)
    task_id: str
    results: dict[str, list[str]] = field(default_factory=dict)  # result_id -> pages (paginate)
    summarize: SummarizeFn | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def new_result_id(self) -> str:
        return "r" + uuid.uuid4().hex[:6]


class ToolRuntime:
    def __init__(self, tools: list[Tool], delivery, ctx: ToolContext, tool_timeout_s: float = 60.0):
        self.tools = {t.spec.name: t for t in tools}
        self.delivery = delivery
        self.ctx = ctx
        self.tool_timeout_s = tool_timeout_s

    @property
    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self.tools.values()]

    async def execute(self, call: ToolCall, step: int) -> tuple[ToolResultBlock, ToolCallRecord]:
        t0 = time.perf_counter()
        tool = self.tools.get(call.name)
        if tool is None:
            raw = RawToolOutput(
                text=f"Error: unknown tool {call.name!r}. Available tools: {', '.join(self.tools)}.",
                is_error=True,
                error_type="UnknownTool",
                deliverable=False,
            )
        else:
            try:
                raw = await asyncio.wait_for(tool.run(call.arguments, self.ctx), self.tool_timeout_s)
            except asyncio.TimeoutError:
                raw = RawToolOutput(
                    text=f"Error: tool {call.name} timed out after {self.tool_timeout_s:.0f}s. Try a different input.",
                    is_error=True,
                    error_type="ToolTimeout",
                    deliverable=False,
                )
            except Exception as e:  # tool bugs must not kill the trial; they are logged
                raw = RawToolOutput(
                    text=f"Error: {type(e).__name__}: {e}",
                    is_error=True,
                    error_type=type(e).__name__,
                    deliverable=False,
                )
        if raw.deliverable and not raw.is_error:
            delivered, meta, s_usage, s_cost = await self.delivery.apply(raw.text, self.ctx)
        else:
            delivered, meta, s_usage, s_cost = raw.text, self.delivery.passthrough_meta(raw.text), None, 0.0
        latency = time.perf_counter() - t0
        rec = ToolCallRecord(
            call_id=call.id,
            step=step,
            tool=call.name,
            tool_input=call.arguments,
            raw_text=raw.text,
            delivered_text=delivered,
            delivery_meta={**meta, **({"tool_meta": raw.meta} if raw.meta else {})},
            is_error=raw.is_error,
            error_type=raw.error_type,
            latency_seconds=round(latency, 3),
            urls=raw.urls,
            docids=raw.docids,
            cost_usd=round(raw.cost_usd + s_cost, 6),
            summarizer_usage=s_usage,
        )
        return ToolResultBlock(call.id, delivered, raw.is_error), rec


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """Strict-mode compatible object schema."""
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)
