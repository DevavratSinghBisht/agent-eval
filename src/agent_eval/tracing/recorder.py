"""Per-task recorder: the single source of truth for everything logged about one task.

Invariants enforced here (checked again by the validator):
  * one trajectory step per model call, same order/phase/usage, plus one closing step
  * call costs are computed from the versioned price catalog
  * task totals = sum of model-call totals (+ tool fees for cost)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any

from ..core.pricing import PriceCatalog
from ..core.types import AssistantTurn, HarnessPhase, Usage
from ..tools.base import ToolCallRecord


class TaskRecorder:
    def __init__(self, run_id: str, task_id: str, model: str, catalog: PriceCatalog):
        self.run_id, self.task_id, self.model, self.catalog = run_id, task_id, model, catalog
        self.model_calls: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.raw_api: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.usage = Usage()
        self.model_cost = 0.0
        self.tool_cost = 0.0
        self.exception_count = 0
        self.errors: list[str] = []
        self.t0 = time.perf_counter()
        self.started_at = time.time()

    # ---- model calls -------------------------------------------------------------------
    def record_call(
        self,
        phase: HarnessPhase,
        turn: AssistantTurn | None,
        latency: float,
        request_delta: dict[str, Any] | None,
        exception_type: str | None = None,
        error: str | None = None,
        model: str | None = None,
    ) -> int:
        idx = len(self.model_calls)
        model = model or self.model
        usage = turn.usage if turn else Usage()
        cost = self.catalog.cost(model, usage)
        self.usage = self.usage + usage
        self.model_cost += cost
        if exception_type:
            self.exception_count += 1
            if error:
                self.errors.append(f"{exception_type}: {error[:300]}")
        call_id = f"{self.run_id}:{self.task_id}:{idx}"
        self.model_calls.append(
            {
                "call_id": call_id,
                "run_id": self.run_id,
                "task_id": self.task_id,
                "call_index": idx,
                "harness_phase": phase.value,
                "model_name": model,
                "prompt_tokens": usage.input_tokens,
                "cached_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "completion_tokens": usage.output_tokens,
                "stop_reason": turn.stop_reason.value if turn else None,
                "latency_seconds": round(latency, 3),
                "exception_type": exception_type,
                "total_cost_usd": cost,
                "n_tool_calls_emitted": len(turn.tool_calls) if turn else 0,
                "request_id": turn.request_id if turn else None,
            }
        )
        self.steps.append(
            {
                "step": idx,
                "harness_phase": phase.value,
                "model_name": model,
                "reasoning": turn.reasoning if turn else "",
                "text": turn.text if turn else "",
                "tool_calls": [],
                "raw_response": turn.raw_response if turn else None,
                "prompt_tokens": usage.input_tokens,
                "cached_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "completion_tokens": usage.output_tokens,
                "stop_reason": turn.stop_reason.value if turn else None,
                "latency_seconds": round(latency, 3),
                "exception_type": exception_type,
                "error": error,
            }
        )
        self.raw_api.append({"call_index": idx, "request_delta": request_delta, "response": turn.raw_response if turn else None})
        return idx

    def set_phase(self, call_index: int, phase: HarnessPhase) -> None:
        self.model_calls[call_index]["harness_phase"] = phase.value
        self.steps[call_index]["harness_phase"] = phase.value

    # ---- tool calls --------------------------------------------------------------------
    def record_tool(self, rec: ToolCallRecord) -> None:
        d = asdict(rec)
        s_usage = d.pop("summarizer_usage")
        d.update({"run_id": self.run_id, "task_id": self.task_id})
        if s_usage:
            d.update({f"summarizer_{k}": v for k, v in s_usage.items()})
        self.tool_cost += rec.cost_usd
        self.tool_calls.append(d)
        self.steps[rec.step]["tool_calls"].append(
            {
                "call_id": rec.call_id,
                "tool": rec.tool,
                "tool_input": rec.tool_input,
                "tool_output_raw": rec.raw_text,
                "tool_output_delivered": rec.delivered_text,
                "delivery_meta": rec.delivery_meta,
                "is_error": rec.is_error,
                "error_type": rec.error_type,
                "latency_seconds": rec.latency_seconds,
            }
        )

    def event(self, kind: str, step: int, **data: Any) -> None:
        self.events.append({"kind": kind, "step": step, "t": round(time.perf_counter() - self.t0, 3), **data})

    # ---- totals ------------------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @property
    def cost(self) -> float:
        return round(self.model_cost + self.tool_cost, 6)

    def close(self, outcome: dict[str, Any]) -> dict[str, Any]:
        """Closing trajectory step (no usage) + assembled work record."""
        self.steps.append({"step": len(self.model_calls), "harness_phase": "closing", "closing": True, **outcome})
        return {
            "trace_id": uuid.uuid4().hex,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "trajectory": {"run_id": self.run_id, "task_id": self.task_id, "steps": self.steps, "events": self.events},
            "raw_api": self.raw_api,
        }
