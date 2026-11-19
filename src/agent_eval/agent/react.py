"""Strategy 1 — direct tool-calling (ReAct) agent.

Loop: model call -> execute all tool calls concurrently -> append results (one user message)
-> repeat, until final_answer, a cap, a refusal, or an unrecoverable error. Every model call,
tool call and harness intervention is recorded by the TaskRecorder.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from ..core.config import Limits
from ..core.types import HarnessPhase, Message, StopReason
from ..providers.base import CallOptions, ModelClient, ProviderError
from ..tools.base import ToolRuntime
from ..tracing.recorder import TaskRecorder
from . import prompts

MAX_CONSECUTIVE_PROVIDER_ERRORS = 3
MAX_NO_TOOL_NUDGES = 2


@dataclass
class AgentOutcome:
    termination_reason: str  # final_answer|step_cap|token_cap|time_cap|cost_cap|refusal|exception|no_final_answer
    final_answer: dict[str, Any] | None
    fallback_text: str
    final_phase: str
    n_steps: int


class ReactAgent:
    harness_name = "direct_tool"

    def __init__(self, client: ModelClient, options: CallOptions, limits: Limits):
        self.client, self.options, self.limits = client, options, limits

    async def run(
        self,
        system: str,
        user_prompt: str,
        runtime: ToolRuntime,
        rec: TaskRecorder,
    ) -> AgentOutcome:
        messages: list[Message] = [Message("user", text=user_prompt)]
        tools = runtime.specs
        lim = self.limits
        nudged = False
        no_tool_nudges = 0
        consecutive_errors = 0
        last_had_error = False
        last_text = ""
        sent_upto = 0  # for raw-API request deltas
        last_phase = HarnessPhase.EXECUTION

        while True:
            step = len(rec.model_calls)
            # ---- caps (checked before each call) ----------------------------------------
            cap = self._cap_hit(step, rec)
            if cap:
                rec.event("cap_hit", step, cap=cap)
                return AgentOutcome(cap, None, last_text, last_phase.value, step)
            if not nudged and self._used_fraction(step, rec) >= lim.wrap_up_fraction:
                nudged = True
                messages = _append_user_text(messages, prompts.WRAP_UP_NUDGE)
                rec.event("wrap_up_nudge", step)

            phase = HarnessPhase.RECOVERY if last_had_error else HarnessPhase.EXECUTION
            delta = {"messages_from": sent_upto, "n_messages": len(messages)}
            if step == 0:
                delta.update({"system": system, "tools": [t.__dict__ for t in tools], "first_user": user_prompt})
            t0 = time.perf_counter()
            try:
                turn, _req = await self.client.complete(system, messages, tools, self.options)
            except ProviderError as e:
                rec.record_call(phase, None, time.perf_counter() - t0, delta, e.exception_type, str(e))
                consecutive_errors += 1
                last_had_error, last_phase = True, phase
                if not e.retryable or consecutive_errors >= MAX_CONSECUTIVE_PROVIDER_ERRORS:
                    rec.event("abort_provider_error", step, exception_type=e.exception_type)
                    return AgentOutcome("exception", None, last_text, phase.value, step + 1)
                await asyncio.sleep(min(30, 2**consecutive_errors))
                continue
            consecutive_errors = 0
            sent_upto = len(messages)
            idx = rec.record_call(phase, turn, time.perf_counter() - t0, delta)
            messages.append(
                Message("assistant", text=turn.text, tool_calls=turn.tool_calls, native=turn.native,
                        native_provider=self.client.provider)
            )
            if turn.text:
                last_text = turn.text

            if turn.stop_reason == StopReason.REFUSAL:
                rec.event("refusal", idx, category=turn.refusal_category)
                return AgentOutcome("refusal", None, last_text, phase.value, idx + 1)

            if not turn.tool_calls:
                if turn.stop_reason == StopReason.MAX_TOKENS:
                    rec.event("max_tokens_nudge", idx)
                    messages.append(Message("user", text=prompts.MAX_TOKENS_NUDGE))
                    last_had_error, last_phase = True, phase
                    continue
                if turn.stop_reason == StopReason.PAUSE_TURN:
                    continue  # server paused; resend history unchanged
                no_tool_nudges += 1
                if no_tool_nudges > MAX_NO_TOOL_NUDGES:
                    rec.set_phase(idx, HarnessPhase.FINALIZATION)
                    rec.event("no_final_answer", idx)
                    return AgentOutcome("no_final_answer", None, last_text, HarnessPhase.FINALIZATION.value, idx + 1)
                rec.event("no_tool_nudge", idx)
                messages.append(Message("user", text=prompts.NO_TOOL_NUDGE))
                last_had_error, last_phase = True, phase
                continue

            # ---- execute tools concurrently; results go back in ONE user message ----------
            results = await asyncio.gather(*(runtime.execute(c, idx) for c in turn.tool_calls))
            for _, trec in results:
                rec.record_tool(trec)
            last_had_error = any(r.is_error for r, _ in results)
            fa = runtime.ctx.state.get("final_answer")
            if fa is not None:
                rec.set_phase(idx, HarnessPhase.FINALIZATION)
                return AgentOutcome("final_answer", fa, last_text, HarnessPhase.FINALIZATION.value, idx + 1)
            last_phase = phase
            messages.append(Message("user", tool_results=[r for r, _ in results]))

    # ---- budgets ----------------------------------------------------------------------
    def _cap_hit(self, step: int, rec: TaskRecorder) -> str | None:
        lim = self.limits
        if step >= lim.step_cap:
            return "step_cap"
        if rec.usage.total_tokens >= lim.token_budget_per_task:
            return "token_cap"
        if rec.elapsed >= lim.time_cap_s:
            return "time_cap"
        if rec.cost >= lim.cost_cap_usd:
            return "cost_cap"
        return None

    def _used_fraction(self, step: int, rec: TaskRecorder) -> float:
        lim = self.limits
        return max(
            step / lim.step_cap,
            rec.usage.total_tokens / lim.token_budget_per_task,
            rec.elapsed / lim.time_cap_s,
            rec.cost / lim.cost_cap_usd,
        )


def _append_user_text(messages: list[Message], text: str) -> list[Message]:
    """Add harness text to the pending user turn (after tool results) without editing history
    that the model has already seen."""
    if messages and messages[-1].role == "user":
        last = messages[-1]
        merged = Message("user", text=(last.text + "\n\n" + text).strip(), tool_results=last.tool_results)
        return messages[:-1] + [merged]
    return messages + [Message("user", text=text)]
