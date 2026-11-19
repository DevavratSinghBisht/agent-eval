"""Anthropic Messages API client (manual tool loop; the harness owns the loop for logging).

- Adaptive thinking + explicit effort (Haiku 5.5 defaults to `medium`; always sent explicitly).
- Top-level automatic prompt caching so the growing history prefix is cached each turn.
- History is append-only: assistant turns are replayed with their native content blocks
  (thinking blocks included), never edited.
- Parallel tool results go back in ONE user message, tool_result blocks first.
"""

from __future__ import annotations

from typing import Any

import anthropic

from ..core.types import (
    AssistantTurn,
    Message,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)
from .base import CallOptions, ProviderError

_STOP_MAP = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
    "pause_turn": StopReason.PAUSE_TURN,
    "stop_sequence": StopReason.STOP_SEQUENCE,
}


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str, max_retries: int = 4, timeout: float = 600.0):
        self.model = model
        self._client = anthropic.AsyncAnthropic(max_retries=max_retries, timeout=timeout)

    # ---- conversion -------------------------------------------------------------------
    @staticmethod
    def tool_to_api(t: ToolSpec) -> dict[str, Any]:
        d: dict[str, Any] = {"name": t.name, "description": t.description, "input_schema": t.parameters}
        if t.strict:
            d["strict"] = True
        return d

    def messages_to_api(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant":
                if m.native is not None and m.native_provider == self.provider:
                    content = m.native
                else:
                    content = []
                    if m.text:
                        content.append({"type": "text", "text": m.text})
                    for tc in m.tool_calls:
                        content.append(
                            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                        )
                out.append({"role": "assistant", "content": content})
            else:
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_call_id,
                        "content": r.content,
                        **({"is_error": True} if r.is_error else {}),
                    }
                    for r in m.tool_results
                ]
                if m.text:
                    content.append({"type": "text", "text": m.text})
                out.append({"role": "user", "content": content})
        return out

    # ---- call -------------------------------------------------------------------------
    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        options: CallOptions,
    ) -> tuple[AssistantTurn, dict[str, Any]]:
        thinking: dict[str, Any] = {"type": "adaptive"}
        if options.thinking_display:
            thinking["display"] = options.thinking_display
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": options.max_tokens,
            "messages": self.messages_to_api(messages),
            "thinking": thinking,
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            req["system"] = [{"type": "text", "text": system}]
        if tools:
            req["tools"] = [self.tool_to_api(t) for t in tools]
        if options.effort:
            req["output_config"] = {"effort": options.effort}
        req.update(options.extra)

        try:
            resp = await self._client.messages.create(**req)
        except anthropic.RateLimitError as e:
            raise ProviderError("RateLimitError", str(e), retryable=True) from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("APIConnectionError", str(e), retryable=True) from e
        except anthropic.BadRequestError as e:
            raise ProviderError("BadRequestError", str(e)) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(
                f"APIStatusError{e.status_code}", str(e), retryable=e.status_code >= 500
            ) from e

        raw = resp.to_dict()
        text_parts, reasoning_parts, calls = [], [], []
        for block in raw.get("content", []):
            bt = block.get("type")
            if bt == "text":
                text_parts.append(block.get("text", ""))
            elif bt == "thinking":
                if block.get("thinking"):
                    reasoning_parts.append(block["thinking"])
            elif bt == "tool_use":
                calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("input") or {}))
        u = raw.get("usage", {}) or {}
        usage = Usage(
            input_tokens=u.get("input_tokens") or 0,
            output_tokens=u.get("output_tokens") or 0,
            cache_read_tokens=u.get("cache_read_input_tokens") or 0,
            cache_write_tokens=u.get("cache_creation_input_tokens") or 0,
        )
        stop_details = raw.get("stop_details") or {}
        turn = AssistantTurn(
            text="\n".join(text_parts).strip(),
            reasoning="\n".join(reasoning_parts).strip(),
            tool_calls=calls,
            stop_reason=_STOP_MAP.get(raw.get("stop_reason") or "", StopReason.OTHER),
            usage=usage,
            native=raw.get("content", []),
            raw_response=raw,
            request_id=getattr(resp, "_request_id", None),
            refusal_category=stop_details.get("category") if isinstance(stop_details, dict) else None,
        )
        return turn, req
