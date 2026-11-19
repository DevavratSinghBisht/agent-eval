"""OpenAI Chat Completions client — also serves every OpenAI-compatible endpoint (Gemini's
OpenAI-compatible API, DeepSeek, OpenRouter, Ollama, vLLM, ...) via provider presets or an
explicit base_url.

Same neutral interface as AnthropicClient: tool calling, usage (cached prompt tokens split out),
stop-reason mapping, and errors normalised to ProviderError. History is rebuilt from neutral
messages every call (provider-native blocks are not replayed across endpoints).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import openai

from ..core.types import AssistantTurn, Message, StopReason, ToolCall, ToolSpec, Usage
from .base import CallOptions, ProviderError


@dataclass(frozen=True)
class Preset:
    base_url: str | None
    api_key_env: str | None
    strict_tools: bool = False            # OpenAI-only strict JSON-schema tool arguments
    max_tokens_param: str = "max_tokens"  # OpenAI reasoning models require max_completion_tokens
    effort_param: str | None = None       # request field that receives ModelConfig.effort


PRESETS: dict[str, Preset] = {
    "openai": Preset(None, "OPENAI_API_KEY", strict_tools=True, max_tokens_param="max_completion_tokens",
                     effort_param="reasoning_effort"),
    "gemini": Preset("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
                     effort_param="reasoning_effort"),
    "deepseek": Preset("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openrouter": Preset("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": Preset("http://localhost:11434/v1", None),
    "vllm": Preset(None, None),               # base_url required in the config
    "openai_compatible": Preset(None, None),  # base_url required in the config
}

_FINISH = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


class OpenAIChatClient:
    def __init__(self, provider: str, model: str, base_url: str | None = None, api_key_env: str | None = None,
                 max_retries: int = 4, timeout: float = 600.0, http_client=None):
        if provider not in PRESETS:
            raise ValueError(f"unknown OpenAI-compatible provider {provider!r}; presets: {sorted(PRESETS)}")
        self.provider, self.model = provider, model
        self.preset = PRESETS[provider]
        base_url = base_url or self.preset.base_url
        if provider in ("vllm", "openai_compatible") and not base_url:
            raise ValueError(f"provider {provider!r} needs base_url in the model/grader config")
        key_env = api_key_env or self.preset.api_key_env
        api_key = os.environ.get(key_env) if key_env else None
        if key_env and not api_key:
            raise ProviderError("MissingAPIKey", f"{key_env} is not set (needed for provider {provider!r})")
        self._client = openai.AsyncOpenAI(
            api_key=api_key or "not-needed", base_url=base_url, max_retries=max_retries, timeout=timeout,
            **({"http_client": http_client} if http_client is not None else {}),
        )

    # ---- conversion -------------------------------------------------------------------
    def tool_to_api(self, t: ToolSpec) -> dict[str, Any]:
        fn: dict[str, Any] = {"name": t.name, "description": t.description, "parameters": t.parameters}
        if t.strict and self.preset.strict_tools:
            fn["strict"] = True
        return {"type": "function", "function": fn}

    @staticmethod
    def messages_to_api(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                        for tc in m.tool_calls
                    ]
                out.append(msg)
            else:
                # tool results are separate role=tool messages; any harness text follows as a user message
                for r in m.tool_results:
                    content = ("Error: " + r.content) if r.is_error and not r.content.startswith("Error") else r.content
                    out.append({"role": "tool", "tool_call_id": r.tool_call_id, "content": content})
                if m.text:
                    out.append({"role": "user", "content": m.text})
        return out

    # ---- call -------------------------------------------------------------------------
    async def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                       options: CallOptions) -> tuple[AssistantTurn, dict[str, Any]]:
        req: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages_to_api(system, messages),
            self.preset.max_tokens_param: options.max_tokens,
        }
        if tools:
            req["tools"] = [self.tool_to_api(t) for t in tools]
        if options.effort and self.preset.effort_param:
            req[self.preset.effort_param] = options.effort
        req.update(options.extra)
        try:
            resp = await self._client.chat.completions.create(**req)
        except openai.RateLimitError as e:
            raise ProviderError("RateLimitError", str(e), retryable=True) from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise ProviderError("APIConnectionError", str(e), retryable=True) from e
        except openai.BadRequestError as e:
            raise ProviderError("BadRequestError", str(e)) from e
        except openai.APIStatusError as e:
            raise ProviderError(f"APIStatusError{e.status_code}", str(e), retryable=e.status_code >= 500) from e
        return self._parse(resp.model_dump(), getattr(resp, "_request_id", None)), req

    def _parse(self, raw: dict[str, Any], request_id: str | None) -> AssistantTurn:
        choices = raw.get("choices") or []
        if not choices:
            raise ProviderError("EmptyResponse", f"no choices in response from {self.provider}/{self.model}")
        choice = choices[0]
        msg = choice.get("message") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {"_unparseable_arguments": fn.get("arguments")}
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": fn.get("arguments")}  # the tool reports BadArgs to the model
            calls.append(ToolCall(id=tc.get("id") or f"call_{len(calls)}", name=fn.get("name", ""), arguments=args))
        u = raw.get("usage") or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        prompt = u.get("prompt_tokens") or 0
        usage = Usage(input_tokens=max(0, prompt - cached), output_tokens=u.get("completion_tokens") or 0,
                      cache_read_tokens=cached, cache_write_tokens=0)
        finish = choice.get("finish_reason") or ""
        stop = _FINISH.get(finish, StopReason.OTHER)
        if calls:
            stop = StopReason.TOOL_USE
        refusal = msg.get("refusal")
        if refusal:
            stop = StopReason.REFUSAL
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return AssistantTurn(
            text=(msg.get("content") or "").strip(),
            reasoning=reasoning.strip() if isinstance(reasoning, str) else "",
            tool_calls=calls,
            stop_reason=stop,
            usage=usage,
            native=None,  # rebuilt from neutral messages each call
            raw_response=raw,
            request_id=request_id,
            refusal_category="content_filter" if finish == "content_filter" else (refusal and "refusal"),
        )
