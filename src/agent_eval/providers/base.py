"""Provider abstraction. Each provider turns neutral messages + tool specs into its own
request format and returns a neutral AssistantTurn (also keeping the native payload)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.types import AssistantTurn, Message, ToolSpec


@dataclass
class CallOptions:
    max_tokens: int = 16000
    effort: str | None = None
    thinking_display: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A model call failed after the provider's own retries. `exception_type` is logged."""

    def __init__(self, exception_type: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.exception_type = exception_type
        self.retryable = retryable


class ModelClient(Protocol):
    provider: str
    model: str

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        options: CallOptions,
    ) -> tuple[AssistantTurn, dict[str, Any]]:
        """Returns (turn, request_record). request_record is the provider request body
        (used for raw-API logging)."""
        ...
