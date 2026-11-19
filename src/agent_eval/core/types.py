"""Provider-neutral conversation and tool types.

Every provider client translates to/from these types. Assistant turns also keep the
provider-native content (`native`) so that history can be replayed byte-for-byte to the
same provider (required for thinking-block / thought-signature preservation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    PAUSE_TURN = "pause_turn"
    STOP_SEQUENCE = "stop_sequence"
    OTHER = "other"


class HarnessPhase(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    RECOVERY = "recovery"
    FINALIZATION = "finalization"
    SUMMARIZATION = "summarization"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    """Token usage for one model call.

    `input_tokens` is UNCACHED input only; cached reads/writes are counted separately,
    matching the convention of the example loss-analysis data (prompt_tokens excludes cached).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class Message:
    """One conversation message.

    role: "user" | "assistant".
    text: plain text content (user prompt, nudges, assistant visible text).
    tool_calls: assistant tool calls.
    tool_results: user-side tool results.
    native: provider-native content blocks for assistant turns (replayed verbatim).
    native_provider: which provider produced `native`.
    """

    role: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResultBlock] = field(default_factory=list)
    native: Any = None
    native_provider: str | None = None


@dataclass
class AssistantTurn:
    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    stop_reason: StopReason
    usage: Usage
    native: Any
    raw_response: dict[str, Any]
    request_id: str | None = None
    refusal_category: str | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema (object)
    strict: bool = True
