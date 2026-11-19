"""Offline checks of the Anthropic request builder (no network)."""

from agent_eval.core.types import Message, ToolCall, ToolResultBlock
from agent_eval.providers.anthropic_client import AnthropicClient
from agent_eval.tools.final_answer import FinalAnswerTool


def make():
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    return AnthropicClient("claude-haiku-5-5")


def test_native_replay_and_tool_result_order():
    c = make()
    native = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "tool_use", "id": "t1", "name": "web_search", "input": {"query": "x"}}]
    msgs = [
        Message("user", text="Q"),
        Message("assistant", tool_calls=[ToolCall("t1", "web_search", {"query": "x"})], native=native, native_provider="anthropic"),
        Message("user", text="[Budget notice]", tool_results=[ToolResultBlock("t1", "res", False)]),
    ]
    api = c.messages_to_api(msgs)
    assert api[1]["content"] is native  # thinking blocks replayed verbatim (append-only history)
    assert [b["type"] for b in api[2]["content"]] == ["tool_result", "text"]  # results first, then text


def test_foreign_native_is_rebuilt_and_errors_flagged():
    c = make()
    msgs = [
        Message("user", text="Q"),
        Message("assistant", text="hi", tool_calls=[ToolCall("t1", "x", {})], native=[{"bogus": 1}], native_provider="openai"),
        Message("user", tool_results=[ToolResultBlock("t1", "boom", True)]),
    ]
    api = c.messages_to_api(msgs)
    assert api[1]["content"] == [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t1", "name": "x", "input": {}}]
    assert api[2]["content"][0]["is_error"] is True


def test_strict_tool_schema():
    t = AnthropicClient.tool_to_api(FinalAnswerTool.spec)
    assert t["strict"] is True
    s = t["input_schema"]
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
