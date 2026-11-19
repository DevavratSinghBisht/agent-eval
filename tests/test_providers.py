"""OpenAI-compatible provider tested offline through the real `openai` SDK with a mocked HTTP
transport: request shape, tool calls, usage, stop reasons, errors, and a campaign whose agent and
judge come from different providers."""

import asyncio
import json

import httpx2
import pandas as pd
import pytest

from agent_eval.core.config import ExperimentConfig, GraderConfig
from agent_eval.core.pricing import PricingError, load_catalog
from agent_eval.core.types import Message, StopReason, ToolCall, ToolResultBlock
from agent_eval.providers import CallOptions, ProviderError, make_client
from agent_eval.providers.mock_client import MockClient
from agent_eval.providers.openai_client import OpenAIChatClient
from agent_eval.tools.final_answer import FinalAnswerTool


def completion(message: dict, finish: str = "stop", prompt=120, cached=100, completion_tokens=30) -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion_tokens, "total_tokens": prompt + completion_tokens,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }


def client_with(handler, provider="openai", **kw) -> tuple[OpenAIChatClient, list]:
    seen = []

    def wrapped(request: httpx2.Request):
        seen.append(json.loads(request.content))
        return handler(request)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(wrapped))
    return OpenAIChatClient(provider, "m", http_client=http, max_retries=0, **kw), seen


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(k, "test-key")


def test_request_shape_tool_calls_and_usage():
    reply = completion({"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "final_answer", "arguments": '{"explanation":"e","exact_answer":"4","confidence":90}'}}]},
        finish="tool_calls")
    client, seen = client_with(lambda r: httpx2.Response(200, json=reply))
    history = [
        Message("user", text="Q"),
        Message("assistant", text="thinking aloud", tool_calls=[ToolCall("c0", "web_search", {"query": "x"})]),
        Message("user", text="[Budget notice]", tool_results=[ToolResultBlock("c0", "results", False)]),
    ]
    turn, _ = asyncio.run(client.complete("SYS", history, [FinalAnswerTool.spec], CallOptions(max_tokens=500, effort="low")))
    req = seen[0]
    assert [m["role"] for m in req["messages"]] == ["system", "user", "assistant", "tool", "user"]
    assert req["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'
    assert req["messages"][3] == {"role": "tool", "tool_call_id": "c0", "content": "results"}
    assert req["max_completion_tokens"] == 500 and req["reasoning_effort"] == "low"
    assert req["tools"][0]["function"]["strict"] is True
    assert turn.stop_reason == StopReason.TOOL_USE and turn.tool_calls[0].arguments["exact_answer"] == "4"
    assert (turn.usage.input_tokens, turn.usage.cache_read_tokens, turn.usage.output_tokens) == (20, 100, 30)


def test_compatible_preset_and_bad_arguments():
    reply = completion({"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{not json"}}]}, finish="tool_calls")
    client, seen = client_with(lambda r: httpx2.Response(200, json=reply), provider="openai_compatible",
                               base_url="http://localhost:8000/v1")
    turn, _ = asyncio.run(client.complete("", [Message("user", text="hi")], [FinalAnswerTool.spec], CallOptions(max_tokens=100, effort="high")))
    req = seen[0]
    assert "max_tokens" in req and "reasoning_effort" not in req and "strict" not in req["tools"][0]["function"]
    assert turn.tool_calls[0].arguments == {"_unparseable_arguments": "{not json"}  # surfaced to the tool, not dropped


def test_finish_reasons_and_errors():
    client, _ = client_with(lambda r: httpx2.Response(200, json=completion({"content": "cut"}, finish="length")))
    turn, _ = asyncio.run(client.complete("", [Message("user", text="hi")], [], CallOptions()))
    assert turn.stop_reason == StopReason.MAX_TOKENS
    client, _ = client_with(lambda r: httpx2.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(ProviderError) as e:
        asyncio.run(client.complete("", [Message("user", text="hi")], [], CallOptions()))
    assert e.value.exception_type == "RateLimitError" and e.value.retryable
    client, _ = client_with(lambda r: httpx2.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(ProviderError) as e:
        asyncio.run(client.complete("", [Message("user", text="hi")], [], CallOptions()))
    assert e.value.exception_type == "BadRequestError" and not e.value.retryable


def test_make_client_presets_and_missing_key(monkeypatch):
    assert isinstance(make_client("gemini", "g"), OpenAIChatClient)
    assert str(make_client("gemini", "g")._client.base_url).startswith("https://generativelanguage.googleapis.com")
    with pytest.raises(ValueError, match="needs base_url"):
        make_client("vllm", "m")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        make_client("openai", "m")
    with pytest.raises(ValueError, match="unknown provider"):
        make_client("nope", "m")


def test_price_catalog_glob_and_unpriced():
    cat = load_catalog()
    with pytest.raises(PricingError):
        cat.model("gpt-unpriced-model")
    from agent_eval.core.pricing import ModelPrice
    cat.models["qwen3*"] = ModelPrice(0, 0, 0, 0)
    assert cat.cost("qwen3:32b", __import__("agent_eval.core.types", fromlist=["Usage"]).Usage(1000, 1000)) == 0


async def test_campaign_agent_and_judge_from_different_providers(tmp_path):
    """Agent = mock provider, judge = OpenAI (mocked HTTP): both are called and logged separately."""
    from agent_eval.benchmarks.browsecomp import Task
    from agent_eval.core.pricing import ModelPrice
    from agent_eval.runner.campaign import Campaign
    from agent_eval.runner.grading import Grader
    from agent_eval.tracing.store import finalize

    judge_reply = completion({"content": "extracted_final_answer: Paris\nreasoning: same\ncorrect: yes\nconfidence: 90"})
    judge_client, judge_seen = client_with(lambda r: httpx2.Response(200, json=judge_reply))
    exp = ExperimentConfig(
        campaign_id="xprov",
        benchmark={"id": "browsecomp"},
        grader=GraderConfig(provider="openai", model="judge-model", effort=None),
        defaults={"model": {"provider": "mock", "name": "mock-model", "effort": None, "thinking_display": None},
                  "condition": "closed_book"},
    )
    catalog = load_catalog()
    catalog.models["judge-model"] = ModelPrice(1.0, 4.0, 0.5, 1.0)
    grader = Grader(exp.grader, catalog, judge_client)
    agent = lambda m, t: {"tool_calls": [("final_answer", {"explanation": "e", "exact_answer": "Paris", "confidence": 90})]}  # noqa: E731
    camp = Campaign(exp, root=tmp_path, client_factory=lambda p, m, **kw: MockClient(m, agent), grader=grader,
                    tasks=[Task("t1", "Capital of France?", "Paris")])
    camp.catalog = catalog
    await camp.run()
    finalize(camp.dir)
    runs, tasks = pd.read_csv(camp.dir / "runs.csv"), pd.read_csv(camp.dir / "tasks.csv")
    assert (runs.provider.iloc[0], runs.grader_provider.iloc[0]) == ("mock", "openai")
    assert tasks.passed.iloc[0] and tasks.grader_model.iloc[0] == "judge-model"
    assert tasks.grader_cost_usd.iloc[0] > 0 and "reasoning_effort" not in judge_seen[0]
