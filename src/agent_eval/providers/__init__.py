from __future__ import annotations

from .base import CallOptions, ModelClient, ProviderError


def make_client(provider: str, model: str) -> ModelClient:
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model)
    if provider == "mock":
        from .mock_client import MockClient

        return MockClient(model)
    raise ValueError(
        f"provider {provider!r} not implemented yet (Phase 4 adds openai, gemini, openai_compatible)"
    )


__all__ = ["CallOptions", "ModelClient", "ProviderError", "make_client"]
