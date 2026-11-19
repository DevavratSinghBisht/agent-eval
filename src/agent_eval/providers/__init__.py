from __future__ import annotations

from .base import CallOptions, ModelClient, ProviderError

# provider -> implementation. Everything except anthropic/mock goes through the OpenAI-compatible
# Chat Completions client (official `openai` SDK) with a preset base_url and API-key variable.
OPENAI_COMPATIBLE = ("openai", "gemini", "deepseek", "openrouter", "ollama", "vllm", "openai_compatible")
PROVIDERS = ("anthropic", *OPENAI_COMPATIBLE, "mock")


def make_client(provider: str, model: str, base_url: str | None = None, api_key_env: str | None = None) -> ModelClient:
    """Build a client for `provider`/`model`. `base_url` / `api_key_env` override the provider preset
    (needed for vllm / openai_compatible, optional elsewhere)."""
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model)
    if provider in OPENAI_COMPATIBLE:
        from .openai_client import OpenAIChatClient

        return OpenAIChatClient(provider, model, base_url=base_url, api_key_env=api_key_env)
    if provider == "mock":
        from .mock_client import MockClient

        return MockClient(model)
    raise ValueError(f"unknown provider {provider!r}; available: {PROVIDERS}")


__all__ = ["CallOptions", "ModelClient", "ProviderError", "make_client", "PROVIDERS"]
