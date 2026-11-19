"""Tool registry and named toolsets ("conditions").

A *toolset* is a named list of tool names, e.g. `search_fetch: [web_search, web_fetch]`. The run's
`condition` picks one. Toolsets come from (later overrides earlier):
  DEFAULT_TOOLSETS  <  Benchmark.toolsets()  <  experiment YAML `toolsets:`
Tools are built by factories registered with `@register_tool(name)`; a plugin can add its own
tools the same way. The benchmark's answer tool (final_answer) is always appended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.pricing import PriceCatalog
from .base import Tool
from .delivery import DeliveryPolicy

DEFAULT_TOOLSETS: dict[str, list[str]] = {
    "closed_book": [],
    "search_snippets": ["web_search_snippets"],
    "search_fetch": ["web_search", "web_fetch"],
    "local_docs": ["search_docs", "read_doc"],
    "search_docs": ["web_search", "web_fetch", "search_docs", "read_doc"],
    "oracle_docs": ["read_doc_oracle"],
}


@dataclass
class ToolBuildContext:
    """Everything a tool factory may need. Factories take what they use and ignore the rest."""

    catalog: PriceCatalog
    search_backend: str = "exa"
    search_k: int = 5
    corpus_dir: Any = None                 # Path | None, from Benchmark.corpus_dir()
    oracle_doc_ids: list[str] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=list)   # benchmark leak patterns
    cache: dict[str, Any] = field(default_factory=dict)  # shared objects (search backend, corpus)


ToolFactory = Callable[[ToolBuildContext], Tool]
_FACTORIES: dict[str, ToolFactory] = {}


def register_tool(name: str):
    def deco(fn: ToolFactory) -> ToolFactory:
        _FACTORIES[name] = fn
        return fn

    return deco


def available_tools() -> list[str]:
    return sorted(_FACTORIES)


def resolve_toolsets(*layers: dict[str, list[str]] | None) -> dict[str, list[str]]:
    out = dict(DEFAULT_TOOLSETS)
    for layer in layers:
        out.update(layer or {})
    return out


def build_tools(tool_names: list[str], ctx: ToolBuildContext, delivery: DeliveryPolicy, answer_tool: Tool) -> list[Tool]:
    unknown = [n for n in tool_names if n not in _FACTORIES]
    if unknown:
        raise ValueError(f"unknown tools {unknown}; registered: {available_tools()}")
    tools = [_FACTORIES[n](ctx) for n in tool_names]
    if tool_names:  # delivery helper tools (read_more) are meaningless without other tools
        tools.extend(delivery.extra_tools())
    tools.append(answer_tool)
    return tools


# ---- built-in tools ---------------------------------------------------------------------
def _backend(ctx: ToolBuildContext):
    from .web import make_backend

    if "search_backend" not in ctx.cache:
        ctx.cache["search_backend"] = make_backend(ctx.search_backend)
    return ctx.cache["search_backend"]


def _corpus(ctx: ToolBuildContext):
    from .local_docs import Corpus

    if ctx.corpus_dir is None:
        raise ValueError("this toolset needs a local corpus, but the benchmark has no corpus_dir()")
    if "corpus" not in ctx.cache:
        ctx.cache["corpus"] = Corpus.load(ctx.corpus_dir)
    return ctx.cache["corpus"]


@register_tool("web_search")
def _web_search(ctx: ToolBuildContext) -> Tool:
    from .web import WebSearchTool

    return WebSearchTool(_backend(ctx), ctx.catalog.search_cost(ctx.search_backend), k=ctx.search_k, blocklist=ctx.blocklist)


@register_tool("web_search_snippets")
def _web_search_snippets(ctx: ToolBuildContext) -> Tool:
    from .web import WebSearchTool

    return WebSearchTool(_backend(ctx), ctx.catalog.search_cost(ctx.search_backend), k=ctx.search_k,
                         snippets_only=True, blocklist=ctx.blocklist)


@register_tool("web_fetch")
def _web_fetch(ctx: ToolBuildContext) -> Tool:
    from .web import WebFetchTool

    return WebFetchTool(blocklist=ctx.blocklist)


@register_tool("search_docs")
def _search_docs(ctx: ToolBuildContext) -> Tool:
    from .local_docs import SearchDocsTool

    return SearchDocsTool(_corpus(ctx), k=ctx.search_k)


@register_tool("read_doc")
def _read_doc(ctx: ToolBuildContext) -> Tool:
    from .local_docs import ReadDocTool

    return ReadDocTool(_corpus(ctx))


@register_tool("read_doc_oracle")
def _read_doc_oracle(ctx: ToolBuildContext) -> Tool:
    from .local_docs import ReadDocTool

    return ReadDocTool(_corpus(ctx), allowed=set(ctx.oracle_doc_ids))
