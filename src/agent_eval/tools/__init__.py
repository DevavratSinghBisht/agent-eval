"""Toolset assembly per experimental condition."""

from __future__ import annotations

from ..core.pricing import PriceCatalog
from .base import Tool
from .delivery import DeliveryPolicy
from .final_answer import FinalAnswerTool

CONDITION_TOOLS = {
    "closed_book": [],
    "search_snippets": ["web_search_snippets"],
    "search_fetch": ["web_search", "web_fetch"],
    "local_docs": ["search_docs", "read_doc"],
    "search_docs": ["web_search", "web_fetch", "search_docs", "read_doc"],
    "oracle_docs": ["read_doc_oracle"],
}


def build_tools(
    condition: str,
    delivery: DeliveryPolicy,
    catalog: PriceCatalog,
    search_backend: str,
    search_k: int,
    data_dir: str,
    oracle_docids: set[str] | None = None,
) -> list[Tool]:
    names = CONDITION_TOOLS[condition]
    tools: list[Tool] = []
    backend = None
    corpus = None
    for n in names:
        if n.startswith("web_search"):
            from .web import WebSearchTool, make_backend

            backend = backend or make_backend(search_backend)
            tools.append(
                WebSearchTool(backend, catalog.search_cost(search_backend), k=search_k, snippets_only=n.endswith("_snippets"))
            )
        elif n == "web_fetch":
            from .web import WebFetchTool

            tools.append(WebFetchTool())
        elif n in ("search_docs", "read_doc", "read_doc_oracle"):
            from .local_docs import Corpus, ReadDocTool, SearchDocsTool

            corpus = corpus or Corpus.load(data_dir)
            if n == "search_docs":
                tools.append(SearchDocsTool(corpus, k=search_k))
            elif n == "read_doc":
                tools.append(ReadDocTool(corpus))
            else:
                tools.append(ReadDocTool(corpus, allowed=oracle_docids or set()))
    if names:  # delivery-only helper tools (read_more) are meaningless without other tools
        tools.extend(delivery.extra_tools())
    tools.append(FinalAnswerTool())
    return tools
