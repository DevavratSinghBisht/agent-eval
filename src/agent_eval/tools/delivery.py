"""Tool-result delivery policies: what part of a raw tool output the model actually sees.

Token counts here are a deterministic, provider-independent approximation
(1 token ~= 4 characters) so the same policy truncates identically for every model.
"""

from __future__ import annotations

import math
from typing import Any

from ..core.types import ToolSpec, Usage
from .base import RawToolOutput, Tool, ToolContext, schema

CHARS_PER_TOKEN = 4
FULL_CEILING_TOKENS = 100_000  # safety ceiling for `full`; hits are logged as truncation


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


class DeliveryPolicy:
    name = "base"

    def _meta(self, raw: str, delivered: str, truncated: bool, **extra) -> dict[str, Any]:
        return {
            "policy": self.name,
            "raw_tokens": approx_tokens(raw),
            "delivered_tokens": approx_tokens(delivered),
            "raw_chars": len(raw),
            "delivered_chars": len(delivered),
            "truncated": truncated,
            **extra,
        }

    def passthrough_meta(self, raw: str) -> dict[str, Any]:
        return self._meta(raw, raw, False, passthrough=True)

    async def apply(self, raw: str, ctx: ToolContext) -> tuple[str, dict[str, Any], Usage | None, float]:
        raise NotImplementedError

    def extra_tools(self) -> list[Tool]:
        return []


class FullPolicy(DeliveryPolicy):
    name = "full"

    def __init__(self, ceiling_tokens: int = FULL_CEILING_TOKENS):
        self.ceiling_chars = ceiling_tokens * CHARS_PER_TOKEN

    async def apply(self, raw, ctx):
        if len(raw) <= self.ceiling_chars:
            return raw, self._meta(raw, raw, False), None, 0.0
        shown = raw[: self.ceiling_chars]
        delivered = shown + _trunc_marker(len(shown), len(raw))
        return delivered, self._meta(raw, delivered, True, kept_chars=len(shown), ceiling_hit=True), None, 0.0


class TruncatePolicy(DeliveryPolicy):
    def __init__(self, max_tokens: int, mode: str = "head"):
        self.max_chars = max_tokens * CHARS_PER_TOKEN
        self.mode = mode
        self.name = f"truncate_{max_tokens // 1000}k"

    async def apply(self, raw, ctx):
        if len(raw) <= self.max_chars:
            return raw, self._meta(raw, raw, False, kept_chars=len(raw)), None, 0.0
        if self.mode == "head_tail":
            half = self.max_chars // 2
            shown = raw[:half] + "\n[...]\n" + raw[-half:]
            kept = 2 * half
        else:
            shown = raw[: self.max_chars]
            kept = len(shown)
        delivered = shown + _trunc_marker(kept, len(raw))
        return delivered, self._meta(raw, delivered, True, kept_chars=kept, mode=self.mode), None, 0.0


class PaginatePolicy(DeliveryPolicy):
    """First page is delivered; the agent may call read_more(result_id, page) for the rest.
    Pages concatenate back to exactly the raw text."""

    def __init__(self, page_tokens: int):
        self.page_chars = page_tokens * CHARS_PER_TOKEN
        self.name = f"paginate_{page_tokens // 1000}k"

    def paginate(self, raw: str) -> list[str]:
        pages, i = [], 0
        while i < len(raw):
            j = min(len(raw), i + self.page_chars)
            if j < len(raw):  # prefer to break on whitespace in the last 10% of the page
                k = raw.rfind("\n", i + int(self.page_chars * 0.9), j)
                if k == -1:
                    k = raw.rfind(" ", i + int(self.page_chars * 0.9), j)
                if k > i:
                    j = k + 1
            pages.append(raw[i:j])
            i = j
        return pages or [""]

    async def apply(self, raw, ctx):
        pages = self.paginate(raw)
        if len(pages) == 1:
            return raw, self._meta(raw, raw, False, n_pages=1, page=1, kept_chars=len(raw)), None, 0.0
        rid = ctx.new_result_id()
        ctx.results[rid] = pages
        delivered = pages[0] + (
            f"\n\n[Page 1 of {len(pages)} of this result (about {approx_tokens(raw)} tokens in total). "
            f"Call read_more with result_id=\"{rid}\" and page=2..{len(pages)} to read the rest.]"
        )
        return (
            delivered,
            self._meta(raw, delivered, True, n_pages=len(pages), page=1, result_id=rid, kept_chars=len(pages[0])),
            None,
            0.0,
        )

    def extra_tools(self) -> list[Tool]:
        return [ReadMoreTool()]


class SummarizePolicy(DeliveryPolicy):
    """Outputs above `threshold_tokens` are replaced by a task-focused LLM summary."""

    def __init__(self, threshold_tokens: int = 2000):
        self.threshold_chars = threshold_tokens * CHARS_PER_TOKEN
        self.name = "summarize"

    async def apply(self, raw, ctx):
        if len(raw) <= self.threshold_chars or ctx.summarize is None:
            return raw, self._meta(raw, raw, False, summarized=False), None, 0.0
        summary, usage, cost = await ctx.summarize(raw, ctx.prompt)
        delivered = (
            f"[Summary of a {approx_tokens(raw)}-token result, written for the current task. "
            f"Details not in this summary are not shown.]\n{summary}"
        )
        return delivered, self._meta(raw, delivered, True, summarized=True), usage, cost


class ReadMoreTool(Tool):
    spec = ToolSpec(
        name="read_more",
        description=(
            "Read another page of a long tool result that was split into pages. Use the result_id "
            "shown at the end of the first page and a page number from 2 up to the page count. "
            "Use it when the information you need may be further into the document."
        ),
        parameters=schema(
            {
                "result_id": {"type": "string", "description": "The result_id from the page footer."},
                "page": {"type": "integer", "description": "1-based page number to read."},
            }
        ),
    )

    async def run(self, args, ctx):
        rid, page = args.get("result_id", ""), int(args.get("page", 0))
        pages = ctx.results.get(rid)
        if not pages:
            return RawToolOutput(f"Error: unknown result_id {rid!r}.", is_error=True, error_type="BadResultId", deliverable=False)
        if not 1 <= page <= len(pages):
            return RawToolOutput(
                f"Error: page must be between 1 and {len(pages)} for result {rid}.",
                is_error=True,
                error_type="BadPage",
                deliverable=False,
            )
        text = pages[page - 1] + f"\n\n[Page {page} of {len(pages)} of result {rid}.]"
        return RawToolOutput(text, deliverable=False, meta={"result_id": rid, "page": page, "n_pages": len(pages)})


def _trunc_marker(kept: int, total: int) -> str:
    return (
        f"\n\n[Truncated: showing the first ~{approx_tokens(' ' * kept)} of ~{approx_tokens(' ' * total)} tokens. "
        "The rest of this result is not available.]"
    )


def make_policy(name: str) -> DeliveryPolicy:
    if name == "full":
        return FullPolicy()
    if name.startswith("truncate_"):
        return TruncatePolicy(int(name.removeprefix("truncate_").removesuffix("k")) * 1000)
    if name.startswith("paginate_"):
        return PaginatePolicy(int(name.removeprefix("paginate_").removesuffix("k")) * 1000)
    if name == "summarize":
        return SummarizePolicy()
    raise ValueError(f"unknown delivery policy {name!r}")
