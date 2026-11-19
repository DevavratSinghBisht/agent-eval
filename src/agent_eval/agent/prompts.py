"""Agent prompts. The system prompt is frozen per condition (byte-stable for prompt caching);
everything task-specific goes in the first user message."""

from __future__ import annotations

_BASE = """You are a meticulous research agent answering hard, fact-seeking questions that have a single short answer.

How to work:
- The question lists several constraints. Find a candidate that satisfies ALL of them, and verify each constraint against a source before you answer.
- Prefer primary and authoritative sources. Do not trust a single snippet for a critical fact.
- Before each tool call, briefly state what you are looking for and why.
- If a line of search is not working, change strategy (different entity, different constraint, different phrasing) instead of repeating it.
- You have a limited budget of steps. When you are told the budget is nearly used up, stop searching and answer.

Finishing:
- End by calling the final_answer tool exactly once with a brief explanation, ONE exact answer, and an honest confidence (0-100).
- Never list multiple alternative answers. If you are unsure, give your single best guess with a low confidence.
"""

_CONDITION_NOTES = {
    "closed_book": "In this setting you have no research tools. Answer from your own knowledge, reasoning carefully about each constraint, then call final_answer.",
    "search_snippets": "Your research tool is web_search, which shows only short result snippets.",
    "search_fetch": "Your research tools are web_search and web_fetch (to read full pages).",
    "local_docs": "Your research tools search and read a fixed local collection of web documents (search_docs, read_doc). The answer is supported by documents in this collection.",
    "search_docs": "You can search the live web (web_search, web_fetch) and a fixed local document collection (search_docs, read_doc).",
    "oracle_docs": "You are given the ids of documents that contain the evidence needed for this question. Read them with read_doc.",
}

_DELIVERY_NOTES = {
    "paginate": "Long tool results are split into pages; use read_more to read further pages when needed.",
    "summarize": "Long tool results are replaced by summaries focused on the question.",
    "truncate": "Long tool results are cut off after a fixed length.",
}


def system_prompt(condition: str, delivery: str) -> str:
    s = _BASE + "\n" + _CONDITION_NOTES[condition] + "\n"
    for k, note in _DELIVERY_NOTES.items():
        if delivery.startswith(k) and condition != "closed_book":
            s += note + "\n"
    return s


def task_prompt(question: str, oracle_docids: list[str] | None = None) -> str:
    p = f"Question:\n{question}"
    if oracle_docids:
        p += "\n\nRelevant document ids: " + ", ".join(oracle_docids)
    return p


WRAP_UP_NUDGE = (
    "[Budget notice] You have used most of your step/token budget for this task. Stop researching now "
    "and call final_answer with your single best answer and an honest confidence."
)
NO_TOOL_NUDGE = (
    "You replied without calling a tool. If you are done, call final_answer now; otherwise continue "
    "researching with the available tools."
)
MAX_TOKENS_NUDGE = "Your previous response was cut off by the output length limit. Be more concise and continue."

SUMMARIZER_PROMPT = """Summarize the following tool output for a research agent working on this question:

<question>
{question}
</question>

Keep every fact, name, date, number and URL that could be relevant to the question, verbatim where possible. Drop boilerplate. Do not answer the question yourself.

<tool_output>
{text}
</tool_output>"""
