"""Deterministic failure signals computed from a trial's trajectory (no LLM needed)."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

URL_RE = re.compile(r"https?://\S+")
LEAK_NGRAM = 15  # consecutive question words found verbatim in a tool output => likely a dump of the question


def leak_check(work: dict[str, Any]) -> tuple[bool, str | None]:
    """Flag trials whose tool outputs look like benchmark data: a blocked / dataset-file URL, or a
    long verbatim run of the question text. Leak suspects are excluded from performance stats."""
    from ..tools.web import is_blocked
    from .evidence import normalize

    q = normalize(work.get("prompt") or work["trajectory"].get("prompt", "")).split()
    grams = {" ".join(q[i : i + LEAK_NGRAM]) for i in range(len(q) - LEAK_NGRAM + 1)}
    patterns = work.get("leak_patterns") or []
    for s in work["trajectory"]["steps"]:
        for tc in s.get("tool_calls", []):
            raw = tc.get("tool_output_raw") or ""
            for url in URL_RE.findall(raw):
                if is_blocked(url.rstrip(").,]"), patterns):
                    return True, f"blocked URL in {tc['tool']} output at step {s['step']}: {url[:120]}"
            if grams:
                text = " " + normalize(raw) + " "
                if any(f" {g} " in text for g in grams):
                    return True, f"{LEAK_NGRAM}-word verbatim question text in {tc['tool']} output at step {s['step']}"
    return False, None


HEDGE_RE =re.compile(r"\b(or|either|possibly|maybe|perhaps)\b|;|\bvs\.?\b", re.IGNORECASE)


def trial_signals(work: dict[str, Any]) -> dict[str, Any]:
    task = work["task"]
    steps = [s for s in work["trajectory"]["steps"] if not s.get("closing")]
    calls = [tc for s in steps for tc in s.get("tool_calls", [])]
    research = [c for c in calls if c["tool"] != "final_answer"]
    keys = Counter((c["tool"], json.dumps(c["tool_input"], sort_keys=True)) for c in research)
    max_repeat = max(keys.values(), default=0)
    searches = [c for c in research if c["tool"] in ("web_search", "search_docs")]
    queries = [str(c["tool_input"].get("query", "")).lower().strip() for c in searches]
    fetches = [c for c in research if c["tool"] in ("web_fetch", "read_doc")]
    empty_fetch = sum(
        1 for c in fetches if (c.get("delivery_meta") or {}).get("tool_meta", {}).get("empty") or c["is_error"]
    )
    truncated = sum(1 for c in research if (c.get("delivery_meta") or {}).get("truncated"))
    read_more = sum(1 for c in calls if c["tool"] == "read_more")
    errors = sum(1 for c in calls if c["is_error"])
    events = Counter(e["kind"] for e in work["trajectory"].get("events", []))
    exact = str(task.get("answer") or "")
    conf = task.get("confidence")
    hedged = bool(exact) and (len(exact) > 80 or bool(HEDGE_RE.search(exact)))
    leak, leak_reason = leak_check(work)
    return {
        "leak_suspect": leak,
        "leak_reason": leak_reason,
        "n_research_calls": len(research),
        "n_searches": len(searches),
        "n_unique_queries": len(set(queries)),
        "n_fetches": len(fetches),
        "max_identical_call_repeats": max_repeat,
        "doom_loop": max_repeat >= 3,
        "tool_error_rate": round(errors / len(calls), 3) if calls else 0.0,
        "empty_or_failed_fetches": empty_fetch,
        "n_truncated_results": truncated,
        "n_read_more": read_more,
        "wrap_up_nudged": events.get("wrap_up_nudge", 0) > 0,
        "no_tool_nudges": events.get("no_tool_nudge", 0),
        "cap_hit": task.get("termination_reason") in ("step_cap", "token_cap", "time_cap", "cost_cap"),
        "hedged_answer": hedged,
        "overconfident_wrong": task.get("passed") is False and conf is not None and float(conf) >= 80,
        "premature_answer": task.get("termination_reason") == "final_answer" and len(research) == 0
        and bool(work.get("toolset_tools")),
    }
