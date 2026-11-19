"""Evidence-consumption funnel: retrieved -> delivered -> attended -> correct.

For each trial we look for the gold evidence in what the tools returned (raw), in what the
model was actually shown (delivered, after the delivery policy), and in what the model did
afterwards (reasoning, text, tool inputs, final answer). Each failure is attributed to the
first broken link.

Evidence definitions:
  * answer-string evidence (all BrowseComp tasks): normalized gold answer appears in the text
  * gold-doc evidence (BrowseComp-Plus): a gold/evidence docid appears in the tool output's docids
    (search results count as "retrieved"; read_doc of that docid counts as "delivered" if the
    answer string or a large part of the doc survived delivery)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def answer_variants(answer: str) -> list[str]:
    """Normalized forms of the gold answer worth searching for in text."""
    a = normalize(answer)
    out = {a}
    # numbers: "1,234" -> "1234"; "$5 million" -> "5 million"
    out.add(normalize(answer.replace(",", "")))
    # drop leading articles
    out.add(re.sub(r"^(the|a|an) ", "", a))
    # parenthetical aliases: "Foo (Bar)" -> "foo", "bar"
    for part in re.split(r"[()/;]", answer):
        p = normalize(part)
        if len(p) >= 4:
            out.add(p)
    return [v for v in out if len(v) >= 2]


def contains_answer(text: str, variants: list[str]) -> bool:
    if not text:
        return False
    t = " " + normalize(text) + " "
    return any(f" {v} " in t for v in variants)


def analyze_trial(work: dict[str, Any]) -> dict[str, Any]:
    """Returns task-level funnel fields and per-tool-call evidence flags."""
    task = work["task"]
    answer = task.get("correct_answer") or ""
    variants = answer_variants(answer)
    meta = work.get("task_metadata") or {}
    gold_docs = set(meta.get("gold_docids") or []) | set(meta.get("evidence_docids") or [])
    steps = [s for s in work["trajectory"]["steps"] if not s.get("closing")]

    per_call: dict[str, dict[str, Any]] = {}
    first_retrieved = first_delivered = None
    gold_doc_retrieved = gold_doc_read = False
    for s in steps:
        for tc in s.get("tool_calls", []):
            if tc["tool"] == "final_answer":
                continue
            raw_hit = contains_answer(tc.get("tool_output_raw", ""), variants)
            del_hit = contains_answer(tc.get("tool_output_delivered", ""), variants)
            tcrec = next((t for t in work["tool_calls"] if t["call_id"] == tc["call_id"]), {})
            docids = set(tcrec.get("docids") or [])
            gold_in = bool(docids & gold_docs)
            if gold_in:
                gold_doc_retrieved = True
                if tc["tool"] == "read_doc":
                    gold_doc_read = True
            per_call[tc["call_id"]] = {
                "answer_in_raw": raw_hit,
                "answer_in_delivered": del_hit,
                "gold_doc_in_output": gold_in,
            }
            if raw_hit and first_retrieved is None:
                first_retrieved = s["step"]
            if del_hit and first_delivered is None:
                first_delivered = s["step"]

    attended = False
    if first_delivered is not None:
        for s in steps:
            if s["step"] <= first_delivered:
                continue
            texts = [s.get("reasoning", ""), s.get("text", "")] + [
                str(tc.get("tool_input", "")) for tc in s.get("tool_calls", [])
            ]
            if any(contains_answer(t, variants) for t in texts):
                attended = True
                break
    final_has_answer = contains_answer(task.get("exact_answer") or "", variants)
    if first_delivered is not None and final_has_answer:
        attended = True

    passed = task.get("passed")
    condition = work.get("condition_id")
    if task.get("task_status") != "completed":
        category = "harness_error" if task.get("task_status") == "errored" else "grading_error"
    elif passed:
        category = "success"
    elif task.get("termination_reason") not in ("final_answer",):
        category = f"no_answer_{task.get('termination_reason')}"
    elif condition == "closed_book":
        category = "knowledge_miss"
    elif first_retrieved is None:
        category = "retrieval_miss"
    elif first_delivered is None:
        category = "truncation_loss"
    elif not attended:
        category = "comprehension_miss"
    elif final_has_answer:
        category = "format_or_grading"  # gold string in the answer but graded wrong
    else:
        category = "synthesis_error"

    return {
        "task": {
            "retrieved_hit": first_retrieved is not None,
            "delivered_hit": first_delivered is not None,
            "attended": attended,
            "first_retrieved_step": first_retrieved,
            "first_delivered_step": first_delivered,
            "gold_doc_retrieved": gold_doc_retrieved if gold_docs else None,
            "gold_doc_read": gold_doc_read if gold_docs else None,
            "funnel_category": category,
        },
        "tool_calls": per_call,
    }
