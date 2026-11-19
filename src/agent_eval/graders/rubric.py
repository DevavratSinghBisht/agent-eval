"""Rubric judge: binary criteria graded by an LLM (APEX-style). score = fraction of criteria met,
passed = all criteria met. The judge sees the task, the response and the rubric — not the trajectory."""

from __future__ import annotations

import json
import re

from ..core.task import GradeResult
from .base import Judge, JudgeError

RUBRIC_TEMPLATE = """You are grading a response against a rubric of binary criteria.

[task]:
{task}

[response]:
{response}

[criteria]:
{criteria}

For EACH criterion decide whether the response clearly meets it. Judge only what is written in the
response; do not give credit for implied or missing content.

Reply with JSON only: {{"criteria": [{{"id": <int>, "met": true|false, "reason": "<short>"}}, ...]}}"""


async def grade_rubric(judge: Judge, task: str, response: str, criteria: list[str]) -> GradeResult:
    listing = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, 1))
    try:
        reply = await judge.ask(RUBRIC_TEMPLATE.format(task=task, response=response, criteria=listing))
    except JudgeError as e:
        return GradeResult(None, grader_model=judge.model, error=str(e))
    parsed = parse_rubric_reply(reply.text, len(criteria))
    if parsed is None:
        return GradeResult(None, details={"raw": reply.text, "parse_ok": False}, grader_model=judge.model,
                           cost_usd=reply.cost_usd, error="unparseable rubric judgement")
    met = sum(parsed)
    return GradeResult(
        passed=met == len(criteria),
        score=met / len(criteria) if criteria else 0.0,
        details={"criteria_met": met, "criteria_total": len(criteria), "per_criterion": parsed, "parse_ok": True},
        grader_model=judge.model,
        cost_usd=reply.cost_usd,
    )


def parse_rubric_reply(text: str, n: int) -> list[bool] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        items = json.loads(m.group(0)).get("criteria", [])
    except json.JSONDecodeError:
        return None
    met = {int(it["id"]): bool(it.get("met")) for it in items if isinstance(it, dict) and "id" in it}
    if set(met) != set(range(1, n + 1)):
        return None
    return [met[i] for i in range(1, n + 1)]
