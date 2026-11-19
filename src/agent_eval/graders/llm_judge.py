"""Template LLM judge: format a prompt from the task/response/reference, ask the judge model, parse
a yes/no verdict with a regex. BrowseComp's official grader is one instance of this."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.task import GradeResult
from .base import Judge, JudgeError

DEFAULT_TEMPLATE = """Judge whether the [response] to the [task] is correct given the [reference].

[task]: {task}

[response]: {response}

[reference]: {reference}

Explain briefly, then end with a line exactly of the form:
correct: yes
or
correct: no"""


@dataclass
class TemplateJudge:
    template: str = DEFAULT_TEMPLATE
    verdict_re: str = r"correct:\s*\**\s*(yes|no)"
    yes_value: str = "yes"
    # extra named regexes whose first group is stored in GradeResult.details
    extract: dict[str, str] = field(default_factory=dict)

    async def __call__(self, judge: Judge, **fields) -> GradeResult:
        prompt = self.template.format(**fields)
        try:
            reply = await judge.ask(prompt)
        except JudgeError as e:
            return GradeResult(None, grader_model=judge.model, error=str(e))
        m = re.search(self.verdict_re, reply.text, re.IGNORECASE)
        details = {"parse_ok": bool(m), "raw": reply.text}
        for name, rx in self.extract.items():
            mm = re.search(rx, reply.text, re.IGNORECASE)
            details[name] = mm.group(1).strip() if mm else None
        passed = bool(m) and m.group(1).lower() == self.yes_value  # unparseable -> counted as not passed
        return GradeResult(passed, details=details, grader_model=judge.model, cost_usd=reply.cost_usd)
