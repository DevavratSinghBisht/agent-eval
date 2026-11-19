"""BrowseComp (OpenAI, 2025) — reference benchmark plugin; faithful port of openai/simple-evals
`browsecomp_eval.py`.

Differences from the reference implementation (both are bug fixes, documented on purpose):
  1. simple-evals returns `match.group(0)` ("correct: yes") and compares it to "yes", so
     nothing is ever scored correct; we compare the captured group instead.
  2. simple-evals' HTML report shows the encrypted answer; we store the decrypted answer.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from ..core.task import Evidence, GradeResult, Task
from ..graders import Judge, TemplateJudge
from .base import Benchmark, register

CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

# Verbatim from simple-evals (originally from the HLE repo).
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

GRADE_RE = re.compile(r"correct:\s*\**\s*(yes|no)", re.IGNORECASE)
EXTRACT_RE = re.compile(r"extracted_final_answer:\s*(.*)", re.IGNORECASE)
CONF_RE = re.compile(r"confidence:\s*\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

RESEARCH_SYSTEM_PROMPT = """You are a meticulous research agent answering hard, fact-seeking questions that have a single short answer.

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


def derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def encrypt(plaintext: str, password: str) -> str:
    data = plaintext.encode()
    key = derive_key(password, len(data))
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


def parse_grade(text: str) -> dict[str, Any]:
    m = GRADE_RE.search(text)
    e = EXTRACT_RE.search(text)
    c = CONF_RE.search(text)
    return {
        "verdict": m.group(1).lower() if m else "no",
        "parse_ok": bool(m),
        "extracted_final_answer": e.group(1).strip() if e else None,
        "grader_confidence": float(c.group(1)) if c else None,
    }


@register("browsecomp")
class BrowseComp(Benchmark):
    """1,266 encrypted fact-seeking questions; live-web browsing; LLM judge vs the gold answer."""

    leak_patterns = [
        "browsecomp", "browse_comp", "browse-comp", "bcplus", "bc_plus", "bc-plus",
        "simple-evals", "simple_evals", "github.com/openai/simple-evals", "openaipublic.blob.core.windows.net",
    ]
    judge_template = TemplateJudge(
        template=GRADER_TEMPLATE,
        verdict_re=GRADE_RE.pattern,
        extract={"extracted_final_answer": EXTRACT_RE.pattern, "grader_confidence": CONF_RE.pattern},
    )

    # ---- data ----------------------------------------------------------------------------
    def csv_path(self) -> Path:
        return self.data_dir / "browsecomp" / "browse_comp_test_set.csv"

    def prepare(self) -> None:
        p = self.csv_path()
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with httpx.Client(timeout=120, follow_redirects=True) as c:
                r = c.get(CSV_URL)
                r.raise_for_status()
                p.write_bytes(r.content)

    def load_tasks(self) -> list[Task]:
        self.prepare()
        df = pd.read_csv(self.csv_path())
        tasks = []
        for i, row in enumerate(df.to_dict("records")):
            canary = row.get("canary", "")
            answer = decrypt(row.get("answer", ""), canary)
            tasks.append(
                Task(
                    task_id=f"bc-{i:04d}",
                    prompt=decrypt(row.get("problem", ""), canary),
                    reference=answer,
                    metadata={"topic": row.get("problem_topic")},
                    evidence=Evidence(answer_strings=[answer]),
                )
            )
        return tasks

    # ---- agent-facing --------------------------------------------------------------------
    def base_prompt(self) -> str:
        return RESEARCH_SYSTEM_PROMPT

    def task_prompt(self, task: Task, condition: str) -> str:
        return "Question:\n" + super().task_prompt(task, condition).removeprefix("Task:\n")

    # ---- grading -------------------------------------------------------------------------
    def format_response(self, answer: dict | None, fallback_text: str = "") -> str:
        """The official Explanation / Exact Answer / Confidence response format."""
        if not answer:
            return fallback_text or "Explanation: None\nExact Answer: None\nConfidence: 0%"
        conf = answer.get("confidence")
        return (
            f"Explanation: {answer.get('explanation', '')}\n"
            f"Exact Answer: {answer.get('answer', '')}\n"
            f"Confidence: {conf if conf is not None else 100}%"
        )

    async def grade(self, task: Task, response: str, answer: dict | None, judge: Judge) -> GradeResult:
        return await self.judge_template(judge, question=task.prompt, correct_answer=task.reference, response=response)
