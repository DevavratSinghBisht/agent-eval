"""Benchmark-neutral task and grade types.

A benchmark plugin turns its dataset into `Task`s and grades responses into `GradeResult`s; the
rest of the harness (agent, tools, logging, analysis) only sees these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    """What counts as the evidence needed to solve a task (optional; powers the evidence funnel).

    answer_strings: strings whose appearance in a tool output means the answer was retrieved
                    (gold answer + aliases for QA; key facts for other tasks).
    doc_ids:        ids of documents in the benchmark's local corpus that contain the evidence.
    """

    answer_strings: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)


@dataclass
class Task:
    task_id: str
    prompt: str                      # what the agent is asked (the benchmark decides the wording)
    reference: Any = None            # gold answer / rubric / test spec — NEVER shown to the agent
    metadata: dict[str, Any] = field(default_factory=dict)   # topic, difficulty, split, ...
    evidence: Evidence = field(default_factory=Evidence)


@dataclass
class GradeResult:
    """Outcome of grading one response.

    passed: binary success used for pass rates (None when grading itself failed).
    score:  optional continuous score in [0, 1] (e.g. fraction of rubric criteria met);
            defaults to float(passed).
    details: grader-specific fields, flattened into tasks.csv as `grade_<key>` columns.
    """

    passed: bool | None
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    grader_model: str | None = None
    cost_usd: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.score is None and self.passed is not None:
            self.score = float(self.passed)
