"""TEMPLATE — copy this file to add a benchmark. Not registered (the name starts with "_").

Minimum: implement `load_tasks` and `grade`, then either
  * add `@register("my_bench")` and import the module in `base._import_builtins()`, or
  * point the experiment at it without registering: `benchmark: {id: "agent_eval.benchmarks.my_bench:MyBench"}`.

Everything below `grade` is optional; delete what you don't need.
"""

from __future__ import annotations

from pathlib import Path

from ..core.task import Evidence, GradeResult, Task
from ..graders import Judge, TemplateJudge, exact_match, grade_rubric  # noqa: F401  (pick one)
from .base import Benchmark


class MyBench(Benchmark):
    # URL substrings that would reveal this benchmark's answers (blocked in search + flagged as leaks)
    leak_patterns = ["my-bench", "mybench"]

    # ---- required ------------------------------------------------------------------------
    def load_tasks(self) -> list[Task]:
        """Return ALL tasks; sampling (num_examples / seed / task_ids) is done by the harness.
        `self.params` holds `benchmark.params` from the YAML; `self.data_dir` the data directory."""
        rows = [{"id": "ex-1", "q": "What is the capital of France?", "a": "Paris"}]  # load your data here
        return [
            Task(
                task_id=r["id"],                       # stable across runs (used for resume & pairing)
                prompt=r["q"],                         # what the agent sees
                reference=r["a"],                      # gold — never shown to the agent
                metadata={"difficulty": "easy"},       # becomes columns in tasks.csv analysis
                evidence=Evidence(answer_strings=[r["a"]]),  # optional: powers the evidence funnel
            )
            for r in rows
        ]

    async def grade(self, task: Task, response: str, answer: dict | None, judge: Judge) -> GradeResult:
        """`response` = format_response(answer); `answer` = final_answer tool args (or None).
        Use a deterministic grader when you can; `judge.ask(prompt)` gives priced LLM access."""
        return exact_match((answer or {}).get("answer", ""), [task.reference])
        # LLM judge:   return await TemplateJudge()(judge, task=task.prompt, response=response, reference=task.reference)
        # rubric:      return await grade_rubric(judge, task.prompt, response, criteria=task.reference)

    # ---- optional ------------------------------------------------------------------------
    def prepare(self) -> None:
        """One-time download/preprocessing (`agent-eval prepare <id>`)."""

    def base_prompt(self) -> str:
        """Domain-specific role and instructions (default: a generic careful-agent prompt)."""
        return super().base_prompt()

    def task_prompt(self, task: Task, condition: str) -> str:
        return super().task_prompt(task, condition)

    def corpus_dir(self) -> Path | None:
        """Return a directory with corpus.jsonl to enable the local_docs / oracle_docs toolsets."""
        return None

    def toolsets(self) -> dict[str, list[str]]:
        """Add or override named toolsets, e.g. {"calc_only": ["python"]} (tools must be registered)."""
        return {}

    def format_response(self, answer: dict | None, fallback_text: str = "") -> str:
        return super().format_response(answer, fallback_text)
