"""Benchmark plugin interface + registry.

To add a benchmark, subclass `Benchmark`, implement `load_tasks` and `grade`, and register it:

    @register("my_bench")
    class MyBench(Benchmark): ...

or reference it from an experiment YAML without registering: `benchmark: {id: "my_pkg.my_mod:MyBench"}`.
Everything else (prompts, answer tool, toolsets, corpus, leak patterns) has a sensible default
that a plugin may override. See `benchmarks/_template.py` and docs/ADDING_A_BENCHMARK.md.
"""

from __future__ import annotations

import importlib
import random
from pathlib import Path
from typing import Any

from ..core.task import GradeResult, Task
from ..tools.base import Tool
from ..tools.final_answer import FinalAnswerTool

_REGISTRY: dict[str, type["Benchmark"]] = {}

GENERIC_SYSTEM_PROMPT = """You are a careful, autonomous agent solving a task with the tools provided.

How to work:
- Read the task carefully and identify every requirement before acting.
- Before each tool call, briefly state what you are looking for and why.
- Verify important facts with your tools instead of guessing. If an approach is not working, change strategy rather than repeating it.
- You have a limited budget of steps. When you are told the budget is nearly used up, stop and answer.

Finishing:
- End by calling the final_answer tool exactly once with your answer, a short explanation and an honest confidence (0-100).
"""

TOOLSET_NOTES = {
    "closed_book": "In this setting you have no tools other than final_answer. Answer from your own knowledge.",
    "search_snippets": "Your research tool is web_search, which shows only short result snippets.",
    "search_fetch": "Your research tools are web_search and web_fetch (to read full pages).",
    "local_docs": "Your research tools search and read a fixed local document collection (search_docs, read_doc).",
    "search_docs": "You can search the live web (web_search, web_fetch) and a fixed local document collection (search_docs, read_doc).",
    "oracle_docs": "You are given the ids of documents that contain the evidence needed for this task. Read them with read_doc.",
}

DELIVERY_NOTES = {
    "paginate": "Long tool results are split into pages; use read_more to read further pages when needed.",
    "summarize": "Long tool results are replaced by summaries focused on the task.",
    "truncate": "Long tool results are cut off after a fixed length.",
}


class Benchmark:
    """Base class. Required: `load_tasks`, `grade`. Everything else is an overridable default."""

    id: str = "base"
    #: URL substrings that indicate this benchmark's data leaked onto the web (blocked + flagged).
    leak_patterns: list[str] = []

    def __init__(self, cfg: Any):
        self.cfg = cfg                       # BenchmarkConfig
        self.params: dict[str, Any] = dict(getattr(cfg, "params", {}) or {})
        self.data_dir = Path(getattr(cfg, "data_dir", "data"))

    # ---- data ----------------------------------------------------------------------------
    def prepare(self) -> None:
        """One-time download / preprocessing (`agent-eval prepare <id>`). Optional."""

    def load_tasks(self) -> list[Task]:
        raise NotImplementedError

    def tasks(self) -> list[Task]:
        """Tasks for this experiment after task_ids / num_examples sampling (deterministic by seed)."""
        tasks = sorted(self.load_tasks(), key=lambda t: t.task_id)
        if self.cfg.task_ids:
            wanted = set(self.cfg.task_ids)
            return [t for t in tasks if t.task_id in wanted]
        if self.cfg.num_examples:
            return random.Random(self.cfg.seed).sample(tasks, min(self.cfg.num_examples, len(tasks)))
        return tasks

    def corpus_dir(self) -> Path | None:
        """Directory with corpus.jsonl (+ BM25 index) for local_docs/oracle_docs toolsets, if any."""
        return None

    # ---- what the agent sees -------------------------------------------------------------
    def base_prompt(self) -> str:
        """Role + working instructions. Override for domain-specific guidance."""
        return GENERIC_SYSTEM_PROMPT

    def system_prompt(self, condition: str, delivery: str) -> str:
        """base_prompt() + notes describing the active toolset and delivery policy.
        Kept byte-stable per (condition, delivery) so prompt caching works."""
        s = self.base_prompt() + "\n" + TOOLSET_NOTES.get(condition, "") + "\n"
        if condition != "closed_book":
            for k, note in DELIVERY_NOTES.items():
                if delivery.startswith(k):
                    s += note + "\n"
        return s

    def task_prompt(self, task: Task, condition: str) -> str:
        p = f"Task:\n{task.prompt}"
        if condition == "oracle_docs" and self.oracle_doc_ids(task):
            p += "\n\nRelevant document ids: " + ", ".join(self.oracle_doc_ids(task))
        return p

    def answer_tool(self) -> Tool:
        """The termination tool. Override for structured answers (it must set ctx.state['final_answer'])."""
        return FinalAnswerTool()

    def toolsets(self) -> dict[str, list[str]]:
        """Extra or overriding named toolsets (condition -> tool names) for this benchmark."""
        return {}

    def oracle_doc_ids(self, task: Task) -> list[str]:
        return list(dict.fromkeys(task.evidence.doc_ids))

    # ---- grading -------------------------------------------------------------------------
    def format_response(self, answer: dict | None, fallback_text: str = "") -> str:
        """Render the agent's final answer as the text the grader sees."""
        if not answer:
            return fallback_text or "(no answer)"
        return str(answer.get("answer", ""))

    def confidence(self, answer: dict | None) -> float | None:
        c = (answer or {}).get("confidence")
        try:
            return None if c is None else max(0.0, min(100.0, float(c)))
        except (TypeError, ValueError):
            return None

    async def grade(self, task: Task, response: str, answer: dict | None, judge: "Judge") -> GradeResult:
        raise NotImplementedError


def register(benchmark_id: str):
    def deco(cls: type[Benchmark]) -> type[Benchmark]:
        cls.id = benchmark_id
        _REGISTRY[benchmark_id] = cls
        return cls

    return deco


def available() -> list[str]:
    _import_builtins()
    return sorted(_REGISTRY)


def get_benchmark(cfg: Any) -> Benchmark:
    """Resolve `cfg.id`: a registered id, or an import path "package.module:ClassName"."""
    _import_builtins()
    bid = cfg.id
    if bid in _REGISTRY:
        return _REGISTRY[bid](cfg)
    if ":" in bid:
        mod, cls_name = bid.split(":", 1)
        cls = getattr(importlib.import_module(mod), cls_name)
        if not issubclass(cls, Benchmark):
            raise TypeError(f"{bid} is not a Benchmark subclass")
        return cls(cfg)
    raise ValueError(f"unknown benchmark {bid!r}; registered: {sorted(_REGISTRY)} (or use 'module:Class')")


def _import_builtins() -> None:
    from . import browsecomp, browsecomp_plus, jsonl_qa  # noqa: F401  (registration side effects)


# imported late to avoid a cycle (graders use the Task/GradeResult types only)
from ..graders.base import Judge  # noqa: E402
