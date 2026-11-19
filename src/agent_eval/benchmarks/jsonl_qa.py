"""Bring-your-own-dataset QA benchmark: any JSONL file of questions and answers, no code needed.

Experiment YAML:
  benchmark:
    id: jsonl_qa
    params:
      path: examples/data/sample_qa.jsonl   # required
      id_field: id                          # default "id" (falls back to the line number)
      prompt_field: question                # default "question"
      answer_field: answer                  # default "answer"
      aliases_field: aliases                # optional list of accepted alternative answers
      grader: exact_match                   # exact_match (default, free) | llm_judge
      numeric_tolerance: 0.0                # exact_match only, relative tolerance for numbers
      corpus_dir: path/to/corpus            # optional: enables local_docs / oracle_docs toolsets
      evidence_docs_field: evidence_docids  # optional: per-question doc ids in that corpus
      leak_patterns: [my-dataset-name]      # optional: URL substrings to block / flag as leaks
Every other field of a record is kept as task metadata (e.g. topic, difficulty).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.task import Evidence, GradeResult, Task
from ..graders import Judge, TemplateJudge, exact_match
from .base import Benchmark, register


@register("jsonl_qa")
class JsonlQA(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)
        if "path" not in self.params:
            raise ValueError("jsonl_qa needs benchmark.params.path")
        self.leak_patterns = list(self.params.get("leak_patterns", []))

    def corpus_dir(self) -> Path | None:
        d = self.params.get("corpus_dir")
        return Path(d) if d else None

    def prepare(self) -> None:
        d = self.corpus_dir()
        if d and (d / "corpus.jsonl").exists() and not (d / "bm25_index").exists():
            from ..tools.local_docs import Corpus

            Corpus.build_index(d)

    def load_tasks(self) -> list[Task]:
        p = self.params
        f_id, f_prompt, f_ans = p.get("id_field", "id"), p.get("prompt_field", "question"), p.get("answer_field", "answer")
        f_alias, f_docs = p.get("aliases_field", "aliases"), p.get("evidence_docs_field")
        tasks = []
        with open(p["path"], encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                r = json.loads(line)
                answer = str(r[f_ans])
                aliases = [str(a) for a in (r.get(f_alias) or [])]
                meta = {k: v for k, v in r.items() if k not in {f_id, f_prompt, f_ans, f_alias, f_docs}}
                tasks.append(Task(
                    task_id=str(r.get(f_id, f"q{i:05d}")),
                    prompt=str(r[f_prompt]),
                    reference={"answer": answer, "aliases": aliases},
                    metadata=meta,
                    evidence=Evidence(answer_strings=[answer] + aliases,
                                      doc_ids=[str(d) for d in (r.get(f_docs) or [])] if f_docs else []),
                ))
        return tasks

    async def grade(self, task: Task, response: str, answer: dict | None, judge: Judge) -> GradeResult:
        refs = [task.reference["answer"], *task.reference["aliases"]]
        if self.params.get("grader", "exact_match") == "llm_judge":
            return await TemplateJudge()(judge, task=task.prompt, response=response, reference=" | ".join(refs))
        return exact_match((answer or {}).get("answer", "") or response, refs,
                           numeric_tolerance=float(self.params.get("numeric_tolerance", 0.0)))
