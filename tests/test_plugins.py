"""Benchmark plugin API: registry / import-path resolution, the generic JSONL benchmark, and the
reusable graders (exact match, template LLM judge, rubric judge)."""

import asyncio
import json

import pytest

from agent_eval.benchmarks import Benchmark, available, get_benchmark
from agent_eval.benchmarks.browsecomp import BrowseComp
from agent_eval.core.config import BenchmarkConfig
from agent_eval.core.pricing import load_catalog
from agent_eval.core.task import Task
from agent_eval.graders import Judge, TemplateJudge, exact_match, grade_rubric
from agent_eval.graders.rubric import parse_rubric_reply
from agent_eval.providers.mock_client import MockClient


def judge_replying(text: str) -> Judge:
    return Judge(MockClient("mock-model", lambda m, t: {"text": text, "stop": "end_turn"}), "mock-model", load_catalog())


def test_registry_and_import_path():
    assert {"browsecomp", "browsecomp_plus", "jsonl_qa"} <= set(available())
    assert isinstance(get_benchmark(BenchmarkConfig(id="browsecomp")), BrowseComp)
    b = get_benchmark(BenchmarkConfig(id="agent_eval.benchmarks.browsecomp:BrowseComp"))
    assert isinstance(b, BrowseComp)
    with pytest.raises(ValueError, match="unknown benchmark"):
        get_benchmark(BenchmarkConfig(id="nope"))


def test_jsonl_qa_loading_sampling_and_exact_match(tmp_path):
    p = tmp_path / "qa.jsonl"
    rows = [{"id": f"q{i}", "question": f"Q{i}?", "answer": str(i), "aliases": [f"number {i}"], "topic": "math"} for i in range(10)]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cfg = BenchmarkConfig(id="jsonl_qa", params={"path": str(p), "leak_patterns": ["my-qa"]}, num_examples=4, seed=1)
    b = get_benchmark(cfg)
    tasks = b.tasks()
    assert len(tasks) == 4 and tasks == get_benchmark(cfg).tasks()  # deterministic sampling
    t = next(x for x in b.load_tasks() if x.task_id == "q3")
    assert t.metadata == {"topic": "math"} and t.evidence.answer_strings == ["3", "number 3"]
    assert b.leak_patterns == ["my-qa"]
    ok = asyncio.run(b.grade(t, "3", {"answer": "Number 3"}, judge=None))
    bad = asyncio.run(b.grade(t, "4", {"answer": "4"}, judge=None))
    assert ok.passed and ok.score == 1.0 and not bad.passed


def test_exact_match_numeric_tolerance():
    assert exact_match("1,000", ["1000"]).passed
    assert exact_match("101", ["100"], numeric_tolerance=0.02).passed
    assert not exact_match("110", ["100"], numeric_tolerance=0.02).passed


def test_template_judge_and_failure_modes():
    g = asyncio.run(TemplateJudge()(judge_replying("reasons...\ncorrect: yes"), task="t", response="r", reference="x"))
    assert g.passed and g.details["parse_ok"] and g.grader_model == "mock-model"
    g = asyncio.run(TemplateJudge()(judge_replying("no verdict here"), task="t", response="r", reference="x"))
    assert g.passed is False and g.details["parse_ok"] is False
    failing = Judge(MockClient("mock-model", lambda m, t: {"raise": "BadRequestError"}), "mock-model", load_catalog())
    g = asyncio.run(TemplateJudge()(failing, task="t", response="r", reference="x"))
    assert g.passed is None and "BadRequestError" in g.error  # grading failure != wrong answer


def test_browsecomp_plugin_grades_with_official_template():
    b = BrowseComp(BenchmarkConfig(id="browsecomp"))
    task = Task("bc-x", "Which island?", "Zanzibar")
    resp = b.format_response({"explanation": "e", "answer": "Zanzibar", "confidence": 80})
    assert "Exact Answer: Zanzibar" in resp and "Confidence: 80%" in resp
    reply = "extracted_final_answer: Zanzibar\nreasoning: same\ncorrect: yes\nconfidence: 80"
    g = asyncio.run(b.grade(task, resp, None, judge_replying(reply)))
    assert g.passed and g.details["extracted_final_answer"] == "Zanzibar"
    assert "meticulous research agent" in b.system_prompt("search_fetch", "full")
    assert b.task_prompt(task, "search_fetch").startswith("Question:\nWhich island?")


def test_rubric_judge():
    reply = '{"criteria": [{"id": 1, "met": true}, {"id": 2, "met": false}, {"id": 3, "met": true}]}'
    g = asyncio.run(grade_rubric(judge_replying(reply), "task", "resp", ["a", "b", "c"]))
    assert g.passed is False and g.score == pytest.approx(2 / 3) and g.details["criteria_met"] == 2
    assert parse_rubric_reply('{"criteria": [{"id": 1, "met": true}]}', 2) is None  # missing criterion


def test_base_class_requires_load_and_grade():
    b = Benchmark(BenchmarkConfig(id="x"))
    with pytest.raises(NotImplementedError):
        b.load_tasks()
    with pytest.raises(NotImplementedError):
        asyncio.run(b.grade(Task("t", "p"), "", None, None))
