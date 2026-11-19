"""End-to-end campaign with a scripted model, mock web and a tiny local corpus.

Checks: every condition and delivery policy runs; logging invariants hold; the evidence
funnel attributes the planted truncation loss correctly; resume works; the validator catches
injected faults."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from agent_eval.analysis.report import build_report
from agent_eval.analysis.validate import validate
from agent_eval.benchmarks.browsecomp import Task
from agent_eval.core.config import ExperimentConfig, GraderConfig
from agent_eval.core.pricing import load_catalog
from agent_eval.providers.mock_client import MockClient
from agent_eval.runner.campaign import Campaign
from agent_eval.runner.grading import Grader
from agent_eval.tools.local_docs import Corpus
from agent_eval.tools.web import MockBackend
from agent_eval.tracing.store import finalize, get_trajectory

ANSWER = "Zanzibar"
# answer planted ~40K chars in: survives full, lost by truncate_2k/8k, reachable via paginate read_more
LONG_PAGE = ("filler text about islands and spices. " * 1100) + f" The festival was first held in {ANSWER} in 1998. " + ("more filler. " * 200)


def agent_script(messages, tools):
    names = {t.name for t in tools}
    seen = "\n".join(r.content for m in messages for r in m.tool_results)
    n_tool_msgs = sum(1 for m in messages if m.tool_results)
    final = lambda ans, c: {"tool_calls": [("final_answer", {"explanation": "found", "exact_answer": ans, "confidence": c})]}  # noqa: E731
    if ANSWER in seen:
        return final(ANSWER, 90)
    page = re.search(r'result_id="(r[0-9a-f]+)" and page=2\.\.(\d+)', seen)
    if page and "read_more" in names:
        rid, last = page.group(1), int(page.group(2))
        done = len(re.findall(rf"of result {rid}", seen))
        if done + 2 <= last:
            return {"text": "reading further", "tool_calls": [("read_more", {"result_id": rid, "page": done + 2})]}
    if n_tool_msgs >= 4 or names == {"final_answer"}:
        return final("Madagascar", 60)
    if "web_search" in names and "web_fetch" in names and n_tool_msgs == 0:
        return {"text": "searching", "reasoning": "look up festival", "tool_calls": [("web_search", {"query": "festival islands spices first held"})]}
    if "web_fetch" in names and n_tool_msgs == 1:
        url = re.search(r"URL: (\S+)", seen).group(1)
        return {"tool_calls": [("web_fetch", {"url": url})]}
    if "web_search" in names and n_tool_msgs == 0:
        return {"tool_calls": [("web_search", {"query": "festival islands spices"})]}
    if "search_docs" in names and not any("docid:" in r.content for m in messages for r in m.tool_results):
        return {"tool_calls": [("search_docs", {"query": "festival islands spices first held"})]}
    if "read_doc" in names:
        m = re.search(r"docid: (\S+)", seen)
        docid = m.group(1) if m else "d1"
        return {"tool_calls": [("read_doc", {"docid": docid})]}
    return final("Madagascar", 60)


def oracle_aware_script(messages, tools):
    # oracle_docs: docids are listed in the first user message
    first = messages[0].text
    m = re.search(r"Relevant document ids: (.+)", first)
    if m and not any(mm.tool_results for mm in messages):
        return {"tool_calls": [("read_doc", {"docid": m.group(1).split(",")[0].strip()})]}
    return agent_script(messages, tools)


def grader_script(messages, tools):
    resp = messages[0].text
    exact = re.search(r"Exact Answer: (.*)", resp.split("[response]:", 1)[1]).group(1)
    ok = ANSWER.lower() in exact.lower()
    return {"text": f"extracted_final_answer: {exact}\nreasoning: compare\ncorrect: {'yes' if ok else 'no'}\nconfidence: 90", "stop": "end_turn"}


@pytest.fixture
def setup(tmp_path):
    MockBackend.pages = {
        "https://example.org/festival": ("Island spice festival history", LONG_PAGE),
        "https://example.org/other": ("Unrelated islands page", "islands spices " * 50),
    }
    data_dir = tmp_path / "data"
    corpus = Corpus.from_records(
        [
            {"docid": "d1", "text": "title: Spice festival\n" + LONG_PAGE, "url": "https://example.org/festival"},
            {"docid": "d2", "text": "title: Other\nislands " * 30, "url": "https://example.org/other"},
        ]
    )
    Corpus._instances[str(Path(data_dir).resolve())] = corpus
    tasks = [
        Task("bcp-1", "Which island first held the spice festival?", ANSWER, {"gold_docids": ["d1"], "evidence_docids": ["d1"]}),
        Task("bcp-2", "Which island first held the spice festival (variant)?", ANSWER, {"gold_docids": ["d1"], "evidence_docids": []}),
    ]
    return tmp_path, data_dir, tasks


def make_campaign(tmp_path, data_dir, tasks, matrix, campaign_id="e2e", repeats=1):
    exp = ExperimentConfig(
        campaign_id=campaign_id,
        benchmark={"id": "browsecomp_plus", "data_dir": str(data_dir)},
        grader=GraderConfig(provider="mock", model="mock-model"),
        defaults={
            "model": {"provider": "mock", "name": "mock-model", "effort": None, "thinking_display": None},
            "search_backend": "mock",
            "summarizer_model": "mock-model",
            "limits": {"step_cap": 12, "token_budget_per_task": 10_000_000, "time_cap_s": 120, "cost_cap_usd": 10},
        },
        matrix=matrix,
        repeats=repeats,
        concurrency=4,
        max_cost_usd=100,
    )

    def factory(provider, model):
        return MockClient(model, oracle_aware_script)

    grader = Grader(exp.grader, load_catalog(), MockClient("mock-model", grader_script))
    return Campaign(exp, root=tmp_path / "runs", client_factory=factory, grader=grader, tasks=tasks)


async def test_full_matrix_logging_and_funnel(setup):
    tmp_path, data_dir, tasks = setup
    camp = make_campaign(
        tmp_path, data_dir, tasks,
        {
            "condition": ["closed_book", "search_fetch", "search_snippets", "local_docs", "oracle_docs"],
            "delivery": ["full", "truncate_2k", "paginate_8k", "summarize"],
        },
    )
    await camp.run()
    finalize(camp.dir)
    res = validate(camp.dir)
    assert res["ok"], [f for f in res["findings"] if f["severity"] == "error"]

    runs = pd.read_csv(camp.dir / "runs.csv")
    tasks_df = pd.read_csv(camp.dir / "tasks.csv")
    calls = pd.read_csv(camp.dir / "model_calls.csv")
    assert len(runs) == 20 and len(tasks_df) == 40
    assert set(runs.run_status) == {"finished"}
    assert runs.runner_version.notna().all() and runs.price_catalog_version.notna().all()
    assert set(calls.harness_phase) <= {"execution", "recovery", "finalization"}

    t = tasks_df.merge(runs[["run_id", "condition_id", "delivery_policy"]], on="run_id")
    cell = lambda c, d: t[(t.condition_id == c) & (t.delivery_policy == d)]  # noqa: E731
    assert cell("search_fetch", "full").passed.all()
    assert (cell("search_fetch", "truncate_2k").funnel_category == "truncation_loss").all()
    assert cell("search_fetch", "paginate_8k").passed.all()          # recovered via read_more
    assert (cell("search_fetch", "paginate_8k").n_read_more > 0).all()
    assert (cell("closed_book", "full").funnel_category == "knowledge_miss").all()
    assert cell("oracle_docs", "full").passed.all()
    assert cell("local_docs", "full").gold_doc_read.all()

    # trajectory archive: one step per call + closing step, fetchable by id
    r0 = runs.run_id.iloc[0]
    traj = get_trajectory(camp.dir, r0, "bcp-1")
    n_calls = len(calls[(calls.run_id == r0) & (calls.task_id == "bcp-1")])
    assert len(traj["steps"]) == n_calls + 1 and traj["steps"][-1]["closing"]
    assert build_report(camp.dir).exists()


async def test_resume_partial_and_fault_injection(setup):
    tmp_path, data_dir, tasks = setup
    camp = make_campaign(tmp_path, data_dir, tasks, {"condition": ["search_fetch"]}, campaign_id="resume", repeats=2)
    await camp.run()
    # simulate a crash: delete one task record -> partial run with a not_executed row
    victim = next(camp.work.glob("*/tasks/bcp-2.json"))
    victim.unlink()
    finalize(camp.dir)
    runs = pd.read_csv(camp.dir / "runs.csv")
    assert (runs.run_status == "partial").sum() == 1
    tasks_df = pd.read_csv(camp.dir / "tasks.csv")
    assert (tasks_df.task_status == "not_executed").sum() == 1
    res = validate(camp.dir)
    assert any(f["check"] == "partial_run" for f in res["findings"])

    # resume: only the missing trial reruns
    camp2 = make_campaign(tmp_path, data_dir, tasks, {"condition": ["search_fetch"]}, campaign_id="resume", repeats=2)
    await camp2.run()
    finalize(camp2.dir)
    assert set(pd.read_csv(camp2.dir / "runs.csv").run_status) == {"finished"}
    assert validate(camp2.dir)["ok"]
    # repeats share config_hash, have distinct run_ids
    runs = pd.read_csv(camp2.dir / "runs.csv")
    assert runs.config_hash.nunique() == 1 and runs.run_id.nunique() == 2

    # fault injection: each example-data fault class must be caught
    d = camp2.dir
    calls = pd.read_csv(d / "model_calls.csv")
    calls.loc[0, "total_cost_usd"] = calls.loc[0, "total_cost_usd"] / 500  # mispriced call
    calls.to_csv(d / "model_calls.csv", index=False)
    runs.loc[0, "price_catalog_version"] = None  # missing catalog
    runs.to_csv(d / "runs.csv", index=False)
    res = validate(d)
    checks = {f["check"] for f in res["findings"]}
    assert not res["ok"]
    assert {"missing_metadata", "cost_rollup"} <= checks

    # unverifiable run: passes recorded but call log removed
    finalize(d)
    calls = pd.read_csv(d / "model_calls.csv")
    run_with_pass = pd.read_csv(d / "runs.csv").query("tasks_passed > 0").run_id.iloc[0]
    calls[calls.run_id != run_with_pass].to_csv(d / "model_calls.csv", index=False)
    checks = {f["check"] for f in validate(d)["findings"]}
    assert {"unverifiable_run", "call_count_mismatch"} <= checks


async def test_provider_failures_are_logged_as_infrastructure(setup):
    tmp_path, data_dir, tasks = setup
    camp = make_campaign(tmp_path, data_dir, tasks, {"condition": ["search_fetch"]}, campaign_id="infra")
    camp.client_factory = lambda p, m: MockClient(m, lambda msgs, tools: {"raise": "BadRequestError"})
    await camp.run()
    finalize(camp.dir)
    res = validate(camp.dir)
    assert any(f["check"] == "infrastructure_failure" for f in res["findings"])
    assert res["run_validity"][next(iter(res["run_validity"]))]["valid_perf"] is False
    tasks_df = pd.read_csv(camp.dir / "tasks.csv")
    assert (tasks_df.termination_reason == "exception").all()
    calls = pd.read_csv(camp.dir / "model_calls.csv")
    assert calls.exception_type.eq("BadRequestError").all()
    # trajectory still has one step per (failed) call + closing
    traj = json.loads(next((camp.work).glob("*/tasks/bcp-1.json")).read_text())["trajectory"]
    assert len(traj["steps"]) == 2
