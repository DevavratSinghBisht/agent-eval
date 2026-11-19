"""Judge agreement: statistics against hand-computed values, degenerate cases, and the multi-judge /
rejudge pipeline end to end (offline, scripted judges)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from agent_eval.analysis.agreement import (
    cohen_kappa, compute_agreement, gwet_ac1, krippendorff_alpha_nominal, mcnemar_p, pair_stats, write_agreement,
)
from agent_eval.benchmarks.browsecomp import Task
from agent_eval.core.config import ExperimentConfig, GraderConfig
from agent_eval.core.pricing import load_catalog
from agent_eval.providers.mock_client import MockClient
from agent_eval.runner.campaign import Campaign
from agent_eval.runner.grading import Grader
from agent_eval.tracing.store import finalize


def textbook():
    # 50 trials: both pass 20, only A 5, only B 10, both fail 15
    a = np.array([1] * 20 + [1] * 5 + [0] * 10 + [0] * 15, bool)
    b = np.array([1] * 20 + [0] * 5 + [1] * 10 + [0] * 15, bool)
    return a, b


def test_kappa_ac1_mcnemar_textbook():
    a, b = textbook()
    assert cohen_kappa(a, b) == pytest.approx(0.4)          # po=.70, pe=.50
    assert gwet_ac1(a, b) == pytest.approx(0.205 / 0.505)    # pi=.55, pe=.495
    assert mcnemar_p(a, b) == pytest.approx(0.3018, abs=1e-4)
    s = pair_stats(a, b)
    assert s["percent_agreement"] == pytest.approx(0.7) and (s["a_only"], s["b_only"]) == (5, 10)
    assert s["kappa_ci"][0] < 0.4 < s["kappa_ci"][1]


def test_degenerate_cases_kappa_paradox():
    same = np.zeros(20, bool)
    assert cohen_kappa(same, same) != cohen_kappa(same, same)  # NaN: both judges constant
    assert gwet_ac1(np.ones(5, bool), np.ones(5, bool)) == 1.0  # AC1 stays defined where kappa is not
    assert math.isnan(cohen_kappa(np.ones(5, bool), np.ones(5, bool)))
    a, b = np.zeros(20, bool), np.zeros(20, bool)
    b[0] = True  # 95% raw agreement on an all-fail set
    s = pair_stats(a, b)
    assert s["cohen_kappa"] == pytest.approx(0.0) and s["gwet_ac1"] > 0.9  # kappa collapses, AC1 does not
    assert cohen_kappa(np.array([], bool), np.array([], bool)) != 0  # NaN for empty


def test_krippendorff_alpha_with_missing():
    assert krippendorff_alpha_nominal(np.array([[1, 1], [0, 0], [1, 0]], float)) == pytest.approx(4 / 9)
    with_missing = np.array([[1, 1, np.nan], [0, 0, 0], [1, 0, np.nan], [np.nan, 1, np.nan]], float)
    assert not math.isnan(krippendorff_alpha_nominal(with_missing))  # unit with a single rating is ignored


def test_compute_agreement_long_format_and_contested():
    rows = []
    for i, (va, vb) in enumerate([(True, True), (True, False), (False, False), (False, False)]):
        rows += [{"run_id": "r", "task_id": f"t{i}", "judge": "A", "role": "primary", "passed": va},
                 {"run_id": "r", "task_id": f"t{i}", "judge": "B", "role": "rejudge", "passed": vb}]
    rows.append({"run_id": "r", "task_id": "t9", "judge": "B", "role": "rejudge", "passed": None})  # grading error
    res = compute_agreement(pd.DataFrame(rows), primary="A")
    p = res["pairs"][0]
    assert p["n"] == 4 and p["percent_agreement"] == 0.75
    assert [c["task_id"] for c in res["contested"]] == ["t1"]
    assert res["score_range_on_common"]["spread"] == pytest.approx(0.25)


def _verdict_script(rule):
    def script(messages, tools):
        exact = messages[0].text.split("Exact Answer: ", 1)[1].split("\n", 1)[0]
        return {"text": f"extracted_final_answer: {exact}\nreasoning: r\ncorrect: {'yes' if rule(exact) else 'no'}\nconfidence: 90",
                "stop": "end_turn"}
    return script


async def test_secondary_judges_and_rejudge_pipeline(tmp_path):
    answers = {"t1": "Paris", "t2": "Lyon", "t3": "Nice", "t4": "Paris"}
    agent = lambda m, t: {"tool_calls": [("final_answer", {"explanation": "e", "exact_answer": answers[  # noqa: E731
        next(k for k in answers if k in m[0].text)], "confidence": 70})]}
    strict = _verdict_script(lambda a: a == "Paris")               # primary: only Paris is right
    lenient = _verdict_script(lambda a: a in ("Paris", "Lyon"))    # secondary: also accepts Lyon
    exp = ExperimentConfig(
        campaign_id="judges", benchmark={"id": "browsecomp"},
        grader=GraderConfig(provider="mock", model="mock-model", name="strict"),
        secondary_graders=[GraderConfig(provider="mock", model="mock-model", name="lenient")],
        defaults={"model": {"provider": "mock", "name": "mock-model", "effort": None, "thinking_display": None},
                  "condition": "closed_book"},
    )
    clients = {"strict": strict, "lenient": lenient}
    camp = Campaign(exp, root=tmp_path, client_factory=lambda p, m, **kw: MockClient(m, lambda ms, ts: agent(ms, ts)),
                    tasks=[Task(t, f"question {t}", "Paris") for t in answers])
    camp.grader = Grader(exp.grader, load_catalog(), MockClient("mock-model", strict))
    camp.secondary_graders = [Grader(exp.secondary_graders[0], load_catalog(), MockClient("mock-model", clients["lenient"]))]
    await camp.run()
    finalize(camp.dir)
    j = pd.read_csv(camp.dir / "judgements.csv")
    assert sorted(j.judge.unique()) == ["lenient", "strict"] and len(j) == 8
    res = write_agreement(camp.dir)
    p = res["pairs"][0]
    assert p["n"] == 4 and p["percent_agreement"] == 0.75 and p["a_only"] + p["b_only"] == 1  # only Lyon differs
    assert pd.read_csv(camp.dir / "tasks.csv").passed.sum() == 2  # primary decides `passed`

    # rejudge with a third judge; second call resumes without duplicating rows
    from agent_eval.runner.rejudge import rejudge
    third = GraderConfig(provider="mock", model="mock-model", name="everything")
    always = MockClient("mock-model", _verdict_script(lambda a: True))
    s1 = await rejudge(camp.dir, third, client=always)
    s2 = await rejudge(camp.dir, third, client=always)
    assert (s1["graded"], s2["graded"]) == (4, 0)
    finalize(camp.dir)
    res = write_agreement(camp.dir)
    assert len(res["pairs"]) == 3 and "krippendorff_alpha" in res
    assert json.loads((camp.dir / "agreement.json").read_text())["judges"] == ["everything", "lenient", "strict"]


def test_duplicate_judge_ids_rejected(tmp_path):
    exp = ExperimentConfig(campaign_id="dup", benchmark={"id": "browsecomp"},
                           grader=GraderConfig(provider="mock", model="mock-model"),
                           secondary_graders=[GraderConfig(provider="mock", model="mock-model")],
                           defaults={"model": {"provider": "mock", "name": "mock-model"}})
    with pytest.raises(ValueError, match="judge ids must be unique"):
        Campaign(exp, root=tmp_path, client_factory=lambda p, m, **kw: MockClient(m), tasks=[Task("t", "q", "a")])
