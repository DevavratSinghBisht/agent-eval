import asyncio
import math

import numpy as np
import pytest

from agent_eval.analysis.evidence import answer_variants, contains_answer
from agent_eval.analysis.metrics import calibration_error, pass_at_k, pass_hat_k, wilson
from agent_eval.benchmarks.browsecomp import GRADER_TEMPLATE, QUERY_TEMPLATE, decrypt, encrypt, parse_grade
from agent_eval.core.pricing import PricingError, load_catalog
from agent_eval.core.types import Usage
from agent_eval.tools.base import ToolContext
from agent_eval.tools.delivery import PaginatePolicy, TruncatePolicy, approx_tokens, make_policy


# ---- BrowseComp port ------------------------------------------------------------------
def test_decrypt_roundtrip():
    canary = "BENCHMARK DATA SHOULD NEVER APPEAR canary GUID test"
    for s in ["Zanzibar", "Ünïcödé answer 42", "x" * 1000]:
        assert decrypt(encrypt(s, canary), canary) == s


def test_templates_verbatim_markers():
    assert "Exact Answer: {{your succinct, final answer}}" not in QUERY_TEMPLATE.format(Question="q")
    assert "Exact Answer: {your succinct, final answer}" in QUERY_TEMPLATE.format(Question="q")
    assert "0|\\%| and 100|\\%|" in GRADER_TEMPLATE
    GRADER_TEMPLATE.format(question="q", response="r", correct_answer="a")  # no stray braces


def test_parse_grade_uses_captured_group():
    # simple-evals compared match.group(0) ('correct: yes') to 'yes' -> never correct. We must not.
    g = parse_grade("extracted_final_answer: Paris\nreasoning: same\ncorrect: yes\nconfidence: 90")
    assert g["verdict"] == "yes" and g["parse_ok"] and g["extracted_final_answer"] == "Paris"
    assert parse_grade("correct: **no**")["verdict"] == "no"
    bad = parse_grade("I think it matches")
    assert bad["verdict"] == "no" and not bad["parse_ok"]


# ---- pricing ----------------------------------------------------------------------------
def test_pricing_tiers_and_unknown_model():
    cat = load_catalog()
    small = Usage(input_tokens=50_000, output_tokens=1_000)
    assert math.isclose(cat.cost("claude-haiku-5-5", small), (50_000 * 0.10 + 1_000 * 0.50) / 1e6)
    big = Usage(input_tokens=60_000, cache_read_tokens=50_000, output_tokens=1_000)  # 110K prompt -> long tier
    exp = (60_000 * 0.50 + 50_000 * 0.50 * 0.1 + 1_000 * 2.50) / 1e6
    assert math.isclose(cat.cost("claude-haiku-5-5", big), round(exp, 6))
    with pytest.raises(PricingError):
        cat.cost("gpt-unknown", small)


# ---- delivery policies --------------------------------------------------------------------
def test_truncate_boundary():
    pol = TruncatePolicy(2000)
    ctx = ToolContext("q", "t")
    short = "a" * 8000
    d, m, _, _ = asyncio.run(pol.apply(short, ctx))
    assert d == short and not m["truncated"]
    long = "a" * 8001
    d, m, _, _ = asyncio.run(pol.apply(long, ctx))
    assert m["truncated"] and m["kept_chars"] == 8000 and d.startswith("a" * 8000) and "[Truncated" in d


@pytest.mark.parametrize("n", [1, 100, 32_000, 32_001, 100_003])
def test_paginate_reconstructs_exactly(n):
    rng = np.random.default_rng(n)
    words = rng.choice(["alpha", "beta", "gamma\n", "delta"], size=max(1, n // 5))
    raw = " ".join(words)[:n]
    pol = PaginatePolicy(8000)
    pages = pol.paginate(raw)
    assert "".join(pages) == raw
    assert all(len(p) <= 32_000 for p in pages)


def test_make_policy_names():
    for name in ["full", "truncate_2k", "truncate_8k", "truncate_32k", "paginate_8k", "summarize"]:
        assert make_policy(name).name == name
    assert approx_tokens("abcd") == 1 and approx_tokens("abcde") == 2


# ---- metrics ------------------------------------------------------------------------------
def test_pass_at_k_and_pass_hat_k():
    assert pass_at_k(3, 0, 1) == 0.0 and pass_at_k(3, 3, 3) == 1.0
    assert math.isclose(pass_at_k(3, 1, 1), 1 / 3)
    assert math.isclose(pass_at_k(3, 1, 3), 1.0)
    assert math.isclose(pass_hat_k(3, 2, 2), 1 / 3)
    assert math.isclose(pass_hat_k(4, 3, 3), 0.25)


def test_calibration_and_wilson():
    conf = np.array([100.0] * 100)
    assert math.isclose(calibration_error(conf, np.ones(100)), 0.0)
    assert math.isclose(calibration_error(conf, np.zeros(100)), 1.0)
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi


# ---- evidence matching ----------------------------------------------------------------------
def test_answer_matching():
    v = answer_variants("The Battle of Hastings (1066)")
    assert contains_answer("...known as the battle of hastings, fought...", v)
    assert contains_answer("in 1066 the", v)
    assert not contains_answer("hastingsville", answer_variants("Hastings"))
    assert contains_answer("population 1234567 people", answer_variants("1,234,567"))
