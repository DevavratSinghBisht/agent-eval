"""LLM grader for BrowseComp (official GRADER_TEMPLATE). The grader never sees the trajectory,
only question, final response and correct answer — exactly as in simple-evals."""

from __future__ import annotations

import time
from typing import Any

from ..benchmarks.browsecomp import GRADER_TEMPLATE, parse_grade
from ..core.config import GraderConfig
from ..core.pricing import PriceCatalog
from ..core.types import Message
from ..providers import CallOptions, ProviderError, make_client


class Grader:
    def __init__(self, cfg: GraderConfig, catalog: PriceCatalog, client=None):
        self.cfg, self.catalog = cfg, catalog
        self.client = client or make_client(cfg.provider, cfg.model, base_url=cfg.base_url, api_key_env=cfg.api_key_env)
        catalog.model(cfg.model)  # fail fast if the grader is unpriced

    @property
    def judge_id(self) -> str:
        return self.cfg.judge_id

    def judgement(self, grade: dict[str, Any], run_id: str, task_id: str, role: str) -> dict[str, Any]:
        """One row of judgements.csv (long format: one row per trial x judge)."""
        return {
            "run_id": run_id, "task_id": task_id, "judge": self.judge_id, "role": role,
            "provider": self.cfg.provider, "model": self.cfg.model,
            "passed": (grade["verdict"] == "yes") if grade.get("verdict") else None,
            "verdict": grade.get("verdict"), "parse_ok": grade.get("parse_ok"),
            "extracted_answer": grade.get("extracted_final_answer"),
            "cost_usd": grade.get("grader_cost_usd", 0.0), "latency_seconds": grade.get("grader_latency_seconds"),
            "error": grade.get("grader_error"), "raw": grade.get("grader_raw"),
        }

    async def grade(self, question: str, correct_answer: str, response: str) -> dict[str, Any]:
        prompt = GRADER_TEMPLATE.format(question=question, correct_answer=correct_answer, response=response)
        opts = CallOptions(max_tokens=self.cfg.max_tokens, effort=self.cfg.effort, extra=self.cfg.extra)
        t0 = time.perf_counter()
        last_err = None
        for _attempt in range(3):
            try:
                turn, _ = await self.client.complete("", [Message("user", text=prompt)], [], opts)
                parsed = parse_grade(turn.text)
                return {
                    **parsed,
                    "grader_model": self.cfg.model,
                    "grader_provider": self.cfg.provider,
                    "grader_raw": turn.text,
                    "grader_cost_usd": self.catalog.cost(self.cfg.model, turn.usage),
                    "grader_latency_seconds": round(time.perf_counter() - t0, 3),
                    "grader_error": None,
                }
            except ProviderError as e:
                last_err = e
                if not e.retryable:
                    break
        return {
            "verdict": None,
            "parse_ok": False,
            "extracted_final_answer": None,
            "grader_confidence": None,
            "grader_model": self.cfg.model,
            "grader_provider": self.cfg.provider,
            "grader_raw": None,
            "grader_cost_usd": 0.0,
            "grader_latency_seconds": round(time.perf_counter() - t0, 3),
            "grader_error": f"{getattr(last_err, 'exception_type', 'Error')}: {last_err}",
        }
