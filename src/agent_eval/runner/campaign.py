"""Campaign runner: expand the experiment matrix into runs, execute (run, task) trials with
bounded concurrency, write one atomic work record per trial, and support resume.

Work layout (consolidated into CSVs/archives by `finalize`):
  runs/<campaign>/_work/<run_id>/run.json
  runs/<campaign>/_work/<run_id>/tasks/<task_id>.json
  runs/<campaign>/_work/<run_id>/raw_api/<task_id>.jsonl.zst
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import zstandard
from rich.console import Console

from ..agent import prompts
from ..agent.react import ReactAgent
from ..benchmarks import Task, load_tasks
from ..benchmarks.browsecomp import parse_response_confidence
from ..core.config import ExperimentConfig, RunConfig, runner_version
from ..core.pricing import PriceCatalog, load_catalog
from ..core.types import Message
from ..providers import CallOptions, ModelClient, make_client
from ..tools import build_tools
from ..tools.base import ToolContext, ToolRuntime
from ..tools.delivery import make_policy
from ..tools.final_answer import format_response
from ..tracing.recorder import TaskRecorder
from .grading import Grader

console = Console()


@dataclass
class RunSpec:
    run_id: str
    repeat_index: int
    cfg: RunConfig


def plan_runs(exp: ExperimentConfig) -> list[RunSpec]:
    specs = []
    for rc in exp.expand():
        h = rc.config_hash()
        for k in range(exp.repeats):
            specs.append(RunSpec(f"run_{h}_rep{k}", k, rc))
    return specs


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Campaign:
    def __init__(
        self,
        exp: ExperimentConfig,
        root: str | Path | None = None,
        client_factory: Callable[[str, str], ModelClient] = make_client,
        grader: Grader | None = None,
        tasks: list[Task] | None = None,
    ):
        self.exp = exp
        self.dir = Path(root or exp.output_dir) / exp.campaign_id
        self.work = self.dir / "_work"
        self.catalog: PriceCatalog = load_catalog(exp.price_catalog)
        self.runner_version = runner_version()
        self.client_factory = client_factory
        self.grader = grader or Grader(exp.grader, self.catalog, client_factory(exp.grader.provider, exp.grader.model))
        self.tasks = tasks if tasks is not None else load_tasks(exp.benchmark)
        self.specs = plan_runs(exp)
        self.spent = 0.0
        self._lock = asyncio.Lock()
        self.stop_reason: str | None = None

    # ---- bookkeeping -------------------------------------------------------------------
    def task_path(self, run_id: str, task_id: str) -> Path:
        return self.work / run_id / "tasks" / f"{task_id}.json"

    def _write_run_manifest(self, spec: RunSpec) -> None:
        p = self.work / spec.run_id / "run.json"
        prior = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        rv = prior.get("runner_versions", [])
        if self.runner_version not in rv:
            rv.append(self.runner_version)
        meta = {
            "run_id": spec.run_id,
            "campaign_id": self.exp.campaign_id,
            "benchmark_id": spec.cfg.benchmark.id,
            "model_name": spec.cfg.model.name,
            "provider": spec.cfg.model.provider,
            "harness_name": spec.cfg.harness,
            "condition_id": spec.cfg.condition,
            "delivery_policy": spec.cfg.delivery,
            "effort": spec.cfg.model.effort,
            "search_backend": spec.cfg.search_backend,
            "token_budget_per_task": spec.cfg.limits.token_budget_per_task,
            "step_cap": spec.cfg.limits.step_cap,
            "repeat_index": spec.repeat_index,
            "repeat_group_id": f"grp_{spec.cfg.config_hash()}",
            "config_hash": spec.cfg.config_hash(),
            "runner_version": self.runner_version,
            "runner_versions": rv,
            "price_catalog_version": self.catalog.version,
            "seed": spec.cfg.benchmark.seed,
            "grader_model": self.exp.grader.model,
            "config": spec.cfg.model_dump(mode="json"),
            "task_ids": [t.task_id for t in self.tasks],
            "started_at": prior.get("started_at", time.time()),
        }
        atomic_write(p, json.dumps(meta, indent=1).encode())

    # ---- execution ---------------------------------------------------------------------
    async def run(self) -> None:
        if not self.runner_version or not self.catalog.version:
            raise RuntimeError("runner_version and price_catalog_version are mandatory")
        for spec in self.specs:
            self.catalog.model(spec.cfg.model.name)  # fail fast on unpriced models
            if spec.cfg.delivery == "summarize":
                self.catalog.model(spec.cfg.summarizer_model)
            self._write_run_manifest(spec)
        todo = [
            (spec, task)
            for spec in self.specs
            for task in self.tasks
            if not self.task_path(spec.run_id, task.task_id).exists()
        ]
        done = len(self.specs) * len(self.tasks) - len(todo)
        console.print(
            f"[bold]{self.exp.campaign_id}[/]: {len(self.specs)} runs x {len(self.tasks)} tasks; "
            f"{done} already done, {len(todo)} to run (concurrency {self.exp.concurrency}, "
            f"campaign cap ${self.exp.max_cost_usd:.2f})"
        )
        sem = asyncio.Semaphore(self.exp.concurrency)

        async def worker(spec: RunSpec, task: Task):
            async with sem:
                if self.stop_reason:
                    return
                if self.spent >= self.exp.max_cost_usd:
                    self.stop_reason = f"campaign max_cost_usd ${self.exp.max_cost_usd} reached"
                    console.print(f"[red]stopping: {self.stop_reason}[/]")
                    return
                await self.run_trial(spec, task)

        await asyncio.gather(*(worker(s, t) for s, t in todo))

    async def run_trial(self, spec: RunSpec, task: Task) -> dict[str, Any]:
        cfg = spec.cfg
        rec = TaskRecorder(spec.run_id, task.task_id, cfg.model.name, self.catalog)
        status, error = "completed", None
        outcome = None
        try:
            client = self.client_factory(cfg.model.provider, cfg.model.name)
            delivery = make_policy(cfg.delivery)
            oracle = None
            if cfg.condition == "oracle_docs":
                oracle = list(dict.fromkeys(task.metadata.get("evidence_docids", []) + task.metadata.get("gold_docids", [])))
            ctx = ToolContext(question=task.question, task_id=task.task_id)
            if cfg.delivery == "summarize":
                ctx.summarize = self._make_summarizer(cfg)
            tools = build_tools(
                cfg.condition, delivery, self.catalog, cfg.search_backend, cfg.search_max_results,
                cfg.benchmark.data_dir, set(oracle) if oracle else None,
            )
            runtime = ToolRuntime(tools, delivery, ctx)
            options = CallOptions(
                max_tokens=cfg.model.max_tokens, effort=cfg.model.effort,
                thinking_display=cfg.model.thinking_display, extra=cfg.model.extra,
            )
            agent = ReactAgent(client, options, cfg.limits)
            outcome = await agent.run(
                prompts.system_prompt(cfg.condition, cfg.delivery),
                prompts.task_prompt(task.question, oracle),
                runtime,
                rec,
            )
        except Exception as e:  # harness bug or setup failure: the trial is logged as errored
            status, error = "errored", f"{type(e).__name__}: {e}"
            rec.errors.append(traceback.format_exc()[-2000:])
            rec.exception_count += 1

        response = format_response(outcome.final_answer if outcome else None, outcome.fallback_text if outcome else "")
        grade: dict[str, Any] = {}
        if status == "completed":
            grade = await self.grader.grade(task.question, task.answer, response)
            if grade["verdict"] is None:
                status = "grading_error"
        fa = outcome.final_answer if outcome else None
        conf = (fa or {}).get("confidence")
        if conf is None:
            conf = parse_response_confidence(response)
        task_row = {
            "run_id": spec.run_id,
            "task_id": task.task_id,
            "topic": task.metadata.get("topic"),
            "task_status": status,
            "passed": (grade.get("verdict") == "yes") if grade.get("verdict") else None,
            "exception_count": rec.exception_count,
            "error": error or ("; ".join(rec.errors)[:1000] if rec.errors else None),
            "call_count": len(rec.model_calls),
            "tool_call_count": len(rec.tool_calls),
            "prompt_tokens": rec.usage.input_tokens,
            "cached_tokens": rec.usage.cache_read_tokens,
            "cache_write_tokens": rec.usage.cache_write_tokens,
            "completion_tokens": rec.usage.output_tokens,
            "model_cost_usd": round(rec.model_cost, 6),
            "tool_cost_usd": round(rec.tool_cost, 6),
            "cost_usd": rec.cost,
            "latency_seconds": round(rec.elapsed, 3),
            "final_phase": outcome.final_phase if outcome else None,
            "termination_reason": outcome.termination_reason if outcome else "exception",
            "exact_answer": (fa or {}).get("exact_answer"),
            "confidence": conf,
            "response": response,
            "correct_answer": task.answer,
            "grader_verdict": grade.get("verdict"),
            "grader_extracted_answer": grade.get("extracted_final_answer"),
            "grader_parse_ok": grade.get("parse_ok"),
            "grader_model": grade.get("grader_model"),
            "grader_cost_usd": grade.get("grader_cost_usd", 0.0),
            "grader_raw": grade.get("grader_raw"),
            "grader_error": grade.get("grader_error"),
            "finished_at": time.time(),
        }
        closing = {
            "final_answer": fa,
            "response": response,
            "termination_reason": task_row["termination_reason"],
            "passed": task_row["passed"],
            "grader_verdict": task_row["grader_verdict"],
        }
        work = rec.close(closing)
        raw_api = work.pop("raw_api")
        work["task"] = task_row
        work["task_metadata"] = task.metadata
        work["question"] = task.question
        work["trajectory"]["question"] = task.question
        work["trajectory"]["correct_answer"] = task.answer
        cctx = zstandard.ZstdCompressor(level=6)
        atomic_write(
            self.work / spec.run_id / "raw_api" / f"{task.task_id}.jsonl.zst",
            cctx.compress("\n".join(json.dumps(r, default=str) for r in raw_api).encode()),
        )
        atomic_write(self.task_path(spec.run_id, task.task_id), json.dumps(work, default=str).encode())
        async with self._lock:
            self.spent += rec.cost + task_row["grader_cost_usd"]
        mark = {True: "[green]PASS[/]", False: "[red]fail[/]", None: "[yellow]n/a[/]"}[task_row["passed"]]
        console.print(
            f"{mark} {spec.run_id} {task.task_id} {cfg.condition}/{cfg.delivery} "
            f"steps={task_row['call_count']} term={task_row['termination_reason']} "
            f"${task_row['cost_usd']:.4f} (campaign ${self.spent:.2f})"
        )
        return work

    def _make_summarizer(self, cfg: RunConfig):
        client = self.client_factory(cfg.model.provider, cfg.summarizer_model)

        async def summarize(text: str, question: str):
            turn, _ = await client.complete(
                "",
                [Message("user", text=prompts.SUMMARIZER_PROMPT.format(question=question, text=text))],
                [],
                CallOptions(max_tokens=4000, effort="low"),
            )
            return turn.text, turn.usage, self.catalog.cost(cfg.summarizer_model, turn.usage)

        return summarize
