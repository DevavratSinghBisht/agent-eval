"""Campaign runner: expand the experiment matrix into runs, execute (run, task) trials with
bounded concurrency, write one atomic work record per trial, and support resume.

Benchmark-neutral: the benchmark plugin supplies tasks, prompts, the answer tool, toolsets,
corpus and grading; the runner owns the agent loop, budgets, logging and bookkeeping.

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
from ..benchmarks import Benchmark, Task, get_benchmark
from ..core.config import ExperimentConfig, RunConfig, runner_version
from ..core.pricing import PriceCatalog, load_catalog
from ..core.task import GradeResult
from ..core.types import Message
from ..graders import Judge
from ..providers import CallOptions, ModelClient, make_client
from ..tools import ToolBuildContext, build_tools, resolve_toolsets
from ..tools.base import ToolContext, ToolRuntime
from ..tools.delivery import make_policy
from ..tracing.recorder import TaskRecorder

console = Console()


@dataclass
class RunSpec:
    run_id: str
    repeat_index: int
    cfg: RunConfig


def plan_runs(exp: ExperimentConfig, toolsets: dict[str, list[str]]) -> list[RunSpec]:
    """Resolve each run's toolset (so its tool list is part of config_hash) and assign run ids."""
    specs = []
    for rc in exp.expand():
        if rc.condition not in toolsets:
            raise ValueError(f"unknown condition/toolset {rc.condition!r}; available: {sorted(toolsets)}")
        rc = rc.model_copy(update={"toolset_tools": list(toolsets[rc.condition])})
        h = rc.config_hash()
        for k in range(exp.repeats):
            specs.append(RunSpec(f"run_{h}_rep{k}", k, rc))
    return specs


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _scalar(v: Any) -> Any:
    return v if v is None or isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False, default=str)


class Campaign:
    def __init__(
        self,
        exp: ExperimentConfig,
        root: str | Path | None = None,
        client_factory: Callable[[str, str], ModelClient] = make_client,
        benchmark: Benchmark | None = None,
        judge: Judge | None = None,
    ):
        self.exp = exp
        self.dir = Path(root or exp.output_dir) / exp.campaign_id
        self.work = self.dir / "_work"
        self.catalog: PriceCatalog = load_catalog(exp.price_catalog)
        self.runner_version = runner_version()
        self.client_factory = client_factory
        self.benchmark = benchmark or get_benchmark(exp.benchmark)
        g = exp.grader
        self.judge = judge or Judge(client_factory(g.provider, g.model), g.model, self.catalog, g.effort, g.max_tokens)
        self.tasks = self.benchmark.tasks()
        self.toolsets = resolve_toolsets(self.benchmark.toolsets(), exp.toolsets)
        self.specs = plan_runs(exp, self.toolsets)
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
            "benchmark_id": self.benchmark.id,
            "model_name": spec.cfg.model.name,
            "provider": spec.cfg.model.provider,
            "harness_name": spec.cfg.harness,
            "condition_id": spec.cfg.condition,
            "toolset_tools": ",".join(spec.cfg.toolset_tools or []),
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
            "grader_model": self.judge.model,
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
            f"[bold]{self.exp.campaign_id}[/] ({self.benchmark.id}): {len(self.specs)} runs x {len(self.tasks)} tasks; "
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
        cfg, bench = spec.cfg, self.benchmark
        rec = TaskRecorder(spec.run_id, task.task_id, cfg.model.name, self.catalog)
        status, error = "completed", None
        outcome = None
        try:
            client = self.client_factory(cfg.model.provider, cfg.model.name)
            delivery = make_policy(cfg.delivery)
            ctx = ToolContext(prompt=task.prompt, task_id=task.task_id)
            if cfg.delivery == "summarize":
                ctx.summarize = self._make_summarizer(cfg)
            build_ctx = ToolBuildContext(
                catalog=self.catalog, search_backend=cfg.search_backend, search_k=cfg.search_max_results,
                corpus_dir=bench.corpus_dir(), oracle_doc_ids=bench.oracle_doc_ids(task),
                blocklist=list(bench.leak_patterns),
            )
            tools = build_tools(cfg.toolset_tools or [], build_ctx, delivery, bench.answer_tool())
            runtime = ToolRuntime(tools, delivery, ctx)
            options = CallOptions(
                max_tokens=cfg.model.max_tokens, effort=cfg.model.effort,
                thinking_display=cfg.model.thinking_display, extra=cfg.model.extra,
            )
            agent = ReactAgent(client, options, cfg.limits)
            outcome = await agent.run(
                bench.system_prompt(cfg.condition, cfg.delivery), bench.task_prompt(task, cfg.condition), runtime, rec,
            )
        except Exception as e:  # harness bug or setup failure: the trial is logged as errored
            status, error = "errored", f"{type(e).__name__}: {e}"
            rec.errors.append(traceback.format_exc()[-2000:])
            rec.exception_count += 1

        answer = outcome.final_answer if outcome else None
        response = bench.format_response(answer, outcome.fallback_text if outcome else "")
        grade = GradeResult(None)
        if status == "completed":
            try:
                grade = await bench.grade(task, response, answer, self.judge)
            except Exception as e:  # a grader bug must not lose the trial
                grade = GradeResult(None, error=f"{type(e).__name__}: {e}")
            if grade.passed is None:
                status = "grading_error"
        task_row = {
            "run_id": spec.run_id,
            "task_id": task.task_id,
            **{f"meta_{k}": _scalar(v) for k, v in task.metadata.items()},
            "task_status": status,
            "passed": grade.passed,
            "score": grade.score,
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
            "answer": _scalar((answer or {}).get("answer")),
            "confidence": bench.confidence(answer),
            "response": response,
            "reference": _scalar(task.reference),
            "grader_model": grade.grader_model,
            "grader_cost_usd": round(grade.cost_usd, 6),
            "grader_error": grade.error,
            **{f"grade_{k}": _scalar(v) for k, v in grade.details.items()},
            "finished_at": time.time(),
        }
        closing = {
            "final_answer": answer,
            "response": response,
            "termination_reason": task_row["termination_reason"],
            "passed": grade.passed,
            "score": grade.score,
        }
        work = rec.close(closing)
        raw_api = work.pop("raw_api")
        work.update({
            "task": task_row,
            "task_metadata": task.metadata,
            "prompt": task.prompt,
            "evidence": {"answer_strings": task.evidence.answer_strings, "doc_ids": task.evidence.doc_ids},
            "leak_patterns": list(bench.leak_patterns),
            "toolset_tools": cfg.toolset_tools or [],
        })
        work["trajectory"].update({"prompt": task.prompt, "reference": _scalar(task.reference),
                                   "evidence_strings": task.evidence.answer_strings})
        cctx = zstandard.ZstdCompressor(level=6)
        atomic_write(
            self.work / spec.run_id / "raw_api" / f"{task.task_id}.jsonl.zst",
            cctx.compress("\n".join(json.dumps(r, default=str) for r in raw_api).encode()),
        )
        atomic_write(self.task_path(spec.run_id, task.task_id), json.dumps(work, default=str).encode())
        async with self._lock:
            self.spent += rec.cost + grade.cost_usd
        mark = {True: "[green]PASS[/]", False: "[red]fail[/]", None: "[yellow]n/a[/]"}[grade.passed]
        console.print(
            f"{mark} {spec.run_id} {task.task_id} {cfg.condition}/{cfg.delivery} "
            f"steps={task_row['call_count']} term={task_row['termination_reason']} "
            f"${task_row['cost_usd']:.4f} (campaign ${self.spent:.2f})"
        )
        return work

    def _make_summarizer(self, cfg: RunConfig):
        client = self.client_factory(cfg.model.provider, cfg.summarizer_model)

        async def summarize(text: str, prompt: str):
            turn, _ = await client.complete(
                "",
                [Message("user", text=prompts.SUMMARIZER_PROMPT.format(question=prompt, text=text))],
                [],
                CallOptions(max_tokens=4000, effort="low"),
            )
            return turn.text, turn.usage, self.catalog.cost(cfg.summarizer_model, turn.usage)

        return summarize
