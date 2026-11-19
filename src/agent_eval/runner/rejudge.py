"""Re-grade a finished campaign's stored responses with an extra judge (no agent calls).

Verdicts are appended to runs/<campaign>/_work/_rejudge/<judge>.jsonl, one line per trial, so the
command is resumable and never touches the primary grades. `finalize` merges these rows into
judgements.csv and the agreement analysis compares all judges.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from rich.console import Console

from ..core.config import GraderConfig
from ..core.pricing import load_catalog
from ..providers import make_client
from ..tracing.store import iter_runs, load_work, rejudge_dir
from .grading import Grader

console = Console()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


async def rejudge(campaign_dir: str | Path, cfg: GraderConfig, concurrency: int = 8, limit: int | None = None,
                  client=None) -> dict:
    d = Path(campaign_dir)
    catalog = load_catalog()
    grader = Grader(cfg, catalog, client or make_client(cfg.provider, cfg.model, base_url=cfg.base_url,
                                                        api_key_env=cfg.api_key_env))
    items = []
    for meta, run_dir in iter_runs(d):
        if meta.get("grader_model") and f"{meta.get('grader_provider', 'anthropic')}/{meta['grader_model']}" == grader.judge_id:
            raise ValueError(f"{grader.judge_id} is already this campaign's primary judge; pick another model or set a distinct name")
        for task_id in meta["task_ids"]:
            w = load_work(run_dir, task_id, meta)
            if w and w["task"].get("task_status") == "completed":
                items.append(w)
    out = rejudge_dir(d) / f"{_safe(grader.judge_id)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["run_id"], r["task_id"]))
    todo = [w for w in items if (w["task"]["run_id"], w["task"]["task_id"]) not in done]
    if limit:
        todo = todo[:limit]
    console.print(f"rejudge {d.name} with {grader.judge_id}: {len(items)} graded trials, {len(done)} already done, {len(todo)} to grade")
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    stats = {"graded": 0, "errors": 0, "cost_usd": 0.0}

    async def one(w):
        async with sem:
            t = w["task"]
            g = await grader.grade(w.get("question") or w["trajectory"].get("question", ""), t["correct_answer"], t["response"])
            row = grader.judgement(g, t["run_id"], t["task_id"], "rejudge")
            async with lock:
                with open(out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                stats["graded"] += 1
                stats["errors"] += row["passed"] is None
                stats["cost_usd"] += row["cost_usd"] or 0.0

    await asyncio.gather(*(one(w) for w in todo))
    stats["cost_usd"] = round(stats["cost_usd"], 6)
    console.print(f"rejudged {stats['graded']} trials ({stats['errors']} grading errors), ${stats['cost_usd']:.4f}")
    return stats
