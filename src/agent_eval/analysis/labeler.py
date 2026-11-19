"""LLM failure labeller: assigns a MAST/TRAIL-derived category and the first-error step to each
failed trial. The labeller sees a compact digest of the trajectory plus the gold answer
(it is an analysis tool, never part of the agent). Calibrate it on ~50 hand-labelled traces
before trusting its distribution (see README)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pandas as pd

from ..core.pricing import load_catalog
from ..core.types import Message
from ..providers import CallOptions, make_client
from ..tracing.store import get_trajectory

TAXONOMY = {
    "spec_misread": "Misunderstood the question or dropped one of its constraints.",
    "poor_planning": "No coherent strategy; searched for the wrong entity/aspect first and never recovered.",
    "retrieval_failure": "Queries never surfaced the needed source although it plausibly exists.",
    "tool_misuse": "Wrong tool, malformed inputs, or ignored tool errors.",
    "evidence_ignored": "The needed evidence was shown to the agent but it did not use it.",
    "evidence_truncated": "The needed evidence was cut off by truncation/summarization and the agent did not read further.",
    "hallucinated_fact": "Asserted a fact not supported by any tool output.",
    "verification_skipped": "Answered without verifying all constraints against sources.",
    "premature_termination": "Stopped (answered) too early with budget left.",
    "looping": "Repeated the same or near-identical actions without progress.",
    "budget_exhausted": "Ran out of steps/tokens while still making reasonable progress.",
    "answer_formatting": "Found the right entity but reported it in a wrong/incomplete form.",
    "other": "None of the above.",
}

PROMPT = """You are auditing a failed run of a web-research agent. Classify the PRIMARY cause of failure.

Question: {question}
Correct answer: {answer}
Agent's final answer: {final}
Termination: {termination}

Trajectory digest (step: tool(input) -> what the agent was shown, truncated):
{digest}

Categories:
{taxonomy}

Reply with JSON only: {{"category": "<one category key>", "first_error_step": <int or null>, "explanation": "<one or two sentences>"}}"""


def digest(traj: dict, max_chars: int = 400) -> str:
    lines = []
    for s in traj["steps"]:
        if s.get("closing"):
            continue
        if s.get("reasoning"):
            lines.append(f"{s['step']}: [thinking] {s['reasoning'][:max_chars]}")
        if s.get("text"):
            lines.append(f"{s['step']}: [text] {s['text'][:max_chars]}")
        for tc in s.get("tool_calls", []):
            meta = tc.get("delivery_meta") or {}
            flag = " [TRUNCATED]" if meta.get("truncated") else ""
            out = (tc.get("tool_output_delivered") or "").replace("\n", " ")[:max_chars]
            lines.append(f"{s['step']}: {tc['tool']}({json.dumps(tc['tool_input'], ensure_ascii=False)[:200]}){flag} -> {out}")
    return "\n".join(lines)[-30000:]


async def label_campaign(campaign_dir: str | Path, model: str = "claude-haiku-5-5", provider: str = "anthropic",
                         limit: int | None = None, concurrency: int = 4) -> Path:
    d = Path(campaign_dir)
    tasks = pd.read_csv(d / "tasks.csv")
    runs = pd.read_csv(d / "runs.csv")
    tasks = tasks.merge(runs[["run_id", "condition_id"]], on="run_id")
    # closed-book failures have no trajectory to diagnose (category is knowledge_miss by construction)
    failed = tasks[(tasks.task_status == "completed") & (tasks.passed == False) & (tasks.condition_id != "closed_book")]  # noqa: E712
    if limit:
        failed = failed.head(limit)
    client = make_client(provider, model)
    catalog = load_catalog()
    sem = asyncio.Semaphore(concurrency)
    tax = "\n".join(f"- {k}: {v}" for k, v in TAXONOMY.items())

    async def one(row):
        async with sem:
            traj = get_trajectory(d, row.run_id, row.task_id)
            prompt = PROMPT.format(
                question=(traj or {}).get("question", ""),
                answer=row.correct_answer, final=row.exact_answer, termination=row.termination_reason,
                digest=digest(traj) if traj else "(missing)", taxonomy=tax,
            )
            turn, _ = await client.complete("", [Message("user", text=prompt)], [], CallOptions(max_tokens=2000, effort="low"))
            m = re.search(r"\{.*\}", turn.text, re.S)
            try:
                parsed = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                parsed = {}
            cat = parsed.get("category") if parsed.get("category") in TAXONOMY else "other"
            return {"run_id": row.run_id, "task_id": row.task_id, "condition_id": row.condition_id, "category": cat,
                    "first_error_step": parsed.get("first_error_step"), "explanation": parsed.get("explanation"),
                    "labeler_model": model, "labeler_cost_usd": catalog.cost(model, turn.usage)}

    rows = await asyncio.gather(*(one(r) for r in failed.itertuples()))
    out = d / "failure_labels.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
