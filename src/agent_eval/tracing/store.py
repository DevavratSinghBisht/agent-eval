"""Consolidate a campaign's work records into the analysis layout modelled on the example
loss-analysis package:

  runs.csv, tasks.csv (full grid incl. not_executed), model_calls.csv, tool_calls.csv,
  trajectories/archive.jsonl.zst + trajectory_index.csv + selected/*.json,
  traces/otel_spans.jsonl.zst (OpenTelemetry GenAI-shaped spans), manifest.json
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import zstandard

from ..analysis.evidence import analyze_trial
from ..analysis.signals import trial_signals
from ..core.pricing import load_catalog

RUN_META_COLS = [
    "run_id", "campaign_id", "benchmark_id", "model_name", "provider", "harness_name", "condition_id",
    "delivery_policy", "effort", "search_backend", "token_budget_per_task", "step_cap", "repeat_index",
    "repeat_group_id", "config_hash", "runner_version", "price_catalog_version", "seed", "grader_model",
]
TOKEN_COLS = ["prompt_tokens", "cached_tokens", "cache_write_tokens", "completion_tokens"]


def iter_runs(campaign_dir: Path) -> Iterator[tuple[dict[str, Any], Path]]:
    for run_json in sorted((campaign_dir / "_work").glob("*/run.json")):
        yield json.loads(run_json.read_text(encoding="utf-8")), run_json.parent


def load_work(run_dir: Path, task_id: str, run_meta: dict[str, Any]) -> dict[str, Any] | None:
    p = run_dir / "tasks" / f"{task_id}.json"
    if not p.exists():
        return None
    w = json.loads(p.read_text(encoding="utf-8"))
    w["condition_id"] = run_meta["condition_id"]
    w["delivery_policy"] = run_meta["delivery_policy"]
    return w


def finalize(campaign_dir: str | Path, n_selected: int = 10) -> dict[str, Path]:
    campaign_dir = Path(campaign_dir)
    runs, tasks, calls, tools = [], [], [], []
    traj_dir = campaign_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    raw_buf = io.BytesIO()
    spans_buf = io.BytesIO()
    catalog_versions = set()
    record_index = 0
    selected_pool: list[tuple[str, dict[str, Any]]] = []

    for meta, run_dir in iter_runs(campaign_dir):
        catalog_versions.add(meta.get("price_catalog_version"))
        run_tasks = []
        for task_id in meta["task_ids"]:
            w = load_work(run_dir, task_id, meta)
            if w is None:
                run_tasks.append({"run_id": meta["run_id"], "task_id": task_id, "task_status": "not_executed", "passed": None})
                continue
            t = dict(w["task"])
            ev = analyze_trial(w)
            t.update(ev["task"])
            t.update(trial_signals(w))
            run_tasks.append(t)
            calls.extend(w["model_calls"])
            for tc in w["tool_calls"]:
                flags = ev["tool_calls"].get(tc["call_id"], {})
                dm = tc.get("delivery_meta") or {}
                tools.append(
                    {
                        "call_id": tc["call_id"], "run_id": tc["run_id"], "task_id": tc["task_id"], "step": tc["step"],
                        "tool": tc["tool"], "tool_input": json.dumps(tc["tool_input"], ensure_ascii=False),
                        "is_error": tc["is_error"], "error_type": tc["error_type"],
                        "latency_seconds": tc["latency_seconds"], "cost_usd": tc["cost_usd"],
                        "policy": dm.get("policy"), "raw_tokens": dm.get("raw_tokens"),
                        "delivered_tokens": dm.get("delivered_tokens"), "truncated": dm.get("truncated"),
                        "n_pages": dm.get("n_pages"), "summarized": dm.get("summarized"),
                        "n_urls": len(tc.get("urls") or []), "docids": ",".join(tc.get("docids") or []),
                        **{k: flags.get(k) for k in ("answer_in_raw", "answer_in_delivered", "gold_doc_in_output")},
                        **{k: tc.get(k) for k in ("summarizer_input_tokens", "summarizer_output_tokens")},
                    }
                )
            traj = w["trajectory"]
            line = (json.dumps(traj, ensure_ascii=False, default=str) + "\n").encode()
            index_rows.append(
                {
                    "run_id": meta["run_id"], "task_id": task_id, "benchmark_id": meta["benchmark_id"],
                    "n_steps": len(traj["steps"]), "record_index": record_index, "record_offset": raw_buf.tell(),
                }
            )
            raw_buf.write(line)
            record_index += 1
            for span in otel_spans(meta, w):
                spans_buf.write((json.dumps(span, default=str) + "\n").encode())
            selected_pool.append((t.get("funnel_category", ""), {"run_id": meta["run_id"], **traj}))

        executed = [t for t in run_tasks if t["task_status"] != "not_executed"]
        ex_df = pd.DataFrame(executed)
        sums = {c: int(ex_df[c].sum()) if len(ex_df) and c in ex_df else 0 for c in TOKEN_COLS}
        cost = round(float(ex_df["cost_usd"].sum()), 6) if len(ex_df) else 0.0
        runs.append(
            {
                **{k: meta.get(k) for k in RUN_META_COLS},
                "run_status": "finished" if len(executed) == len(run_tasks) else "partial",
                "tasks_expected": len(run_tasks),
                "tasks_executed": len(executed),
                "tasks_passed": int(sum(1 for t in run_tasks if t.get("passed") is True)),
                "tasks_errored": int(sum(1 for t in run_tasks if t["task_status"] in ("errored", "grading_error"))),
                "total_cost_usd": cost,
                "grader_cost_usd": round(float(ex_df["grader_cost_usd"].sum()), 6) if len(ex_df) else 0.0,
                **sums,
                "started_at": meta.get("started_at"),
                "ended_at": max((t.get("finished_at") or 0) for t in executed) if executed else None,
                "runner_versions": "|".join(meta.get("runner_versions", [])),
            }
        )
        tasks.extend(run_tasks)

    out = {
        "runs": campaign_dir / "runs.csv",
        "tasks": campaign_dir / "tasks.csv",
        "model_calls": campaign_dir / "model_calls.csv",
        "tool_calls": campaign_dir / "tool_calls.csv",
    }
    pd.DataFrame(runs).to_csv(out["runs"], index=False)
    pd.DataFrame(tasks).to_csv(out["tasks"], index=False)
    pd.DataFrame(calls).to_csv(out["model_calls"], index=False)
    pd.DataFrame(tools).to_csv(out["tool_calls"], index=False)
    cctx = zstandard.ZstdCompressor(level=10)
    (traj_dir / "archive.jsonl.zst").write_bytes(cctx.compress(raw_buf.getvalue()))
    pd.DataFrame(index_rows).to_csv(traj_dir / "trajectory_index.csv", index=False)
    (campaign_dir / "traces").mkdir(exist_ok=True)
    (campaign_dir / "traces" / "otel_spans.jsonl.zst").write_bytes(cctx.compress(spans_buf.getvalue()))
    _write_selected(traj_dir / "selected", selected_pool, n_selected)
    catalog = load_catalog()
    manifest = {
        "campaign_id": campaign_dir.name,
        "finalized_at": time.time(),
        "n_runs": len(runs),
        "n_task_rows": len(tasks),
        "n_model_calls": len(calls),
        "n_tool_calls": len(tools),
        "price_catalog_versions": sorted(v for v in catalog_versions if v),
        "current_price_catalog": catalog.version,
        "notes": "prompt_tokens exclude cached tokens; task cost_usd = model calls + tool fees (search, summarizer); "
        "grader cost is reported separately (grader_cost_usd).",
    }
    (campaign_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def _write_selected(d: Path, pool: list[tuple[str, dict]], n: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.json"):
        f.unlink()
    by_cat: dict[str, list[dict]] = {}
    for cat, traj in pool:
        by_cat.setdefault(cat, []).append(traj)
    chosen, cats = [], sorted(by_cat)
    while len(chosen) < n and any(by_cat.values()):
        for c in cats:
            if by_cat[c] and len(chosen) < n:
                chosen.append((c, by_cat[c].pop(0)))
    for cat, traj in chosen:
        (d / f"{traj['run_id']}__{traj['task_id']}__{cat}.json").write_text(
            json.dumps(traj, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
        )


def otel_spans(meta: dict[str, Any], w: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenTelemetry GenAI-convention-shaped spans: invoke_agent -> chat -> execute_tool."""
    trace_id = w.get("trace_id")
    root = f"{trace_id[:16]}"
    t = w["task"]
    spans = [
        {
            "trace_id": trace_id, "span_id": root, "parent_span_id": None,
            "name": f"invoke_agent {meta['harness_name']}",
            "attributes": {
                "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": meta["harness_name"],
                "gen_ai.request.model": meta["model_name"], "gen_ai.provider.name": meta["provider"],
                "agent_eval.run_id": meta["run_id"], "agent_eval.task_id": t["task_id"],
                "agent_eval.condition": meta["condition_id"], "agent_eval.delivery": meta["delivery_policy"],
                "agent_eval.termination_reason": t.get("termination_reason"), "agent_eval.passed": t.get("passed"),
            },
            "duration_s": t.get("latency_seconds"),
        }
    ]
    for c in w["model_calls"]:
        sid = f"{root}-c{c['call_index']}"
        spans.append(
            {
                "trace_id": trace_id, "span_id": sid, "parent_span_id": root, "name": f"chat {c['model_name']}",
                "attributes": {
                    "gen_ai.operation.name": "chat", "gen_ai.request.model": c["model_name"],
                    "gen_ai.usage.input_tokens": c["prompt_tokens"] + c["cached_tokens"] + c["cache_write_tokens"],
                    "gen_ai.usage.output_tokens": c["completion_tokens"],
                    "gen_ai.response.finish_reasons": [c["stop_reason"]],
                    "agent_eval.harness_phase": c["harness_phase"], "agent_eval.cost_usd": c["total_cost_usd"],
                    "error.type": c["exception_type"],
                },
                "duration_s": c["latency_seconds"],
            }
        )
    for tc in w["tool_calls"]:
        spans.append(
            {
                "trace_id": trace_id, "span_id": f"{root}-t{tc['call_id']}", "parent_span_id": f"{root}-c{tc['step']}",
                "name": f"execute_tool {tc['tool']}",
                "attributes": {
                    "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tc["tool"],
                    "gen_ai.tool.call.id": tc["call_id"], "error.type": tc["error_type"],
                    "agent_eval.delivery.truncated": (tc.get("delivery_meta") or {}).get("truncated"),
                },
                "duration_s": tc["latency_seconds"],
            }
        )
    return spans


def get_trajectory(campaign_dir: str | Path, run_id: str, task_id: str) -> dict[str, Any] | None:
    campaign_dir = Path(campaign_dir)
    idx = pd.read_csv(campaign_dir / "trajectories" / "trajectory_index.csv")
    row = idx[(idx.run_id == run_id) & (idx.task_id == task_id)]
    if row.empty:
        return None
    off = int(row.iloc[0]["record_offset"])
    data = zstandard.ZstdDecompressor().decompress((campaign_dir / "trajectories" / "archive.jsonl.zst").read_bytes(), max_output_size=2**34)
    end = data.find(b"\n", off)
    return json.loads(data[off:end])
