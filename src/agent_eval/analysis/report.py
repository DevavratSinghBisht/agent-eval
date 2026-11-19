"""Markdown report for a finalized + validated campaign."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import read_csv_safe
from .metrics import consumption_table, funnel_table, score_configs, score_runs

SIGNAL_COLS = [
    "leak_suspect", "doom_loop", "cap_hit", "hedged_answer", "overconfident_wrong", "premature_answer", "wrap_up_nudged",
]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in df.columns]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, float):
                cells.append(floatfmt.format(v) if pd.notna(v) else "")
            else:
                cells.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def build_report(campaign_dir: str | Path) -> Path:
    d = Path(campaign_dir)
    runs = pd.read_csv(d / "runs.csv")
    tasks = pd.read_csv(d / "tasks.csv")
    tools = read_csv_safe(d / "tool_calls.csv")
    val = json.loads((d / "validation.json").read_text()) if (d / "validation.json").exists() else {}
    validity = val.get("run_validity", {})

    parts = [f"# Campaign report: {d.name}\n"]
    if val:
        parts.append(
            f"**Validation:** {'OK' if val['ok'] else 'FAILED'} — {val['n_errors']} errors, {val['n_warnings']} warnings "
            "(see validation.json). Runs flagged invalid are excluded from performance tables.\n"
        )
        errs = [f for f in val["findings"] if f["severity"] == "error"][:15]
        if errs:
            parts.append("Top validation errors:\n" + "".join(f"- `{e['check']}` {e['run_id'] or ''} {e['task_id'] or ''}: {e['detail']}\n" for e in errs))

    parts.append("\n## Configurations (task-level bootstrap 95% CI over per-task mean pass rate)\n")
    parts.append(md_table(score_configs(runs, tasks, validity)))
    parts.append("\n## Runs (score = tasks_passed / tasks_expected)\n")
    sr = score_runs(runs, tasks, validity)
    parts.append(md_table(sr[["run_id", "condition_id", "delivery_policy", "repeat_index", "run_status", "score",
                              "score_excl_leaks", "n_leak_suspect", "executed_only_accuracy", "tasks_executed", "tasks_expected", "cost_usd", "cost_per_success",
                              "valid_perf", "flags"]]))

    parts.append("\n## Evidence-consumption funnel (share of executed tasks)\n")
    parts.append(
        "Categories: success; retrieval_miss (answer never in any tool output); truncation_loss (in raw output but cut "
        "by the delivery policy); comprehension_miss (shown to the model but never used); synthesis_error (used but final "
        "answer wrong); knowledge_miss (closed-book); no_answer_* (cap/refusal/exception); format_or_grading.\n\n"
    )
    parts.append(md_table(funnel_table(tasks, runs)))
    parts.append("\n## Tool-result consumption (raw vs delivered)\n")
    parts.append(md_table(consumption_table(tools, runs)))

    ex = tasks[tasks.task_status != "not_executed"].merge(runs[["run_id", "condition_id", "delivery_policy"]], on="run_id")
    present = [c for c in SIGNAL_COLS if c in ex]
    if present:
        parts.append("\n## Failure signals (rate among executed tasks)\n")
        sig = ex.groupby(["condition_id", "delivery_policy"])[present].mean().round(3)
        parts.append(md_table(sig))
        parts.append("\n## Termination reasons\n")
        parts.append(md_table(ex.groupby(["condition_id", "delivery_policy", "termination_reason"]).size().unstack(fill_value=0)))

    labels = d / "failure_labels.csv"
    if labels.exists():
        lab = pd.read_csv(labels)
        parts.append("\n## LLM failure labels (taxonomy)\n")
        parts.append(md_table(lab.groupby("category").size().rename("n").reset_index().sort_values("n", ascending=False)))

    out = d / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
