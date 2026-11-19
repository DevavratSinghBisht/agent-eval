"""Campaign integrity validator.

Each check corresponds to a data fault class seen in the example loss-analysis campaign
(mispriced runs without a catalog, runs with passes but no call log, all-parse-failure
runs, partial runs scored on the wrong denominator, repeat labels that don't match
configuration, cached-token accounting anomalies) or to our own logging invariants.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from . import read_csv_safe
from ..core.pricing import PricingError, load_catalog
from ..core.types import Usage

TOL = 1e-5
KNOWN_HARNESSES = {"direct_tool", "planner_evaluator"}


@dataclass
class Finding:
    severity: str  # error | warning | info
    check: str
    run_id: str | None
    task_id: str | None
    detail: str


def validate(campaign_dir: str | Path) -> dict:
    d = Path(campaign_dir)
    runs = pd.read_csv(d / "runs.csv")
    tasks = pd.read_csv(d / "tasks.csv")
    calls = read_csv_safe(d / "model_calls.csv")
    tools = read_csv_safe(d / "tool_calls.csv")
    tindex = read_csv_safe(d / "trajectories" / "trajectory_index.csv")
    F: list[Finding] = []

    def add(sev, check, detail, run_id=None, task_id=None):
        F.append(Finding(sev, check, run_id, task_id, detail))

    catalog = load_catalog()

    # ---- run metadata ------------------------------------------------------------------
    for _, r in runs.iterrows():
        for col in ("runner_version", "price_catalog_version", "config_hash", "model_name", "harness_name"):
            if pd.isna(r.get(col)) or r.get(col) == "":
                add("error", "missing_metadata", f"{col} is missing", r.run_id)
        if r.get("harness_name") not in KNOWN_HARNESSES:
            add("error", "unknown_harness", f"harness_name {r.get('harness_name')!r} not in {sorted(KNOWN_HARNESSES)}", r.run_id)
        try:
            catalog.model(r.model_name)
        except PricingError as e:
            add("error", "unpriced_model", str(e), r.run_id)
        if str(r.get("price_catalog_version")) != catalog.version:
            add("warning", "catalog_version_mismatch",
                f"run priced with catalog {r.get('price_catalog_version')}, current is {catalog.version}; repricing check skipped", r.run_id)
        if r.run_status == "partial":
            add("warning", "partial_run", f"{r.tasks_executed}/{r.tasks_expected} tasks executed; score uses tasks_expected", r.run_id)
        if "|" in str(r.get("runner_versions", "")):
            add("warning", "mixed_runner_versions", f"run resumed across runner versions {r.runner_versions}", r.run_id)

    # ---- referential integrity -----------------------------------------------------------
    run_ids = set(runs.run_id)
    orphan = tasks[~tasks.run_id.isin(run_ids)]
    for _, t in orphan.iterrows():
        add("error", "orphan_task", "task row has no run", t.run_id, t.task_id)
    if not calls.empty:
        keys = set(zip(tasks.run_id, tasks.task_id))
        for _, c in calls[[k not in keys for k in zip(calls.run_id, calls.task_id)]].iterrows():
            add("error", "orphan_call", "model call has no task row", c.run_id, c.task_id)

    # ---- rollups -----------------------------------------------------------------------
    executed = tasks[tasks.task_status != "not_executed"]
    cg = calls.groupby(["run_id", "task_id"]) if not calls.empty else None
    tg = tools.groupby(["run_id", "task_id"]) if not tools.empty else None
    for _, t in executed.iterrows():
        key = (t.run_id, t.task_id)
        c = cg.get_group(key) if cg is not None and key in cg.groups else pd.DataFrame()
        tl = tg.get_group(key) if tg is not None and key in tg.groups else pd.DataFrame()
        n_calls = len(c)
        if int(t.call_count) != n_calls:
            add("error", "call_count_mismatch", f"call_count={t.call_count} but {n_calls} call rows", *key)
        for col in ("prompt_tokens", "cached_tokens", "cache_write_tokens", "completion_tokens"):
            s = int(c[col].sum()) if n_calls else 0
            if int(t[col]) != s:
                add("error", "token_rollup", f"{col}: task={t[col]} calls={s}", *key)
        cost = (float(c.total_cost_usd.sum()) if n_calls else 0.0) + (float(tl.cost_usd.sum()) if len(tl) else 0.0)
        if abs(cost - float(t.cost_usd)) > TOL * max(1, n_calls):
            add("error", "cost_rollup", f"task cost {t.cost_usd} != calls+tools {cost:.6f}", *key)
        if not tindex.empty:
            ti = tindex[(tindex.run_id == t.run_id) & (tindex.task_id == t.task_id)]
            if ti.empty:
                add("error", "missing_trajectory", "no trajectory record", *key)
            elif int(ti.iloc[0].n_steps) != n_calls + 1:
                add("error", "trajectory_step_mismatch", f"{ti.iloc[0].n_steps} steps for {n_calls} calls (+1 closing expected)", *key)
        if t.task_status == "grading_error":
            add("warning", "grading_error", "grader failed; re-run `agent-eval regrade`", *key)
        if t.task_status == "completed" and t.get("grader_parse_ok") is False:
            add("warning", "grader_parse_fail", "grader output had no 'correct: yes|no'; scored as no", *key)
        if t.get("leak_suspect") is True:
            add("warning", "leak_suspect", f"excluded from performance stats: {t.get('leak_reason')}", *key)

    for _, r in runs.iterrows():
        tr = executed[executed.run_id == r.run_id]
        for col in ("prompt_tokens", "cached_tokens", "completion_tokens"):
            if int(tr[col].sum()) != int(r[col]):
                add("error", "run_token_rollup", f"{col}: run={r[col]} tasks={int(tr[col].sum())}", r.run_id)
        if abs(float(tr.cost_usd.sum()) - float(r.total_cost_usd)) > TOL * max(1, len(tr)):
            add("error", "run_cost_rollup", f"run cost {r.total_cost_usd} != tasks {tr.cost_usd.sum():.6f}", r.run_id)
        if int((tasks[(tasks.run_id == r.run_id)].passed == True).sum()) != int(r.tasks_passed):  # noqa: E712
            add("error", "passed_rollup", "tasks_passed does not match task rows", r.run_id)

    # ---- repricing (only when the catalog version matches) --------------------------------
    if not calls.empty:
        cur = runs[runs.price_catalog_version.astype(str) == catalog.version].run_id
        for _, c in calls[calls.run_id.isin(cur)].iterrows():
            u = Usage(int(c.prompt_tokens), int(c.completion_tokens), int(c.cached_tokens), int(c.cache_write_tokens))
            exp = catalog.cost(c.model_name, u)
            if abs(exp - float(c.total_cost_usd)) > TOL:
                add("error", "mispriced_call", f"logged {c.total_cost_usd} vs catalog {exp}", c.run_id, c.task_id)

    # ---- run validity flags ----------------------------------------------------------------
    validity = {}
    for _, r in runs.iterrows():
        rc = calls[calls.run_id == r.run_id] if not calls.empty else pd.DataFrame()
        flags = []
        if int(r.tasks_passed) > 0 and len(rc) == 0:
            flags.append("unverifiable_no_calls")
            add("error", "unverifiable_run", "tasks passed but zero model calls logged", r.run_id)
        if len(rc) and rc.exception_type.notna().mean() > 0.5:
            flags.append("infrastructure_failure")
            add("error", "infrastructure_failure", f"{rc.exception_type.notna().mean():.0%} of calls raised", r.run_id)
        rt = tasks[tasks.run_id == r.run_id]
        if len(rt) and (rt.task_status == "errored").mean() > 0.5:
            flags.append("infrastructure_failure")
            add("error", "infrastructure_failure", "more than half of tasks errored", r.run_id)
        validity[r.run_id] = {"valid_perf": not flags, "flags": sorted(set(flags))}

    # ---- repeats: cached-token share anomalies ------------------------------------------------
    rs = runs.assign(
        cached_share=runs.cached_tokens / (runs.prompt_tokens + runs.cached_tokens + runs.cache_write_tokens).clip(lower=1)
    )
    for h, g in rs.groupby("config_hash"):
        if len(g) < 2:
            continue
        med = g.cached_share.median()
        for _, r in g.iterrows():
            if abs(r.cached_share - med) > 0.3:
                add("warning", "cached_share_anomaly",
                    f"cached share {r.cached_share:.2f} vs repeat median {med:.2f} (config {h})", r.run_id)

    errors = sum(f.severity == "error" for f in F)
    result = {
        "campaign": d.name,
        "ok": errors == 0,
        "n_errors": errors,
        "n_warnings": sum(f.severity == "warning" for f in F),
        "run_validity": validity,
        "findings": [asdict(f) for f in F],
    }
    (d / "validation.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result
