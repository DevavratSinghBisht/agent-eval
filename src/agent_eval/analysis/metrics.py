"""Scoring and statistics.

Conventions (following the example loss-analysis notebook):
  * Headline run score = tasks_passed / tasks_expected (not-executed and errored tasks count as
    failures). Executed-only accuracy is reported alongside, never instead.
  * Two runs are repeats only if their config_hash matches; configs are compared on the tasks
    they share, with task-level bootstrap CIs over per-task mean pass rates.
  * Runs flagged by the validator (unverifiable / infrastructure failure) are excluded from
    performance statistics but kept for cost/ops analysis.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

CONFIG_COLS = ["config_hash", "benchmark_id", "model_name", "harness_name", "condition_id", "delivery_policy", "effort"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: probability that k independent trials ALL succeed (reliability)."""
    if n < k:
        return float("nan")
    return comb(c, k) / comb(n, k)


def calibration_error(conf: np.ndarray, correct: np.ndarray, bin_size: int = 100) -> float:
    """RMS calibration error with equal-count bins (as in the HLE/BrowseComp evaluations)."""
    if len(conf) == 0:
        return float("nan")
    order = np.argsort(conf)
    conf, correct = conf[order] / 100.0, correct[order].astype(float)
    total, err = len(conf), 0.0
    for i in range(0, total, bin_size):
        c, y = conf[i : i + bin_size], correct[i : i + bin_size]
        err += len(c) / total * (c.mean() - y.mean()) ** 2
    return float(np.sqrt(err))


def score_runs(runs: pd.DataFrame, tasks: pd.DataFrame, validity: dict | None = None) -> pd.DataFrame:
    rows = []
    for _, r in runs.iterrows():
        t = tasks[tasks.run_id == r.run_id]
        passed = int((t.passed == True).sum())  # noqa: E712
        leak = (t.leak_suspect == True) if "leak_suspect" in t else pd.Series(False, index=t.index)  # noqa: E712
        n_leak, leak_passed = int(leak.sum()), int((leak & (t.passed == True)).sum())  # noqa: E712
        n_exp = len(t)
        ex = t[t.task_status != "not_executed"]
        lo, hi = wilson(passed, n_exp)
        v = (validity or {}).get(r.run_id, {"valid_perf": True, "flags": []})
        rows.append(
            {
                **{c: r.get(c) for c in ["run_id", *CONFIG_COLS, "repeat_index", "run_status"]},
                "score": passed / n_exp if n_exp else float("nan"),
                "score_ci_lo": lo,
                "score_ci_hi": hi,
                "executed_only_accuracy": passed / len(ex) if len(ex) else float("nan"),
                "n_leak_suspect": n_leak,
                "score_excl_leaks": (passed - leak_passed) / (n_exp - n_leak) if n_exp > n_leak else float("nan"),
                "tasks_expected": n_exp,
                "tasks_executed": len(ex),
                "tasks_passed": passed,
                "cost_usd": r.total_cost_usd,
                "cost_per_task": r.total_cost_usd / len(ex) if len(ex) else float("nan"),
                "cost_per_success": r.total_cost_usd / passed if passed else float("inf"),
                "valid_perf": v["valid_perf"],
                "flags": ",".join(v["flags"]),
            }
        )
    return pd.DataFrame(rows)


def score_configs(runs: pd.DataFrame, tasks: pd.DataFrame, validity: dict | None = None, ks=(1, 3)) -> pd.DataFrame:
    valid = [rid for rid in runs.run_id if (validity or {}).get(rid, {"valid_perf": True})["valid_perf"]]
    r = runs[runs.run_id.isin(valid)]
    t = tasks.merge(r[["run_id", *CONFIG_COLS]], on="run_id")
    n_leak = t.groupby("config_hash").leak_suspect.apply(lambda x: int((x == True).sum())) if "leak_suspect" in t else None  # noqa: E712
    if "leak_suspect" in t:
        t = t[t.leak_suspect != True]  # noqa: E712  leak suspects are excluded from performance stats
    t = t.assign(ok=(t.passed == True).astype(float))  # noqa: E712
    rows = []
    for keys, g in t.groupby(CONFIG_COLS, dropna=False):
        per_task = g.groupby("task_id").ok.agg(["sum", "count"])
        means = (per_task["sum"] / per_task["count"]).to_numpy()
        lo, hi = bootstrap_ci(means)
        row = dict(zip(CONFIG_COLS, keys))
        row.update(
            {
                "n_runs": g.run_id.nunique(),
                "n_tasks": len(per_task),
                "n_leak_excluded": int(n_leak.get(row["config_hash"], 0)) if n_leak is not None else 0,
                "accuracy": float(means.mean()) if len(means) else float("nan"),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
        for k in ks:
            if (per_task["count"] >= k).all() and k > 1:
                row[f"pass@{k}"] = float(np.mean([pass_at_k(int(n), int(c), k) for c, n in per_task.to_numpy()]))
                row[f"pass^{k}"] = float(np.mean([pass_hat_k(int(n), int(c), k) for c, n in per_task.to_numpy()]))
        conf = g.dropna(subset=["confidence"])
        conf = conf[conf.task_status == "completed"]
        row["calibration_error"] = calibration_error(conf.confidence.to_numpy(float), conf.ok.to_numpy()) if len(conf) else float("nan")
        ex = g[g.task_status != "not_executed"]
        row["cost_per_task"] = float(ex.cost_usd.mean()) if len(ex) else float("nan")
        row["tokens_per_task"] = float((ex.prompt_tokens + ex.cached_tokens + ex.cache_write_tokens + ex.completion_tokens).mean()) if len(ex) else float("nan")
        row["tool_calls_per_task"] = float(ex.tool_call_count.mean()) if len(ex) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def paired_compare(tasks: pd.DataFrame, runs: pd.DataFrame, hash_a: str, hash_b: str, seed: int = 0) -> dict:
    """Difference in per-task mean pass rate (B - A) on shared tasks, paired bootstrap CI and
    an exact McNemar test on per-task majority outcomes."""
    from scipy.stats import binomtest

    t = tasks.merge(runs[["run_id", "config_hash"]], on="run_id")
    t = t.assign(ok=(t.passed == True).astype(float))  # noqa: E712
    a = t[t.config_hash == hash_a].groupby("task_id").ok.mean()
    b = t[t.config_hash == hash_b].groupby("task_id").ok.mean()
    shared = a.index.intersection(b.index)
    d = (b[shared] - a[shared]).to_numpy()
    lo, hi = bootstrap_ci(d, seed=seed)
    aw, bw = (a[shared] >= 0.5), (b[shared] >= 0.5)
    n01, n10 = int((~aw & bw).sum()), int((aw & ~bw).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"n_shared_tasks": len(shared), "diff": float(d.mean()) if len(d) else float("nan"),
            "ci_lo": lo, "ci_hi": hi, "b_only_wins": n01, "a_only_wins": n10, "mcnemar_p": float(p)}


def funnel_table(tasks: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    t = tasks[tasks.task_status != "not_executed"].merge(runs[["run_id", "condition_id", "delivery_policy"]], on="run_id")
    if "funnel_category" not in t:
        return pd.DataFrame()
    tab = t.groupby(["condition_id", "delivery_policy", "funnel_category"]).size().unstack(fill_value=0)
    return tab.div(tab.sum(axis=1), axis=0).round(3)


def consumption_table(tool_calls: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    if tool_calls.empty:
        return pd.DataFrame()
    tc = tool_calls[tool_calls.tool != "final_answer"].merge(runs[["run_id", "condition_id", "delivery_policy"]], on="run_id")
    g = tc.groupby(["condition_id", "delivery_policy"])
    return pd.DataFrame(
        {
            "n_tool_calls": g.size(),
            "mean_raw_tokens": g.raw_tokens.mean().round(0),
            "mean_delivered_tokens": g.delivered_tokens.mean().round(0),
            "delivered_fraction": (g.delivered_tokens.sum() / g.raw_tokens.sum().clip(lower=1)).round(3),
            "truncated_rate": g.truncated.mean().round(3),
            "answer_in_raw_rate": g.answer_in_raw.mean().round(3),
            "answer_in_delivered_rate": g.answer_in_delivered.mean().round(3),
            "answer_lost_to_delivery": ((tc.answer_in_raw == True) & (tc.answer_in_delivered != True)).groupby(  # noqa: E712
                [tc.condition_id, tc.delivery_policy]).sum(),
        }
    ).reset_index()
