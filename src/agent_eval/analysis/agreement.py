"""Inter-rater reliability between judges (LLM graders and, optionally, human labels).

Input is the long-format `judgements.csv` (one row per trial x judge, `passed` in {True, False}
or empty when that judge failed to grade). For every judge pair, overall and per condition:

  * percent agreement
  * Cohen's kappa with a bootstrap 95% CI (chance-corrected agreement)
  * Gwet's AC1 — stays meaningful when pass rates are extreme (the "kappa paradox": with ~0% or
    ~100% passes, kappa collapses or is undefined even when judges agree almost always)
  * 2x2 confusion counts, McNemar exact test (is one judge systematically more lenient?),
    prevalence and bias indices (why kappa is low despite high raw agreement)
For 3+ judges: Krippendorff's alpha (nominal), which tolerates missing verdicts.
Plus: accuracy under each judge on the shared trials (how much the headline score depends on the
judge) and the list of contested trials for human review.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KAPPA_WARN, AC1_WARN = 0.6, 0.7  # below these the validator warns (Landis & Koch "substantial" boundary for kappa)


# ---- pairwise statistics (binary verdicts) ------------------------------------------------
def confusion(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    a, b = a.astype(bool), b.astype(bool)
    return {"both_pass": int((a & b).sum()), "a_only": int((a & ~b).sum()),
            "b_only": int((~a & b).sum()), "both_fail": int((~a & ~b).sum())}


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    a, b = a.astype(bool), b.astype(bool)
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if math.isclose(pe, 1.0) else float((po - pe) / (1 - pe))


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    a, b = a.astype(bool), b.astype(bool)
    po = float((a == b).mean())
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    return float("nan") if math.isclose(pe, 1.0) else float((po - pe) / (1 - pe))


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import binomtest

    c = confusion(a, b)
    k, n = c["a_only"], c["a_only"] + c["b_only"]
    return 1.0 if n == 0 else float(binomtest(k, n, 0.5).pvalue)


def bootstrap_stat(fn, a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI over resampled trials; resamples where the statistic is undefined are dropped."""
    if len(a) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    vals = np.array([fn(a[i], b[i]) for i in idx])
    vals = vals[~np.isnan(vals)]
    if len(vals) < n_boot * 0.5:  # mostly undefined (near-constant ratings): no meaningful CI
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def pair_stats(a: np.ndarray, b: np.ndarray, with_ci: bool = True) -> dict[str, Any]:
    n = len(a)
    c = confusion(a, b)
    k = cohen_kappa(a, b)
    out: dict[str, Any] = {
        "n": n,
        "percent_agreement": float((a.astype(bool) == b.astype(bool)).mean()) if n else float("nan"),
        "cohen_kappa": k,
        "gwet_ac1": gwet_ac1(a, b),
        **c,
        "mcnemar_p": mcnemar_p(a, b),
        "prevalence_index": abs(c["both_pass"] - c["both_fail"]) / n if n else float("nan"),
        "bias_index": abs(c["a_only"] - c["b_only"]) / n if n else float("nan"),
        "pass_rate_a": float(a.mean()) if n else float("nan"),
        "pass_rate_b": float(b.mean()) if n else float("nan"),
    }
    if math.isnan(k) and n:
        out["kappa_note"] = "undefined: both judges gave (nearly) the same verdict to every trial; use AC1 / percent agreement"
    if with_ci:
        out["kappa_ci"] = bootstrap_stat(cohen_kappa, a, b)
        out["ac1_ci"] = bootstrap_stat(gwet_ac1, a, b)
    return out


# ---- multi-rater ----------------------------------------------------------------------------
def krippendorff_alpha_nominal(units: np.ndarray) -> float:
    """units: (n_units, n_raters) float array with values {0,1} and NaN for missing."""
    pairable = [row[~np.isnan(row)] for row in units]
    pairable = [r for r in pairable if len(r) >= 2]
    if not pairable:
        return float("nan")
    cats = sorted({v for r in pairable for v in r})
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for r in pairable:
        m = len(r)
        for i, j in itertools.permutations(range(m), 2):
            o[idx[r[i]], idx[r[j]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    d_o = (o.sum() - np.trace(o)) / n
    d_e = (n_c.sum() ** 2 - (n_c ** 2).sum()) / (n * (n - 1))
    return float("nan") if d_e == 0 else float(1 - d_o / d_e)


# ---- campaign-level ---------------------------------------------------------------------------
def _wide(judgements: pd.DataFrame) -> pd.DataFrame:
    j = judgements.dropna(subset=["passed"]).copy()
    j["passed"] = j["passed"].map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))
    return j.pivot_table(index=["run_id", "task_id"], columns="judge", values="passed", aggfunc="first")


def compute_agreement(judgements: pd.DataFrame, runs: pd.DataFrame | None = None, primary: str | None = None) -> dict[str, Any]:
    wide = _wide(judgements)
    judges = list(wide.columns)
    result: dict[str, Any] = {"judges": judges, "primary_judge": primary, "pairs": [], "by_condition": [],
                              "judge_accuracy": {}, "contested": []}
    if len(judges) < 2:
        result["note"] = "fewer than two judges with verdicts; nothing to compare"
        return result
    cond = None
    if runs is not None and "condition_id" in runs:
        cond = runs.set_index("run_id")["condition_id"]
    for ja, jb in itertools.combinations(judges, 2):
        both = wide[[ja, jb]].dropna()
        a, b = both[ja].to_numpy(bool), both[jb].to_numpy(bool)
        result["pairs"].append({"judge_a": ja, "judge_b": jb, **pair_stats(a, b)})
        if cond is not None:
            conds = both.index.get_level_values("run_id").map(cond)
            for c in sorted(set(conds.dropna())):
                m = np.asarray(conds == c)
                result["by_condition"].append({"judge_a": ja, "judge_b": jb, "condition_id": c,
                                               **pair_stats(a[m], b[m], with_ci=False)})
    common = wide.dropna()
    for j in judges:
        col = wide[j].dropna()
        result["judge_accuracy"][j] = {"n_judged": int(len(col)), "pass_rate": float(col.astype(bool).mean()),
                                       "pass_rate_on_common": float(common[j].astype(bool).mean()) if len(common) else float("nan")}
    rates = [v["pass_rate_on_common"] for v in result["judge_accuracy"].values() if not math.isnan(v["pass_rate_on_common"])]
    result["score_range_on_common"] = {"n_common": int(len(common)), "min": min(rates) if rates else None,
                                       "max": max(rates) if rates else None,
                                       "spread": (max(rates) - min(rates)) if rates else None}
    if len(judges) >= 3:
        result["krippendorff_alpha"] = krippendorff_alpha_nominal(wide.astype(float).to_numpy())
    disagree = wide[wide.nunique(axis=1, dropna=True) > 1]
    result["contested"] = [{"run_id": r, "task_id": t, **{j: (None if pd.isna(v) else bool(v)) for j, v in row.items()}}
                           for (r, t), row in disagree.iterrows()]
    return result


def write_agreement(campaign_dir: str | Path, human_labels: str | Path | None = None) -> dict[str, Any] | None:
    """Compute agreement for a finalized campaign (judgements.csv). Human labels (CSV with run_id,
    task_id, verdict) are added as one more rater named 'human'. Returns None if no judgements."""
    from . import read_csv_safe

    d = Path(campaign_dir)
    j = read_csv_safe(d / "judgements.csv")
    if j.empty:
        return None
    if human_labels:
        h = pd.read_csv(human_labels)
        col = "verdict" if "verdict" in h else "passed"
        h = h.assign(judge="human", role="human",
                     passed=h[col].map(lambda v: str(v).strip().lower() in ("true", "1", "yes", "pass")))
        j = pd.concat([j, h[["run_id", "task_id", "judge", "role", "passed"]]], ignore_index=True)
    runs = read_csv_safe(d / "runs.csv")
    primary = j.loc[j.role == "primary", "judge"].iloc[0] if (j.role == "primary").any() else None
    res = compute_agreement(j, runs if not runs.empty else None, primary)
    (d / "agreement.json").write_text(json.dumps(res, indent=1, default=_json_default), encoding="utf-8")
    return res


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
