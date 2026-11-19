"""Experiment / run configuration.

An experiment YAML defines `defaults` plus an optional `matrix` (cartesian product of
overrides). Each resolved combination x repeat is one *run*. `config_hash` is computed from
every setting that defines the setup (not the repeat index or campaign), so repeats are
identified by what was configured, not by a launch label.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .. import __version__

DELIVERY_POLICIES = (
    "full",
    "truncate_2k",
    "truncate_8k",
    "truncate_32k",
    "paginate_8k",
    "summarize",
)


class ModelConfig(BaseModel):
    provider: str = "anthropic"
    name: str = "claude-haiku-5-5"
    effort: str | None = "medium"
    max_tokens: int = 16000
    thinking_display: str | None = "summarized"
    extra: dict[str, Any] = Field(default_factory=dict)


class Limits(BaseModel):
    step_cap: int = 50
    token_budget_per_task: int = 1_500_000
    time_cap_s: float = 1200.0
    cost_cap_usd: float = 1.0
    wrap_up_fraction: float = 0.85


class GraderConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-haiku-5-5"
    effort: str | None = "low"
    max_tokens: int = 4000


class BenchmarkConfig(BaseModel):
    """`id` is a registered benchmark (see `agent-eval benchmarks`) or "package.module:ClassName"."""

    id: str = "browsecomp"
    params: dict[str, Any] = Field(default_factory=dict)  # plugin-specific settings
    num_examples: int | None = None
    seed: int = 0
    task_ids: list[str] | None = None
    data_dir: str = "data"


class RunConfig(BaseModel):
    """Fully-resolved configuration of one run (minus repeat index)."""

    benchmark: BenchmarkConfig
    model: ModelConfig
    harness: str = "direct_tool"
    condition: str = "search_fetch"          # name of a toolset (see tools.DEFAULT_TOOLSETS)
    toolset_tools: list[str] | None = None   # resolved tool list for `condition` (part of the hash)
    delivery: str = "full"
    search_backend: str = "exa"
    search_max_results: int = 5
    context_policy: str = "none"
    summarizer_model: str = "claude-haiku-5-5"
    limits: Limits = Field(default_factory=Limits)

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


class ExperimentConfig(BaseModel):
    campaign_id: str
    description: str = ""
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    grader: GraderConfig = Field(default_factory=GraderConfig)
    toolsets: dict[str, list[str]] = Field(default_factory=dict)  # extra/overriding named toolsets
    defaults: dict[str, Any] = Field(default_factory=dict)
    matrix: dict[str, list[Any]] = Field(default_factory=dict)
    repeats: int = 1
    concurrency: int = 4
    max_cost_usd: float = 10.0
    output_dir: str = "runs"
    price_catalog: str | None = None

    def expand(self) -> list[RunConfig]:
        keys = list(self.matrix)
        combos = itertools.product(*(self.matrix[k] for k in keys)) if keys else [()]
        out: list[RunConfig] = []
        for combo in combos:
            merged = _deep_merge(self.defaults, {})
            for k, v in zip(keys, combo):
                _set_dotted(merged, k, v)
            merged["benchmark"] = self.benchmark.model_dump()
            rc = RunConfig.model_validate(merged)
            if rc.delivery not in DELIVERY_POLICIES:
                raise ValueError(f"unknown delivery {rc.delivery!r}; choose from {DELIVERY_POLICIES}")
            out.append(rc)
        return out


def load_experiment(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)


def runner_version() -> str:
    """Package version + git SHA (+ '-dirty'). Never empty."""
    sha = "nogit"
    try:
        root = Path(__file__).resolve().parents[3]
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        ).stdout.strip() or "nogit"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if dirty and sha != "nogit":
            sha += "-dirty"
    except Exception:
        pass
    return f"{__version__}+{sha}"


def _deep_merge(a: dict, b: dict) -> dict:
    out = json.loads(json.dumps(a))
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value
