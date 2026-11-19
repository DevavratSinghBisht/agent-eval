"""Versioned price catalog. Costs are ALWAYS computed from a catalog entry; there is no
fallback rate. A missing model or catalog raises, so mispriced runs cannot be produced
silently (a fault class found in the example loss-analysis data)."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

import yaml

from .types import Usage

DEFAULT_CATALOG = Path(__file__).resolve().parents[3] / "configs" / "price_catalog.yaml"


class PricingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float
    cache_write_multiplier: float
    long_context_threshold: int | None = None
    long_input_per_mtok: float | None = None
    long_output_per_mtok: float | None = None

    def cost(self, u: Usage) -> float:
        inp, out = self.input_per_mtok, self.output_per_mtok
        if (
            self.long_context_threshold is not None
            and u.total_prompt_tokens > self.long_context_threshold
        ):
            inp = self.long_input_per_mtok or inp
            out = self.long_output_per_mtok or out
        return (
            u.input_tokens * inp
            + u.cache_read_tokens * inp * self.cache_read_multiplier
            + u.cache_write_tokens * inp * self.cache_write_multiplier
            + u.output_tokens * out
        ) / 1_000_000


@dataclass(frozen=True)
class PriceCatalog:
    version: str
    models: dict[str, ModelPrice]
    search_per_call: dict[str, float]

    def model(self, name: str) -> ModelPrice:
        """Exact entry first, then the first glob pattern that matches (e.g. "qwen3*" for a
        self-hosted family). No match is a hard error: runs are never priced with a guess."""
        if name in self.models:
            return self.models[name]
        for pattern, price in self.models.items():
            if any(ch in pattern for ch in "*?[") and fnmatchcase(name, pattern):
                return price
        raise PricingError(
            f"model {name!r} not in price catalog {self.version}; add it (or a matching pattern) before running"
        )

    def cost(self, model: str, usage: Usage) -> float:
        return round(self.model(model).cost(usage), 6)

    def search_cost(self, backend: str) -> float:
        return self.search_per_call.get(backend, 0.0)


def load_catalog(path: str | Path | None = None) -> PriceCatalog:
    p = Path(path) if path else DEFAULT_CATALOG
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not data.get("version"):
        raise PricingError(f"price catalog {p} has no version")
    models = {name: ModelPrice(**spec) for name, spec in data["models"].items()}
    return PriceCatalog(
        version=str(data["version"]),
        models=models,
        search_per_call={k: float(v) for k, v in (data.get("search_per_call") or {}).items()},
    )
