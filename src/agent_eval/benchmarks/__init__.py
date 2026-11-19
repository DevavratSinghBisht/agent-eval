from __future__ import annotations

from ..core.config import BenchmarkConfig
from .browsecomp import Task


def load_tasks(cfg: BenchmarkConfig) -> list[Task]:
    if cfg.id == "browsecomp":
        from .browsecomp import load_tasks as f
    elif cfg.id == "browsecomp_plus":
        from .browsecomp_plus import load_tasks as f
    else:
        raise ValueError(f"unknown benchmark {cfg.id}")
    return f(cfg.data_dir, cfg.num_examples, cfg.seed, cfg.task_ids)


__all__ = ["Task", "load_tasks"]
