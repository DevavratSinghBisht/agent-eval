"""Benchmark plugins. Built-ins: browsecomp, browsecomp_plus (reference implementations) and
jsonl_qa (bring-your-own JSONL dataset). Add new ones from `_template.py`."""

from ..core.task import Evidence, GradeResult, Task
from .base import Benchmark, available, get_benchmark, register

__all__ = ["Benchmark", "Evidence", "GradeResult", "Task", "available", "get_benchmark", "register"]
