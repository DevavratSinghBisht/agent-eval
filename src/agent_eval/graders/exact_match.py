"""Deterministic answer matching (no LLM): normalized exact match against the reference and
aliases, with optional numeric tolerance."""

from __future__ import annotations

import re

from ..analysis.evidence import normalize
from ..core.task import GradeResult

_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def exact_match(answer: str, references: list[str], numeric_tolerance: float = 0.0, contains: bool = False) -> GradeResult:
    a = normalize(answer.replace(",", ""))
    refs = [normalize(r.replace(",", "")) for r in references if r is not None]
    for r in refs:
        if a == r or (contains and r and f" {r} " in f" {a} "):
            return GradeResult(True, details={"matched": r, "method": "exact"})
        if numeric_tolerance and _NUM.match(a) and _NUM.match(r):
            if abs(float(a) - float(r)) <= numeric_tolerance * max(1.0, abs(float(r))):
                return GradeResult(True, details={"matched": r, "method": "numeric"})
    return GradeResult(False, details={"matched": None, "method": "exact"})
