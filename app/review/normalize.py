"""
Normalization helpers for raw LLM finding fields.

LLMs are inconsistent about the shape of the values they return
(``"line_number": "42"``, ``"~50"``, ``"N/A"``, floats; severities like
``"blocker"`` or ``"moderate"``). These helpers coerce those values into
the exact types/values the database columns and downstream code expect,
so a sloppy model response cannot crash the review pipeline.
"""

from __future__ import annotations

import re

# Severity / category vocabularies accepted by the schema + publisher.
_ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}
_SEVERITY_ALIASES = {
    "blocker": "critical",
    "severe": "critical",
    "major": "high",
    "moderate": "medium",
    "minor": "low",
    "info": "low",
    "informational": "low",
    "nit": "low",
    "trivial": "low",
    "warning": "medium",
    "": "medium",
}

_ALLOWED_CATEGORIES = {
    "correctness",
    "security",
    "error_handling",
    "performance",
    "quality",
    "testing",
}
_CATEGORY_ALIASES = {
    "bug": "correctness",
    "logic": "correctness",
    "logic_error": "correctness",
    "vulnerability": "security",
    "auth": "security",
    "error handling": "error_handling",
    "errors": "error_handling",
    "exception": "error_handling",
    "exceptions": "error_handling",
    "perf": "performance",
    "efficiency": "performance",
    "style": "quality",
    "readability": "quality",
    "maintainability": "quality",
    "code_quality": "quality",
    "code quality": "quality",
    "tests": "testing",
    "test": "testing",
    "test_coverage": "testing",
    "": "quality",
}


def coerce_line_number(value: object) -> int | None:
    """Return a positive int line number, or ``None`` if not usable.

    Accepts ints, floats, and strings that contain digits
    (``"42"``, ``" 42 "``, ``"L42"``, ``"line 42"``). Anything else
    (``None``, ``"N/A"``, ``"multiple"``, ``0``, negatives) → ``None``.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value >= 1 else None
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if not match:
            return None
        try:
            n = int(match.group(0))
        except ValueError:
            return None
        return n if n > 0 else None
    return None


def coerce_confidence(value: object, default: float = 0.5) -> float:
    """Return a float confidence clamped to [0.0, 1.0]."""
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if n != n:  # NaN
        return default
    # Some models return 0-100 instead of 0-1.
    if n > 1.0:
        n = n / 100.0 if n <= 100.0 else 1.0
    if n < 0.0:
        return 0.0
    return min(1.0, n)


def normalize_severity(value: object) -> str:
    """Map an arbitrary severity string to one of the allowed values."""
    text = str(value or "").strip().lower()
    if text in _ALLOWED_SEVERITIES:
        return text
    return _SEVERITY_ALIASES.get(text, "medium")


def normalize_category(value: object) -> str:
    """Map an arbitrary category string to one of the allowed values."""
    text = str(value or "").strip().lower().replace("-", "_")
    if text in _ALLOWED_CATEGORIES:
        return text
    if text in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[text]
    return _CATEGORY_ALIASES.get(text.replace("_", " "), "quality")
