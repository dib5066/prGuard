"""
Evidence validation for AI-generated review findings.

Validates findings against actual PR file content to catch:
- Hallucinated file paths (file doesn't exist in the PR)
- Wrong line numbers (line not in the diff range)
- Fabricated evidence (code snippet doesn't match actual file)
- Low-confidence findings that should be filtered

Pipeline:
    Raw Findings → File Check → Line Check → Evidence Check → Calibrate → Filter
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from app.review.normalize import coerce_line_number
from app.services.github_service import PRContext

logger = logging.getLogger(__name__)


# ============================================================================
# Validation Status
# ============================================================================


class ValidationStatus(str, Enum):
    """Validation status for a finding."""

    VALID = "valid"
    FILE_NOT_IN_PR = "file_not_in_pr"
    LINE_NOT_IN_DIFF = "line_not_in_diff"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    LOW_CONFIDENCE = "low_confidence"


# ============================================================================
# Validated Finding
# ============================================================================


@dataclass
class ValidatedFinding:
    """A finding with validation metadata."""

    # Original finding fields
    severity: str
    category: str
    title: str
    description: str
    file_path: str
    line_number: int | None
    evidence: str | None
    confidence: float
    agent: str = ""

    # Validation metadata
    validation_status: ValidationStatus = ValidationStatus.VALID
    original_confidence: float = 0.0
    validation_notes: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if the finding passed validation."""
        return (
            self.validation_status == ValidationStatus.VALID
            and self.confidence >= 0.3
        )

    def to_dict(self) -> dict:
        """Convert to dict for storage / publishing."""
        return {
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "evidence": self.evidence,
            "confidence": float(self.confidence),
            "agent": self.agent,
            "validation_status": self.validation_status.value,
            "validation_notes": self.validation_notes,
        }


# ============================================================================
# Path matching helper
# ============================================================================


def _normalize_path(path: str) -> str:
    """Normalize a file path for tolerant comparison.

    Strips a leading "./", collapses backslashes to forward slashes,
    and lowercases so that minor formatting differences between the
    LLM output and the GitHub file list do not cause false negatives.
    """
    if not path:
        return ""
    cleaned = path.strip().replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lower()


def _paths_match(path_a: str, path_b: str) -> bool:
    """Return True if two paths plausibly refer to the same file.

    Exact match after normalization, or one path is a path-suffix of
    the other (handles repo-root vs. sub-directory-relative paths).
    """
    a = _normalize_path(path_a)
    b = _normalize_path(path_b)
    if not a or not b:
        return False
    if a == b:
        return True
    return a.endswith("/" + b) or b.endswith("/" + a)


# ============================================================================
# Validation Pipeline Stages
# ============================================================================


def _check_file_exists(
    finding: ValidatedFinding,
    pr_context: PRContext,
) -> None:
    """Check if the finding's file exists in the PR.

    Args:
        finding: The finding to validate.
        pr_context: PR context with file paths.

    Side effects:
        Sets finding.validation_status and adjusts confidence.
    """
    # Compare against PR changed files with tolerant path matching.
    file_in_pr = any(
        _paths_match(pr_file.filename, finding.file_path)
        for pr_file in pr_context.files
    )

    if not file_in_pr:
        finding.validation_status = ValidationStatus.FILE_NOT_IN_PR
        finding.confidence *= 0.3  # Heavy penalty
        finding.validation_notes.append(
            f"File '{finding.file_path}' not found in PR changed files"
        )
        logger.debug(
            "Finding '%s' references file not in PR: %s",
            finding.title,
            finding.file_path,
        )


def _check_line_number(
    finding: ValidatedFinding,
    pr_context: PRContext,
) -> None:
    """Check if the finding's line number is within the diff range.

    Args:
        finding: The finding to validate.
        pr_context: PR context with parsed diff hunks.

    Side effects:
        Sets finding.validation_status and adjusts confidence.
    """
    if finding.line_number is None:
        return  # No line number = nothing to validate

    if not pr_context.parsed_diff:
        return  # No parsed diff available

    # Find the file in the parsed diff
    for diff_file in pr_context.parsed_diff.files:
        if _paths_match(diff_file.path, finding.file_path):
            # Check if line falls within any hunk
            line_in_diff = False
            for hunk in diff_file.hunks:
                # DiffHunk exposes new_start_line / new_line_count.
                # The hunk covers new_start_line .. new_start_line + new_line_count
                # on the RIGHT (post-change) side of the diff.
                hunk_start = hunk.new_start_line
                hunk_end = hunk.new_start_line + hunk.new_line_count
                if hunk_start <= finding.line_number <= hunk_end:
                    line_in_diff = True
                    break

            if not line_in_diff:
                finding.validation_status = ValidationStatus.LINE_NOT_IN_DIFF
                finding.confidence *= 0.5  # Moderate penalty
                finding.validation_notes.append(
                    f"Line {finding.line_number} not within diff range "
                    f"for {finding.file_path}"
                )
                logger.debug(
                    "Finding '%s' line %d not in diff for %s",
                    finding.title,
                    finding.line_number,
                    finding.file_path,
                )
            break


def _strip_diff_markers(patch: str) -> str:
    """Return patch body text with unified-diff noise removed.

    Drops hunk headers (``@@ ... @@``) and file headers (``diff``,
    ``index``, ``+++``, ``---``), strips the leading ``+``/``-``/`` ``
    column from body lines, then lowercases and collapses whitespace so
    that clean LLM-supplied code snippets can be matched against it.
    """
    kept: list[str] = []
    for raw in patch.splitlines():
        if raw.startswith(("@@", "diff ", "index ", "+++", "---")):
            continue
        if raw[:1] in ("+", "-", " "):
            raw = raw[1:]
        kept.append(raw)
    text = "\n".join(kept).lower()
    return " ".join(text.split())


def _evidence_matches(evidence: str, haystack: str) -> bool:
    """Heuristically decide whether ``evidence`` is present in ``haystack``.

    ``haystack`` is the marker-stripped patch text. Matching is
    deliberately forgiving because the model rarely reproduces
    whitespace exactly:

    1. whole normalized snippet is a substring, or
    2. any non-trivial single line of the snippet is a substring, or
    3. >= 60% of the snippet's word tokens (len >= 3) appear.
    """
    norm = " ".join(evidence.lower().split())
    if not norm:
        return False
    if norm in haystack:
        return True

    for line in evidence.lower().splitlines():
        line_norm = " ".join(line.split())
        if len(line_norm) >= 10 and line_norm in haystack:
            return True

    tokens = [t for t in re.findall(r"[a-z0-9_]{3,}", norm)]
    if not tokens:
        return False
    hits = sum(1 for t in set(tokens) if t in haystack)
    return hits / len(set(tokens)) >= 0.6


def _check_evidence(
    finding: ValidatedFinding,
    pr_context: PRContext,
) -> None:
    """Check if the evidence snippet matches actual changed code.

    The evidence is compared against the marker-stripped patch text.
    A verified match boosts confidence; a miss applies a *soft*
    confidence penalty but does NOT hard-fail the finding, because the
    match is heuristic and false negatives are common.

    Args:
        finding: The finding to validate.
        pr_context: PR context with file patches.

    Side effects:
        Adjusts confidence and appends a validation note.
    """
    if not finding.evidence:
        return  # No evidence = nothing to verify

    if not pr_context.files:
        return  # No file data available

    for pr_file in pr_context.files:
        if not _paths_match(pr_file.filename, finding.file_path):
            continue
        if not pr_file.patch:
            return  # No patch available

        haystack = _strip_diff_markers(pr_file.patch)

        if _evidence_matches(finding.evidence, haystack):
            finding.confidence = min(1.0, finding.confidence * 1.1)
            finding.validation_notes.append(
                "Evidence verified against changed code"
            )
            logger.debug("Finding '%s' evidence verified", finding.title)
        else:
            # Soft penalty only — heuristic match, keep status VALID and
            # let the confidence threshold make the final call.
            finding.confidence *= 0.7
            finding.validation_notes.append(
                "Evidence could not be verified against the patch"
            )
            logger.debug("Finding '%s' evidence not verified", finding.title)
        return


# ============================================================================
# Main Validation Pipeline
# ============================================================================


def validate_findings(
    findings: list[dict],
    pr_context: PRContext,
    min_confidence: float = 0.3,
) -> list[ValidatedFinding]:
    """Validate all findings against actual PR content.

    Pipeline:
    1. Convert raw findings to ValidatedFinding objects
    2. Check file existence for each finding
    3. Check line numbers against diff hunks
    4. Verify evidence snippets against file patches
    5. Filter out findings below confidence threshold

    Args:
        findings: Raw finding dicts from the review agents.
        pr_context: PR context with file data and parsed diff.
        min_confidence: Minimum confidence threshold (default 0.3).

    Returns:
        List of ValidatedFinding objects that passed validation.
    """
    validated: list[ValidatedFinding] = []

    for finding_dict in findings:
        try:
            # Create ValidatedFinding
            vf = ValidatedFinding(
                severity=finding_dict.get("severity", "low"),
                category=finding_dict.get("category", "quality"),
                title=finding_dict.get("title", "Untitled"),
                description=finding_dict.get("description", ""),
                file_path=finding_dict.get("file_path", "") or "",
                line_number=coerce_line_number(finding_dict.get("line_number")),
                evidence=finding_dict.get("evidence"),
                confidence=float(finding_dict.get("confidence", 0.5)),
                agent=finding_dict.get("agent", ""),
            )
            vf.original_confidence = vf.confidence

            # Run validation pipeline
            # Skip later checks if an earlier critical check already failed
            _check_file_exists(vf, pr_context)
            if vf.validation_status == ValidationStatus.VALID:
                _check_line_number(vf, pr_context)
            if vf.validation_status == ValidationStatus.VALID:
                _check_evidence(vf, pr_context)

            # Apply final confidence filter
            # Only set LOW_CONFIDENCE if no other validation failure occurred
            if vf.confidence < min_confidence:
                if vf.validation_status == ValidationStatus.VALID:
                    vf.validation_status = ValidationStatus.LOW_CONFIDENCE
                vf.validation_notes.append(
                    f"Confidence {vf.confidence:.2f} below threshold "
                    f"{min_confidence}"
                )

            validated.append(vf)

        except Exception as error:
            # A validator bug must never sink the whole review. Fall back
            # to treating the finding as valid-but-unverified.
            logger.warning(
                "Validation raised for finding %r; passing it through "
                "unvalidated: %s",
                finding_dict.get("title", "Untitled"),
                error,
                exc_info=True,
            )
            fallback = ValidatedFinding(
                severity=str(finding_dict.get("severity", "low")),
                category=str(finding_dict.get("category", "quality")),
                title=str(finding_dict.get("title", "Untitled")),
                description=str(finding_dict.get("description", "")),
                file_path=str(finding_dict.get("file_path", "") or ""),
                line_number=coerce_line_number(
                    finding_dict.get("line_number")
                ),
                evidence=finding_dict.get("evidence"),
                confidence=float(finding_dict.get("confidence", 0.5) or 0.5),
                agent=str(finding_dict.get("agent", "")),
            )
            fallback.validation_notes.append(
                "Validator error — finding not verified"
            )
            validated.append(fallback)

    # Summary logging
    total = len(validated)
    valid = sum(1 for v in validated if v.is_valid)
    invalid = total - valid
    logger.info(
        "Validated %d findings: %d valid, %d filtered",
        total,
        valid,
        invalid,
    )

    return validated
