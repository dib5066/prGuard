"""
Pydantic schemas for API responses.

These schemas define the shape of JSON responses returned by
the backend API endpoints consumed by the Next.js frontend.
"""

from datetime import datetime
from pydantic import BaseModel


# ============================================================================
# Repository Schemas
# ============================================================================


class RepoResponse(BaseModel):
    """Repository API response."""

    id: int
    github_id: int
    name: str
    full_name: str
    installation_id: int
    is_indexed: bool
    last_indexed_at: datetime | None
    created_at: datetime
    pr_count: int = 0
    review_count: int = 0


# ============================================================================
# Pull Request Schemas
# ============================================================================


class PullRequestResponse(BaseModel):
    """Pull request API response."""

    id: int
    repository_id: int
    number: int
    title: str
    state: str
    base_sha: str
    head_sha: str
    user_login: str
    created_at: datetime
    latest_review_status: str | None = None
    latest_review_id: int | None = None
    review_count: int = 0


# ============================================================================
# Review Schemas
# ============================================================================


class ReviewResponse(BaseModel):
    """Review API response."""

    id: int
    pull_request_id: int
    status: str
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None
    finding_count: int = 0
    severity_counts: dict[str, int] = {}


# ============================================================================
# Finding Schemas
# ============================================================================


class FindingResponse(BaseModel):
    """Finding API response."""

    id: int
    review_id: int
    severity: str
    category: str
    title: str
    description: str
    file_path: str
    line_number: int | None
    evidence: str | None
    confidence: float
    validation_status: str | None = None
    is_published: bool = False
    created_at: datetime


# ============================================================================
# Review Run (Agent Metrics) Schemas
# ============================================================================


class ReviewRunResponse(BaseModel):
    """Review run (agent metrics) API response."""

    id: int
    review_id: int
    agent_name: str
    latency_ms: int | None
    tokens_used: int | None
    created_at: datetime


# ============================================================================
# Dashboard Stats Schema
# ============================================================================


class DashboardStatsResponse(BaseModel):
    """Dashboard aggregate statistics."""

    total_repos: int
    indexed_repos: int
    total_prs: int
    total_reviews: int
    completed_reviews: int
    failed_reviews: int
    total_findings: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int
    avg_latency_ms: int | None
    avg_tokens_used: int | None
