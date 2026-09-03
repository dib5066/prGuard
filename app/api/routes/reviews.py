"""
Review API endpoints.

Provides REST API for retrieving review details, findings, and agent metrics
consumed by the Next.js frontend review detail page.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.auth.deps import get_current_user
from app.core.database import AsyncSessionLocal
from app.core.events import subscribe, subscriber_count
from app.models.user import User
from app.repositories.repository_repo import RepositoryRepo
from app.repositories.review_repo import ReviewRepo, FindingRepo, ReviewRunRepo
from app.api.schemas import (
    ReviewResponse,
    FindingResponse,
    ReviewRunResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reviews", tags=["reviews"])

# States in which no further progress events will be produced.
_TERMINAL_STATUSES = {"COMPLETED", "FAILED"}
_HEARTBEAT_SECONDS = 15


async def _load_owned_review(session, review_id: int, user_id: int):
    """Fetch a review, 404-ing if it isn't the user's."""
    review = await ReviewRepo(session).get_by_id(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    owned = await RepositoryRepo(session).owned_repo_ids(user_id)
    if review.repository_id is None or review.repository_id not in owned:
        raise HTTPException(status_code=404, detail="Review not found")
    return review


@router.get("/{review_id}", response_model=ReviewResponse)
async def get_review(
    review_id: int, user: User = Depends(get_current_user)
):
    """
    Get review details with severity breakdown.

    Returns the review metadata along with finding counts by severity.
    """
    async with AsyncSessionLocal() as session:
        finding_repo = FindingRepo(session)

        review = await _load_owned_review(session, review_id, user.id)

        findings = await finding_repo.get_by_review(review_id)
        # Counts reflect what was actually published to GitHub. While a
        # review is still running nothing is published yet, so fall back to
        # all findings so the UI isn't empty mid-run.
        published = [f for f in findings if f.is_published]
        counted = published or (
            findings if review.status not in _TERMINAL_STATUSES else []
        )
        severity_counts: dict[str, int] = {}
        for f in counted:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        return ReviewResponse(
            id=review.id,
            pull_request_id=review.pull_request_id,
            status=review.status,
            error_message=review.error_message,
            created_at=review.created_at,
            completed_at=review.completed_at,
            finding_count=len(counted),
            severity_counts=severity_counts,
        )


@router.get("/{review_id}/findings", response_model=list[FindingResponse])
async def list_review_findings(
    review_id: int,
    published_only: bool = True,
    user: User = Depends(get_current_user),
):
    """
    List findings for a review.

    By default returns only the findings that passed evidence validation
    and were published to GitHub (so the dashboard matches the PR review).
    Pass ``?published_only=false`` to also see findings the validator
    filtered out.
    """
    async with AsyncSessionLocal() as session:
        finding_repo = FindingRepo(session)

        review = await _load_owned_review(session, review_id, user.id)

        findings = await finding_repo.get_by_review(review_id)

        # Only hide filtered findings once the review has finished — while
        # it runs, nothing is published yet.
        if published_only and review.status in _TERMINAL_STATUSES:
            findings = [f for f in findings if f.is_published]

        return [
            FindingResponse(
                id=f.id,
                review_id=f.review_id,
                severity=f.severity,
                category=f.category,
                title=f.title,
                description=f.description,
                file_path=f.file_path,
                line_number=f.line_number,
                evidence=f.evidence,
                confidence=float(f.confidence),
                validation_status=f.validation_status,
                is_published=f.is_published,
                created_at=f.created_at,
            )
            for f in findings
        ]


@router.get("/{review_id}/metrics", response_model=list[ReviewRunResponse])
async def list_review_metrics(
    review_id: int, user: User = Depends(get_current_user)
):
    """
    List agent metrics for a review.

    Returns per-agent latency and token usage data.
    """
    async with AsyncSessionLocal() as session:
        run_repo = ReviewRunRepo(session)

        await _load_owned_review(session, review_id, user.id)

        runs = await run_repo.get_by_review(review_id)

        return [
            ReviewRunResponse(
                id=r.id,
                review_id=r.review_id,
                agent_name=r.agent_name,
                latency_ms=r.latency_ms,
                tokens_used=r.tokens_used,
                created_at=r.created_at,
            )
            for r in runs
        ]


# ---------------------------------------------------------------------------
# Review Phase Status (for frontend polling)
# ---------------------------------------------------------------------------


@router.get("/{review_id}/status")
async def get_review_status(
    review_id: int, user: User = Depends(get_current_user)
):
    """
    Get the current phase and progress message of a review.

    The frontend polls this endpoint every 2 seconds while a review
    is in progress to display a loading indicator with the current phase.
    """
    async with AsyncSessionLocal() as session:
        review = await _load_owned_review(session, review_id, user.id)

        return {
            "id": review.id,
            "status": review.status,
            "current_phase": review.current_phase,
            "phase_message": review.phase_message,
            "created_at": review.created_at,
            "completed_at": review.completed_at,
        }


# ---------------------------------------------------------------------------
# Live progress stream (Server-Sent Events)
# ---------------------------------------------------------------------------


def _sse(event: dict) -> str:
    """Format a dict as one SSE ``data:`` frame."""
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.get("/{review_id}/events")
async def stream_review_events(
    review_id: int,
    request: Request,
    user: User = Depends(get_current_user),
):
    """Stream review progress to the browser as Server-Sent Events.

    The client (``EventSource``) receives:

    - one ``snapshot`` event with the current DB state on connect,
    - ``phase`` events as the worker advances through the pipeline,
    - ``agents_started`` / ``agent`` events for per-agent progress,
    - ``findings_ready`` once findings are stored,
    - a final ``done`` event, after which the stream closes.

    If the review is already finished when the client connects, it gets
    the snapshot + a ``done`` event and the stream ends immediately.
    """

    async def event_generator():
        # Subscribe BEFORE reading the snapshot so no event raised between
        # the snapshot read and the live loop can be missed.
        with subscribe(review_id) as queue:
            logger.info(
                "SSE client attached to review %d (%d total)",
                review_id,
                subscriber_count(review_id),
            )

            # 1. Initial snapshot straight from the database (owner-checked).
            async with AsyncSessionLocal() as session:
                try:
                    review = await _load_owned_review(
                        session, review_id, user.id
                    )
                except HTTPException:
                    review = None

            if review is None:
                yield _sse({"type": "error", "message": "Review not found"})
                return

            yield _sse(
                {
                    "type": "snapshot",
                    "status": review.status,
                    "phase": review.current_phase,
                    "message": review.phase_message,
                }
            )

            if review.status in _TERMINAL_STATUSES:
                yield _sse({"type": "done", "status": review.status})
                return

            # 2. Live events until the review finishes or the client leaves.
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=_HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Comment line keeps the connection (and any proxy) alive.
                    yield ": ping\n\n"
                    continue

                yield _sse(event)
                if event.get("type") == "done":
                    break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering (nginx / ngrok) so events flush live.
            "X-Accel-Buffering": "no",
        },
    )
