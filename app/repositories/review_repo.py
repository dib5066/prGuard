"""
Repositories for reviews, findings, and review runs.

These classes handle database operations for the PRGuard review system.

- ReviewRepository: Handles the review lifecycle.
- FindingRepository: Handles issues found during a review.
- ReviewRunRepository: Stores information about individual AI agent runs.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.review import Finding, Review, ReviewRun
from app.repositories.base import BaseRepository


# ============================================================================
# Review Repository
# ============================================================================

class ReviewRepo(BaseRepository[Review]):
    """
    Handles database operations for reviews.

    A review represents one complete PRGuard review process.

    Lifecycle:
        PENDING → RUNNING → COMPLETED
        RUNNING → FAILED (if error occurs)
    """

    def __init__(self, session: AsyncSession):
        """Create a ReviewRepository."""
        super().__init__(Review, session)

    # ------------------------------------------------------------------------
    # Find latest review
    # ------------------------------------------------------------------------

    async def get_latest_by_pull_request(self, pull_request_id: int) -> Review | None:
        """
        Get the latest review for a pull request.

        A pull request can have multiple reviews (e.g., when the PR changes).

        Args:
            pull_request_id: Database ID of the pull request.

        Returns:
            The latest review if one exists, else None.
        """
        query = (
            select(Review)
            .where(Review.pull_request_id == pull_request_id)
            .order_by(Review.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------------
    # Get reviews by pull request
    # ------------------------------------------------------------------------

    async def get_by_pull_request(self, pull_request_id: int) -> list[Review]:
        """
        Get all reviews for a pull request, ordered by creation time (newest first).

        Args:
            pull_request_id: Database ID of the pull request.

        Returns:
            List of reviews.
        """
        query = (
            select(Review)
            .where(Review.pull_request_id == pull_request_id)
            .order_by(Review.created_at.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    # ------------------------------------------------------------------------
    # Get review with findings
    # ------------------------------------------------------------------------

    async def get_by_id_with_findings(self, review_id: int) -> tuple[Review | None, list[Finding]]:
        """
        Get a review together with all findings created by that review.

        Args:
            review_id: Database ID of the review.

        Returns:
            A tuple (review, findings). If the review does not exist,
            returns (None, []).
        """
        review = await self.get_by_id(review_id)
        if not review:
            return None, []

        query = select(Finding).where(Finding.review_id == review_id).order_by(Finding.created_at)
        result = await self.session.execute(query)
        findings = list(result.scalars().all())
        return review, findings

    # ------------------------------------------------------------------------
    # Create review
    # ------------------------------------------------------------------------

    async def create_review(
        self,
        pull_request_id: int,
        head_sha: str | None = None,
        repository_id: int | None = None,
    ) -> Review:
        """
        Create a new review with status PENDING.

        Args:
            pull_request_id: Database ID of the pull request.
            head_sha: Commit SHA this review targets (enables reuse).
            repository_id: Denormalized repo id for per-user scoping.

        Returns:
            The newly created review.
        """
        return await self.create(
            pull_request_id=pull_request_id,
            status="PENDING",
            head_sha=head_sha,
            repository_id=repository_id,
        )

    # ------------------------------------------------------------------------
    # Reusable review lookup (idempotency by commit)
    # ------------------------------------------------------------------------

    async def get_reusable_review(
        self,
        pull_request_id: int,
        head_sha: str,
    ) -> Review | None:
        """
        Find the latest review for this PR + commit that can be reused
        instead of re-running the agents.

        A review is reusable when it targets the same ``head_sha`` and is
        either still ``RUNNING`` (another worker owns it) or ``COMPLETED``
        (re-publish only). ``FAILED`` reviews are never reused.
        """
        query = (
            select(Review)
            .where(
                Review.pull_request_id == pull_request_id,
                Review.head_sha == head_sha,
                Review.status.in_(("RUNNING", "COMPLETED")),
            )
            .order_by(Review.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------------
    # Review status: RUNNING
    # ------------------------------------------------------------------------

    async def mark_running(self, review: Review) -> Review:
        """Mark a review as RUNNING (AI review process started)."""
        return await self.update(review, status="RUNNING")

    # ------------------------------------------------------------------------
    # Review phase tracking
    # ------------------------------------------------------------------------

    async def update_phase(
        self,
        review: Review,
        phase: str,
        message: str,
    ) -> Review:
        """
        Update the current phase of a running review.

        Args:
            review: The review to update.
            phase: Short phase identifier (e.g. 'fetching_pr', 'running_agents').
            message: Human-readable progress message.

        Returns:
            Updated review.
        """
        return await self.update(
            review,
            current_phase=phase,
            phase_message=message,
        )

    # ------------------------------------------------------------------------
    # Review status: COMPLETED
    # ------------------------------------------------------------------------

    async def mark_completed(self, review: Review) -> Review:
        """Mark a review as COMPLETED and store the completion time."""
        return await self.update(review, status="COMPLETED", completed_at=datetime.utcnow())

    # ------------------------------------------------------------------------
    # Review status: FAILED
    # ------------------------------------------------------------------------

    async def mark_failed(self, review: Review, error_message: str) -> Review:
        """
        Mark a review as FAILED and store the error and completion time.

        Args:
            review: The review that failed.
            error_message: Description of the error.

        Returns:
            Updated review.
        """
        return await self.update(
            review,
            status="FAILED",
            error_message=error_message,
            completed_at=datetime.utcnow(),
        )


# ============================================================================
# Finding Repository
# ============================================================================

class FindingRepo(BaseRepository[Finding]):
    """
    Handles database operations for review findings.

    A finding represents one issue discovered by an AI review agent,
    such as a security issue, bug, code quality issue, etc.
    """

    def __init__(self, session: AsyncSession):
        """Create a FindingRepository."""
        super().__init__(Finding, session)

    # ------------------------------------------------------------------------
    # Get findings
    # ------------------------------------------------------------------------

    async def get_by_review(self, review_id: int) -> list[Finding]:
        """
        Get all findings belonging to a review, ordered by creation time.

        Args:
            review_id: Database ID of the review.

        Returns:
            List of findings.
        """
        query = select(Finding).where(Finding.review_id == review_id).order_by(Finding.created_at)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    # ------------------------------------------------------------------------
    # Create finding
    # ------------------------------------------------------------------------

    async def create_finding(
        self,
        review_id: int,
        *,
        severity: str,
        category: str,
        title: str,
        description: str,
        file_path: str,
        confidence: float,
        line_number: int | None = None,
        evidence: str | None = None,
    ) -> Finding:
        """
        Create a finding for a review.

        Args:
            review_id: Database ID of the review.
            severity: Severity level (critical, high, medium, low).
            category: Type of issue (e.g., correctness, security, quality).
            title: Short title describing the issue.
            description: Detailed explanation.
            file_path: Path of the file containing the issue.
            confidence: AI confidence score (0.0–1.0).
            line_number: Line number (optional).
            evidence: Supporting code or evidence (optional).

        Returns:
            The newly created finding.
        """
        return await self.create(
            review_id=review_id,
            severity=severity,
            category=category,
            title=title,
            description=description,
            file_path=file_path,
            line_number=line_number,
            evidence=evidence,
            confidence=confidence,
        )

    # ------------------------------------------------------------------------
    # Update finding confidence (Phase 8)
    # ------------------------------------------------------------------------

    async def update_finding_confidence(
        self,
        finding: Finding,
        new_confidence: float,
    ) -> Finding:
        """Update a finding's confidence score after validation.

        Args:
            finding: The finding to update.
            new_confidence: New confidence value (0.0–1.0).

        Returns:
            Updated finding.
        """
        return await self.update(finding, confidence=new_confidence)

    # ------------------------------------------------------------------------
    # Persist evidence-validation results (Phase 8 write-back)
    # ------------------------------------------------------------------------

    async def update_finding_validation(
        self,
        finding: Finding,
        *,
        confidence: float,
        validation_status: str,
        validation_notes: list | None,
        is_published: bool,
    ) -> Finding:
        """Persist the validator's calibrated confidence + status on a finding.

        Keeps the dashboard in sync with what was actually published to
        GitHub (``is_published`` is True only for findings that passed
        validation and became inline comments).
        """
        return await self.update(
            finding,
            confidence=confidence,
            validation_status=validation_status,
            validation_notes=validation_notes,
            is_published=is_published,
        )

    # ------------------------------------------------------------------------
    # Delete invalid findings (Phase 8)
    # ------------------------------------------------------------------------

    async def delete_invalid_findings(
        self,
        review_id: int,
        max_confidence: float = 0.3,
    ) -> int:
        """Delete findings below the confidence threshold.

        Args:
            review_id: The review to clean up.
            max_confidence: Maximum confidence to keep
                (findings at or below are deleted).

        Returns:
            Number of findings deleted.
        """
        query = select(Finding).where(
            Finding.review_id == review_id,
            Finding.confidence <= max_confidence,
        )
        result = await self.session.execute(query)
        findings_to_delete = list(result.scalars().all())

        for finding in findings_to_delete:
            await self.delete(finding)

        return len(findings_to_delete)


# ============================================================================
# Review Run Repository
# ============================================================================

class ReviewRunRepo(BaseRepository[ReviewRun]):
    """
    Handles database operations for individual review runs.

    A review run represents one AI agent execution (e.g., security agent,
    correctness agent, quality agent) within a review.
    """

    def __init__(self, session: AsyncSession):
        """Create a ReviewRunRepository."""
        super().__init__(ReviewRun, session)

    # ------------------------------------------------------------------------
    # Get review runs by review
    # ------------------------------------------------------------------------

    async def get_by_review(self, review_id: int) -> list[ReviewRun]:
        """
        Get all review runs (agent metrics) for a review.

        Args:
            review_id: Database ID of the review.

        Returns:
            List of review runs.
        """
        query = (
            select(ReviewRun)
            .where(ReviewRun.review_id == review_id)
            .order_by(ReviewRun.created_at)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    # ------------------------------------------------------------------------
    # Create review run
    # ------------------------------------------------------------------------

    async def create_run(
        self,
        review_id: int,
        *,
        agent_name: str,
        latency_ms: int | None = None,
        tokens_used: int | None = None,
    ) -> ReviewRun:
        """
        Store information about one AI agent execution.

        Args:
            review_id: Database ID of the review.
            agent_name: Name of the AI agent (e.g., security_agent).
            latency_ms: Time taken by the agent in milliseconds.
            tokens_used: Number of LLM tokens used.

        Returns:
            The newly created review run.
        """
        return await self.create(
            review_id=review_id,
            agent_name=agent_name,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )


# Aliases for backward compatibility
ReviewRepository = ReviewRepo
FindingRepository = FindingRepo
ReviewRunRepository = ReviewRunRepo