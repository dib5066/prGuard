"""
Pull request API endpoints (scoped to the logged-in user).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import get_current_user
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.repositories.pull_request_repo import PullRequestRepo
from app.repositories.repository_repo import RepositoryRepo
from app.repositories.review_repo import FindingRepo, ReviewRepo
from app.api.schemas import PullRequestResponse, ReviewResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prs", tags=["pull_requests"])


async def _load_owned_pr(session, pr_id: int, user_id: int):
    pr = await PullRequestRepo(session).get_by_id(pr_id)
    if pr is None:
        raise HTTPException(status_code=404, detail="Pull request not found")
    owned = await RepositoryRepo(session).owned_repo_ids(user_id)
    if pr.repository_id not in owned:
        raise HTTPException(status_code=404, detail="Pull request not found")
    return pr


@router.get("/{pr_id}", response_model=PullRequestResponse)
async def get_pull_request(
    pr_id: int, user: User = Depends(get_current_user)
):
    async with AsyncSessionLocal() as session:
        pr = await _load_owned_pr(session, pr_id, user.id)
        reviews = await ReviewRepo(session).get_by_pull_request(pr.id)
        latest = reviews[0] if reviews else None
        return PullRequestResponse(
            id=pr.id,
            repository_id=pr.repository_id,
            number=pr.number,
            title=pr.title,
            state=pr.state,
            base_sha=pr.base_sha,
            head_sha=pr.head_sha,
            user_login=pr.user_login,
            created_at=pr.created_at,
            latest_review_status=latest.status if latest else None,
            latest_review_id=latest.id if latest else None,
            review_count=len(reviews),
        )


@router.get("/{pr_id}/reviews", response_model=list[ReviewResponse])
async def list_pr_reviews(
    pr_id: int, user: User = Depends(get_current_user)
):
    """All reviews for a pull request, newest first."""
    async with AsyncSessionLocal() as session:
        pr = await _load_owned_pr(session, pr_id, user.id)
        review_repo = ReviewRepo(session)
        finding_repo = FindingRepo(session)

        reviews = await review_repo.get_by_pull_request(pr.id)
        result = []
        for review in reviews:
            findings = await finding_repo.get_by_review(review.id)
            published = [f for f in findings if f.is_published]
            counted = published or (
                findings if review.status not in ("COMPLETED", "FAILED") else []
            )
            severity_counts: dict[str, int] = {}
            for f in counted:
                severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
            result.append(
                ReviewResponse(
                    id=review.id,
                    pull_request_id=review.pull_request_id,
                    status=review.status,
                    error_message=review.error_message,
                    created_at=review.created_at,
                    completed_at=review.completed_at,
                    finding_count=len(counted),
                    severity_counts=severity_counts,
                )
            )

        logger.info("Listed %d reviews for PR %d", len(result), pr_id)
        return result
