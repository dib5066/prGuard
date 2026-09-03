"""
Dashboard statistics — aggregate counts scoped to the logged-in user.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from app.auth.deps import get_current_user
from app.core.database import AsyncSessionLocal
from app.models.pull_request import PullRequest
from app.models.repository import Repository
from app.models.review import Finding, Review, ReviewRun
from app.models.user import User
from app.repositories.repository_repo import RepositoryRepo
from app.api.schemas import DashboardStatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stats", tags=["stats"])

_EMPTY = DashboardStatsResponse(
    total_repos=0, indexed_repos=0, total_prs=0, total_reviews=0,
    completed_reviews=0, failed_reviews=0, total_findings=0,
    critical_findings=0, high_findings=0, medium_findings=0, low_findings=0,
    avg_latency_ms=None, avg_tokens_used=None,
)


@router.get("", response_model=DashboardStatsResponse)
async def get_dashboard_stats(user: User = Depends(get_current_user)):
    async with AsyncSessionLocal() as session:
        repo_ids = list(await RepositoryRepo(session).owned_repo_ids(user.id))
        if not repo_ids:
            return _EMPTY

        async def scalar(stmt) -> int:
            return (await session.scalar(stmt)) or 0

        repo_count = await scalar(
            select(func.count(Repository.id)).where(Repository.id.in_(repo_ids))
        )
        indexed_count = await scalar(
            select(func.count(Repository.id)).where(
                Repository.id.in_(repo_ids), Repository.is_indexed.is_(True)
            )
        )
        pr_count = await scalar(
            select(func.count(PullRequest.id)).where(
                PullRequest.repository_id.in_(repo_ids)
            )
        )

        rv = select(func.count(Review.id)).where(Review.repository_id.in_(repo_ids))
        review_count = await scalar(rv)
        completed_count = await scalar(rv.where(Review.status == "COMPLETED"))
        failed_count = await scalar(rv.where(Review.status == "FAILED"))

        fbase = (
            select(func.count(Finding.id))
            .join(Review, Review.id == Finding.review_id)
            .where(Review.repository_id.in_(repo_ids), Finding.is_published.is_(True))
        )
        finding_count = await scalar(fbase)
        by_sev = {
            sev: await scalar(fbase.where(Finding.severity == sev))
            for sev in ("critical", "high", "medium", "low")
        }

        metric_join = (
            select(func.avg(ReviewRun.latency_ms))
            .join(Review, Review.id == ReviewRun.review_id)
            .where(Review.repository_id.in_(repo_ids))
        )
        avg_latency = await session.scalar(metric_join)
        avg_tokens = await session.scalar(
            select(func.avg(ReviewRun.tokens_used))
            .join(Review, Review.id == ReviewRun.review_id)
            .where(Review.repository_id.in_(repo_ids))
        )

        return DashboardStatsResponse(
            total_repos=repo_count,
            indexed_repos=indexed_count,
            total_prs=pr_count,
            total_reviews=review_count,
            completed_reviews=completed_count,
            failed_reviews=failed_count,
            total_findings=finding_count,
            critical_findings=by_sev["critical"],
            high_findings=by_sev["high"],
            medium_findings=by_sev["medium"],
            low_findings=by_sev["low"],
            avg_latency_ms=int(avg_latency) if avg_latency else None,
            avg_tokens_used=int(avg_tokens) if avg_tokens else None,
        )
