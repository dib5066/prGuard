"""
Repository API endpoints (scoped to the logged-in user's installations).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import get_current_user
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.repositories.repository_repo import RepositoryRepo
from app.repositories.pull_request_repo import PullRequestRepo
from app.repositories.review_repo import ReviewRepo
from app.api.schemas import PullRequestResponse, RepoResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/repos", tags=["repos"])


def _repo_payload(repo, pr_count: int, review_count: int) -> RepoResponse:
    return RepoResponse(
        id=repo.id,
        github_id=repo.github_id,
        name=repo.name,
        full_name=repo.full_name,
        installation_id=repo.installation_id,
        is_indexed=repo.is_indexed,
        last_indexed_at=repo.last_indexed_at,
        created_at=repo.created_at,
        pr_count=pr_count,
        review_count=review_count,
    )


@router.get("", response_model=list[RepoResponse])
async def list_repositories(user: User = Depends(get_current_user)):
    """List the current user's repositories with PR and review counts."""
    async with AsyncSessionLocal() as session:
        repo_repo = RepositoryRepo(session)
        pr_repo = PullRequestRepo(session)
        review_repo = ReviewRepo(session)

        repos = await repo_repo.list_for_user(user.id)
        result = []
        for repo in repos:
            prs = await pr_repo.get_by_repository(repo.id)
            review_count = 0
            for pr in prs:
                review_count += len(await review_repo.get_by_pull_request(pr.id))
            result.append(_repo_payload(repo, len(prs), review_count))

        logger.info("Listed %d repositories for user %d", len(result), user.id)
        return result


@router.get("/{repo_id}", response_model=RepoResponse)
async def get_repository(
    repo_id: int, user: User = Depends(get_current_user)
):
    async with AsyncSessionLocal() as session:
        repo_repo = RepositoryRepo(session)
        pr_repo = PullRequestRepo(session)
        review_repo = ReviewRepo(session)

        repo = await repo_repo.get_by_id(repo_id)
        if not repo or repo.id not in await repo_repo.owned_repo_ids(user.id):
            raise HTTPException(status_code=404, detail="Repository not found")

        prs = await pr_repo.get_by_repository(repo.id)
        review_count = 0
        for pr in prs:
            review_count += len(await review_repo.get_by_pull_request(pr.id))
        return _repo_payload(repo, len(prs), review_count)


@router.get("/{repo_id}/prs", response_model=list[PullRequestResponse])
async def list_repository_prs(
    repo_id: int, user: User = Depends(get_current_user)
):
    """List a repository's pull requests with their latest review."""
    async with AsyncSessionLocal() as session:
        repo_repo = RepositoryRepo(session)
        pr_repo = PullRequestRepo(session)
        review_repo = ReviewRepo(session)

        repo = await repo_repo.get_by_id(repo_id)
        if not repo or repo.id not in await repo_repo.owned_repo_ids(user.id):
            raise HTTPException(status_code=404, detail="Repository not found")

        prs = await pr_repo.get_by_repository(repo.id)
        result = []
        for pr in prs:
            reviews = await review_repo.get_by_pull_request(pr.id)
            latest = reviews[0] if reviews else None
            result.append(
                PullRequestResponse(
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
            )

        logger.info("Listed %d PRs for repo %d", len(result), repo_id)
        return result
