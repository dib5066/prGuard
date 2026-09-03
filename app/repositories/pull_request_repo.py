"""
Repository for pull requests.

This module contains database operations related to GitHub pull requests.
It inherits common CRUD operations from BaseRepository and adds
pull‑request‑specific operations.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pull_request import PullRequest
from app.repositories.base import BaseRepository


class PullRequestRepository(BaseRepository[PullRequest]):
    """
    Repository for PullRequest records.

    Inherits from BaseRepository:
        - get_by_id()
        - get_all()
        - create()
        - update()
        - delete()

    Adds:
        - get_by_repository_and_number()
        - upsert()
    """

    def __init__(self, session: AsyncSession):
        """
        Create a PullRequestRepository.

        Args:
            session: Active asynchronous database session.
        """
        super().__init__(PullRequest, session)

    # -----------------------------------------------------------------------
    # Find pull request
    # -----------------------------------------------------------------------

    async def get_by_repository(self, repository_id: int) -> list[PullRequest]:
        """
        Get all pull requests for a repository, ordered by creation time (newest first).

        Args:
            repository_id: Database ID of the repository.

        Returns:
            List of pull requests.
        """
        query = (
            select(PullRequest)
            .where(PullRequest.repository_id == repository_id)
            .order_by(PullRequest.created_at.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_repository_and_number(
        self,
        repository_id: int,
        pull_request_number: int,
    ) -> PullRequest | None:
        """
        Find a pull request using its repository ID and PR number.

        A GitHub PR number is only unique inside a repository.

        Args:
            repository_id: Database ID of the repository.
            pull_request_number: GitHub pull request number.

        Returns:
            The pull request if found, else None.
        """
        query = select(PullRequest).where(
            PullRequest.repository_id == repository_id,
            PullRequest.number == pull_request_number,
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    # -----------------------------------------------------------------------
    # Create or update pull request
    # -----------------------------------------------------------------------

    async def upsert(
        self,
        repository_id: int,
        pull_request_number: int,
        title: str,
        state: str,
        base_sha: str,
        head_sha: str,
        user_login: str,
        created_at: datetime,
    ) -> PullRequest:
        """
        Create a new pull request or update an existing one.

        If the pull request already exists for the repository,
        its latest information is updated. Otherwise, a new record is created.

        Args:
            repository_id: Database ID of the repository.
            pull_request_number: GitHub pull request number.
            title: Pull request title.
            state: Pull request state, e.g., "open" or "closed".
            base_sha: Commit SHA of the base branch.
            head_sha: Commit SHA of the pull request branch.
            user_login: GitHub username of the pull request author.
            created_at: Time when the pull request was created.

        Returns:
            The created or updated pull request.
        """
        pull_request = await self.get_by_repository_and_number(
            repository_id,
            pull_request_number,
        )

        if pull_request:
            return await self.update(
                pull_request,
                title=title,
                state=state,
                base_sha=base_sha,
                head_sha=head_sha,
                user_login=user_login,
            )

        return await self.create(
            repository_id=repository_id,
            number=pull_request_number,
            title=title,
            state=state,
            base_sha=base_sha,
            head_sha=head_sha,
            user_login=user_login,
            created_at=created_at,
        )


# Alias for convenience
PullRequestRepo = PullRequestRepository