"""
Repository for GitHub repositories.

This module contains database operations related to GitHub repositories.
It inherits common CRUD operations from BaseRepository and adds
repository‑specific operations.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.installation import GitHubInstallation
from app.models.repository import Repository
from app.repositories.base import BaseRepository


class RepositoryRepository(BaseRepository[Repository]):
    """
    Repository for Repository records.

    Inherits from BaseRepository:
        - get_by_id()
        - get_all()
        - create()
        - update()
        - delete()

    Adds:
        - get_by_full_name()
        - get_by_github_id()
        - upsert()
    """

    def __init__(self, session: AsyncSession):
        """
        Create a RepositoryRepository.

        Args:
            session: Active asynchronous database session.
        """
        super().__init__(Repository, session)

    # -----------------------------------------------------------------------
    # Find repository by full name
    # -----------------------------------------------------------------------

    async def get_by_full_name(self, full_name: str) -> Repository | None:
        """
        Find a repository using its full name.

        Example:
            octocat/Hello-World

        Args:
            full_name: GitHub repository name in owner/repository format.

        Returns:
            The repository if found, else None.
        """
        query = select(Repository).where(Repository.full_name == full_name)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    # -----------------------------------------------------------------------
    # Find repository by GitHub ID
    # -----------------------------------------------------------------------

    async def get_by_github_id(self, github_id: int) -> Repository | None:
        """
        Find a repository using its GitHub repository ID.

        Args:
            github_id: Unique ID assigned to the repository by GitHub.

        Returns:
            The repository if found, else None.
        """
        query = select(Repository).where(Repository.github_id == github_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    # -----------------------------------------------------------------------
    # Create or update repository
    # -----------------------------------------------------------------------

    async def upsert(
        self,
        github_id: int,
        name: str,
        full_name: str,
        installation_id: int,
    ) -> Repository:
        """
        Create a new repository or update an existing one.

        If the repository already exists, its latest information is updated.
        Otherwise, a new repository is created.

        Args:
            github_id: Unique GitHub repository ID.
            name: Repository name.
            full_name: Repository name in owner/repository format.
            installation_id: GitHub App installation ID.

        Returns:
            The created or updated repository.
        """
        repository = await self.get_by_github_id(github_id)

        if repository:
            return await self.update(
                repository,
                name=name,
                full_name=full_name,
                installation_id=installation_id,
            )

        return await self.create(
            github_id=github_id,
            name=name,
            full_name=full_name,
            installation_id=installation_id,
        )


    # -----------------------------------------------------------------------
    # Per-user scoping
    # -----------------------------------------------------------------------

    async def list_for_user(self, user_id: int) -> list[Repository]:
        """Repositories reachable by a user through their installations."""
        query = (
            select(Repository)
            .join(
                GitHubInstallation,
                GitHubInstallation.installation_id == Repository.installation_id,
            )
            .where(
                GitHubInstallation.user_id == user_id,
                GitHubInstallation.deleted_at.is_(None),
            )
            .order_by(Repository.full_name)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def owned_repo_ids(self, user_id: int) -> set[int]:
        """Set of repository PKs the user may see (for cheap 404 checks)."""
        return {r.id for r in await self.list_for_user(user_id)}

    # -----------------------------------------------------------------------
    # Mark repository as indexed
    # -----------------------------------------------------------------------

    async def mark_indexed(self, repo: Repository) -> Repository:
        """
        Mark a repository as successfully indexed.

        Updates is_indexed to True and sets last_indexed_at to now.

        Args:
            repo: Repository record to update.

        Returns:
            Updated repository record.
        """
        from datetime import datetime, timezone

        return await self.update(
            repo,
            is_indexed=True,
            last_indexed_at=datetime.now(timezone.utc),
        )


# Alias for convenience
RepositoryRepo = RepositoryRepository