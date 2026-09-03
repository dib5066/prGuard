"""
Repository for GitHub App installations.

This module contains database operations related to GitHub App installations.
It inherits common CRUD operations from BaseRepository and adds
installation‑specific operations.
"""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.installation import GitHubInstallation
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class InstallationRepository(BaseRepository[GitHubInstallation]):
    """
    Repository for GitHubInstallation records.

    Inherits from BaseRepository:
        - get_by_id()
        - get_all()
        - create()
        - update()
        - delete()

    Adds:
        - get_by_installation_id()
        - upsert()
    """

    def __init__(self, session: AsyncSession):
        """
        Create an InstallationRepository.

        Args:
            session: Active asynchronous database session.
        """
        super().__init__(GitHubInstallation, session)

    # -----------------------------------------------------------------------
    # Find installation by GitHub installation ID
    # -----------------------------------------------------------------------

    async def get_by_installation_id(self, installation_id: int) -> Optional[GitHubInstallation]:
        """
        Find a GitHub App installation by its installation ID.

        Args:
            installation_id: GitHub App installation ID.

        Returns:
            The installation record if found, else None.
        """
        query = select(GitHubInstallation).where(
            GitHubInstallation.installation_id == installation_id
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    # -----------------------------------------------------------------------
    # Create or update installation
    # -----------------------------------------------------------------------
    # If exists → UPDATE
    # If doesn't exist → INSERT
    async def upsert(
        self,
        installation_id: int,
        account_name: str,
        account_type: str,
        *,
        user_id: int | None = None,
        account_id: int | None = None,
        installer_github_id: int | None = None,
        clear_deleted: bool = False,
    ) -> GitHubInstallation:
        """
        Create a new installation or update an existing one.

        ``user_id`` / ``account_id`` are only written when provided, and an
        existing non-null ``user_id`` is never overwritten with ``None`` —
        so the lazy upsert inside the review worker can't unlink an
        installation that the setup callback already claimed.

        Args:
            installation_id: GitHub App installation ID.
            account_name: GitHub username or organization name.
            account_type: "User" or "Organization".
            user_id: PRGuard user that owns this installation (setup flow).
            account_id: GitHub numeric account id.
            clear_deleted: reset ``deleted_at`` (re-install of a prior one).

        Returns:
            The created or updated installation record.
        """
        installation = await self.get_by_installation_id(installation_id)

        if installation:
            fields: dict = {
                "account_name": account_name,
                "account_type": account_type,
            }
            if user_id is not None:
                fields["user_id"] = user_id
            if account_id is not None:
                fields["account_id"] = account_id
            if installer_github_id is not None and installation.installer_github_id is None:
                fields["installer_github_id"] = installer_github_id
            if clear_deleted:
                fields["deleted_at"] = None
                fields["suspended_at"] = None
            return await self.update(installation, **fields)

        return await self.create(
            installation_id=installation_id,
            account_name=account_name,
            account_type=account_type,
            user_id=user_id,
            account_id=account_id,
            installer_github_id=installer_github_id,
        )

    # -----------------------------------------------------------------------
    # Per-user queries
    # -----------------------------------------------------------------------

    async def list_for_user(self, user_id: int) -> list[GitHubInstallation]:
        """Active (not deleted) installations owned by a user."""
        query = select(GitHubInstallation).where(
            GitHubInstallation.user_id == user_id,
            GitHubInstallation.deleted_at.is_(None),
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def mark_deleted(self, installation: GitHubInstallation) -> GitHubInstallation:
        from datetime import datetime, timezone

        return await self.update(
            installation, deleted_at=datetime.now(timezone.utc)
        )

    async def list_unclaimed(self) -> list[GitHubInstallation]:
        """Active installations not yet linked to any user."""
        query = select(GitHubInstallation).where(
            GitHubInstallation.user_id.is_(None),
            GitHubInstallation.deleted_at.is_(None),
        ).order_by(GitHubInstallation.created_at.desc())
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def claim(
        self, installation: GitHubInstallation, user_id: int
    ) -> GitHubInstallation:
        return await self.update(installation, user_id=user_id, deleted_at=None)


# Alias used elsewhere in the codebase
InstallationRepo = InstallationRepository