"""Repository for application users (GitHub OAuth identities)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.base import BaseRepository


class UserRepo(BaseRepository[User]):
    def __init__(self, session: AsyncSession):
        super().__init__(User, session)

    async def get_by_github_id(self, github_user_id: int) -> User | None:
        query = select(User).where(User.github_user_id == github_user_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def upsert_from_github(
        self,
        *,
        github_user_id: int,
        github_username: str | None,
        email: str | None,
        avatar_url: str | None,
    ) -> User:
        """Create the user on first login, refresh profile fields on later ones."""
        user = await self.get_by_github_id(github_user_id)
        if user is not None:
            return await self.update(
                user,
                github_username=github_username,
                email=email,
                avatar_url=avatar_url,
            )
        return await self.create(
            github_user_id=github_user_id,
            github_username=github_username,
            email=email,
            avatar_url=avatar_url,
        )


UserRepository = UserRepo
