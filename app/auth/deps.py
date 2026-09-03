"""FastAPI auth dependencies — resolve the logged-in user from the session cookie."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import decode_session_token
from app.core.config import settings
from app.core.database import get_db
from app.models.user import User
from app.repositories.user_repo import UserRepo

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
)


async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> User:
    """The logged-in user, or 401."""
    token = request.cookies.get(settings.SESSION_COOKIE_NAME)
    if not token:
        raise _UNAUTHORIZED

    user_id = decode_session_token(token)
    if user_id is None:
        raise _UNAUTHORIZED

    user = await UserRepo(session).get_by_id(user_id)
    if user is None:
        raise _UNAUTHORIZED
    return user


async def get_current_user_optional(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> User | None:
    try:
        return await get_current_user(request, session)
    except HTTPException:
        return None
