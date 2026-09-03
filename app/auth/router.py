"""Auth endpoints — "Sign in with GitHub" (user-to-server OAuth).

    GET  /api/auth/github/login     → redirect to GitHub's authorize page
    GET  /api/auth/github/callback  → exchange code, create session, → /dashboard
    POST /api/auth/logout
    GET  /api/auth/me

The session is an HS256 JWT in an httpOnly cookie.
"""

from __future__ import annotations

import logging
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.auth.security import (
    create_session_token,
    decode_session_token,
    new_state,
)
from app.core.config import settings
from app.core.database import get_db
from app.github.app_auth import exchange_oauth_code, get_oauth_identity
from app.models.user import User
from app.repositories.installation_repo import InstallationRepository
from app.repositories.user_repo import UserRepo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

_OAUTH_STATE_COOKIE = "prguard_oauth_state"
_GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"


class UserResponse(BaseModel):
    id: int
    github_username: str | None = None
    email: str | None = None
    avatar_url: str | None = None
    has_installation: bool = False


def _redirect_uri() -> str:
    return f"{settings.FRONTEND_URL}/api/auth/github/callback"


def _set_session_cookie(response: Response, user_id: int) -> None:
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=create_session_token(user_id),
        max_age=settings.SESSION_TTL_HOURS * 3600,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        path="/",
    )


async def _to_response(user: User, session: AsyncSession) -> UserResponse:
    installs = await InstallationRepository(session).list_for_user(user.id)
    return UserResponse(
        id=user.id,
        github_username=user.github_username,
        email=user.email,
        avatar_url=user.avatar_url,
        has_installation=len(installs) > 0,
    )


# --------------------------------------------------------------------------
# GitHub OAuth
# --------------------------------------------------------------------------

@router.get("/github/login")
async def github_login():
    """Kick off "Sign in with GitHub"."""
    state = new_state()
    params = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "redirect_uri": _redirect_uri(),
        "state": state,
        "scope": "read:user user:email",
    }
    resp = RedirectResponse(f"{_GITHUB_AUTHORIZE}?{urlencode(params)}", 302)
    resp.set_cookie(
        _OAUTH_STATE_COOKIE,
        state,
        max_age=600,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        path="/",
    )
    return resp


@router.get("/github/callback")
async def github_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    session: AsyncSession = Depends(get_db),
):
    dashboard = f"{settings.FRONTEND_URL}/dashboard"
    login = f"{settings.FRONTEND_URL}/login"

    cookie_state = request.cookies.get(_OAUTH_STATE_COOKIE)
    if not code or not state or state != cookie_state:
        logger.warning("OAuth callback: missing/mismatched state")
        return RedirectResponse(f"{login}?error=oauth", 302)

    try:
        token = await exchange_oauth_code(code, _redirect_uri())
        if not token:
            raise RuntimeError("no access token")
        ident = await get_oauth_identity(token)
    except Exception as error:
        logger.error("OAuth exchange failed: %s", error)
        return RedirectResponse(f"{login}?error=oauth", 302)

    if not ident.get("id"):
        return RedirectResponse(f"{login}?error=oauth", 302)

    users = UserRepo(session)
    user = await users.upsert_from_github(
        github_user_id=int(ident["id"]),
        github_username=ident.get("login"),
        email=ident.get("email"),
        avatar_url=ident.get("avatar_url"),
    )

    # Link any installation this GitHub user already created.
    from app.api.github_app import reconcile_installations

    await reconcile_installations(session, user)
    await session.commit()

    resp = RedirectResponse(dashboard, 302)
    resp.delete_cookie(_OAUTH_STATE_COOKIE, path="/")
    _set_session_cookie(resp, user.id)
    logger.info("GitHub login: user %s (%s)", user.id, user.github_username)
    return resp


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------

@router.post("/logout")
async def logout():
    resp = Response(status_code=204)
    resp.delete_cookie(
        key=settings.SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
    )
    return resp


@router.get("/me", response_model=UserResponse)
async def me(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    return await _to_response(user, session)
