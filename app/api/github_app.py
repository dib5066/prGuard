"""
GitHub App install flow (user is already signed in via GitHub OAuth).

    1. `GET /api/github/install-url` → link to
       `github.com/apps/<slug>/installations/new?state=<signed user id>`.
    2. User installs the App.
    3. GitHub redirects to the App's **Setup URL**
       `GET /api/github/setup?installation_id=..&state=..` → we link the
       installation to the user (session cookie, or the signed `state`),
       sync its repos, and bounce to `/dashboard`.

Installs are also linked automatically:
    - the `installation` webhook records `installer_github_id`, and
    - `reconcile_installations()` (run on every login) links any row whose
      installer / account id matches the logged-in GitHub user.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user, get_current_user_optional
from app.auth.security import create_install_state, decode_install_state
from app.core.config import settings
from app.core.database import get_db
from app.github.app_auth import get_app_installation
from app.github.client import GitHubClient
from app.models.installation import GitHubInstallation
from app.models.user import User
from app.repositories.installation_repo import InstallationRepository
from app.repositories.repository_repo import RepositoryRepo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/github", tags=["github-app"])


async def _sync_installation_repos(session: AsyncSession, installation_id: int) -> int:
    """Pull the installation's repository list from GitHub into our DB."""
    try:
        async with GitHubClient(installation_id) as gh:
            repos = await gh.list_installation_repositories()
    except Exception as error:
        logger.warning(
            "Could not list repos for installation %s: %s", installation_id, error
        )
        return 0

    repo_repo = RepositoryRepo(session)
    for r in repos:
        await repo_repo.upsert(
            github_id=r["id"],
            name=r["name"],
            full_name=r["full_name"],
            installation_id=installation_id,
        )
    logger.info("Synced %d repos for installation %s", len(repos), installation_id)
    return len(repos)


async def reconcile_installations(session: AsyncSession, user: User) -> int:
    """Link unlinked installations that belong to this GitHub user.

    Matches on the installer's GitHub id or the account id (personal
    installs: account id == the user's GitHub id).
    """
    q = select(GitHubInstallation).where(
        GitHubInstallation.user_id.is_(None),
        GitHubInstallation.deleted_at.is_(None),
        (GitHubInstallation.installer_github_id == user.github_user_id)
        | (GitHubInstallation.account_id == user.github_user_id),
    )
    rows = (await session.execute(q)).scalars().all()
    repo = InstallationRepository(session)
    for inst in rows:
        await repo.update(inst, user_id=user.id)
        await _sync_installation_repos(session, inst.installation_id)
    if rows:
        logger.info("Linked %d installation(s) to user %s", len(rows), user.id)
    return len(rows)


@router.get("/install-url")
async def install_url(user: User = Depends(get_current_user)):
    if not settings.GITHUB_APP_SLUG:
        raise HTTPException(500, "GITHUB_APP_SLUG is not configured on the server.")
    state = create_install_state(user.id)
    url = (
        f"https://github.com/apps/{settings.GITHUB_APP_SLUG}"
        f"/installations/new?state={state}"
    )
    return {"url": url}


@router.get("/setup")
async def setup_callback(
    installation_id: int,
    state: str | None = None,
    setup_action: str | None = None,
    session: AsyncSession = Depends(get_db),
    session_user: User | None = Depends(get_current_user_optional),
):
    """GitHub App **Setup URL** target — link the installation to the user."""
    from app.repositories.user_repo import UserRepo

    user = session_user
    if user is None and state:
        uid = decode_install_state(state)
        if uid is not None:
            user = await UserRepo(session).get_by_id(uid)

    account_name = f"installation-{installation_id}"
    account_type = "User"
    account_id = None
    try:
        data = await get_app_installation(installation_id)
        account = data.get("account") or {}
        account_name = account.get("login") or account_name
        account_type = account.get("type") or account_type
        account_id = account.get("id")
    except Exception as error:  # pragma: no cover
        logger.warning("Could not fetch installation %s: %s", installation_id, error)

    await InstallationRepository(session).upsert(
        installation_id=installation_id,
        account_name=account_name,
        account_type=account_type,
        user_id=user.id if user else None,
        account_id=account_id,
        clear_deleted=True,
    )
    if user is not None:
        await _sync_installation_repos(session, installation_id)
    await session.commit()

    logger.info(
        "Install setup: installation=%s account=%s user=%s",
        installation_id,
        account_name,
        user.id if user else None,
    )

    dest = f"{settings.FRONTEND_URL}/dashboard?installed=1"
    if user is None:
        dest += "&unlinked=1"
    return RedirectResponse(dest, status_code=302)
