"""
GitHub App authentication utilities.

This module handles:
- Generating GitHub App JWTs.
- Exchanging App JWTs for installation access tokens.
- Caching installation tokens in memory.

Credentials are loaded from application settings.

JWT:
    "I am PRGuard."

Installation access token:
    "PRGuard is installed on this GitHub account/repository,
     and I want permission to access it."
"""

import logging
import time
from datetime import datetime
from typing import Final

import httpx
import jwt

from app.core.config import settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JWT generation
# ---------------------------------------------------------------------------

# GitHub App JWT can be valid for a maximum of 10 minutes.
JWT_EXPIRY_SECONDS: Final[int] = 600
TOKEN_REFRESH_MARGIN_SECONDS: Final[int] = 600
GITHUB_API_BASE_URL: Final[str] = "https://api.github.com"
GITHUB_API_VERSION: Final[str] = "2022-11-28"


def generate_jwt() -> str:
    """
    Generate a short-lived JWT for the GitHub App.

    The JWT is signed using the GitHub App's private RSA key.

    Returns:
        str: JWT token.
    """

    current_timestamp = int(time.time())

    payload = {
        "iat": current_timestamp,
        "exp": current_timestamp + JWT_EXPIRY_SECONDS,
        "iss": settings.GITHUB_APP_ID,
    }

    private_key = settings.github_private_key

    token = jwt.encode(
        payload,
        private_key,
        algorithm="RS256",
    )

    logger.debug(
        "Generated GitHub App JWT (app_id=%s)",
        settings.GITHUB_APP_ID,
    )

    return token


# ---------------------------------------------------------------------------
# Installation token cache
# ---------------------------------------------------------------------------

# installation_id -> (token, expiry_timestamp)
#  for example -- >{
#     123456: (
#         "ghs_xxxxxxxxx",
#         1788000000.0
#     )
#   }
_installation_token_cache: dict[int, tuple[str, float]] = {}

# aysnc because it make HTTP request to the github

async def get_installation_token(installation_id: int) -> str:
    """
    Get an installation access token for a GitHub App installation.

    If a valid token already exists in the cache, it is reused.
    Otherwise, a new token is requested from GitHub.

    Args:
        installation_id: GitHub App installation ID.

    Returns:
        str: GitHub installation access token.
    """

    current_timestamp = time.time()

    # ---------------------------------------------------------
    # 1. Check cache
    # ---------------------------------------------------------

    cached_token_data = _installation_token_cache.get(installation_id)

    if cached_token_data is not None:
        cached_token, expires_at_timestamp = cached_token_data

        # Refresh 10 minutes before actual expiration.
        if current_timestamp < expires_at_timestamp - TOKEN_REFRESH_MARGIN_SECONDS:
            logger.debug(
                "Using cached installation token for installation %d",
                installation_id,
            )

            return cached_token

    # ---------------------------------------------------------
    # 2. Generate GitHub App JWT
    # ---------------------------------------------------------

    app_jwt = generate_jwt()

    # ---------------------------------------------------------
    # 3. Request installation access token
    # ---------------------------------------------------------

    url = (
        f"{GITHUB_API_BASE_URL}"
        f"/app/installations/{installation_id}"
        f"/access_tokens"
    )

    headers = {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient() as http_client:

        response = await http_client.post(
            url,
            headers=headers,
        )

        # return status --> 401,404,501 etc
        response.raise_for_status()

    # ---------------------------------------------------------
    # 4. Read response
    # ---------------------------------------------------------

    token_response = response.json()

    installation_token: str = token_response["token"]
    expires_at_string: str = token_response["expires_at"]

    # ---------------------------------------------------------
    # 5. Convert expiry time to Unix timestamp
    # ---------------------------------------------------------

    expires_at_datetime = datetime.fromisoformat(
        expires_at_string.replace("Z", "+00:00")
    )

    expires_at = expires_at_datetime.timestamp()

    # ---------------------------------------------------------
    # 6. Store token in cache
    # ---------------------------------------------------------

    _installation_token_cache[installation_id] = (
        installation_token,
        expires_at,
    )

    logger.info(
        "Obtained new installation token for installation %d "
        "(expires %s)",
        installation_id,
        expires_at_string,
    )

    return installation_token


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

def invalidate_token_cache(
    installation_id: int | None = None,
) -> None:
    """
    Remove cached installation tokens.

    Args:
        installation_id:
            If provided, only that installation's token is removed.

            If None, the entire token cache is cleared.
    """

    if installation_id is None:
        _installation_token_cache.clear()
        logger.info(
            "Cleared entire installation token cache"
        )
    else:
        _installation_token_cache.pop(
            installation_id,
            None,
        )
        logger.info(
            "Cleared cached token for installation %d",
            installation_id,
        )


# ---------------------------------------------------------------------------
# App-level installation lookup (used by the install-flow setup callback)
# ---------------------------------------------------------------------------

async def get_app_installation(installation_id: int) -> dict:
    """Fetch installation metadata using the App JWT (not an installation token).

    Returns the raw GitHub payload — `account.login`, `account.id`,
    `account.type`, `repository_selection`, etc.
    """
    app_jwt = generate_jwt()
    url = f"{GITHUB_API_BASE_URL}/app/installations/{installation_id}"
    headers = {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# User-to-server OAuth ("Sign in with GitHub")
# ---------------------------------------------------------------------------

async def exchange_oauth_code(code: str, redirect_uri: str | None = None) -> str | None:
    """Exchange an OAuth `code` for a GitHub user access token."""
    from app.core.config import settings

    if not (settings.GITHUB_CLIENT_ID and settings.GITHUB_CLIENT_SECRET):
        return None
    data = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "client_secret": settings.GITHUB_CLIENT_SECRET,
        "code": code,
    }
    if redirect_uri:
        data["redirect_uri"] = redirect_uri
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data=data,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")


async def get_oauth_identity(user_token: str) -> dict:
    """Return {'id', 'login', 'email', 'avatar_url'} for the OAuth user.

    `email` is the primary *verified* email, or None.
    """
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        user = await client.get(f"{GITHUB_API_BASE_URL}/user", headers=headers)
        user.raise_for_status()
        u = user.json()

        email = u.get("email")
        try:
            emails = await client.get(
                f"{GITHUB_API_BASE_URL}/user/emails", headers=headers
            )
            emails.raise_for_status()
            for e in emails.json():
                if e.get("primary") and e.get("verified"):
                    email = e.get("email")
                    break
        except Exception:
            pass

    return {
        "id": u.get("id"),
        "login": u.get("login"),
        "email": email,
        "avatar_url": u.get("avatar_url"),
    }