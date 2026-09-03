"""
Login-session tokens (HS256 JWT, kept in an httpOnly cookie) and the
short-lived signed `state` used by the OAuth / install redirects.

Distinct from `app/github/app_auth.py` — that JWT authenticates PRGuard *to
GitHub* as an App. This one authenticates a *human* to PRGuard's own API.
"""

from __future__ import annotations

import secrets
import time

import jwt

from app.core.config import settings

_ALGO = "HS256"


# --------------------------------------------------------------------------
# Session tokens (put in an httpOnly cookie)
# --------------------------------------------------------------------------

def create_session_token(user_id: int) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + settings.SESSION_TTL_HOURS * 3600,
        "typ": "session",
    }
    return jwt.encode(payload, settings.SESSION_SECRET, algorithm=_ALGO)


def decode_session_token(token: str) -> int | None:
    """Return the user id, or None if the token is invalid/expired."""
    try:
        payload = jwt.decode(token, settings.SESSION_SECRET, algorithms=[_ALGO])
        if payload.get("typ") != "session":
            return None
        return int(payload["sub"])
    except Exception:
        return None


# --------------------------------------------------------------------------
# OAuth / install `state` (CSRF token; may also carry a user id)
# --------------------------------------------------------------------------

def new_state() -> str:
    """Opaque random CSRF state for the login redirect."""
    return secrets.token_urlsafe(24)


def create_install_state(user_id: int, ttl_seconds: int = 900) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + ttl_seconds,
        "typ": "install_state",
    }
    return jwt.encode(payload, settings.SESSION_SECRET, algorithm=_ALGO)


def decode_install_state(token: str) -> int | None:
    try:
        payload = jwt.decode(token, settings.SESSION_SECRET, algorithms=[_ALGO])
        if payload.get("typ") != "install_state":
            return None
        return int(payload["sub"])
    except Exception:
        return None
