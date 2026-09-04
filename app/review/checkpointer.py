"""
LangGraph Postgres checkpointer for the multi-agent review workflow.

Persisting graph state per review means a retried / resumed review picks
up from the last completed agent instead of re-running all five agents
(which, with LLM non-determinism and partial rate-limit failures, is why
the same PR produced different findings each run).

State is stored in its own ``checkpoints*`` tables in the same Postgres
database. Those tables are created by ``AsyncPostgresSaver.setup()`` and
are intentionally *not* managed by Alembic — LangGraph owns their schema.

``AsyncPostgresSaver`` must be constructed inside a running event loop, so
it is built lazily in :func:`setup_checkpointer` (called from the app
lifespan), not at import time. If setup fails the app still runs — reviews
just execute without checkpoint/resume.
"""

from __future__ import annotations

import logging
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# asyncpg-style ssl values → libpq/psycopg `sslmode`
_SSL_TO_SSLMODE = {
    "true": "require", "require": "require", "1": "require",
    "false": "disable", "disable": "disable", "0": "disable",
    "prefer": "prefer", "allow": "allow",
    "verify-ca": "verify-ca", "verify-full": "verify-full",
}


def _psycopg_dsn() -> str:
    """A plain libpq DSN psycopg3 accepts.

    Strips the SQLAlchemy driver suffix and rewrites asyncpg's ``?ssl=``
    query param to libpq's ``?sslmode=`` (Neon/Supabase URLs use ``ssl=require``).
    """
    url = settings.DATABASE_URL
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if url.startswith(prefix):
            url = "postgresql://" + url[len(prefix):]
            break

    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    ssl_val = query.pop("ssl", None)
    if ssl_val is not None and "sslmode" not in query:
        query["sslmode"] = _SSL_TO_SSLMODE.get(str(ssl_val).lower(), "require")

    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


# The pool can be built at import time (no event loop needed); the saver
# cannot.
#
# Neon (and its PgBouncer -pooler endpoint) drop idle server connections
# fairly aggressively; without a checkout check the pool hands back a dead
# socket and the next query fails with "SSL connection has been closed
# unexpectedly". `check` validates each connection on checkout and quietly
# replaces a broken one; `max_idle`/`max_lifetime` recycle connections
# before the server would kill them; `min_size=0` means nothing is kept
# open between reviews.
_pool: AsyncConnectionPool = AsyncConnectionPool(
    conninfo=_psycopg_dsn(),
    min_size=0,
    max_size=4,
    open=False,
    check=AsyncConnectionPool.check_connection,
    max_idle=120.0,
    max_lifetime=600.0,
    reconnect_timeout=30.0,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    },
)
_saver: AsyncPostgresSaver | None = None
_opened = False


def get_checkpointer() -> AsyncPostgresSaver | None:
    """The shared checkpointer, or ``None`` if setup hasn't run / failed.

    Callers must handle ``None`` (compile the graph without a checkpointer
    and skip the ``thread_id`` config).
    """
    return _saver


async def setup_checkpointer() -> None:
    """Open the pool, build the saver, create tables. Call once at startup."""
    global _saver, _opened
    try:
        if not _opened:
            await _pool.open(wait=True)
            _opened = True
        saver = AsyncPostgresSaver(_pool)  # needs a running loop
        await saver.setup()
        _saver = saver
        logger.info("LangGraph Postgres checkpointer ready")
    except Exception as error:  # pragma: no cover - startup diagnostics
        logger.warning(
            "Checkpointer setup failed (%s). Reviews will run without "
            "checkpointing / resume.",
            error,
        )
        _saver = None


async def close_checkpointer() -> None:
    global _opened
    if _opened:
        await _pool.close()
        _opened = False
