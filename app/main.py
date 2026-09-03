import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.webhooks import router as webhooks_router
from app.api.routes.repos import router as repos_router
from app.api.routes.pull_requests import router as prs_router
from app.api.routes.reviews import router as reviews_router
from app.api.routes.stats import router as stats_router
from app.api.github_app import router as github_app_router
from app.auth.router import router as auth_router
from app.core.config import settings
from app.core.database import engine

_ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"

# Configure application logging so all logger.info / logger.error calls
# are visible in the console (stderr). Without this, all app logs are
# silently discarded and only uvicorn access lines appear.
# force=True so it still wins if the ASGI server pre-configured logging.
def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("prguard").setLevel(logging.INFO)


_configure_logging()

logger = logging.getLogger("prguard.startup")


def _run_alembic_upgrade_sync() -> None:
    """Blocking ``alembic upgrade head`` — call via asyncio.to_thread.

    Alembic's async env.py uses ``asyncio.run()`` internally, which cannot
    run inside the already-running lifespan loop; a worker thread has none.
    """
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config(str(_ALEMBIC_INI)), "head")


async def _ensure_migrations() -> None:
    """Bring the DB schema to head (or just warn), based on config.

    ``RUN_MIGRATIONS_ON_STARTUP`` defaults to True — handy for local dev
    where the Postgres container is frequently recreated empty.
    """
    try:
        if settings.RUN_MIGRATIONS_ON_STARTUP:
            await asyncio.to_thread(_run_alembic_upgrade_sync)
            # Alembic's env.py runs fileConfig() which disables existing
            # loggers — restore the app's logging afterwards.
            _configure_logging()
            logger.info("Ran `alembic upgrade head` on startup")
        async with engine.connect() as conn:
            row = await conn.execute(
                text("SELECT version_num FROM alembic_version")
            )
            current = {r[0] for r in row}
        logger.info("Database schema at Alembic revision %s", current or "NONE")
        if not current:
            logger.warning(
                "DATABASE HAS NO SCHEMA — run `cd backend && alembic "
                "upgrade head` (or leave RUN_MIGRATIONS_ON_STARTUP=true)."
            )
    except Exception as error:
        logger.warning("Could not ensure database migration state: %s", error)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _ensure_migrations()

    from app.review.checkpointer import close_checkpointer, setup_checkpointer

    await setup_checkpointer()
    try:
        yield
    finally:
        await close_checkpointer()


app = FastAPI(title="PRGuard API", version="1.0.0", lifespan=lifespan)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(webhooks_router)
app.include_router(auth_router)
app.include_router(github_app_router)
app.include_router(repos_router)
app.include_router(prs_router)
app.include_router(reviews_router)
app.include_router(stats_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "prguard"}