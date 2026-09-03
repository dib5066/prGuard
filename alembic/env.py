import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

# Import all models so they're registered on Base.metadata
from app.core.database import Base
from app.models.user import User
from app.models.installation import GitHubInstallation
from app.models.repository import Repository
from app.models.pull_request import PullRequest
from app.models.review import Review, Finding, ReviewRun

# Import settings to get the database URL
from app.core.config import settings

# Alembic Config object
config = context.config

# Set the database URL from our settings (rewrite to async driver)
database_url = settings.DATABASE_URL
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
config.set_main_option("sqlalchemy.url", database_url)

# Interpret the config file for Python logging.
# disable_existing_loggers=False so running migrations in-process (from the
# FastAPI lifespan) does not silence the application's own loggers.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Target metadata for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def _include_object(obj, name, type_, reflected, compare_to):
    """Keep LangGraph's checkpoint* tables out of autogenerate diffs.

    They live in the same database but are owned by AsyncPostgresSaver.
    """
    if type_ == "table" and name.startswith("checkpoint"):
        return False
    return True


def do_run_migrations(connection):
    """Run migrations with a given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        # Required when the URL points at a PgBouncer (transaction-mode)
        # endpoint such as Neon's -pooler host.
        connect_args={"statement_cache_size": 0}
        if "+asyncpg" in config.get_main_option("sqlalchemy.url", "")
        else {},
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
