from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import settings

# Automatically swap to the asyncpg driver if postgresql:// is used
database_url = settings.DATABASE_URL
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Neon / Supabase / any PgBouncer-fronted Postgres pools in "transaction"
# mode, which is incompatible with asyncpg's prepared-statement cache and
# with SQLAlchemy holding long-lived pooled connections across event loops
# (the latter surfaces as `MissingGreenlet` on pool_pre_ping). Use NullPool
# and let the external pooler own connection pooling; disable the statement
# cache so PgBouncer can multiplex safely.
_is_asyncpg = "+asyncpg" in database_url
_connect_args = {"statement_cache_size": 0} if _is_asyncpg else {}

engine = create_async_engine(
    database_url,
    echo=False,
    poolclass=NullPool,
    connect_args=_connect_args,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)

class Base(DeclarativeBase):
    """Base class for all database models."""
    pass

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to retrieve database session.

    Neon / PgBouncer can drop the underlying socket mid-request, after
    which SQLAlchemy's own rollback-on-close raises
    ``InterfaceError: the underlying connection is closed``. Neither the
    defensive rollback nor the close is allowed to turn an
    otherwise-successful response into a 500, and neither may mask the
    real exception from the route.
    """
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception:
        try:
            await session.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            await session.close()
        except Exception:
            pass
