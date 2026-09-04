#!/bin/sh
# Railway start command. Run DB migrations first, in a clean subprocess
# (its own asyncio loop), retrying a few times so a cold/slow Neon wake
# doesn't fail the whole deploy. Then hand off to uvicorn.
set -e

attempt=1
max_attempts=5
until alembic upgrade head; do
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "alembic upgrade head failed after ${max_attempts} attempts" >&2
    exit 1
  fi
  echo "alembic upgrade head failed (attempt ${attempt}/${max_attempts}); retrying in 5s..." >&2
  attempt=$((attempt + 1))
  sleep 5
done

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
