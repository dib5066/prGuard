# PRGuard — Production Risk Audit (Railway backend + Vercel frontend)

Date: 2026-09-04
Scope: everything that can break or degrade PRGuard once it is live on
Railway (backend) and Vercel (frontend). Ordered by severity. Each item has
**Symptom → Cause → Fix**. Frontend is healthy; its few items are at the end.

Legend: 🔴 critical (will fail under normal use) · 🟠 high · 🟡 medium · ⚪ low

---

## 0. Railway environment-variable checklist (do this first)

`.env` is git-ignored, so nothing is inherited — every value must be set in
the Railway dashboard. Missing ones fail *deep inside a review*, not at boot.

| Variable | Why it must be set | If missing |
|---|---|---|
| `DATABASE_URL` | Neon connection string (`postgresql+asyncpg://…-pooler…?ssl=require`) | Falls back to `localhost:5435` → nothing works |
| `GEMINI_API_KEY` | LLM provider | `ReviewService.__init__` raises → **every review FAILS** |
| `GEMINI_MODEL` | Code default is `""` | `ChatGoogleGenerativeAI(model="")` → every agent call errors |
| `SESSION_SECRET` | Signs the login cookie | Empty default → **no one can log in** |
| `GITHUB_PRIVATE_KEY` | App JWT (paste full PEM, real newlines). Do **not** also set `GITHUB_PRIVATE_KEY_PATH` | `generate_jwt()` raises `FileNotFoundError` → no GitHub calls, no reviews |
| `GITHUB_APP_ID` | App JWT `iss` | JWT rejected by GitHub |
| `GITHUB_WEBHOOK_SECRET` | HMAC verify | Every webhook rejected |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | "Sign in with GitHub" | Login broken |
| `GITHUB_APP_SLUG` | "Install App" link | Install CTA can't be built |
| `FRONTEND_URL` | `https://pr-guard-frontend.vercel.app` (no trailing slash) — CORS origin **and** OAuth `redirect_uri` base | Login redirect mismatch |
| `COOKIE_SECURE` | `true` (Vercel/Railway are HTTPS) | Browser drops the session cookie → login "works" then bounces |
| `RUN_MIGRATIONS_ON_STARTUP` | `false` — see 🔴 #3 | Fragile in-process migration path runs and can crash startup |
| `QDRANT_URL` / `QDRANT_API_KEY` | RAG vector store | Indexing fails (caught; RAG disabled) |
| `EMBIDDING_API_KEY` (sic) or `EMBEDDING_API_KEY` | HF embeddings | Indexing fails (caught) — see 🟠 #7 |
| `GEMINI_AGENT_CONCURRENCY` | Keep `1` on Gemini free tier — see 🟡 #10 | 429 rate-limit storms |
| `TIKTOKEN_CACHE_DIR` | Optional; avoids a runtime download — see 🟠 #9 | Indexing can fail on first run |

---

## 🔴 1. A DB transaction is held open for the entire multi-minute LLM run

**Symptom** — `psycopg.OperationalError: SSL connection has been closed
unexpectedly`, `asyncpg InterfaceError: the underlying connection is closed`,
cascading `MissingGreenlet` during/after `graph.astream`. Reviews randomly
die mid-run; some get stuck in `RUNNING`.

**Cause** — `run_review` (`app/workers/review_worker.py`) opens **one**
`async with AsyncSessionLocal() as session` and keeps it for the whole
pipeline: clone → index → embeddings → **5 LLM agents** → validation →
publish. Repository methods only `flush()`, never `commit()`
(`app/repositories/base.py` — "transaction controlled by caller"). Inside
`ReviewService.run_multi_agent_review`, `mark_running()` / `update_phase()`
open a Postgres transaction, then `graph.astream(...)` runs for minutes with
**no DB activity**, so that transaction — and its connection — sits idle.
Neon (PgBouncer `-pooler`, `idle_in_transaction_session_timeout`, compute
autosuspend) kills idle / idle-in-transaction connections. The next query
(the findings-insert loop) then hits a dead socket.

The already-applied `NullPool` + checkpointer-pool hardening + crash-safe
`get_db` reduce the blast radius but **do not fix this** — the fix is to not
hold the session across the LLM run.

**Fix** (pick one, top is best):
1. Restructure `run_review` so each DB phase uses its own short
   `async with AsyncSessionLocal()` block, and the LLM graph run holds
   **no** session/transaction. Pass plain data (ids, dataclasses) between
   phases, not ORM objects.
2. Interim: `await session.commit()` immediately before STEP 12, and wrap
   STEP 5/6 (findings insert) and STEP 16 (mark complete) each in their own
   fresh `AsyncSessionLocal()` so a stale connection is never reused after
   the long gap. Also `commit()` inside `run_multi_agent_review` right after
   `mark_running`/`update_phase` and before `graph.astream`.
3. Make repo write helpers `commit()` (or add an explicit
   `await session.commit()` after every `update_phase` in the service).

---

## 🔴 2. Background reviews are bare `asyncio.create_task` in the web process

**Symptom** — on every Railway redeploy/crash/OOM, in-flight reviews vanish
with no log and stay `RUNNING` forever. A burst of PRs can exhaust
memory / DB connections / LLM quota all at once.

**Cause** — `app/api/webhooks.py::_schedule_review` does
`asyncio.create_task(run_review_in_background(...))`. No queue, no
persistence, no retry, no cross-restart recovery, **no global concurrency
limit**. The `_active_reviews` dedupe dict is in-memory (lost on restart,
useless with >1 instance). The `graph_timeout` can't save a review whose
process was killed.

**Fix**:
- Short term:
  - Global `asyncio.Semaphore(2–3)` around `run_review` so N simultaneous
    PRs don't fan out to N clones + N×5 LLM calls.
  - Startup sweep: mark every review `RUNNING` for > ~15 min as `FAILED`
    ("interrupted by restart") so the dashboard/SSE don't hang.
- Proper: move reviews to a real worker — a separate Railway service running
  `arq` / `dramatiq` (Redis) or a Railway cron that drains a `queued` table.
  Webhook just enqueues and returns 200.

---

## 🔴 3. `RUN_MIGRATIONS_ON_STARTUP` defaults to `True`

**Symptom** — the `MissingGreenlet` seen during boot; migrations run twice.

**Cause** — even though `railpack.json` now runs `alembic upgrade head`
before uvicorn, `app/main.py::_ensure_migrations` still runs it again via
`asyncio.to_thread(_run_alembic_upgrade_sync)` → `asyncio.run()` nested
inside the lifespan loop. Config default is `True`
(`app/core/config.py:29`).

**Fix** — set `RUN_MIGRATIONS_ON_STARTUP=false` in Railway **and** flip the
code default to `False` so a fresh deploy without the env var is still safe.

---

## 🔴 4. Startup command fails the whole deploy on a transient DB blip

**Symptom** — deploy marked failed / crash-loop when Neon is cold or has a
momentary network hiccup.

**Cause** — `railpack.json` start command is
`alembic upgrade head && uvicorn …`. `&&` means any non-zero exit from
alembic (including "couldn't connect because Neon was autosuspended and slow
to wake") stops uvicorn from ever starting.

**Fix** — wrap migration in a small retry, e.g. a `start.sh`:
```sh
for i in 1 2 3 4 5; do alembic upgrade head && break || sleep 5; done
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
```
and point `startCommand` at `sh start.sh`. (Keep failing the deploy if
migrations genuinely can't apply — just not on the first slow connect.)

---

## 🟠 5. Single-instance assumptions everywhere

**Symptom** — if Railway replicas are ever set > 1: SSE clients get no
events for reviews running on another instance; duplicate reviews run for
one PR; installation-token cache thrash.

**Cause** — `app/core/events.py` pub/sub is an in-process dict (its own
docstring says so). `_active_reviews`, `_installation_token_cache`, the
compiled LangGraph, and the Gemini clients are all per-process.

**Fix** — keep **replicas = 1** on Railway until this moves to Redis
pub/sub. Write it in the README so nobody scales it by accident.

---

## 🟠 6. LLM timeouts vs. a "thinking" model

**Symptom** — `RuntimeError: Multi-agent review timed out after 240s` →
review `FAILED` on larger PRs.

**Cause** — `GEMINI_TIMEOUT_SECONDS=60`, and
`graph_timeout = GEMINI_TIMEOUT_SECONDS * 4 = 240s`
(`app/services/review_service.py:925`). `gemini-3-flash-preview` is a
reasoning model; with 5 agents at concurrency 2, `max_retries=3` backoff on
any 429, and a big prompt, one review easily exceeds 240 s. Railway's shared
CPU makes JSON parsing / tiktoken slower too.

**Fix** — raise `GEMINI_TIMEOUT_SECONDS` to ~120 and the graph multiplier to
×6–8, **or** shrink prompts (see #10), **or** drop to `concurrency=1` so
backoff is the only variable. Also confirm langchain's own retry isn't
stacking on top of the outer `asyncio.wait_for`.

---

## 🟠 7. Embeddings use HuggingFace's retired free Inference API

**Symptom** — every index attempt logs `Repository indexing failed … 404`
(or 401) and RAG is silently off; reviews run on the diff only.

**Cause** — `EMBEDDING_USE_LOCAL=false` (default) →
`HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")`
in `app/rag/embeddings.py`. HF removed free serverless feature-extraction
for most models. And `torch` / `sentence-transformers` are **not in
`uv.lock`**, so flipping `EMBEDDING_USE_LOCAL=true` → `ImportError` at
runtime (and even if added, the model + torch won't fit a 512 MB Railway
instance).

**Fix** — choose one:
- **(a)** Switch embeddings to a hosted API that still works:
  Google `text-embedding-004` (you already have a Gemini key),
  OpenAI `text-embedding-3-small`, Voyage, or Cohere. Update
  `EMBEDDING_DIMENSION` and re-create the Qdrant collection.
- **(b)** Add `langchain-huggingface[sentence-transformers]` + `torch`
  (CPU wheel) to deps and bump the Railway plan to ≥ 2 GB RAM.
- **(c)** Accept RAG-disabled in prod; stop calling the indexer so reviews
  don't waste time cloning.

---

## 🟠 8. Indexing can OOM / fill disk on Railway

**Symptom** — container killed (OOM) or `No space left on device` during
`index_repository`; review then continues without RAG (caught) — or the
whole worker dies.

**Cause** — `app/rag/indexer.py` full-clones the repo into `/tmp`
(Railway ephemeral disk), then `load_repository` reads **every** supported
source file into memory as `Document` objects, then tiktoken-splits them.
Only guards are per-file `INDEX_MAX_FILE_SIZE_KB=500` and an extension
allow-list — no cap on file count, total bytes, or repo size. A mid-size
monorepo blows a 512 MB instance.

**Fix** — shallow clone (`git clone --depth 1 --filter=blob:none`), hard
caps (e.g. skip indexing if repo > ~50 MB or > ~2000 eligible files),
skip `node_modules`/`dist`/`.next`/vendored dirs, and stream files instead
of loading all `Document`s at once.

---

## 🟠 9. `tiktoken` downloads its vocab at runtime

**Symptom** — first index on a fresh container fails in
`create_code_splitter()` with a network error.

**Cause** — tiktoken fetches `cl100k_base` (or similar) from
`openaipublic.blob.core.windows.net` on first use. Egress can be slow or
blocked.

**Fix** — set `TIKTOKEN_CACHE_DIR` and commit the cached encoder file into
the image, or catch the failure and fall back to a character/`RecursiveCharacterTextSplitter`.

---

## 🟡 10. Token / rate-limit budget — "is 15000 enough?"

**Per review** the worker makes **5 agent calls** (not the single baseline —
that path is dead code). Each call =
`system prompt (~1–2k tok)` + `user prompt` + up to `GEMINI_MAX_TOKENS`
output.

`user prompt` (`app/review/agents/base.py::build_user_prompt`) =
PR metadata + PR body (≤ 2000 chars) + for **every** changed file its patch
(≤ **5000 chars each**) + RAG context. A 15-file PR ≈ 75k chars ≈ ~20k input
tokens **per agent**, ×5 agents ≈ 100k+ tokens/review, plus retries.

Free-tier Gemini (esp. *preview* models) has low RPM (often 5–15) and daily
caps. 5 parallel calls + langchain retry on 429 → rate-limit storms → agents
return `[]` (handled) but the review finishes with missing findings and may
time out (#6).

**Recommendations:**
| Setting | Now | Prod suggestion | Why |
|---|---|---|---|
| `GEMINI_MODEL` | `gemini-3-flash-preview` | a GA model w/ known quota (`gemini-2.0-flash`) unless preview quota is confirmed | predictable limits |
| `GEMINI_AGENT_CONCURRENCY` | 2 | **1** on free tier | each agent is a separate request; 1 respects RPM |
| per-file patch slice | 5000 chars | 2000–3000 | biggest lever on input tokens |
| files included | all | cap (e.g. first ~25 by size); skip lock/generated/vendored | avoid 100k-token prompts |
| `GEMINI_MAX_TOKENS` | 8192 (`.env` 10096) | **raise to ~15000 only if** you see `_salvage_json` "recovered from malformed JSON" warnings (thinking model truncating) | higher = more cost + latency + TPM pressure; findings JSON rarely needs > 8k |
| billing | free key | **paid Gemini key** for production | removes the whole class of 429 failures |

So: 15000 output tokens is *fine to set* and harmless-ish, but it is not the
bottleneck — **input** size and **requests-per-minute** are. Fix those first.

---

## 🟡 11. No LLM provider fallback

**Symptom** — Gemini outage or quota exhaustion = every review `FAILED`.

**Cause** — the Groq config (`GROQ_*`) is still present and the comments
promise a fallback, but `run_multi_agent_review` and `run_agent` only ever
construct `ChatGoogleGenerativeAI`.

**Fix** — wire the Groq fallback: on Gemini `ResourceExhausted` / repeated
timeout, retry the agent with `ChatGroq`. Or at least degrade to fewer
agents instead of failing the whole review.

---

## 🟡 12. Neon free tier operational limits

- Autosuspend after ~5 min idle → first webhook/migration after a quiet
  period is slow; any held connection dies (feeds #1).
- ~0.5 GB storage; `findings` and `review_runs` grow unbounded — add a
  retention job (e.g. delete runs/findings for reviews older than N days).
- Pooler connection cap — with `NullPool` every request + every review phase
  opens a fresh connection; fine at low volume, watch under load.
- **Consider** Railway's own Postgres plugin instead: co-located (no
  cross-region latency), no autosuspend, simpler `DATABASE_URL`.

---

## 🟡 13. `/api/stats` = ~15 sequential round trips per call

**Symptom** — slow dashboard, elevated Neon compute, especially with the
frontend polling.

**Cause** — `app/api/routes/stats.py` runs count/avg queries one after
another, each a cross-region round trip to Neon.

**Fix** — combine into 2–3 grouped queries (`GROUP BY status`,
`GROUP BY severity`) and/or cache the response ~30 s per user.

---

## 🟡 14. Webhook idempotency is in-memory only

**Symptom** — after a restart, a GitHub delivery retry can start a second
review (new row, real LLM spend) for a PR already reviewed.

**Cause** — dedupe is the `_active_reviews` dict.
`get_reusable_review(pr_id, head_sha)` catches the *same-commit* case (good),
but there's a window between review-row creation and that check.

**Fix** — persist a `delivery_id` (GitHub sends `X-GitHub-Delivery`) unique
row, or check for an existing non-terminal review for `(pr_id, head_sha)`
before creating one.

---

## 🟡 15. Verify Gemini JSON mode is actually on

`ChatGoogleGenerativeAI(..., response_mime_type="application/json")` is
passed as a top-level constructor kwarg; `langchain-google-genai` generally
wants this in `model_kwargs` / `generation_config`. If it's being dropped,
responses may arrive fenced/with prose — currently salvaged by
`_salvage_json`, so it's *degraded, not broken*. Confirm and move it to the
right place.

---

## ⚪ 16. Smaller items

- **Health check has no DB probe** — Railway marking the service healthy
  doesn't mean Neon is reachable. Add `/health/ready` that does
  `SELECT 1`.
- **`run_baseline_review`** (single-agent) is dead code — the worker uses
  `run_multi_agent_review`. Extra surface to maintain.
- **`alembic.ini`** hard-codes a localhost `sqlalchemy.url`; `env.py`
  overrides it from `DATABASE_URL`, so it works but is confusing —
  set it to a comment/placeholder.
- **CORS middleware** is effectively unused (browser only talks to the
  Vercel origin, which server-proxies `/api/*`). Harmless, can stay.
- **Logging** goes to stderr via `basicConfig(force=True)` — fine for
  Railway, but the double `_configure_logging()` (import + post-migration)
  is a smell.
- **`extract_text_content` / thinking tokens** — if a Gemini "thinking"
  response spends most of `max_tokens` on reasoning, the JSON can be
  truncated; watch for `_salvage_json` info logs as the signal to raise
  `GEMINI_MAX_TOKENS`.

---

## Frontend (Vercel) — running fine, minor follow-ups

- ⚪ **F1 — `next.config.ts` `BACKEND_URL` normalization was reverted.**
  It's back to `process.env.BACKEND_URL || "http://localhost:8000"`. If
  `BACKEND_URL` is ever set without an `https://` scheme the Vercel build
  fails with `Invalid rewrite found` (you hit this once). Re-add the
  helper that trims trailing `/` and prepends `https://` when no scheme is
  present.
- 🟡 **F2 — SSE through the Next rewrite proxy.** `/api/reviews/:id/events`
  is proxied by a Vercel function; on Hobby the ~10–15 s execution cap cuts
  long streams (Pro ~60–90 s). The `useReviewStatus` polling fallback covers
  correctness, but live updates stall on long reviews. Option: point
  `EventSource` straight at the Railway origin (needs backend CORS +
  `credentials`), or accept polling.
- ⚪ **F3 — `proxy.ts` only checks cookie *presence*.** A stale/expired
  session cookie lets the page render, then every `/api/*` call 401s.
  Cosmetic; the app already handles 401 → `/login`.

---

## Suggested order of work

1. **#3 + #0** — set env vars, `RUN_MIGRATIONS_ON_STARTUP=false`. (minutes)
2. **#1** — stop holding the DB session across the LLM run. (the big one)
3. **#2** — global review semaphore + startup sweep of stuck `RUNNING`.
4. **#6 + #10** — timeout/prompt/concurrency tuning; ideally a paid Gemini key.
5. **#4** — migration retry wrapper in the start command.
6. **#7** — decide embeddings story (hosted API vs. RAG-off).
7. **#8/#9** — indexing caps + tiktoken cache.
8. Everything 🟡/⚪ as cleanup.
