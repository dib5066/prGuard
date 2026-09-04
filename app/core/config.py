import logging
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    GITHUB_APP_ID: str
    # The App's RSA private key. Provide EITHER the PEM contents inline via
    # GITHUB_PRIVATE_KEY (required on hosts with no committed key file, e.g.
    # Railway) OR a path to a .pem file via GITHUB_PRIVATE_KEY_PATH (local dev).
    GITHUB_PRIVATE_KEY: str = ""
    GITHUB_PRIVATE_KEY_PATH: str = ""
    GITHUB_WEBHOOK_SECRET: str
    # The App's URL slug — github.com/apps/<slug>. Used to build the
    # "install this app" link shown to a registered user.
    GITHUB_APP_SLUG: str = ""
    # "Sign in with GitHub" (user-to-server OAuth). From the GitHub App's
    # General tab — Client ID + a generated Client secret. Required for login.
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""

    DATABASE_URL: str = "postgresql://dibyanshu:Dib5066@localhost:5435/prGuardDB"
    # Run `alembic upgrade head` on startup. Off by default: production runs
    # migrations from the start command (a clean subprocess), and the
    # in-process path here nests asyncio.run() inside the lifespan loop.
    # Set true only for local dev where the Postgres container is recreated.
    RUN_MIGRATIONS_ON_STARTUP: bool = False
    # On boot, mark reviews stuck in RUNNING longer than this as FAILED
    # (they were killed by a restart/redeploy and will never resume).
    STUCK_REVIEW_MINUTES: int = 20
    # Hard ceiling on reviews running concurrently in this process. A burst
    # of PRs otherwise fans out to N repo clones + N*5 LLM calls at once.
    MAX_CONCURRENT_REVIEWS: int = 2

    # --- Auth / sessions -----------------------------------------------------
    # HS256 secret for the login session JWT (httpOnly cookie). MUST be set.
    SESSION_SECRET: str = ""
    SESSION_COOKIE_NAME: str = "prguard_session"
    SESSION_TTL_HOURS: int = 24 * 7
    COOKIE_SECURE: bool = False          # True behind HTTPS in production
    COOKIE_SAMESITE: str = "lax"         # "lax" | "none" | "strict"
    # Where the frontend lives — CORS origin + install-flow redirect target.
    FRONTEND_URL: str = "http://localhost:3000"

    # LLM Configuration (Groq) — kept for reference / fallback
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 4096
    GROQ_TEMPERATURE: float = 0.1  # Low temperature for consistent review output
    GROQ_TIMEOUT_SECONDS: float = 60.0  # Per-request timeout for LLM calls
    # Max review agents allowed to call the LLM at once. The 5 agents run
    # as parallel graph nodes; firing all 5 together blows small free-tier
    # token-per-minute limits (Groq free tier is ~8k TPM). Lower to 1 on a
    # tight quota; raise on a paid tier for full parallelism.
    GROQ_AGENT_CONCURRENCY: int = 2
    GROQ_MAX_RETRIES: int = 3
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    # HuggingFace Embedding Configuration
    # Historically this key was misspelled "EMBIDDING_API_KEY" in .env.
    # Both spellings are accepted; see the validator below.
    EMBIDDING_API_KEY: str = ""
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
    EMBEDDING_INFERENCE_URL: str = "https://api-inference.huggingface.co"
    # When true, use a locally-downloaded sentence-transformers model instead
    # of the HuggingFace Inference API (which no longer hosts most of these).
    EMBEDDING_USE_LOCAL: bool = False

    # RAG / Vector Store Configuration
    QDRANT_COLLECTION: str = "prguard_code_chunks"

    # Chunking Configuration
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64
    CHUNK_MIN_TOKENS: int = 50

    # Indexing Configuration
    INDEX_STALENESS_DAYS: int = 7
    INDEX_MAX_FILE_SIZE_KB: int = 500
    # Skip RAG indexing entirely for repos above these limits — a full clone
    # + load-every-file into memory OOMs a small (512 MB) container.
    INDEX_MAX_FILES: int = 1500
    INDEX_SUPPORTED_EXTENSIONS: list[str] = [
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".go", ".java", ".rb", ".rs", ".c", ".cpp", ".h",
    ]

    # LLM Configuration (Google Gemini) — active provider
    # Prefer setting GEMINI_API_KEY in backend/.env rather than hard-coding it.
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = ""
    GEMINI_MAX_TOKENS: int = 8192  # headroom — gemini-3-flash is a thinking model
    GEMINI_TEMPERATURE: float = 0.1
    GEMINI_TIMEOUT_SECONDS: float = 60.0
    GEMINI_AGENT_CONCURRENCY: int = 2
    GEMINI_MAX_RETRIES: int = 3
    # Prompt-size caps for the review agents. Each of the 5 agents is a
    # separate LLM call that carries every changed file's patch, so a large
    # PR otherwise produces a 100k+ token prompt per agent and trips
    # per-minute rate limits. Lower these on a tight quota.
    REVIEW_MAX_PATCH_CHARS: int = 2500
    REVIEW_MAX_FILES_IN_PROMPT: int = 25

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _warn_on_missing_secrets(self) -> "Settings":
        # Surface misconfiguration early and loudly instead of failing
        # deep inside a background review task. Either spelling of the
        # embedding key is accepted via the `embedding_api_key` property.
        if not self.GEMINI_API_KEY:
            logger.warning(
                "GEMINI_API_KEY is not set — the review pipeline will fail "
                "when it tries to call the LLM. Set it in backend/.env."
            )
        if not self.embedding_api_key and not self.EMBEDDING_USE_LOCAL:
            logger.warning(
                "No embedding API key configured and EMBEDDING_USE_LOCAL is "
                "false — repository indexing (RAG) will be skipped."
            )
        if not self.SESSION_SECRET:
            logger.warning(
                "SESSION_SECRET is not set — login/registration will fail. "
                "Set a long random value in backend/.env."
            )
        if not self.GITHUB_APP_SLUG:
            logger.warning(
                "GITHUB_APP_SLUG is not set — the 'Install GitHub App' link "
                "cannot be built. Set it to your app's github.com/apps/<slug>."
            )
        return self

    @property
    def embedding_api_key(self) -> str:
        """The embedding API key under whichever spelling was provided."""
        return self.EMBEDDING_API_KEY or self.EMBIDDING_API_KEY

    @property
    def github_private_key(self) -> str:
        """The GitHub App private key (PEM).

        Prefers the inline ``GITHUB_PRIVATE_KEY`` env var; falls back to
        reading ``GITHUB_PRIVATE_KEY_PATH`` from disk. A PEM pasted into an
        env var often has its line breaks escaped as ``\\n`` — restore them.

        The key is never logged, returned through an API response, or
        exposed to the frontend.
        """
        if self.GITHUB_PRIVATE_KEY:
            return self.GITHUB_PRIVATE_KEY.replace("\\n", "\n")
        if not self.GITHUB_PRIVATE_KEY_PATH:
            raise FileNotFoundError(
                "No GitHub App private key configured — set GITHUB_PRIVATE_KEY "
                "(PEM contents) or GITHUB_PRIVATE_KEY_PATH (path to .pem)."
            )
        key_path = Path(self.GITHUB_PRIVATE_KEY_PATH).expanduser()
        if not key_path.is_absolute():
            key_path = Path.cwd() / key_path
        if not key_path.exists():
            raise FileNotFoundError(
                f"GitHub App private key not found at {key_path}"
            )
        return key_path.read_text(encoding="utf-8")

settings = Settings()
