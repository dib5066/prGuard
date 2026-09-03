"""
Repository indexing orchestrator for the PRGuard RAG pipeline.

This module coordinates the complete repository indexing process:

1. Clone the GitHub repository into a temporary directory.
2. Load supported source files.
3. Split source files into smaller code chunks.
4. Generate embeddings using HuggingFace.
5. Store the chunks and embeddings in Qdrant.
6. Mark the repository as indexed in PostgreSQL.
7. Delete the temporary cloned repository.

The cloned repository is temporary and is removed after indexing.
"""


import asyncio
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import git
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.github.app_auth import get_installation_token
from app.repositories.repository_repo import RepositoryRepo


logger = logging.getLogger(__name__)


def _authenticated_clone_url(clone_url: str, token: str) -> str:
    """Return an HTTPS clone URL that carries an installation token.

    Required for private repositories. The token is never logged.
    """
    parts = urlsplit(clone_url)
    if parts.scheme != "https" or not token:
        return clone_url
    netloc = f"x-access-token:{token}@{parts.hostname}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _clone_and_checkout(clone_url: str, target_dir: str, head_sha: str) -> None:
    """Blocking git clone + checkout. Run this via ``asyncio.to_thread``."""
    repo = git.Repo.clone_from(clone_url, target_dir)
    try:
        repo.git.checkout(head_sha)
    except git.GitCommandError:
        # head_sha may not be reachable from the default branch tip
        # (e.g. a fork PR). Fall back to whatever the clone checked out.
        logger.warning(
            "Could not checkout %s; indexing the default branch tip instead.",
            head_sha[:8],
        )


class RepositoryIndexer:
    """
    Coordinate the repository indexing pipeline.

    The complete pipeline is:

        GitHub repository
              ↓
        Temporary clone
              ↓
        Load source files
              ↓
        Split into chunks
              ↓
        Generate embeddings
              ↓
        Store in Qdrant
              ↓
        Update repository metadata
              ↓
        Delete temporary files
    """

    def __init__(self, session: AsyncSession):
        """
        Create a repository indexer.

        Args:
            session:
                Active asynchronous PostgreSQL database session.
        """

        self.session = session

        # RepositoryRepo handles database operations for
        # repository records.
        self.repository_repository = RepositoryRepo(session)

    # -----------------------------------------------------------------------
    # Main indexing method
    # -----------------------------------------------------------------------

    async def index_repository(
        self,
        repository_id: int,
        clone_url: str,
        head_sha: str,
        *,
        installation_id: int | None = None,
        user_id: int | None = None,
        force: bool = False,
    ) -> dict:
        """
        Index a GitHub repository into Qdrant.

        Args:
            repository_id:
                Database ID of the repository.

            clone_url:
                HTTPS Git clone URL.

            head_sha:
                Commit SHA that should be indexed.

            force:
                If True, index the repository even if it was
                recently indexed.

        Returns:
            Dictionary containing indexing statistics:

                status:
                    "completed", "skipped", or "error"

                files_processed:
                    Number of source files loaded.

                chunks_created:
                    Number of chunks generated.

                embeddings_stored:
                    Number of chunks successfully stored in Qdrant.

                errors:
                    List of indexing errors.
        """

        # Import RAG components here so that this module does not
        # initialize these components until indexing is actually needed.
        from app.rag.embeddings import get_embedding_model
        from app.rag.loader import load_repository
        from app.rag.qdrant_store import QdrantCodeStore, collection_for_user
        from app.rag.splitter import create_code_splitter

        # -------------------------------------------------------------------
        # 1. Check whether the repository needs indexing
        # -------------------------------------------------------------------

        repository = await self.repository_repository.get_by_id(
            repository_id
        )

        if repository and not force:

            if (
                repository.is_indexed
                and repository.last_indexed_at
            ):

                days_since_indexing = (
                    datetime.now(timezone.utc)
                    - repository.last_indexed_at
                ).days

                if (
                    days_since_indexing
                    < settings.INDEX_STALENESS_DAYS
                ):

                    logger.info(
                        "Repository %s was indexed %d days ago. "
                        "Skipping indexing.",
                        repository.full_name,
                        days_since_indexing,
                    )

                    return {
                        "status": "skipped",
                        "reason": "recently indexed",
                        "files_processed": 0,
                        "chunks_created": 0,
                        "embeddings_stored": 0,
                        "errors": [],
                    }

        # -------------------------------------------------------------------
        # Indexing statistics
        # -------------------------------------------------------------------

        indexing_stats = {
            "status": "completed",
            "files_processed": 0,
            "chunks_created": 0,
            "embeddings_stored": 0,
            "errors": [],
        }

        temporary_directory = None

        try:

            # ---------------------------------------------------------------
            # 2. Clone repository
            # ---------------------------------------------------------------

            temporary_directory = tempfile.mkdtemp(
                prefix="prguard_idx_"
            )

            logger.info(
                "Cloning repository %s at commit %s into %s",
                clone_url,  # plain URL, no token
                head_sha[:8],
                temporary_directory,
            )

            effective_url = clone_url
            if installation_id is not None:
                token = await get_installation_token(installation_id)
                effective_url = _authenticated_clone_url(clone_url, token)

            # git clone + checkout are blocking (subprocess + disk I/O).
            # Run them off the event loop so webhooks and other reviews
            # are not frozen while a large repo is cloned.
            await asyncio.to_thread(
                _clone_and_checkout,
                effective_url,
                temporary_directory,
                head_sha,
            )

            # ---------------------------------------------------------------
            # 3. Load source files (disk walk — also blocking)
            # ---------------------------------------------------------------

            documents = await asyncio.to_thread(
                load_repository,
                temporary_directory,
            )

            indexing_stats["files_processed"] = len(documents)

            if not documents:

                logger.warning(
                    "No supported source files found in %s",
                    clone_url,
                )

                return indexing_stats

            logger.info(
                "Loaded %d source files from %s",
                len(documents),
                clone_url,
            )

            # ---------------------------------------------------------------
            # 4. Split source files into code chunks
            # ---------------------------------------------------------------

            code_splitter = create_code_splitter()

            # Splitting is CPU-bound; keep it off the event loop.
            chunks = await asyncio.to_thread(
                code_splitter.split_documents,
                documents,
            )

            indexing_stats["chunks_created"] = len(chunks)

            logger.info(
                "Split %d source files into %d chunks",
                len(documents),
                len(chunks),
            )

            # ---------------------------------------------------------------
            # 5. Generate embeddings and store chunks in Qdrant
            # ---------------------------------------------------------------

            if chunks:

                embedding_model = get_embedding_model()

                qdrant_store = QdrantCodeStore(collection_for_user(user_id))

                try:

                    # Make sure the collection exists before
                    # inserting any vectors.
                    await qdrant_store.ensure_collection()

                    stored_embeddings = (
                        await qdrant_store.upsert_documents(
                            documents=chunks,
                            embedding_model=embedding_model,
                            repository_id=repository_id,
                        )
                    )

                    indexing_stats[
                        "embeddings_stored"
                    ] = stored_embeddings

                finally:

                    # Always close Qdrant connections.
                    await qdrant_store.close()

            # ---------------------------------------------------------------
            # 6. Update repository metadata
            # ---------------------------------------------------------------

            if repository:

                await self.repository_repository.mark_indexed(
                    repository
                )

            logger.info(
                "Repository indexing completed: %s",
                (
                    repository.full_name
                    if repository
                    else repository_id
                ),
            )

            logger.info(
                "Indexing statistics: "
                "%d files, %d chunks, %d embeddings",
                indexing_stats["files_processed"],
                indexing_stats["chunks_created"],
                indexing_stats["embeddings_stored"],
            )

            return indexing_stats

        # -------------------------------------------------------------------
        # Error handling
        # -------------------------------------------------------------------

        except Exception as error:

            logger.error(
                "Repository indexing failed: %s",
                error,
                exc_info=True,
            )

            indexing_stats["status"] = "error"

            indexing_stats["errors"].append(
                str(error)
            )

            raise

        # -------------------------------------------------------------------
        # Temporary directory cleanup
        # -------------------------------------------------------------------

        finally:

            if (
                temporary_directory
                and Path(temporary_directory).exists()
            ):

                shutil.rmtree(
                    temporary_directory,
                    ignore_errors=True,
                )

                logger.debug(
                    "Deleted temporary repository directory: %s",
                    temporary_directory,
                )