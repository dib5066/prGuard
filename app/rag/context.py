"""
Context builder for PRGuard review enrichment.

This module combines pull request changes with related code
retrieved from the Qdrant vector store.

The context builder:

1. Takes the PR diff and changed files.
2. Searches Qdrant for related code.
3. Searches for code related to added symbols.
4. Includes full file contents for small PRs.
5. Removes duplicate chunks.
6. Calculates an approximate context size.
7. Creates a ReviewContext used by the review agents.
"""

import logging
from dataclasses import dataclass, field

from app.core.config import settings
from app.github.tools import extract_added_symbols
from app.rag.embeddings import get_embedding_model
from app.rag.qdrant_store import QdrantCodeStore, collection_for_user
from app.services.github_service import PRContext


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Review context
# ---------------------------------------------------------------------------


@dataclass
class ReviewContext:
    """
    Store all information required to review a pull request.

    The context contains the original PR information together
    with code retrieved from the repository using RAG.
    """

    # Original pull request information.
    pr_context: PRContext

    # Code retrieved from Qdrant.
    related_chunks: list[dict] = field(default_factory=list)

    # Files found in the retrieved context.
    related_files: list[str] = field(default_factory=list)

    # Complete contents of changed files.
    # Used mainly for smaller pull requests.
    full_file_contents: dict[str, str] = field(default_factory=dict)

    # Approximate number of tokens in the retrieved context.
    total_context_tokens: int = 0

    # Sources used to build the RAG context.
    rag_sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """
    Build enriched review context for a pull request.

    The builder combines:

        PR diff
            ↓
        Vector similarity search
            ↓
        Symbol/name search
            ↓
        Full changed files for small PRs
            ↓
        Deduplicated context
            ↓
        ReviewContext
    """

    def __init__(self, user_id: int | None = None):
        """
        Initialize the Qdrant store (scoped to the owner's collection) and
        the embedding model.
        """

        self.qdrant_store = QdrantCodeStore(collection_for_user(user_id))
        self.embedding_model = get_embedding_model()

    # -----------------------------------------------------------------------
    # Build review context
    # -----------------------------------------------------------------------

    async def build_context(
        self,
        pr_context: PRContext,
        repository_id: int,
    ) -> ReviewContext:
        """
        Build RAG-enriched context for a pull request.

        Args:
            pr_context:
                Pull request data retrieved from GitHub.

            repository_id:
                Database ID of the repository.
                This is used to restrict Qdrant searches to
                the current repository.

        Returns:
            ReviewContext containing the PR data and
            related repository code.
        """

        review_context = ReviewContext(
            pr_context=pr_context,
        )

        try:
            # ---------------------------------------------------------------
            # 1. Create a LangChain retriever for this repository
            # ---------------------------------------------------------------

            retriever = self.qdrant_store.get_retriever(
                embedding_model=self.embedding_model,
                repository_id=repository_id,
                top_k=5,
            )

            # ---------------------------------------------------------------
            # 2. Search for related code for every changed file
            # ---------------------------------------------------------------

            for file_data in pr_context.files:

                # There is nothing useful to embed when the file
                # does not contain a patch.
                if not file_data.patch:
                    continue

                try:
                    # Use the changed code as the search query.
                    related_documents = await retriever.ainvoke(
                        file_data.patch
                    )

                    for document in related_documents:

                        chunk = {
                            "content": document.page_content,
                            "score": document.metadata.get(
                                "score",
                                0,
                            ),
                            "file_path": document.metadata.get(
                                "source",
                                "unknown",
                            ),
                            "relative_path": document.metadata.get(
                                "relative_path",
                                "",
                            ),
                            "language": document.metadata.get(
                                "language",
                                "",
                            ),
                        }

                        review_context.related_chunks.append(chunk)

                        # Keep track of files that contributed
                        # related code.
                        file_path = chunk["file_path"]

                        if (
                            file_path
                            not in review_context.related_files
                        ):
                            review_context.related_files.append(
                                file_path
                            )

                except Exception as error:
                    logger.warning(
                        "Failed to retrieve related code for %s: %s",
                        file_data.filename,
                        error,
                    )

            # ---------------------------------------------------------------
            # 3. Search for added symbols
            # ---------------------------------------------------------------
            #
            # For example:
            #
            #     def validate_user()
            #
            #     class PaymentService
            #
            # We can search for these names in the indexed repository.
            #

            if pr_context.parsed_diff:

                added_symbols = extract_added_symbols(
                    pr_context.parsed_diff
                )

                for file_path, symbols in added_symbols.items():

                    for symbol in symbols:

                        matching_chunks = (
                            await self.qdrant_store.search_by_name(
                                symbol_name=symbol.name,
                                repository_id=repository_id,
                                top_k=3,
                            )
                        )

                        for chunk in matching_chunks:
                            review_context.related_chunks.append(
                                chunk
                            )

            # ---------------------------------------------------------------
            # 4. Include complete file contents for small PRs
            # ---------------------------------------------------------------

            if pr_context.total_files_changed <= 20:

                for file_data in pr_context.files:

                    if file_data.content:
                        review_context.full_file_contents[
                            file_data.filename
                        ] = file_data.content

            # ---------------------------------------------------------------
            # 5. Remove duplicate chunks
            # ---------------------------------------------------------------

            unique_chunks = []
            seen_chunk_ids: set[str] = set()

            for chunk in review_context.related_chunks:

                # Use the file path and the beginning of the
                # content to create a simple identifier.
                chunk_id = (
                    f"{chunk.get('file_path', '')}:"
                    f"{chunk.get('content', '')[:50]}"
                )

                if chunk_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(chunk_id)
                unique_chunks.append(chunk)

            review_context.related_chunks = unique_chunks

            # ---------------------------------------------------------------
            # 6. Calculate approximate token count
            # ---------------------------------------------------------------

            review_context.total_context_tokens = sum(
                len(chunk.get("content", "")) // 4
                for chunk in review_context.related_chunks
            )

            # ---------------------------------------------------------------
            # 7. Store RAG source paths
            # ---------------------------------------------------------------

            review_context.rag_sources = [
                chunk.get("file_path", "unknown")
                for chunk in review_context.related_chunks
            ]

            logger.info(
                "Context built for PR %s/%s#%d: "
                "%d related chunks, %d files, ~%d tokens",
                pr_context.owner,
                pr_context.repo,
                pr_context.pr_number,
                len(review_context.related_chunks),
                len(review_context.related_files),
                review_context.total_context_tokens,
            )

        except Exception as error:

            logger.error(
                "Failed to build review context: %s",
                error,
                exc_info=True,
            )

            # If RAG fails, return the original PR context.
            # This keeps the review pipeline functional.
            #
            # The review will simply have less repository context.

        return review_context

    # -----------------------------------------------------------------------
    # Format context for the LLM prompt
    # -----------------------------------------------------------------------

    def format_context_for_prompt(
        self,
        context: ReviewContext,
    ) -> str:
        """
        Convert retrieved repository context into prompt text.

        The resulting text can be appended to the review-agent
        prompt.

        Args:
            context:
                Enriched ReviewContext.

        Returns:
            Formatted repository context.
        """

        if not context.related_chunks:
            return ""

        prompt_sections = [
            "\n\n## Related Code Context (from repository)"
        ]

        prompt_sections.append(
            "The following code from the repository may be "
            "relevant to the changes. Use it to understand "
            "how changed functions are used elsewhere."
        )

        # Limit the number of chunks added to the prompt.
        # This prevents the prompt from becoming unnecessarily large.
        maximum_chunks = 15

        for chunk in context.related_chunks[:maximum_chunks]:

            file_path = chunk.get(
                "file_path",
                "unknown",
            )

            content = chunk.get(
                "content",
                "",
            )

            programming_language = chunk.get(
                "language",
                "",
            )

            relevance_score = chunk.get(
                "score",
                0,
            )

            prompt_sections.append(
                f"\n### {file_path} "
                f"[relevance: {relevance_score:.2f}]"
            )

            prompt_sections.append(
                f"```{programming_language}"
            )

            # Limit every individual chunk.
            prompt_sections.append(
                content[:2000]
            )

            prompt_sections.append(
                "```"
            )

        return "\n".join(prompt_sections)