"""
Qdrant vector store management for PRGuard.

This module manages code embeddings stored in Qdrant.

It uses LangChain's QdrantVectorStore for:
    - Adding documents
    - Generating embeddings
    - Creating LangChain retrievers

It uses the Qdrant client directly for:
    - Creating the collection
    - Searching with repository filters
    - Searching by name
    - Deleting repository chunks
    - Closing connections

Collection metadata:
    - source: Original file path
    - relative_path: Relative file path
    - language: Programming language
    - file_extension: File extension
    - repository_id: Database ID of the repository

Default embedding dimension:
    384 dimensions
"""


import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from app.core.config import settings


logger = logging.getLogger(__name__)


def collection_for_user(user_id: int | None) -> str:
    """One Qdrant collection per PRGuard user (holds all their repos).

    Chunks inside carry ``repository_id`` in their payload, so searches
    still scope to a single repo. Unlinked repos (no user) use the base
    collection.
    """
    if not user_id:
        return settings.QDRANT_COLLECTION
    return f"{settings.QDRANT_COLLECTION}_user_{user_id}"


class QdrantCodeStore:
    """
    Manage a Qdrant collection that stores code embeddings.

    LangChain's QdrantVectorStore is used for normal vector-store
    operations, while the raw Qdrant clients are used when we need
    more direct control over filtering and collection management.
    """

    def __init__(self, collection_name: str | None = None):
        """
        Initialize the Qdrant clients and vector store configuration.

        Args:
            collection_name: override the collection (per-user). Defaults
                to ``settings.QDRANT_COLLECTION``.
        """

        # Name of the Qdrant collection.
        self.collection_name = collection_name or settings.QDRANT_COLLECTION

        # Number of dimensions in each embedding vector.
        self.vector_dimension = settings.EMBEDDING_DIMENSION

        # -------------------------------------------------------------------
        # Async Qdrant client
        # -------------------------------------------------------------------
        #
        # Used for:
        # - Collection management
        # - Filtered searches
        # - Deleting repository data
        #
        self.async_qdrant_client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )

        # -------------------------------------------------------------------
        # Sync Qdrant client
        # -------------------------------------------------------------------
        #
        # LangChain's QdrantVectorStore uses this client.
        #
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )

        # LangChain vector store.
        #
        # It is created when an embedding model is provided.
        self.vector_store: QdrantVectorStore | None = None

    # -----------------------------------------------------------------------
    # LangChain vector store
    # -----------------------------------------------------------------------

    def get_vector_store(
        self,
        embedding_model: Embeddings,
    ) -> QdrantVectorStore:
        """
        Get the LangChain QdrantVectorStore.

        The vector store is created only once and reused afterwards.

        Args:
            embedding_model:
                LangChain embedding model used to create vectors.

        Returns:
            LangChain QdrantVectorStore instance.
        """

        if self.vector_store is None:
            # NOTE: langchain-qdrant's ``QdrantVectorStore`` does not take an
            # ``async_client``. Its ``a*`` methods (aadd_documents /
            # asimilarity_search) are inherited from the base ``VectorStore``
            # and run the sync implementation in a thread-pool executor, so
            # they neither block the event loop nor need an async client.
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=embedding_model,
            )

        return self.vector_store

    # -----------------------------------------------------------------------
    # Collection management
    # -----------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """
        Create the Qdrant collection if it does not already exist.

        The collection uses:
            - The configured embedding dimension
            - COSINE similarity
        """

        collections_response = (
            await self.async_qdrant_client.get_collections()
        )

        existing_collection_names = [
            collection.name
            for collection in collections_response.collections
        ]

        if self.collection_name in existing_collection_names:
            logger.debug(
                "Qdrant collection already exists: %s",
                self.collection_name,
            )
            return

        await self.async_qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dimension,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            "Created Qdrant collection: %s",
            self.collection_name,
        )

    # -----------------------------------------------------------------------
    # Document upsert
    # -----------------------------------------------------------------------

    async def upsert_documents(
        self,
        documents: list[Document],
        embedding_model: Embeddings,
        repository_id: int,
    ) -> int:
        """
        Generate embeddings and store documents in Qdrant.

        Each document receives the repository_id in its metadata so
        that searches can later be restricted to a specific repository.

        Documents are uploaded in batches to avoid using too much memory.

        Args:
            documents:
                List of LangChain Document objects.

            embedding_model:
                LangChain embedding model used to generate vectors.

            repository_id:
                Database ID of the repository.

        Returns:
            Number of documents successfully uploaded.
        """

        if not documents:
            return 0

        # Add repository information to every document.
        for document in documents:
            document.metadata["repository_id"] = repository_id

            # Make sure every document has a source field.
            if "source" not in document.metadata:
                document.metadata["source"] = document.metadata.get(
                    "relative_path",
                    "unknown",
                )

        vector_store = self.get_vector_store(embedding_model)

        # Upload documents in batches.
        batch_size = 100
        total_documents_uploaded = 0

        for start_index in range(
            0,
            len(documents),
            batch_size,
        ):
            document_batch = documents[
                start_index : start_index + batch_size
            ]

            try:
                await vector_store.aadd_documents(document_batch)

                total_documents_uploaded += len(document_batch)

            except Exception as error:
                logger.warning(
                    "Failed to upload Qdrant batch %d-%d: %s",
                    start_index,
                    start_index + len(document_batch),
                    error,
                )

        logger.info(
            "Uploaded %d documents into Qdrant for repository_id=%d",
            total_documents_uploaded,
            repository_id,
        )

        return total_documents_uploaded

    # -----------------------------------------------------------------------
    # Payload helper
    # -----------------------------------------------------------------------

    @staticmethod
    def flatten_payload(point: Any) -> dict:
        """
        Convert a Qdrant point payload into a simple dictionary.

        LangChain stores the document text under page_content and
        metadata under the metadata field.

        This method combines them into one dictionary.

        Args:
            point:
                Qdrant search result point.

        Returns:
            Dictionary containing:
                - content
                - metadata
                - score, when available
        """

        payload = point.payload or {}

        metadata = payload.get("metadata", {})

        result = {
            "content": payload.get("page_content", ""),
            **metadata,
        }

        # Search results contain a similarity score.
        if hasattr(point, "score") and point.score is not None:
            result["score"] = point.score

        return result

    # -----------------------------------------------------------------------
    # Vector search
    # -----------------------------------------------------------------------

    async def search_by_vector(
        self,
        query_vector: list[float],
        repository_id: int,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search for similar code chunks inside one repository.

        The repository_id filter prevents results from other repositories
        from being returned.

        Args:
            query_vector:
                Embedding vector representing the search query.

            repository_id:
                Database ID of the repository to search.

            top_k:
                Maximum number of results to return.

        Returns:
            List of matching code chunks with similarity scores.
        """

        search_results = await self.async_qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.repository_id",
                        match=MatchValue(
                            value=repository_id,
                        ),
                    ),
                ],
            ),
            limit=top_k,
            with_payload=True,
        )

        return [
            self.flatten_payload(point)
            for point in search_results.points
        ]

    # -----------------------------------------------------------------------
    # Name search
    # -----------------------------------------------------------------------

    async def search_by_name(
        self,
        symbol_name: str,
        repository_id: int,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search for code chunks by name.

        The search uses an exact match against the source metadata.

        Args:
            symbol_name:
                Name/path to search for.

            repository_id:
                Database ID of the repository.

            top_k:
                Maximum number of results.

        Returns:
            List of matching code chunks.
        """

        search_results = await self.async_qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.repository_id",
                        match=MatchValue(
                            value=repository_id,
                        ),
                    ),
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(
                            value=symbol_name,
                        ),
                    ),
                ],
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "score": 1.0,
                **self.flatten_payload(point),
            }
            for point in search_results[0]
        ]

    # -----------------------------------------------------------------------
    # LangChain retriever
    # -----------------------------------------------------------------------

    def get_retriever(
        self,
        embedding_model: Embeddings,
        repository_id: int,
        top_k: int = 5,
    ):
        """
        Create a LangChain retriever for one repository.

        The retriever performs similarity search and automatically
        restricts results to the specified repository.

        Args:
            embedding_model:
                LangChain embedding model.

            repository_id:
                Database ID of the repository.

            top_k:
                Number of code chunks to retrieve.

        Returns:
            LangChain retriever.
        """

        vector_store = self.get_vector_store(embedding_model)

        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k,
                "filter": {
                    "must": [
                        {
                            "key": "metadata.repository_id",
                            "match": {
                                "value": repository_id,
                            },
                        },
                    ],
                },
            },
        )

    # -----------------------------------------------------------------------
    # Delete repository data
    # -----------------------------------------------------------------------

    async def delete_repository(
        self,
        repository_id: int,
    ) -> None:
        """
        Delete all indexed chunks belonging to a repository.

        This is useful when a repository needs to be re-indexed.

        Args:
            repository_id:
                Database ID of the repository.
        """

        await self.async_qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.repository_id",
                        match=MatchValue(
                            value=repository_id,
                        ),
                    ),
                ],
            ),
        )

        logger.info(
            "Deleted all Qdrant chunks for repository_id=%d",
            repository_id,
        )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    async def close(self) -> None:
        """
        Close both Qdrant client connections.
        """

        await self.async_qdrant_client.close()
        self.qdrant_client.close()