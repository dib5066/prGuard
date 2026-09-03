"""
HuggingFace embedding model configuration using LangChain.

This module provides two embedding options:

1. HuggingFace Inference API
   - Uses HuggingFace's hosted inference API.
   - Does not download the model locally.
   - Requires a HuggingFace API token.

2. Local HuggingFace model
   - Downloads the model once and caches it locally.
   - Does not require an API key after the model is downloaded.
   - Runs inference on the local machine.

Default model:
    sentence-transformers/all-MiniLM-L6-v2

The default model produces 384-dimensional embeddings.
"""

import logging
from functools import lru_cache

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpointEmbeddings,
)

from app.core.config import settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace API embedding model
# ---------------------------------------------------------------------------


@lru_cache()
def get_embedding_model():
    """
    Get the configured embedding model.

    If ``EMBEDDING_USE_LOCAL`` is set, a locally-run sentence-transformers
    model is used (the HuggingFace Inference API no longer hosts most of
    these models). Otherwise the Inference API is used with whichever
    spelling of the API key was provided.

    The model is created once and reused (``lru_cache``).
    """

    if settings.EMBEDDING_USE_LOCAL:
        return get_local_embedding_model()

    embedding_model = HuggingFaceEndpointEmbeddings(
        model=settings.EMBEDDING_MODEL,
        huggingfacehub_api_token=settings.embedding_api_key,
    )

    logger.info(
        "Initialized HuggingFace API embedding model: %s "
        "(dimension=%d)",
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_DIMENSION,
    )

    return embedding_model


# ---------------------------------------------------------------------------
# Local HuggingFace embedding model
# ---------------------------------------------------------------------------


@lru_cache()
def get_local_embedding_model():
    """
    Get a local HuggingFace embedding model.

    The model is downloaded the first time it is used and then
    cached locally.

    This option does not require a HuggingFace API token.

    The model runs on the CPU and normalizes the generated
    embeddings.

    Returns:
        HuggingFaceEmbeddings:
            LangChain local HuggingFace embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    logger.info(
        "Initialized local HuggingFace embedding model: %s",
        settings.EMBEDDING_MODEL,
    )

    return embedding_model