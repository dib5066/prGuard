"""
LangChain text splitter configuration for code-aware chunking.

This module splits source code into smaller chunks for the RAG pipeline.

It uses LangChain's RecursiveCharacterTextSplitter with
language-aware separators so that code is split around logical
structures such as classes, functions, and blank lines.
"""

import logging
from functools import lru_cache

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from app.core.config import settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


@lru_cache()
def get_token_encoder():
    """
    Get the tiktoken encoder used for counting tokens.

    Returns:
        A tiktoken encoder if tiktoken is installed.
        None if tiktoken is not available.
    """
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")

    except Exception:
        return None


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a piece of text.

    Uses tiktoken when available.

    If tiktoken is not installed, this function uses
    a rough estimate where 4 characters are treated as
    approximately 1 token.

    Args:
        text: Text whose tokens should be counted.

    Returns:
        Approximate number of tokens.
    """
    token_encoder = get_token_encoder()

    if token_encoder is not None:
        return len(token_encoder.encode(text))

    # Rough estimate:
    # 1 token ≈ 4 characters
    return len(text) // 4


# ---------------------------------------------------------------------------
# Default Python code splitter
# ---------------------------------------------------------------------------


@lru_cache()
def create_code_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the default code splitter.

    The default language is Python because Python's
    structure is commonly used in the backend code.

    LangChain provides Python-specific separators that
    try to split code around classes, functions, and
    logical blocks.

    Returns:
        Configured RecursiveCharacterTextSplitter.
    """

    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=settings.CHUNK_MAX_TOKENS,
        chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens,
        keep_separator=True,
    )

    logger.debug(
        "Created Python code splitter: max_tokens=%d, overlap_tokens=%d",
        settings.CHUNK_MAX_TOKENS,
        settings.CHUNK_OVERLAP_TOKENS,
    )

    return code_splitter


# ---------------------------------------------------------------------------
# Language-specific splitters
# ---------------------------------------------------------------------------


LANGUAGE_MAP = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "go": Language.GO,
    "java": Language.JAVA,
    "ruby": Language.RUBY,
    "rust": Language.RUST,
}


def get_splitter_for_language(
    language: str,
) -> RecursiveCharacterTextSplitter:
    """
    Create a code splitter for the requested programming language.

    If LangChain supports a language-specific splitter, that
    splitter is used.

    Otherwise, a generic code splitter is used.

    Args:
        language:
            Programming language name such as:
            "python", "javascript", or "typescript".

    Returns:
        Configured RecursiveCharacterTextSplitter.
    """

    language_name = language.lower()

    language_enum = LANGUAGE_MAP.get(language_name)

    # Use LangChain's built-in language-aware splitter
    # when the language is supported.
    if language_enum is not None:

        return RecursiveCharacterTextSplitter.from_language(
            language=language_enum,
            chunk_size=settings.CHUNK_MAX_TOKENS,
            chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
            length_function=count_tokens,
            keep_separator=True,
        )

    # -----------------------------------------------------------------------
    # Generic code splitter
    # -----------------------------------------------------------------------

    logger.debug(
        "No language-specific splitter found for '%s'. "
        "Using generic code separators.",
        language,
    )

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_MAX_TOKENS,
        chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
        separators=[
            "\nclass ",             # Class declarations
            "\ndef ",               # Function declarations
            "\nfunc ",              # Go functions
            "\nasync def ",         # Python async functions
            "\nasync function ",    # JavaScript async functions
            "\nfunction ",          # JavaScript functions
            "\nexport ",            # JavaScript/TypeScript exports
            "\nconst ",             # JavaScript/TypeScript declarations
            "\n\n",                 # Blank lines
            "\n",                   # Single newlines
            " ",                    # Spaces
            "",                     # Character-level fallback
        ],
        length_function=count_tokens,
        keep_separator=True,
    )