"""
Loads source-code files from a repository.

This module:

1. Finds source-code files in the repository.
2. Skips unnecessary directories.
3. Checks supported file extensions.
4. Skips files that are too large.
5. Loads files using LangChain TextLoader.
6. Adds useful metadata to each Document.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.core.config import settings


logger = logging.getLogger(__name__)


# ============================================================================
# DIRECTORIES TO SKIP
# ============================================================================

SKIP_DIRECTORIES = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "vendor",
    "target",
    "bin",
    "obj",
}


# ============================================================================
# FILE EXTENSION -> PROGRAMMING LANGUAGE
# ============================================================================

FILE_EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
}


def get_programming_language(file_extension: str) -> str:
    """
    Get the programming language for a file extension.

    Example:
        ".py" -> "python"
        ".js" -> "javascript"

    Args:
        file_extension:
            File extension such as ".py" or ".js".

    Returns:
        Programming language name.
        Returns "unknown" if the extension is not supported.
    """

    extension = file_extension.lower()

    return FILE_EXTENSION_TO_LANGUAGE.get(
        extension,
        "unknown",
    )


# ============================================================================
# FILE FILTERING
# ============================================================================


def should_include_file(file_path: Path) -> bool:
    """
    Check whether a file should be indexed.

    A file is included only when:

    1. It is not inside an ignored directory.
    2. Its extension is supported.
    3. Its size is below the configured limit.

    Args:
        file_path:
            Path of the file we want to check.

    Returns:
        True if the file should be indexed.
        False otherwise.
    """

    # ------------------------------------------------------------------------
    # Check ignored directories
    # ------------------------------------------------------------------------

    for directory in file_path.parts:

        if directory in SKIP_DIRECTORIES:
            return False

    # ------------------------------------------------------------------------
    # Check file extension
    # ------------------------------------------------------------------------

    extension = file_path.suffix.lower()

    if extension not in settings.INDEX_SUPPORTED_EXTENSIONS:
        return False

    # ------------------------------------------------------------------------
    # Check file size
    # ------------------------------------------------------------------------

    try:

        file_size_kb = file_path.stat().st_size / 1024

    except OSError:

        logger.warning(
            "Could not read file information: %s",
            file_path,
        )

        return False

    if file_size_kb > settings.INDEX_MAX_FILE_SIZE_KB:

        logger.debug(
            "Skipping large file: %s (%.1f KB)",
            file_path,
            file_size_kb,
        )

        return False

    return True


# ============================================================================
# LOAD REPOSITORY FILES
# ============================================================================


def load_repository(repository_directory: str) -> list[Document]:
    """
    Load supported source-code files from a repository.

    Each file becomes a LangChain Document.

    Metadata added to every document:

        relative_path:
            File path relative to the repository root.

        file_extension:
            File extension such as ".py".

        language:
            Programming language such as "python".

    Args:
        repository_directory:
            Path to the cloned repository.

    Returns:
        A list of LangChain Document objects.

    Raises:
        FileNotFoundError:
            If the repository directory does not exist.
    """

    # ------------------------------------------------------------------------
    # Convert the repository path into a Path object
    # ------------------------------------------------------------------------

    repository_path = Path(repository_directory)

    # ------------------------------------------------------------------------
    # Make sure the repository exists
    # ------------------------------------------------------------------------

    if not repository_path.exists():

        raise FileNotFoundError(
            f"Repository directory not found: {repository_directory}"
        )

    # ------------------------------------------------------------------------
    # Store all loaded documents here
    # ------------------------------------------------------------------------

    documents = []

    # ------------------------------------------------------------------------
    # Find files with supported extensions
    # ------------------------------------------------------------------------

    for extension in settings.INDEX_SUPPORTED_EXTENSIONS:

        file_pattern = f"*{extension}"

        for file_path in repository_path.rglob(file_pattern):

            # ---------------------------------------------------------------
            # Skip files that should not be indexed
            # ---------------------------------------------------------------

            if not should_include_file(file_path):
                continue

            try:

                # -----------------------------------------------------------
                # Load the file using LangChain
                # -----------------------------------------------------------

                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                    autodetect_encoding=True,
                )

                loaded_documents = loader.load()

                # -----------------------------------------------------------
                # Add metadata to every loaded document
                # -----------------------------------------------------------

                for document in loaded_documents:

                    relative_path = file_path.relative_to(
                        repository_path
                    )

                    document.metadata["relative_path"] = str(
                        relative_path
                    )

                    document.metadata["file_extension"] = (
                        file_path.suffix.lower()
                    )

                    document.metadata["language"] = (
                        get_programming_language(
                            file_path.suffix
                        )
                    )

                # -----------------------------------------------------------
                # Add documents to the final list
                # -----------------------------------------------------------

                documents.extend(loaded_documents)

            except Exception as error:

                logger.warning(
                    "Failed to load file %s: %s",
                    file_path,
                    error,
                )

    # ------------------------------------------------------------------------
    # Log the final result
    # ------------------------------------------------------------------------

    logger.info(
        "Loaded %d documents from repository %s",
        len(documents),
        repository_directory,
    )

    return documents