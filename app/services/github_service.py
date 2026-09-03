"""
GitHub Pull Request service.

This module collects all information needed to review a Pull Request.

It gets:
    * Pull Request details
    * Changed files
    * Pull Request diff
    * Commits
    * Full file content when requested

The collected information is stored in PRContext and can be used by
the review agents, RAG pipeline, and publishing service.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.github.client import GitHubClient
from app.github.tools import ParsedDiff, parse_unified_diff

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File extension -> programming language
# ---------------------------------------------------------------------------

# Using a tuple of (extension, language) for faster lookup
FILE_EXTENSION_MAPPING = (
    (".py", "python"),
    (".ts", "typescript"),
    (".tsx", "typescript"),
    (".js", "javascript"),
    (".jsx", "javascript"),
    (".go", "go"),
    (".java", "java"),
    (".kt", "kotlin"),
    (".rb", "ruby"),
    (".rs", "rust"),
    (".c", "c"),
    (".cpp", "cpp"),
    (".h", "c"),
    (".hpp", "cpp"),
    (".cs", "csharp"),
    (".swift", "swift"),
    (".php", "php"),
    (".scala", "scala"),
    (".sh", "shell"),
    (".bash", "shell"),
    (".yaml", "yaml"),
    (".yml", "yaml"),
    (".json", "json"),
    (".toml", "toml"),
    (".md", "markdown"),
    (".sql", "sql"),
    (".html", "html"),
    (".css", "css"),
    (".scss", "scss"),
)

def detect_language(file_path: str) -> Optional[str]:
    """
    Detect the programming language from a file extension.

    Example:
        "main.py" -> "python"
        "app.ts" -> "typescript"

    Returns:
        The programming language name, or None if unknown.
    """
    for ext, lang in FILE_EXTENSION_MAPPING:
        if file_path.endswith(ext):
            return lang
    return None

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PullRequestFile:
    """Information about one file changed in a Pull Request."""
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str
    sha: str
    language: Optional[str] = None
    content: Optional[str] = None

@dataclass
class PRContext:
    """
    All information required to review a Pull Request.

    This object is passed to the review pipeline, RAG pipeline,
    and publishing service.
    """
    # Pull Request identity
    owner: str
    repo: str
    pr_number: int

    # Pull Request information
    title: str
    body: str
    state: str
    author: str

    # Branch and commit information
    base_branch: str
    head_branch: str
    base_sha: str
    head_sha: str

    # Repository identity (from GitHub API head.repo.id)
    repository_id: Optional[int] = None

    # Optional Pull Request metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    html_url: Optional[str] = None

    # Changed files
    files: list[PullRequestFile] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)

    # Parsed diff
    parsed_diff: Optional[ParsedDiff] = None

    # Commits
    commits: list[dict[str, Any]] = field(default_factory=list)

    # Summary information
    total_additions: int = 0
    total_deletions: int = 0
    total_files_changed: int = 0
    total_commits: int = 0

# ---------------------------------------------------------------------------
# GitHub Pull Request service
# ---------------------------------------------------------------------------

class GitHubPRService:
    """
    Service responsible for collecting Pull Request information.

    This class does not review the code. Its job is only to collect and
    prepare the data that the review agents will use later.
    """

    # GitHub API endpoint templates
    _PR_ENDPOINT = "/repos/{owner}/{repo}/pulls/{pr_number}"
    _FILES_ENDPOINT = "/repos/{owner}/{repo}/pulls/{pr_number}/files"
    _COMMITS_ENDPOINT = "/repos/{owner}/{repo}/pulls/{pr_number}/commits"
    _CONTENTS_ENDPOINT = "/repos/{owner}/{repo}/contents/{file_path}"

    def __init__(self, installation_id: int):
        """
        Create a GitHub Pull Request service.

        Args:
            installation_id: GitHub App installation ID.
        """
        self.client = GitHubClient(installation_id)
        self._semaphore = asyncio.Semaphore(20)  # Limit concurrent file fetches

    async def __aenter__(self):
        """Allow the service to be used with 'async with'."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Close the GitHub client when the service is finished."""
        await self.close()

    async def close(self):
        """Explicitly close the underlying GitHub client."""
        await self.client.close()

    # -----------------------------------------------------------------------
    # Main Pull Request method
    # -----------------------------------------------------------------------

    async def get_pr_context(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        *,
        include_file_content: bool = False,
    ) -> PRContext:
        """
        Get all information required for a Pull Request review.

        The following GitHub API requests are made in parallel:
            1. Pull Request details
            2. Changed files
            3. Pull Request diff
            4. Pull Request commits

        Args:
            owner: GitHub username or organization name.
            repo: GitHub repository name.
            pr_number: Pull Request number.
            include_file_content: If True, fetch full content of each changed file.

        Returns:
            A PRContext object containing all Pull Request information.
        """
        logger.info(
            "Fetching PR context for %s/%s#%d",
            owner, repo, pr_number
        )

        # Fetch all four pieces of information concurrently
        (
            pull_request_data,
            changed_files_data,
            diff_text,
            commits_data,
        ) = await asyncio.gather(
            self.client.get_pull_request(owner, repo, pr_number),
            self.client.get_pull_request_changed_files(owner, repo, pr_number),
            self.client.get_pull_request_diff(owner, repo, pr_number),
            self.client.get_pull_request_commits(owner, repo, pr_number),
        )

        head = pull_request_data.get("head", {})
        base = pull_request_data.get("base", {})

        # Build the PR context
        pr_context = PRContext(
            repository_id=head.get("repo", {}).get("id"),
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            title=pull_request_data.get("title", ""),
            body=pull_request_data.get("body", "") or "",
            state=pull_request_data.get("state", "unknown"),
            author=pull_request_data.get("user", {}).get("login", "unknown"),
            base_branch=base.get("ref", ""),
            head_branch=head.get("ref", ""),
            base_sha=base.get("sha", ""),
            head_sha=head.get("sha", ""),
            created_at=pull_request_data.get("created_at"),
            updated_at=pull_request_data.get("updated_at"),
            html_url=pull_request_data.get("html_url"),
            commits=commits_data,
            total_commits=len(commits_data),
        )

        # Build file objects
        files = self._build_file_objects(changed_files_data)
        pr_context.files = files
        pr_context.file_paths = [f.filename for f in files]
        pr_context.total_files_changed = len(files)
        pr_context.total_additions = sum(f.additions for f in files)
        pr_context.total_deletions = sum(f.deletions for f in files)

        # Parse diff and validate consistency
        try:
            pr_context.parsed_diff = parse_unified_diff(diff_text)
            if pr_context.parsed_diff.total_files != pr_context.total_files_changed:
                logger.warning(
                    "Parsed file count (%d) differs from GitHub file count (%d) for PR #%d",
                    pr_context.parsed_diff.total_files,
                    pr_context.total_files_changed,
                    pr_number,
                )
        except Exception as error:
            logger.error("Failed to parse diff for PR #%d: %s", pr_number, error)
            pr_context.parsed_diff = None

        # Optionally fetch full file contents
        if include_file_content and files:
            contents = await self.get_file_contents_batch(
                owner=owner,
                repo=repo,
                file_paths=pr_context.file_paths,
                git_ref=pr_context.head_sha,
            )
            for file_obj in files:
                file_obj.content = contents.get(file_obj.filename)

        logger.info(
            "Fetched PR context: %s/%s#%d - %d files, +%d -%d, %d commits",
            owner,
            repo,
            pr_number,
            pr_context.total_files_changed,
            pr_context.total_additions,
            pr_context.total_deletions,
            pr_context.total_commits,
        )

        return pr_context

    def _build_file_objects(self, changed_files_data: list[dict[str, Any]]) -> list[PullRequestFile]:
        """Convert raw file data from GitHub into PullRequestFile objects."""
        files = []
        for file_data in changed_files_data:
            filename = file_data.get("filename", "")
            files.append(
                PullRequestFile(
                    filename=filename,
                    status=file_data.get("status", "modified"),
                    additions=file_data.get("additions", 0),
                    deletions=file_data.get("deletions", 0),
                    changes=file_data.get("changes", 0),
                    patch=file_data.get("patch", ""),
                    sha=file_data.get("sha", ""),
                    language=detect_language(filename),
                )
            )
        return files

    # -----------------------------------------------------------------------
    # Get one complete file
    # -----------------------------------------------------------------------

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        file_path: str,
        git_ref: str,
    ) -> Optional[str]:
        """
        Get the complete content of one file.

        Args:
            owner: GitHub username or organization.
            repo: GitHub repository name.
            file_path: Path of the file inside the repository.
            git_ref: Commit SHA, branch name, or tag.

        Returns:
            File content as a string, or None if the file cannot be read
            (e.g., is a directory, binary, or not found).
        """
        try:
            # Use the GitHub Contents API
            response = await self.client._get(
                self._CONTENTS_ENDPOINT.format(owner=owner, repo=repo, file_path=file_path),
                params={"ref": git_ref},
            )
            data = response.json()

            # If the path is a directory, GitHub returns a list
            if isinstance(data, list):
                logger.warning("Path '%s' is a directory, not a file", file_path)
                return None

            # Try base64-encoded content first
            encoding = data.get("encoding", "")
            content = data.get("content", "")
            if encoding == "base64" and content:
                try:
                    decoded = base64.b64decode(content)
                    return decoded.decode("utf-8")
                except UnicodeDecodeError:
                    logger.debug("File '%s' is not valid UTF-8", file_path)
                    return None

            # Fallback: download via raw URL if available
            download_url = data.get("download_url")
            if download_url:
                resp = await self.client._http_client.get(
                    download_url,
                    headers=await self.client._headers(),
                )
                resp.raise_for_status()
                try:
                    return resp.text
                except UnicodeDecodeError:
                    logger.debug("File '%s' (raw) is not UTF-8", file_path)
                    return None

            return None

        except Exception as error:
            logger.warning("Failed to fetch file '%s' at %s: %s", file_path, git_ref, error)
            return None

    # -----------------------------------------------------------------------
    # Get multiple files
    # -----------------------------------------------------------------------

    async def get_file_contents_batch(
        self,
        owner: str,
        repo: str,
        file_paths: list[str],
        git_ref: str,
    ) -> dict[str, str]:
        """
        Get the contents of multiple files in parallel, with a concurrency limit.

        Args:
            owner: GitHub username or organization.
            repo: GitHub repository name.
            file_paths: List of file paths to fetch.
            git_ref: Commit SHA, branch name, or tag.

        Returns:
            Dictionary mapping file path -> file content.
        """
        if not file_paths:
            return {}

        logger.debug("Fetching content for %d files at %s", len(file_paths), git_ref[:8])

        # Use a semaphore to limit concurrent requests
        async def fetch_one(file_path: str) -> tuple[str, Optional[str]]:
            async with self._semaphore:
                content = await self.get_file_content(
                    owner=owner,
                    repo=repo,
                    file_path=file_path,
                    git_ref=git_ref,
                )
            return file_path, content

        tasks = [fetch_one(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        file_contents: dict[str, str] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Error during batch fetch: %s", result)
                continue
            file_path, content = result
            if content is not None:
                file_contents[file_path] = content

        logger.debug("Fetched %d/%d file contents", len(file_contents), len(file_paths))
        return file_contents