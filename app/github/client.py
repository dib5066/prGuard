"""
GitHub REST API client used by the PRGuard review pipeline.

This module provides:
- Fetching repository information
- Fetching pull request information
- Fetching changed files
- Fetching pull request diffs
- Fetching pull request commits
- Posting PR comments
- Posting inline review comments
- Creating pull request reviews

Authentication is handled using GitHub App installation tokens.
"""

import logging
from typing import Any, Final, Literal
import httpx

from app.github.app_auth import get_installation_token

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitHub API configuration
# ---------------------------------------------------------------------------

GITHUB_API_BASE_URL: Final[str] = "https://api.github.com"
GITHUB_API_VERSION: Final[str] = "2022-11-28"


class GitHubClient:
    """
    Async client for interacting with the GitHub REST API.
    The caller only needs to provide the GitHub App installation ID.
    Example:
        async with GitHubClient(installation_id=12345) as github:
            repo = await github.get_repository(
                "octocat",
                "Hello-World"
            )
            print(repo["full_name"])
    """

    def __init__(self, installation_id: int) -> None:
        """
        Create a GitHub API client.
        Args:
            installation_id: GitHub App installation ID.
        """

        self._installation_id = installation_id

        self._http_client = httpx.AsyncClient(
            base_url=GITHUB_API_BASE_URL,
            timeout=httpx.Timeout(
                30.0,
                connect=10.0,
            ),
            follow_redirects=True,
        )

    async def close(self) -> None:
        """Close the HTTP client."""

        await self._http_client.aclose()

    # forword refrence
    async def __aenter__(self) -> "GitHubClient":
        """Allow usage with `async with`."""

        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Close the client when leaving the context."""

        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _headers(self) -> dict[str, str]:
        """
        Create authentication headers.

        Gets an installation access token and adds it
        to the Authorization header.
        """

        # calling for the token
        installation_token = await get_installation_token(self._installation_id)

        # returning the tokens
        return {
            "Authorization": f"Bearer {installation_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }

    async def _get(
        self,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make an authenticated GET request.
        """
        headers = await self._headers()
        response = await self._http_client.get(
            path,
            headers=headers,
            **kwargs,
        )
        response.raise_for_status()
        return response

    async def _post(
        self,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make an authenticated POST request.
        """
        headers = await self._headers()
        response = await self._http_client.post(
            path,
            headers=headers,
            **kwargs,
        )
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Repository operations
    # ------------------------------------------------------------------

    async def get_repository(
        self,
        owner: str,
        repository: str,
    ) -> dict[str, Any]:
        """
        Get repository information.

        Args:
            owner: GitHub username or organization.
            repo: Repository name.

        Returns:
            Repository information.
        """
        response = await self._get(f"/repos/{owner}/{repository}")
        logger.debug(
            "Fetched repository %s/%s",
            owner,
            repository,
        )
        return response.json()

    async def list_installation_repositories(self) -> list[dict[str, Any]]:
        """All repos this installation can access (paginated)."""
        all_repos: list[dict[str, Any]] = []
        page = 1
        while True:
            response = await self._get(
                "/installation/repositories",
                params={"per_page": 100, "page": page},
            )
            data = response.json()
            repos = data.get("repositories", [])
            if not repos:
                break
            all_repos.extend(repos)
            if len(repos) < 100:
                break
            page += 1
        logger.debug("Installation has %d repositories", len(all_repos))
        return all_repos

    # ------------------------------------------------------------------
    # Pull request operations
    # ------------------------------------------------------------------

    async def get_pull_request(
        self,
        owner: str,
        repository: str,
        pull_request_number: int,
    ) -> dict[str, Any]:
        """
        Get pull request information.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Pull request number.

        Returns:
            Pull request information.
        """
        response = await self._get(
            f"/repos/{owner}/{repository}/pulls/{pull_request_number}"
        )
        logger.debug(
            "Fetched PR #%d for %s/%s",
            pull_request_number,
            owner,
            repository,
        )
        return response.json()

    async def get_pull_request_changed_files(
        self,
        owner: str,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]:
        """
        Get all files changed in a pull request.

        GitHub returns a maximum number of files per request,
        so pagination is handled automatically.

        Returns:
            List of changed file objects.
        """
        all_files: list[dict[str, Any]] = []
        page = 1
        while True:
            response = await self._get(
                f"/repos/{owner}/{repository}/pulls/{pull_request_number}/files",
                params={
                    "per_page": 100,
                    "page": page,
                },
            )
            files = response.json()
            if not files:
                break
            all_files.extend(files)
            page += 1
        logger.debug(
            "Fetched %d changed files for PR #%d",
            len(all_files),
            pull_request_number,
        )
        return all_files

    async def get_pull_request_diff(
        self,
        owner: str,
        repository: str,
        pull_request_number: int,
    ) -> str:
        """
        Get the raw diff of a pull request.

        Returns:
            Raw unified diff as a string.
        """
        token = await get_installation_token(self._installation_id)
        response = await self._http_client.get(
            f"/repos/{owner}/{repository}/pulls/{pull_request_number}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3.diff",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            },
        )
        response.raise_for_status()
        logger.debug(
            "Fetched diff for PR #%d (%d bytes)",
            pull_request_number,
            len(response.text),
        )
        return response.text

    async def get_pull_request_commits(
        self,
        owner: str,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]:
        """
        Get all commits belonging to a pull request.

        Returns:
            List of commit objects.
        """
        all_commits: list[dict[str, Any]] = []
        page = 1
        while True:
            response = await self._get(
                f"/repos/{owner}/{repository}/pulls/{pull_request_number}/commits",
                params={
                    "per_page": 100,
                    "page": page,
                },
            )
            commits = response.json()
            if not commits:
                break
            all_commits.extend(commits)
            page += 1
        logger.debug(
            "Fetched %d commits for PR #%d",
            len(all_commits),
            pull_request_number,
        )
        return all_commits

    # ------------------------------------------------------------------
    # Comment operations
    # ------------------------------------------------------------------

    async def post_pull_request_comment(
        self,
        owner: str,
        repository: str,
        pull_request_number: int,
        body: str,
    ) -> dict[str, Any]:
        """
        Post a normal top-level comment on a pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: PR number.
            body: Markdown comment.
        """
        response = await self._post(
            f"/repos/{owner}/{repository}/issues/{pull_request_number}/comments",
            json={
                "body": body,
            },
        )
        logger.info(
            "Posted comment on PR #%d in %s/%s",
            pull_request_number,
            owner,
            repository,
        )
        return response.json()

    async def post_inline_review_comment(
        self,
        owner: str,
        repository: str,
        pull_request_number: int,
        body: str,
        *,
        commit_id: str,
        path: str,
        line: int,
        side: Literal["LEFT", "RIGHT"] = "RIGHT",
    ) -> dict[str, Any]:
        """
        Post an inline comment on a specific line of a PR.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: PR number.
            body: Markdown comment.
            commit_id: Commit SHA.
            path: File path.
            line: Line number.
            side: LEFT or RIGHT.
        """
        response = await self._post(
            f"/repos/{owner}/{repository}/pulls/{pull_request_number}/comments",
            json={
                "body": body,
                "commit_id": commit_id,
                "path": path,
                "line": line,
                "side": side,
            },
        )
        logger.info(
            "Posted review comment on PR #%d, %s:L%d",
            pull_request_number,
            path,
            line,
        )
        return response.json()

    # ------------------------------------------------------------------
    # Review operations
    # ------------------------------------------------------------------

    async def create_pull_request_review(
        self,
        owner: str,
        repository: str,
        pull_request_number: int,
        body: str,
        event: Literal[
            "COMMENT",
            "APPROVE",
            "REQUEST_CHANGES",
        ] = "COMMENT",
        *,
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Create a pull request review.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: PR number.
            body: Review message.
            event:
                COMMENT,
                APPROVE,
                or REQUEST_CHANGES.
            comments:
                Optional inline review comments.
        """
        payload: dict[str, Any] = {
            "body": body,
            "event": event,
        }
        if comments:
            payload["comments"] = comments
        response = await self._post(
            f"/repos/{owner}/{repository}/pulls/{pull_request_number}/reviews",
            json=payload,
        )
        logger.info(
            "Created %s review on PR #%d in %s/%s",
            event,
            pull_request_number,
            owner,
            repository,
        )
        return response.json()
