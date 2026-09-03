"""
Repository Layer for PRGuard.

Provides data-access classes that encapsulate all database queries.
Services use repositories instead of importing ORM models directly.
"""

from app.repositories.base import BaseRepository
from app.repositories.installation_repo import InstallationRepository
from app.repositories.repository_repo import RepositoryRepo
from app.repositories.pull_request_repo import PullRequestRepo
from app.repositories.review_repo import ReviewRepo, FindingRepo, ReviewRunRepo

__all__ = [
    "BaseRepository",
    "InstallationRepository",
    "RepositoryRepo",
    "PullRequestRepo",
    "ReviewRepo",
    "FindingRepo",
    "ReviewRunRepo",
]
