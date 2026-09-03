"""
Generic database repository.

This module contains common CRUD operations that can be reused
by all database models in PRGuard.

Example models that can use these methods:
    - UserRepository
    - RepositoryRepository
    - PullRequestRepository
    - ReviewRepository
"""

import logging
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic model type
# ---------------------------------------------------------------------------

# ModelType can represent any SQLAlchemy model that inherits from Base
ModelType = TypeVar("ModelType", bound=Base)

# ---------------------------------------------------------------------------
# Base repository
# ---------------------------------------------------------------------------

class BaseRepository(Generic[ModelType]):
    """
    Common database operations for all PRGuard models.

    The repository handles:
        - Get one record
        - Get multiple records
        - Create a record
        - Update a record
        - Delete a record

    Domain‑specific repositories can inherit from this class and
    add their own custom queries.
    """

    def __init__(self, model: type[ModelType], session: AsyncSession):
        """
        Create a repository.

        Args:
            model: SQLAlchemy model class.
            session: Active asynchronous database session.
        """
        # which database model working with --> user
        self.model = model
        #which database connection should we use ---> database session
        self.session = session

    # -----------------------------------------------------------------------
    # Get one record using primary key
    # -----------------------------------------------------------------------

    async def get_by_id(self, record_id: int) -> ModelType | None:
        """
        Get one database record using its primary key.

        Args:
            record_id: Primary key of the record.

        Returns:
            The database record if found, else None.
        """
        return await self.session.get(self.model, record_id)

    # -----------------------------------------------------------------------
    # Get multiple records
    # -----------------------------------------------------------------------

    # offset mean skip first number of record, limit all you know
    async def get_all(self, *, offset: int = 0, limit: int = 100) -> list[ModelType]:
        """
        Get multiple records with pagination.

        Args:
            offset: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of database records.
        """
        query = select(self.model).offset(offset).limit(limit)
        result = await self.session.execute(query)
        return result.scalars().all()

    # -----------------------------------------------------------------------
    # Create a new databse object
    # -----------------------------------------------------------------------

    async def create(self, **values: object) -> ModelType:
        """
        Create a new database record.

        Example:
            await repository.create(name="PRGuard", full_name="dibyanshu/prguard")

        Args:
            **values: Column values for the new record.

        Returns:
            The newly created database record.
        """
        record = self.model(**values)
        self.session.add(record)

        # Flush to send INSERT to the database; transaction controlled by caller.
        await self.session.flush()

        # Refresh to populate generated fields (e.g., primary key).
        await self.session.refresh(record)

        logger.debug("Created %s with ID %s", self.model.__name__, record.id)
        return record

    # -----------------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------------

    async def update(self, record: ModelType, **values: object) -> ModelType:
        """
        Update an existing database record.

        Example:
            await repository.update(repository, name="new-name")

        Args:
            record: Database record to update.
            **values: Column names and their new values.

        Returns:
            The updated database record.
        """
        for field_name, new_value in values.items():
            setattr(record, field_name, new_value)

        await self.session.flush()
        await self.session.refresh(record)

        logger.debug("Updated %s with ID %s", self.model.__name__, record.id)
        return record

    # -----------------------------------------------------------------------
    # Delete
    # -----------------------------------------------------------------------

    async def delete(self, record: ModelType) -> None:
        """
        Delete a database record.

        Args:
            record: Database record to delete.
        """
        await self.session.delete(record)
        await self.session.flush()

        logger.debug("Deleted %s with ID %s", self.model.__name__, record.id)