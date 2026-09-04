"""
Background worker for the PRGuard review pipeline.

This module orchestrates the complete PR review workflow:

1. Upsert GitHub installation in the database.
2. Upsert repository in the database.
3. Fetch pull request information from GitHub.
4. Upsert pull request in the database.
5. Index the repository for RAG when required.
6. Run the baseline AI review.
7. Fetch generated findings.
8. Publish the review to GitHub.

The worker is triggered by GitHub webhook events such as:

- pull_request.opened
- pull_request.synchronize
"""

import logging
from datetime import datetime, timezone

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.events import publish
from app.repositories.installation_repo import InstallationRepository
from app.repositories.pull_request_repo import PullRequestRepo
from app.repositories.repository_repo import RepositoryRepo
from app.repositories.review_repo import ReviewRepo
from app.services.github_service import GitHubPRService
from app.services.publishing_service import ReviewPublishingService
from app.services.review_service import ReviewService


logger = logging.getLogger(__name__)


async def run_review(
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
    account_type: str = "Organization",
) -> None:
    """
    Run the complete PRGuard review pipeline.

    This function is called as a background task after
    a GitHub pull request webhook is received.

    Args:
        installation_id:
            GitHub App installation ID.

        repo_full_name:
            Full GitHub repository name in the format:
            "owner/repository".

        pr_number:
            Pull request number.

        account_type:
            GitHub account type ("User" or "Organization").
            Defaults to "Organization".
    """

    # =========================================================================
    # STEP 1: Extract repository information
    # =========================================================================

    owner, repository_name = repo_full_name.split("/", 1)

    logger.info(
        "Starting review pipeline for %s#%d "
        "(installation %d)",
        repo_full_name,
        pr_number,
        installation_id,
    )

    # =========================================================================
    # STEP 2: Create database session
    # =========================================================================

    async with AsyncSessionLocal() as session:
        try:
            # =================================================================
            # STEP 3: Create repository objects
            # =================================================================

            installation_repository = InstallationRepository(
                session
            )

            repository_repository = RepositoryRepo(
                session
            )

            pull_request_repository = PullRequestRepo(
                session
            )

            # =================================================================
            # STEP 4: Upsert GitHub installation
            # =================================================================

            installation = await installation_repository.upsert(
                installation_id=installation_id,
                account_name=owner,
                account_type=account_type,
            )

            # Snapshot the fields we need later. The repo-list sync below can
            # call ``session.rollback()``, which expires every attribute on
            # every instance in the session — and an AsyncSession cannot
            # implicitly lazy-reload an expired attribute on access (it
            # raises MissingGreenlet). Reading them now, while the instance
            # is fresh, avoids that entirely.
            installation_pk = installation.installation_id
            installation_user_id = installation.user_id
            installation_deleted_at = installation.deleted_at
            installation_suspended_at = installation.suspended_at

            if installation_deleted_at is not None or installation_suspended_at is not None:
                logger.info(
                    "Installation %s for %s#%d is %s — skipping review",
                    installation_id,
                    repo_full_name,
                    pr_number,
                    "deleted" if installation_deleted_at else "suspended",
                )
                await session.commit()
                return

            # Refresh the accessible-repo list for linked installs so the
            # dashboard stays accurate (one GitHub call).
            if installation_user_id is not None:
                try:
                    from app.api.github_app import _sync_installation_repos

                    await _sync_installation_repos(session, installation_id)
                    await session.commit()
                except Exception as sync_error:
                    logger.warning(
                        "Repo-list sync failed for installation %s: %s",
                        installation_id,
                        sync_error,
                    )
                    await session.rollback()

            # =================================================================
            # STEP 5: Fetch PR context from GitHub
            # =================================================================

            async with GitHubPRService(
                installation_id
            ) as github_service:

                pull_request_context = (
                    await github_service.get_pr_context(
                        owner=owner,
                        repo=repository_name,
                        pr_number=pr_number,
                    )
                )

            logger.info(
                "PR context fetched for %s#%d: "
                "%d files, +%d -%d",
                repo_full_name,
                pr_number,
                pull_request_context.total_files_changed,
                pull_request_context.total_additions,
                pull_request_context.total_deletions,
            )

            # =================================================================
            # STEP 6: Upsert repository
            # =================================================================

            # IMPORTANT:
            # Use the real GitHub repository ID here.
            #
            # Do not use github_id=0 because github_id is unique in
            # the database and multiple repositories would conflict.
            #
            # This assumes PRContext contains the GitHub repository ID.
            repository_github_id = pull_request_context.repository_id

            repository = await repository_repository.upsert(
                github_id=repository_github_id,
                name=repository_name,
                full_name=repo_full_name,
                installation_id=installation_pk,
            )

            # =================================================================
            # STEP 7: Prepare pull request creation date
            # =================================================================

            pull_request_created_at = datetime.now(
                timezone.utc
            )

            if pull_request_context.created_at:
                try:
                    pull_request_created_at = (
                        datetime.fromisoformat(
                            pull_request_context.created_at.replace(
                                "Z",
                                "+00:00",
                            )
                        )
                    )

                except (ValueError, TypeError):
                    logger.warning(
                        "Unable to parse PR creation date "
                        "for %s#%d. Using current time.",
                        repo_full_name,
                        pr_number,
                    )

            # =================================================================
            # STEP 8: Upsert pull request
            # =================================================================

            pull_request = await pull_request_repository.upsert(
                repository_id=repository.id,
                pull_request_number=pr_number,
                title=pull_request_context.title,
                state=pull_request_context.state,
                base_sha=pull_request_context.base_sha,
                head_sha=pull_request_context.head_sha,
                user_login=pull_request_context.author,
                created_at=pull_request_created_at,
            )

            # =================================================================
            # STEP 9: Save database changes
            # =================================================================

            await session.commit()

            logger.info(
                "Repository and pull request records saved "
                "for %s#%d",
                repo_full_name,
                pr_number,
            )

            # =================================================================
            # STEP 9.5: Reuse or create the review record
            # =================================================================
            #
            # If this exact commit was already reviewed (reopen, duplicate
            # webhook delivery, no-op synchronize), do NOT re-run the agents
            # — that only produces different findings each time. Reuse the
            # existing review so GitHub + the dashboard stay stable.

            review_repo = ReviewRepo(session)
            head_sha = pull_request_context.head_sha

            reusable = await review_repo.get_reusable_review(
                pull_request.id, head_sha
            )
            if reusable is not None:
                logger.info(
                    "Reusing %s review %d for %s#%d @ %s — skipping re-run",
                    reusable.status,
                    reusable.id,
                    repo_full_name,
                    pr_number,
                    head_sha[:8],
                )
                publish(
                    reusable.id,
                    {
                        "type": "done",
                        "status": reusable.status,
                        "phase": reusable.current_phase or "completed",
                        "message": "Reused existing review for this commit",
                    },
                )
                return

            review_record = await review_repo.create_review(
                pull_request.id,
                head_sha=head_sha,
                repository_id=repository.id,
            )
            review_record = await review_repo.mark_running(review_record)
            review_id_for_events = review_record.id

            async def emit_phase(phase: str, message: str, **extra) -> None:
                """Persist the phase and push it to any live SSE clients."""
                nonlocal review_record
                review_record = await review_repo.update_phase(
                    review_record,
                    phase=phase,
                    message=message,
                )
                await session.commit()
                publish(
                    review_id_for_events,
                    {
                        "type": "phase",
                        "status": "RUNNING",
                        "phase": phase,
                        "message": message,
                        **extra,
                    },
                )

            await emit_phase(
                "fetching_context",
                "Fetching PR context from GitHub...",
            )

            logger.info(
                "Created review %d for PR %s#%d",
                review_record.id,
                repo_full_name,
                pr_number,
            )

            # =================================================================
            # STEP 10: Index repository for RAG when required
            # =================================================================

            from app.rag.indexer import RepositoryIndexer

            repository_indexer = RepositoryIndexer(
                session
            )

            repository_record = (
                await repository_repository.get_by_github_id(
                    repository.github_id
                )
            )

            should_index_repository = False

            if repository_record is None:
                logger.warning(
                    "Repository record not found after upsert "
                    "for %s",
                    repo_full_name,
                )

            elif not repository_record.is_indexed:
                should_index_repository = True

                logger.info(
                    "Repository %s has not been indexed yet",
                    repo_full_name,
                )

            elif repository_record.last_indexed_at:
                days_since_indexing = (
                    datetime.now(timezone.utc)
                    - repository_record.last_indexed_at
                ).days

                if (
                    days_since_indexing
                    > settings.INDEX_STALENESS_DAYS
                ):
                    should_index_repository = True

                    logger.info(
                        "Repository %s index is stale "
                        "(%d days old)",
                        repo_full_name,
                        days_since_indexing,
                    )

            if should_index_repository:
                logger.info(
                    "Indexing repository %s before review",
                    repo_full_name,
                )

                await emit_phase(
                    "indexing",
                    "Indexing repository for RAG...",
                )

                try:
                    await repository_indexer.index_repository(
                        repository_id=repository_record.id,
                        clone_url=(
                            f"https://github.com/"
                            f"{repo_full_name}.git"
                        ),
                        head_sha=pull_request_context.head_sha,
                        installation_id=installation_id,
                        user_id=installation_user_id,
                    )

                    await session.commit()

                    logger.info(
                        "Repository indexing completed for %s",
                        repo_full_name,
                    )

                except Exception as indexing_error:
                    logger.warning(
                        "Repository indexing failed for %s: %s. "
                        "Continuing with AI review.",
                        repo_full_name,
                        indexing_error,
                    )

                    # ---------------------------------------------------------
                    # RAG is currently optional.
                    #
                    # If indexing fails, the review continues using the
                    # normal PR diff/context.
                    # ---------------------------------------------------------

                    await session.rollback()

            # =================================================================
            # STEP 11: Build RAG context (if available)
            # =================================================================

            await emit_phase(
                "building_rag",
                "Building RAG context...",
            )

            rag_context = None

            if (
                repository_record
                and repository_record.is_indexed
            ):
                try:
                    from app.rag.context import ContextBuilder

                    context_builder = ContextBuilder(
                        user_id=installation_user_id
                    )
                    rag_context = (
                        await context_builder.build_context(
                            pr_context=pull_request_context,
                            repository_id=repository_record.id,
                        )
                    )

                    logger.info(
                        "RAG context built for %s: "
                        "%d related chunks",
                        repo_full_name,
                        len(rag_context.related_chunks),
                    )

                except Exception as rag_error:
                    logger.warning(
                        "RAG context building failed for %s: %s. "
                        "Continuing without RAG.",
                        repo_full_name,
                        rag_error,
                    )

            # =================================================================
            # STEP 12: Run multi-agent AI review
            # =================================================================

            await emit_phase(
                "running_agents",
                "Running 5 AI review agents in parallel...",
            )

            review_service = ReviewService(
                session
            )

            review_id = (
                await review_service.run_multi_agent_review(
                    pr_context=pull_request_context,
                    pull_request_id=pull_request.id,
                    rag_context=rag_context,
                    existing_review_id=review_record.id,
                    repository_id=repository.id,
                )
            )

            await session.commit()

            logger.info(
                "Review completed for %s#%d "
                "(review_id=%d)",
                repo_full_name,
                pr_number,
                review_id,
            )

            # =================================================================
            # STEP 13: Fetch review and findings
            # =================================================================

            review, findings = (
                await review_service.review_repository
                .get_by_id_with_findings(
                    review_id
                )
            )

            # =================================================================
            # STEP 14: Validate findings against actual file content
            # =================================================================

            await emit_phase(
                "validating",
                "Validating findings against file content...",
                raw_findings=len(findings),
            )

            from app.review.validator import validate_findings

            validated_findings = validate_findings(
                findings=[
                    {
                        "severity": f.severity,
                        "category": f.category,
                        "title": f.title,
                        "description": f.description,
                        "file_path": f.file_path,
                        "line_number": f.line_number,
                        "evidence": f.evidence,
                        "confidence": float(f.confidence),
                    }
                    for f in findings
                ],
                pr_context=pull_request_context,
            )

            # Filter to only valid findings for publishing
            valid_findings = [
                f for f in validated_findings if f.is_valid
            ]

            # Persist the validator's calibrated confidence + status back onto
            # the stored findings so the dashboard shows exactly what GitHub
            # shows. `validate_findings` returns one ValidatedFinding per input
            # in order, so we can zip against the DB rows.
            valid_ids = {id(vf) for vf in valid_findings}
            for db_finding, vf in zip(findings, validated_findings):
                try:
                    await review_service.finding_repository.update_finding_validation(
                        db_finding,
                        confidence=float(vf.confidence),
                        validation_status=vf.validation_status.value,
                        validation_notes=vf.validation_notes or None,
                        is_published=id(vf) in valid_ids,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist validation for finding %s",
                        db_finding.id,
                    )
            await session.commit()

            logger.info(
                "Validation complete for %s#%d: "
                "%d/%d findings valid",
                repo_full_name,
                pr_number,
                len(valid_findings),
                len(findings),
            )

            publish(
                review_id_for_events,
                {
                    "type": "findings_ready",
                    "total": len(findings),
                    "valid": len(valid_findings),
                },
            )

            # =================================================================
            # STEP 15: Publish validated findings to GitHub
            # =================================================================

            await emit_phase(
                "publishing",
                "Publishing review to GitHub...",
            )

            if review:
                # Always publish — even with zero valid findings the
                # developer should see that PRGuard ran (APPROVE + a
                # short "no issues found" summary).
                async with ReviewPublishingService(
                    installation_id
                ) as publisher:

                    await publisher.publish_review(
                        owner=owner,
                        repository_name=repository_name,
                        pull_request_number=pr_number,
                        review=review,
                        findings=valid_findings,
                        head_sha=pull_request_context.head_sha,
                        validated_findings=validated_findings,
                    )

                logger.info(
                    "Published review for %s#%d "
                    "(%d valid / %d total findings)",
                    repo_full_name,
                    pr_number,
                    len(valid_findings),
                    len(findings),
                )

            else:
                logger.warning(
                    "No review record available for "
                    "publishing for %s#%d",
                    repo_full_name,
                    pr_number,
                )

            # =================================================================
            # STEP 16: Mark review as completed
            # =================================================================

            review_record = await review_repo.update_phase(
                review_record,
                phase="completed",
                message="Review completed successfully",
            )
            await review_repo.mark_completed(review_record)
            await session.commit()

            publish(
                review_id_for_events,
                {
                    "type": "done",
                    "status": "COMPLETED",
                    "phase": "completed",
                    "message": "Review completed successfully",
                },
            )

        except Exception as error:
            # =================================================================
            # REVIEW PIPELINE FAILED
            # =================================================================

            logger.error(
                "Review pipeline failed for %s#%d: %s",
                repo_full_name,
                pr_number,
                error,
                exc_info=True,
            )

            # The failing exception may already have poisoned this
            # transaction. Roll it back FIRST, then record the FAILED
            # state on a brand-new session so the review never gets
            # stuck in RUNNING.
            try:
                await session.rollback()
            except Exception:
                pass

            # Use the plain int captured at review creation — NOT
            # review_record.id. The rollback above expired that ORM
            # instance, and an attribute access here would trigger an
            # implicit lazy-load (MissingGreenlet) inside the error handler.
            review_record_id = locals().get("review_id_for_events")

            if review_record_id is not None:
                try:
                    async with AsyncSessionLocal() as fail_session:
                        fail_repo = ReviewRepo(fail_session)
                        fresh = await fail_repo.get_by_id(review_record_id)
                        if fresh is not None:
                            await fail_repo.update_phase(
                                fresh,
                                phase="failed",
                                message=f"Review failed: {error}",
                            )
                            await fail_repo.mark_failed(fresh, str(error))
                            await fail_session.commit()
                except Exception:
                    logger.exception(
                        "Could not persist FAILED state for review %s",
                        review_record_id,
                    )

                publish(
                    review_record_id,
                    {
                        "type": "done",
                        "status": "FAILED",
                        "phase": "failed",
                        "message": f"Review failed: {error}",
                    },
                )

            raise