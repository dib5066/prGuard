"""
Baseline AI review service for PRGuard.

This service performs the first version of the PRGuard review pipeline:

1. Create a review record.
2. Mark the review as RUNNING.
3. Build a prompt using pull request information.
4. Send the prompt to ChatGroq.
5. Parse the AI response.
6. Save valid findings in the database.
7. Save agent performance metrics.
8. Mark the review as COMPLETED.

If something goes wrong, the review is marked as FAILED.

This is currently a single-agent baseline reviewer.

Later, this service can be extended with:
- Security Agent
- Correctness Agent
- Error Handling Agent
- Code Quality Agent
- RAG
- LangGraph orchestration
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.events import publish
from app.models.review import Finding, Review, ReviewRun
from app.repositories.review_repo import (
    FindingRepository,
    ReviewRepository,
    ReviewRunRepository,
)
from app.review.agents.base import extract_text_content
from app.review.normalize import (
    coerce_confidence,
    coerce_line_number,
    normalize_category,
    normalize_severity,
)
from app.services.github_service import PRContext


logger = logging.getLogger(__name__)


# ============================================================================
# AI SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """
You are an expert code reviewer for a software project.

Your job is to review a pull request and identify real bugs,
security problems, correctness issues, and meaningful code-quality
problems.

Return your response as a JSON object with a "findings" array.

Each finding must contain:

- "severity":
    One of "critical", "high", "medium", "low"

- "category":
    One of "correctness", "security", "error_handling",
    "performance", "quality", "testing"

- "title":
    Short summary of the issue.
    Maximum 100 characters.

- "description":
    Detailed explanation of the problem.
    Use 2-4 sentences.

- "file_path":
    Path of the file containing the issue.

- "line_number":
    Line number where the issue occurs.
    Use null if there is no specific line.

- "evidence":
    Relevant code snippet that demonstrates the problem.

- "confidence":
    Number between 0.0 and 1.0 representing your confidence
    that the issue is real.

Rules:

- Focus on actual problems, not personal style preferences.
- Only report issues you are confident about.
- Confidence must be greater than 0.5.
- Reference exact files and lines whenever possible.
- Do not invent problems.
- Prioritize correctness and security over style.
- If the pull request has no issues, return:

{
    "findings": []
}
"""


# ============================================================================
# FINDING DATA STRUCTURE
# ============================================================================


@dataclass
class ReviewFinding:
    """
    Represents one finding returned by the AI reviewer.

    This object is used internally before the finding
    is saved into the database.
    """

    severity: str
    category: str
    title: str
    description: str
    file_path: str
    line_number: int | None
    evidence: str | None
    confidence: float


# ============================================================================
# PROMPT BUILDER
# ============================================================================


def build_review_prompt(
    pr_context: PRContext,
    rag_context: str = "",
) -> str:
    """
    Build the prompt sent to the AI reviewer.

    The prompt contains:

    - Pull request information.
    - Pull request description.
    - Changed files.
    - GitHub file patches/diffs.
    - Optional RAG context.

    Args:
        pr_context:
            Pull request information collected from GitHub.

        rag_context:
            Optional repository context retrieved from RAG.

    Returns:
        The complete review prompt.
    """

    prompt_sections: list[str] = []

    # ------------------------------------------------------------------------
    # Pull request information
    # ------------------------------------------------------------------------

    prompt_sections.append(
        f"# Pull Request: {pr_context.title}"
    )

    prompt_sections.append(
        f"Author: {pr_context.author}"
    )

    prompt_sections.append(
        f"Base: {pr_context.base_branch} → "
        f"Head: {pr_context.head_branch}"
    )

    prompt_sections.append(
        f"Files changed: {pr_context.total_files_changed} "
        f"(+{pr_context.total_additions} "
        f"-{pr_context.total_deletions})"
    )

    # ------------------------------------------------------------------------
    # Pull request description
    # ------------------------------------------------------------------------

    if pr_context.body:
        pull_request_description = pr_context.body[:2000]

        if len(pr_context.body) > 2000:
            pull_request_description += "\n... (truncated)"

        prompt_sections.append(
            "\n## PR Description\n"
            f"{pull_request_description}"
        )

    # ------------------------------------------------------------------------
    # Changed files
    # ------------------------------------------------------------------------

    prompt_sections.append("\n## Changed Files")

    for changed_file in pr_context.files:
        prompt_sections.append(
            f"### {changed_file.filename} "
            f"({changed_file.status})"
        )

        # ------------------------------------------------------------
        # Add the GitHub patch when available.
        # ------------------------------------------------------------

        if changed_file.patch:
            file_patch = changed_file.patch[:5000]

            if len(changed_file.patch) > 5000:
                file_patch += "\n... (truncated)"

            prompt_sections.append(
                "```diff\n"
                f"{file_patch}\n"
                "```"
            )

        prompt_sections.append("")

    # ------------------------------------------------------------------------
    # RAG context
    # ------------------------------------------------------------------------

    if rag_context:
        prompt_sections.append(
            "\n## Repository Context\n"
            f"{rag_context}"
        )

    return "\n".join(prompt_sections)


# ============================================================================
# LLM RESPONSE PARSER
# ============================================================================


def parse_llm_response(
    response_text: str,
) -> list[ReviewFinding]:
    """
    Convert the AI response into ReviewFinding objects.

    The AI normally returns JSON, but it may sometimes
    wrap the JSON inside a markdown code fence.

    Supported formats:

    Normal JSON:
        {"findings": [...]}

    Markdown JSON:
        ```json
        {"findings": [...]}
        ```

    Returns:
        A list of valid ReviewFinding objects.

        If parsing fails, an empty list is returned.
    """

    # ------------------------------------------------------------------------
    # Check for an empty response.
    # ------------------------------------------------------------------------

    if not response_text:
        logger.warning(
            "AI reviewer returned an empty response."
        )
        return []

    cleaned_response = response_text.strip()

    # ------------------------------------------------------------------------
    # Remove markdown code fences.
    # ------------------------------------------------------------------------

    if cleaned_response.startswith("```"):
        response_lines = cleaned_response.splitlines()

        # Remove opening fence, such as ```json.
        response_lines = response_lines[1:]

        # Remove closing fence.
        if (
            response_lines
            and response_lines[-1].strip() == "```"
        ):
            response_lines = response_lines[:-1]

        cleaned_response = "\n".join(response_lines).strip()

    # ------------------------------------------------------------------------
    # Parse JSON.
    # ------------------------------------------------------------------------

    try:
        parsed_response = json.loads(cleaned_response)

    except json.JSONDecodeError as error:
        logger.warning(
            "Failed to parse AI response as JSON: %s",
            error,
        )
        return []

    # ------------------------------------------------------------------------
    # Extract findings from the response object.
    # ------------------------------------------------------------------------

    if isinstance(parsed_response, dict):
        if "findings" in parsed_response:
            parsed_response = parsed_response["findings"]

        elif "issues" in parsed_response:
            parsed_response = parsed_response["issues"]

        else:
            logger.warning(
                "AI response does not contain 'findings' or 'issues'. "
                "Keys: %s",
                list(parsed_response.keys()),
            )
            return []

    # ------------------------------------------------------------------------
    # Make sure we received a list.
    # ------------------------------------------------------------------------

    if not isinstance(parsed_response, list):
        logger.warning(
            "AI findings response is not a list. "
            "Received type: %s",
            type(parsed_response).__name__,
        )
        return []

    # ------------------------------------------------------------------------
    # Convert every finding into ReviewFinding.
    # ------------------------------------------------------------------------

    findings: list[ReviewFinding] = []

    for finding_data in parsed_response:
        if not isinstance(finding_data, dict):
            logger.warning(
                "Skipping invalid finding because it is not an object."
            )
            continue

        try:
            confidence = coerce_confidence(
                finding_data.get("confidence"), 0.5
            )

            finding = ReviewFinding(
                severity=normalize_severity(
                    finding_data.get("severity")
                ),
                category=normalize_category(
                    finding_data.get("category")
                ),
                title=str(
                    finding_data.get(
                        "title",
                        "Untitled finding",
                    )
                )[:255],
                description=str(
                    finding_data.get(
                        "description",
                        "",
                    )
                ),
                file_path=str(
                    finding_data.get(
                        "file_path",
                        "",
                    )
                    or ""
                ),
                line_number=coerce_line_number(
                    finding_data.get("line_number")
                ),
                evidence=finding_data.get(
                    "evidence"
                ),
                confidence=confidence,
            )

            findings.append(finding)

        except (ValueError, TypeError) as error:
            logger.warning(
                "Skipping malformed AI finding: %s",
                error,
            )

    return findings


# ============================================================================
# REVIEW SERVICE
# ============================================================================


class ReviewService:
    """
    Handles the complete baseline PR review.

    Current pipeline:

        GitHub
           ↓
        PRContext
           ↓
        Prompt Builder
           ↓
        ChatGroq/ChatGoogleGenerativeAI
           ↓
        Response Parser
           ↓
        Database

    The current implementation uses one AI reviewer.

    Later, this service can become the entry point for
    the LangGraph multi-agent workflow.
    """

    def __init__(
        self,
        session: AsyncSession,
    ):
        """
        Initialize the review service.

        Args:
            session:
                Active asynchronous SQLAlchemy database session.
        """

        self.session = session

        # --------------------------------------------------------------------
        # Database repositories
        # --------------------------------------------------------------------

        self.review_repository = ReviewRepository(
            session
        )

        self.finding_repository = FindingRepository(
            session
        )

        self.review_run_repository = ReviewRunRepository(
            session
        )

        # --------------------------------------------------------------------
        # Gemini chat client (baseline single-shot reviewer)
        # --------------------------------------------------------------------

        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured — cannot run the review "
                "pipeline. Set it in backend/.env."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            api_key=settings.GEMINI_API_KEY,
            temperature=settings.GEMINI_TEMPERATURE,
            max_tokens=settings.GEMINI_MAX_TOKENS,
            timeout=settings.GEMINI_TIMEOUT_SECONDS,
            max_retries=settings.GEMINI_MAX_RETRIES,
            response_mime_type="application/json",
        )

    # ========================================================================
    # RUN BASELINE REVIEW
    # ========================================================================

    async def run_baseline_review(
        self,
        pr_context: PRContext,
        pull_request_id: int,
    ) -> int:
        """
        Run the complete baseline AI review.

        Flow:

        1. Create review.
        2. Mark review as RUNNING.
        3. Build prompt.
        4. Call ChatGroq/ChatGoogleGenerativeAI.
        5. Parse AI response.
        6. Save findings.
        7. Save agent metrics.
        8. Mark review as COMPLETED.

        If an error occurs:

        - Mark the review as FAILED.
        - Store the error message.
        - Re-raise the exception.

        Args:
            pr_context:
                Pull request information fetched from GitHub.

            pull_request_id:
                Database ID of the pull request.

        Returns:
            ID of the created review.
        """

        # ====================================================================
        # STEP 1: Create review record
        # ====================================================================

        review = await self.review_repository.create_review(
            pull_request_id
        )

        review_id = review.id

        logger.info(
            "Created review %d for pull request %d",
            review_id,
            pull_request_id,
        )

        try:
            # =================================================================
            # STEP 2: Mark review as RUNNING
            # =================================================================

            review = await self.review_repository.mark_running(
                review
            )

            # =================================================================
            # STEP 3: Build AI prompt
            # =================================================================

            user_prompt = build_review_prompt(
                pr_context
            )

            logger.info(
                "Built review prompt for PR %s/%s#%d "
                "(%d characters)",
                pr_context.owner,
                pr_context.repo,
                pr_context.pr_number,
                len(user_prompt),
            )

            # =================================================================
            # STEP 4: Send prompt to ChatGroq
            # =================================================================

            start_time = time.monotonic()

            messages = [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=user_prompt
                ),
            ]

            response = await self.llm.ainvoke(
                messages
            )

            # =================================================================
            # STEP 5: Calculate response metrics
            # =================================================================

            latency_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            logger.info(
                "AI response received in %d ms",
                latency_ms,
            )

            tokens_used = None

            if (
                hasattr(response, "usage_metadata")
                and response.usage_metadata
            ):
                tokens_used = response.usage_metadata.get(
                    "total_tokens"
                )

            # =================================================================
            # STEP 6: Get response text
            # =================================================================

            response_text = extract_text_content(response)

            logger.debug(
                "AI response length: %d characters",
                len(response_text),
            )

            # =================================================================
            # STEP 7: Parse AI findings
            # =================================================================

            findings = parse_llm_response(
                response_text
            )

            logger.info(
                "AI reviewer returned %d findings",
                len(findings),
            )

            # =================================================================
            # STEP 8: Save findings
            # =================================================================

            stored_finding_count = 0

            for finding in findings:
                # ------------------------------------------------------------
                # Ignore very low-confidence findings.
                # ------------------------------------------------------------

                if finding.confidence < 0.3:
                    logger.debug(
                        "Skipping low-confidence finding: "
                        "'%s' (confidence %.2f)",
                        finding.title,
                        finding.confidence,
                    )
                    continue

                await self.finding_repository.create_finding(
                    review_id=review_id,
                    severity=finding.severity,
                    category=finding.category,
                    title=finding.title,
                    description=finding.description,
                    file_path=finding.file_path,
                    line_number=finding.line_number,
                    evidence=finding.evidence,
                    confidence=finding.confidence,
                )

                stored_finding_count += 1

            # =================================================================
            # STEP 9: Save agent performance metrics
            # =================================================================

            await self.review_run_repository.create_run(
                review_id=review_id,
                agent_name="baseline_reviewer",
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )

            # =================================================================
            # STEP 10: Mark review as COMPLETED
            # =================================================================

            await self.review_repository.mark_completed(
                review
            )

            logger.info(
                "Baseline review completed successfully. "
                "PR=%s/%s#%d, findings=%d, latency=%dms, tokens=%s",
                pr_context.owner,
                pr_context.repo,
                pr_context.pr_number,
                stored_finding_count,
                latency_ms,
                tokens_used or 0,
            )

            return review_id

        except Exception as error:
            # =================================================================
            # REVIEW FAILED
            # =================================================================

            logger.error(
                "Baseline review failed for PR %s/%s#%d: %s",
                pr_context.owner,
                pr_context.repo,
                pr_context.pr_number,
                error,
                exc_info=True,
            )

            # ----------------------------------------------------------------
            # Store failure information.
            # ----------------------------------------------------------------

            await self.review_repository.mark_failed(
                review,
                str(error),
            )

            # ----------------------------------------------------------------
            # Re-raise the exception.
            # ----------------------------------------------------------------

            raise

    # ========================================================================
    # RUN MULTI-AGENT REVIEW (Phase 7)
    # ========================================================================

    async def run_multi_agent_review(
        self,
        pr_context: PRContext,
        pull_request_id: int,
        rag_context=None,
        existing_review_id: int | None = None,
        resume: bool = False,
        repository_id: int | None = None,
    ) -> int:
        """
        Run the multi-agent LangGraph review pipeline.

        This method uses LangGraph to orchestrate 5 specialized agents
        that run in parallel, each focusing on one review dimension:

            1. Correctness (logic bugs, off-by-one)
            2. Security (injection, auth bypass, secrets)
            3. Error Handling (uncaught exceptions, missing paths)
            4. Code Quality (smells, duplication, complexity)
            5. Testing (missing tests, untested edge cases)

        An aggregator node deduplicates, ranks, and merges findings.

        Flow:

        1. Create review.
        2. Mark review as RUNNING.
        3. Build RAG context string.
        4. Run LangGraph workflow (5 parallel agents + aggregator).
        5. Store findings in DB.
        6. Store per-agent metrics.
        7. Mark review as COMPLETED.

        If an error occurs:

        - Mark the review as FAILED.
        - Store the error message.
        - Re-raise the exception.

        Args:
            pr_context:
                Pull request information fetched from GitHub.

            pull_request_id:
                Database ID of the pull request.

            rag_context:
                Optional ReviewContext from Phase 6 RAG pipeline.

        Returns:
            ID of the created review.
        """
        from app.rag.context import ContextBuilder
        from app.review.graph import get_review_graph
        from app.review.state import ReviewState

        # ====================================================================
        # STEP 1: Create or reuse review record
        # ====================================================================

        if existing_review_id:
            review = await self.review_repository.get_by_id(existing_review_id)
            if not review:
                review = await self.review_repository.create_review(
                    pull_request_id
                )
            review_id = review.id
        else:
            review = await self.review_repository.create_review(
                pull_request_id
            )
            review_id = review.id

        logger.info(
            "Created multi-agent review %d for pull request %d",
            review_id,
            pull_request_id,
        )

        try:
            # =================================================================
            # STEP 2: Mark review as RUNNING
            # =================================================================

            review = await self.review_repository.mark_running(
                review
            )

            review = await self.review_repository.update_phase(
                review,
                phase="running_agents",
                message="Running 5 specialized AI review agents...",
            )

            # Commit and release the DB connection BEFORE the multi-minute
            # LLM graph run. Otherwise this transaction sits idle for the
            # whole run and Neon (PgBouncer + idle-in-transaction timeout +
            # autosuspend) kills the connection, which then blows up the
            # findings-insert below with "SSL connection has been closed".
            await self.session.commit()

            # =================================================================
            # STEP 3: Build RAG context string
            # =================================================================

            rag_context_str = ""

            if rag_context:
                context_builder = ContextBuilder()
                rag_context_str = context_builder.format_context_for_prompt(
                    rag_context
                )

            logger.info(
                "Built RAG context string for PR %s/%s#%d "
                "(%d characters)",
                pr_context.owner,
                pr_context.repo,
                pr_context.pr_number,
                len(rag_context_str),
            )

            # =================================================================
            # STEP 4: Run LangGraph workflow
            # ==================================================================

            graph = get_review_graph()

            # When a checkpointer is active, key graph state by review id so a
            # retried / resumed review continues from the last completed agent
            # instead of re-running all five (the cause of "different findings
            # every run"). `resume=True` continues pending work; otherwise a
            # fresh input starts the run.
            from app.review.checkpointer import get_checkpointer

            graph_config = None
            if get_checkpointer() is not None:
                # Checkpointer thread is scoped by repo (per Task 4).
                thread_id = (
                    f"repo:{repository_id}:review:{review_id}"
                    if repository_id is not None
                    else f"review:{review_id}"
                )
                graph_config = {"configurable": {"thread_id": thread_id}}

            initial_state = ReviewState(
                pr_context=pr_context,
                rag_context=rag_context,
                rag_context_str=rag_context_str,
                review_id=review_id,
                findings=[],
                agent_metrics=[],
            )
            graph_input = None if (resume and graph_config) else initial_state

            logger.info(
                "Running multi-agent review for %s/%s#%d",
                pr_context.owner,
                pr_context.repo,
                pr_context.pr_number,
            )

            # Tell any live SSE clients which agents are about to run.
            agent_names = [
                "correctness_agent",
                "security_agent",
                "error_handling_agent",
                "quality_agent",
                "testing_agent",
            ]
            publish(
                review_id,
                {"type": "agents_started", "agents": agent_names},
            )

            # Hard ceiling on the whole multi-agent run so a hung LLM
            # call can never wedge the review (and its DB session) forever.
            # Stream the graph so each agent's completion is pushed to the
            # frontend as it happens instead of only at the end.
            graph_timeout = settings.GEMINI_TIMEOUT_SECONDS * 6
            final_state: dict = {}

            async def _run_graph_streaming() -> None:
                nonlocal final_state
                async for mode, chunk in graph.astream(
                    graph_input,
                    config=graph_config,
                    stream_mode=["updates", "values"],
                ):
                    if mode == "values":
                        # Full accumulated state after each super-step;
                        # the last one is the authoritative final state.
                        final_state = chunk
                        continue

                    # mode == "updates": {node_name: node_return} — emitted
                    # as each agent finishes, so the frontend sees agents
                    # complete one by one.
                    for node_name, update in (chunk or {}).items():
                        metrics = (update or {}).get("agent_metrics") or []
                        metric = metrics[0] if metrics else {}
                        if node_name == "aggregator":
                            continue
                        publish(
                            review_id,
                            {
                                "type": "agent",
                                "agent": node_name,
                                "status": "done",
                                "findings": metric.get(
                                    "findings_count",
                                    len((update or {}).get("findings") or []),
                                ),
                                "latency_ms": metric.get("latency_ms"),
                            },
                        )

            try:
                await asyncio.wait_for(
                    _run_graph_streaming(), timeout=graph_timeout
                )
            except asyncio.TimeoutError as timeout_error:
                raise RuntimeError(
                    f"Multi-agent review timed out after {graph_timeout:.0f}s"
                ) from timeout_error

            # Prefer the aggregator's deduplicated + ranked output. Fall
            # back to the raw appended list only if the aggregator node
            # did not run (e.g. graph shape changed).
            findings = final_state.get("aggregated_findings")
            if findings is None:
                findings = final_state.get("findings", [])
                logger.warning(
                    "Aggregator output missing; using raw agent findings "
                    "(%d) without deduplication",
                    len(findings),
                )
            agent_metrics = final_state.get("agent_metrics", [])

            logger.info(
                "Multi-agent graph produced %d aggregated findings "
                "(from %d raw)",
                len(findings),
                len(final_state.get("findings", [])),
            )

            # =================================================================
            # STEP 5: Store findings
            # ==================================================================

            stored_finding_count = 0

            for finding in findings:
                if finding.get("confidence", 0) < 0.3:
                    continue

                await self.finding_repository.create_finding(
                    review_id=review_id,
                    severity=finding["severity"],
                    category=finding["category"],
                    title=finding["title"],
                    description=finding["description"],
                    file_path=finding["file_path"],
                    line_number=finding.get("line_number"),
                    evidence=finding.get("evidence"),
                    confidence=finding["confidence"],
                )

                stored_finding_count += 1

            # =================================================================
            # STEP 6: Store per-agent metrics
            # ==================================================================

            for metric in agent_metrics:
                await self.review_run_repository.create_run(
                    review_id=review_id,
                    agent_name=metric["agent_name"],
                    latency_ms=metric["latency_ms"],
                    tokens_used=metric.get("tokens_used"),
                )

            # =================================================================
            # STEP 7: Mark review as COMPLETED
            # ==================================================================

            await self.review_repository.mark_completed(
                review
            )

            logger.info(
                "Multi-agent review completed successfully. "
                "PR=%s/%s#%d, findings=%d, agents=%d",
                pr_context.owner,
                pr_context.repo,
                pr_context.pr_number,
                stored_finding_count,
                len(agent_metrics),
            )

            return review_id

        except Exception as error:
            # =================================================================
            # REVIEW FAILED
            # =================================================================

            logger.error(
                "Multi-agent review failed for PR %s/%s#%d: %s",
                pr_context.owner,
                pr_context.repo,
                pr_context.pr_number,
                error,
                exc_info=True,
            )

            # ----------------------------------------------------------------
            # Store failure information.
            # ----------------------------------------------------------------

            await self.review_repository.mark_failed(
                review,
                str(error),
            )

            # ----------------------------------------------------------------
            # Re-raise the exception.
            # ----------------------------------------------------------------

            raise