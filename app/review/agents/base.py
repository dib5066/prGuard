"""
Base agent runner for PRGuard specialized review agents.

Each specialized agent provides:

1. Agent name.
2. Review domain.
3. Finding category.
4. Domain-specific instructions.

This module handles the shared workflow:

1. Build the system prompt.
2. Build the user prompt.
3. Call ChatGroq.
4. Parse the JSON response.
5. Filter low-confidence findings.
6. Collect latency and token metrics.

The same runner is reused by all specialized review agents.
"""

import asyncio
import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.review.normalize import (
    coerce_confidence,
    coerce_line_number,
    normalize_category,
    normalize_severity,
)
from app.review.state import AgentMetricDict, ReviewFindingDict
from app.services.github_service import PRContext


logger = logging.getLogger(__name__)


# Limits how many agents hit the LLM API at the same time. Created lazily so
# it binds to the running event loop. Size comes from settings.
_llm_semaphore: asyncio.Semaphore | None = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(
            max(1, settings.GEMINI_AGENT_CONCURRENCY)
        )
    return _llm_semaphore


# ============================================================================
# BASE SYSTEM PROMPT
# ============================================================================

BASE_SYSTEM_TEMPLATE = """
You are a specialized code reviewer focused on {domain}.

Your task is to review a pull request and find issues related to {domain}.

{domain_specific_instructions}

You MUST return your response as a JSON object with a "findings" array.

Each finding must have:

- "severity":
    One of "critical", "high", "medium", "low"

- "category":
    "{category}"

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
    Use an integer or null if not applicable.

- "evidence":
    Relevant code snippet that demonstrates the issue.

- "confidence":
    Number between 0.0 and 1.0.

Rules:

- Focus ONLY on {domain} issues.
- Ignore issues belonging to other categories.
- Only report issues you are confident about.
- Confidence should be greater than 0.5.
- Reference exact files and lines whenever possible.
- Do NOT make up issues that do not exist.
- Prioritize real problems over style preferences.
- If no {domain} issues are found, return {{"findings": []}}.
"""


# ============================================================================
# USER PROMPT BUILDER
# ============================================================================


def build_user_prompt(
    pr_context: PRContext,
    rag_context_str: str = "",
) -> str:
    """
    Build the user prompt containing PR information and RAG context.

    Args:
        pr_context:
            Pull request information fetched from GitHub.

        rag_context_str:
            Optional repository context retrieved from the RAG system.

    Returns:
        Complete user prompt for the review agent.
    """

    prompt_sections: list[str] = []

    # ========================================================================
    # Pull request metadata
    # ========================================================================

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

    # ========================================================================
    # Pull request description
    # ========================================================================

    if pr_context.body:
        pull_request_description = pr_context.body[:2000]

        if len(pr_context.body) > 2000:
            pull_request_description += "\n... (truncated)"

        prompt_sections.append(
            "\n## PR Description\n"
            f"{pull_request_description}"
        )

    # ========================================================================
    # Changed files
    # ========================================================================

    prompt_sections.append(
        "\n## Changed Files"
    )

    for changed_file in pr_context.files:
        prompt_sections.append(
            f"### {changed_file.filename} "
            f"({changed_file.status})"
        )

        # --------------------------------------------------------------------
        # Add GitHub patch when available.
        # --------------------------------------------------------------------

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

    # ========================================================================
    # RAG context
    # ========================================================================

    if rag_context_str:
        prompt_sections.append(
            "\n## Repository Context\n"
            f"{rag_context_str}"
        )

    return "\n".join(prompt_sections)


# ============================================================================
# LLM RESPONSE PARSER
# ============================================================================


def extract_text_content(response) -> str:
    """Return the plain text of an LLM response.

    Modern models (Gemini 3, Claude with thinking, etc.) return
    ``response.content`` as a list of typed blocks
    (``{"type": "text", "text": "..."}`` plus reasoning/signature blocks)
    instead of a bare string. ``str(list)`` would produce a Python repr,
    which then fails JSON parsing — so pull the text blocks out here.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Keep text blocks; skip reasoning / tool / image blocks.
                if block.get("type", "text") == "text" and block.get("text"):
                    parts.append(block["text"])
        if parts:
            return "".join(parts)

    return str(content or "")


def _salvage_json(text: str):
    """Best-effort recovery of a JSON object/array from a noisy string.

    Handles the common LLM failure modes:
    - JSON preceded/followed by prose → slice the outermost brackets.
    - A truncated trailing element → keep up to the last complete ``}``
      and close the array.
    Returns the parsed value, or ``None`` if nothing usable is found.
    """
    candidates: list[str] = []

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    # Truncated array (possibly nested in an object): take from the first
    # "[" up to the last complete "}" and close it.
    arr_start = text.find("[")
    last_obj = text.rfind("}")
    if arr_start != -1 and last_obj > arr_start:
        candidates.append(text[arr_start : last_obj + 1] + "]")

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def parse_agent_response(
    response_text: str,
    agent_name: str,
) -> list[ReviewFindingDict]:
    """
    Parse an AI agent response into ReviewFindingDict objects.

    Supports:

    1. JSON object with a "findings" array.
    2. JSON object with an "issues" array.
    3. Plain JSON array.
    4. JSON wrapped in markdown code fences.

    Args:
        response_text:
            Raw response returned by the LLM.

        agent_name:
            Name of the agent that produced the response.

    Returns:
        List of parsed findings.
    """

    # ========================================================================
    # STEP 1: Validate response
    # ========================================================================

    if not response_text:
        logger.warning(
            "Agent %s returned an empty response",
            agent_name,
        )
        return []

    cleaned_response = response_text.strip()

    # ========================================================================
    # STEP 2: Remove markdown code fences
    # ========================================================================

    if cleaned_response.startswith("```"):
        response_lines = cleaned_response.splitlines()

        # Remove opening fence such as ```json.
        response_lines = response_lines[1:]

        # Remove closing fence.
        if (
            response_lines
            and response_lines[-1].strip() == "```"
        ):
            response_lines = response_lines[:-1]

        cleaned_response = "\n".join(
            response_lines
        ).strip()

    # ========================================================================
    # STEP 3: Parse JSON
    # ========================================================================

    try:
        parsed_response = json.loads(
            cleaned_response
        )

    except json.JSONDecodeError:
        # The model sometimes wraps the JSON in prose or truncates a
        # trailing element. Try to salvage the outermost {...} / [...].
        parsed_response = _salvage_json(cleaned_response)
        if parsed_response is None:
            logger.warning(
                "Agent %s returned unparseable JSON (%d chars); "
                "no findings recorded",
                agent_name,
                len(cleaned_response),
            )
            return []
        logger.info(
            "Agent %s: recovered findings from malformed JSON response",
            agent_name,
        )

    # ========================================================================
    # STEP 4: Extract findings list
    # ========================================================================

    if isinstance(parsed_response, dict):
        if "findings" in parsed_response:
            parsed_response = parsed_response["findings"]

        elif "issues" in parsed_response:
            parsed_response = parsed_response["issues"]

        else:
            logger.warning(
                "Agent %s response does not contain "
                "'findings' or 'issues'",
                agent_name,
            )
            return []

    # ========================================================================
    # STEP 5: Validate findings list
    # ========================================================================

    if not isinstance(parsed_response, list):
        logger.warning(
            "Agent %s response is not a list. "
            "Received type: %s",
            agent_name,
            type(parsed_response).__name__,
        )
        return []

    # ========================================================================
    # STEP 6: Convert findings
    # ========================================================================

    findings: list[ReviewFindingDict] = []

    for finding_data in parsed_response:
        if not isinstance(finding_data, dict):
            logger.warning(
                "Agent %s returned an invalid finding object",
                agent_name,
            )
            continue

        try:
            finding: ReviewFindingDict = {
                "severity": normalize_severity(
                    finding_data.get("severity")
                ),
                "category": normalize_category(
                    finding_data.get("category")
                ),
                "title": str(
                    finding_data.get(
                        "title",
                        "Untitled",
                    )
                )[:255],
                "description": str(
                    finding_data.get(
                        "description",
                        "",
                    )
                ),
                "file_path": str(
                    finding_data.get(
                        "file_path",
                        "",
                    )
                    or ""
                ),
                "line_number": coerce_line_number(
                    finding_data.get("line_number")
                ),
                "evidence": finding_data.get(
                    "evidence"
                ),
                "confidence": coerce_confidence(
                    finding_data.get("confidence"), 0.5
                ),
                "agent": agent_name,
            }

            findings.append(finding)

        except (ValueError, TypeError) as error:
            logger.warning(
                "Agent %s returned a malformed finding: %s",
                agent_name,
                error,
            )

    return findings


# ============================================================================
# AGENT RUNNER
# ============================================================================


async def run_agent(
    agent_name: str,
    domain: str,
    category: str,
    domain_instructions: str,
    pr_context: PRContext,
    rag_context_str: str,
) -> tuple[
    list[ReviewFindingDict],
    AgentMetricDict,
]:
    """
    Run one specialized review agent.

    All specialized agents use this shared function.

    Each specialized agent only needs to provide:

    - Agent name.
    - Domain.
    - Finding category.
    - Domain-specific instructions.
    - PR context.
    - RAG context.

    Args:
        agent_name:
            Unique identifier for the agent.

        domain:
            Human-readable review domain.

        category:
            Finding category stored in the result.

        domain_instructions:
            Instructions specific to this review domain.

        pr_context:
            Pull request information from GitHub.

        rag_context_str:
            Formatted repository context from RAG.

    Returns:
        Tuple containing:

        - List of findings.
        - Agent performance metrics.
    """

    # ========================================================================
    # STEP 1: Build prompts
    # ========================================================================

    system_prompt = BASE_SYSTEM_TEMPLATE.format(
        domain=domain,
        category=category,
        domain_specific_instructions=domain_instructions,
    )

    user_prompt = build_user_prompt(
        pr_context=pr_context,
        rag_context_str=rag_context_str,
    )

    # ========================================================================
    # STEP 2: Create LLM client
    # ========================================================================

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        api_key=settings.GEMINI_API_KEY,
        temperature=settings.GEMINI_TEMPERATURE,
        max_tokens=settings.GEMINI_MAX_TOKENS,
        timeout=settings.GEMINI_TIMEOUT_SECONDS,
        max_retries=settings.GEMINI_MAX_RETRIES,
        # Ask Gemini to return raw JSON so parsing is reliable.
        response_mime_type="application/json",
    )

    # ========================================================================
    # STEP 3: Call LLM (bounded concurrency to respect API rate limits)
    # ========================================================================

    start_time = time.monotonic()

    try:
        async with _get_llm_semaphore():
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content=system_prompt
                    ),
                    HumanMessage(
                        content=user_prompt
                    ),
                ]
            )

        # ====================================================================
        # STEP 4: Calculate latency
        # ====================================================================

        latency_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        # ====================================================================
        # STEP 5: Extract token usage
        # ====================================================================

        tokens_used = None

        if (
            hasattr(response, "usage_metadata")
            and response.usage_metadata
        ):
            tokens_used = response.usage_metadata.get(
                "total_tokens"
            )

        # ====================================================================
        # STEP 6: Parse AI response
        # ====================================================================

        response_text = extract_text_content(response)

        findings = parse_agent_response(
            response_text=response_text,
            agent_name=agent_name,
        )

        # ====================================================================
        # STEP 7: Filter low-confidence findings
        # ====================================================================

        filtered_findings = [
            finding
            for finding in findings
            if finding["confidence"] >= 0.3
        ]

        # ====================================================================
        # STEP 8: Build agent metrics
        # ====================================================================

        metrics: AgentMetricDict = {
            "agent_name": agent_name,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "findings_count": len(filtered_findings),
        }

        logger.info(
            "Agent %s completed: "
            "%d findings, %dms, %d tokens",
            agent_name,
            len(filtered_findings),
            latency_ms,
            tokens_used or 0,
        )

        return filtered_findings, metrics

    except Exception as error:
        # ====================================================================
        # AGENT FAILED
        # ====================================================================

        latency_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        logger.error(
            "Agent %s failed after %dms: %s",
            agent_name,
            latency_ms,
            error,
            exc_info=True,
        )

        metrics: AgentMetricDict = {
            "agent_name": agent_name,
            "latency_ms": latency_ms,
            "tokens_used": None,
            "findings_count": 0,
        }

        return [], metrics
        