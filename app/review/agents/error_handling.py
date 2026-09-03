"""
Error handling agent for detecting missing or incorrect error handling.

This agent focuses on error handling problems that can cause:
- Application crashes
- Data loss
- Silent failures
- Resource leaks
- Unreliable behavior
"""

from app.review.agents.base import run_agent
from app.review.state import AgentMetricDict, ReviewFindingDict
from app.services.github_service import PRContext


AGENT_NAME = "error_handling_agent"
DOMAIN = "error handling"
CATEGORY = "error_handling"


DOMAIN_INSTRUCTIONS = """
Focus on the following types of error handling issues:

- Uncaught exceptions:
  - Operations that can raise exceptions without proper handling
  - Incorrect try/except structures
  - Bare except blocks that hide unexpected errors

- Missing error handling for external calls:
  - API requests
  - Database operations
  - File I/O
  - Network operations
  - External services

- Silent failures:
  - except: pass
  - Empty catch blocks
  - Ignoring important errors
  - Swallowing exceptions without logging or recovery

- Incorrect exception types:
  - Catching exceptions that cannot occur
  - Catching overly broad exception types
  - Catching exceptions that should propagate
  - Missing specific exception handling

- Resource cleanup:
  - Missing finally blocks when cleanup is required
  - Resources not closed after failures
  - Database connections or sessions left open
  - Files or network connections left open

- Error messages:
  - Exposing stack traces or internal implementation details
  - Revealing sensitive information in error responses
  - Returning misleading error messages

- Missing retry logic:
  - Transient network failures
  - Temporary API failures
  - Temporary database or service unavailability
  - Retry logic that could cause duplicate operations

- Unhandled async errors:
  - Missing await on coroutines
  - Fire-and-forget tasks without error handling
  - Background tasks whose exceptions are never observed
  - Incorrect async exception handling
"""


async def run_error_handling_agent(
    pr_context: PRContext,
    rag_context_str: str,
) -> tuple[list[ReviewFindingDict], AgentMetricDict]:
    """
    Run the error handling review agent.

    Args:
        pr_context: Pull request metadata, diff, files, and related information.
        rag_context_str: Relevant repository context retrieved by the RAG system.

    Returns:
        A tuple containing:
        - findings: Error handling issues found by the agent.
        - metrics: Agent execution metrics.
    """

    findings, metrics = await run_agent(
        agent_name=AGENT_NAME,
        domain=DOMAIN,
        category=CATEGORY,
        domain_instructions=DOMAIN_INSTRUCTIONS,
        pr_context=pr_context,
        rag_context_str=rag_context_str,
    )

    return findings, metrics