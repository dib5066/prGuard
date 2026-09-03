"""
Correctness agent for detecting logic and runtime correctness issues.

This agent focuses on problems that can cause incorrect behavior, such as:
- Logic bugs
- Off-by-one errors
- Incorrect algorithm usage
- Null/None dereferences
- Type mismatches
- Race conditions in async code
- Resource leaks
- Incorrect string formatting or template rendering
"""

from app.review.agents.base import run_agent
from app.review.state import AgentMetricDict, ReviewFindingDict
from app.services.github_service import PRContext

AGENT_NAME = "correctness_agent"
DOMAIN = "correctness"
CATEGORY = "correctness"

DOMAIN_INSTRUCTIONS = """
Focus on the following types of correctness issues:

- Logic bugs:
  - Wrong conditions
  - Inverted conditions
  - Missing cases
  - Incorrect branching

- Off-by-one errors:
  - Incorrect array or list boundaries
  - Incorrect loop limits
  - Wrong index calculations

- Incorrect algorithm usage:
  - Wrong sorting logic
  - Wrong searching logic
  - Incorrect data structure or algorithm usage

- Null/None dereferences:
  - Accessing attributes or methods on values that may be None/null

- Type mismatches:
  - Passing incorrect types
  - Returning incorrect types
  - Unsafe implicit conversions

- Race conditions:
  - Shared mutable state accessed incorrectly
  - Async operations that can execute in an unsafe order

- Resource leaks:
  - Unclosed files
  - Unclosed database connections
  - Unclosed HTTP connections
  - Unclosed sessions or other resources

- Incorrect string formatting or template rendering
"""

async def run_correctness_agent(
    pr_context: PRContext,
    rag_context_str: str,
) -> tuple[list[ReviewFindingDict], AgentMetricDict]:
    """
    Run the correctness review agent.

    Args:
        pr_context: Pull request metadata, diff, files, and related information.
        rag_context_str: Relevant repository context retrieved from the RAG system.

    Returns:
        A tuple containing:
        - findings: Correctness issues found by the agent.
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