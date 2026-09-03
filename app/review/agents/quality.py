"""
Code quality agent for detecting maintainability and code smell issues.

This agent focuses on problems that make code:
- Difficult to understand
- Difficult to maintain
- Difficult to test
- Difficult to extend
"""

from app.review.agents.base import run_agent
from app.review.state import AgentMetricDict, ReviewFindingDict
from app.services.github_service import PRContext


AGENT_NAME = "quality_agent"
DOMAIN = "code quality"
CATEGORY = "quality"


DOMAIN_INSTRUCTIONS = """
Focus on the following types of code quality issues:

- Code duplication:
  - Copy-pasted logic
  - Repeated code that should be extracted into a reusable function
  - Repeated logic that could lead to inconsistent future changes

- Long or overly complex functions:
  - Functions that are unnecessarily long
  - Functions with too many responsibilities
  - Excessive branching or complex control flow
  - High cognitive complexity

- Dead code:
  - Unused imports
  - Unused variables or functions
  - Unreachable branches
  - Commented-out code that is no longer needed

- Poor naming:
  - Unclear variable names
  - Unclear function or class names
  - Names that do not describe the value or behavior clearly
  - Inconsistent naming conventions

- Missing type hints:
  - Missing useful type annotations in Python
  - Inconsistent or unclear type information
  - Avoid reporting type hints when they would add little value

- Magic numbers and hardcoded strings:
  - Repeated numeric values without explanation
  - Repeated strings that should be constants
  - Configuration values hardcoded directly into business logic

- Deeply nested code:
  - Excessive nesting
  - More than three levels of indentation when it can reasonably be simplified
  - Complex nested conditions that reduce readability

- God classes:
  - Classes responsible for too many unrelated tasks
  - Classes that should reasonably be split into smaller components

- Tight coupling:
  - Modules depending heavily on implementation details of other modules
  - Strong dependencies that make testing or changing one module difficult
  - Unnecessary cross-module knowledge

Only report issues that have a meaningful impact on readability, maintainability,
testability, or extensibility. Do not report harmless stylistic preferences as
quality issues.
"""


async def run_quality_agent(
    pr_context: PRContext,
    rag_context_str: str,
) -> tuple[list[ReviewFindingDict], AgentMetricDict]:
    """
    Run the code quality review agent.

    Args:
        pr_context: Pull request metadata, diff, files, and related information.
        rag_context_str: Relevant repository context retrieved by the RAG system.

    Returns:
        A tuple containing:
        - findings: Code quality issues found by the agent.
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