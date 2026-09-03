"""
Testing agent for detecting missing tests and untested scenarios.

This agent focuses on test coverage gaps that can allow bugs to go
undetected, including:
- Missing tests for changed behavior
- Missing edge case tests
- Missing error path tests
- Weak or ineffective tests
- Test isolation problems
- Missing integration tests
- Untested public API changes
- Incorrect handling of external dependencies in tests
"""

from app.review.agents.base import run_agent
from app.review.state import AgentMetricDict, ReviewFindingDict
from app.services.github_service import PRContext


AGENT_NAME = "testing_agent"
DOMAIN = "testing"
CATEGORY = "testing"


DOMAIN_INSTRUCTIONS = """
Focus on the following types of testing issues:

- Missing tests for changed code:
  - Changed functions without corresponding test coverage
  - New behavior without tests
  - Modified business logic without updated tests

- Missing edge case tests:
  - Empty inputs
  - None/null values
  - Boundary values
  - Minimum and maximum values
  - Empty collections
  - Unexpected but valid input

- Missing error path tests:
  - Exceptions
  - Failed API requests
  - Database failures
  - Invalid input
  - Permission or authentication failures
  - Other expected failure scenarios

- Ineffective tests:
  - Tests that do not contain meaningful assertions
  - Tests that can pass even when the intended behavior is broken
  - Assertions that do not verify the important behavior being tested

- Test isolation issues:
  - Shared mutable state between tests
  - Tests depending on execution order
  - Persistent data that is not cleaned up
  - Tests affecting the result of other tests

- Missing integration tests:
  - Cross-component changes that require integration coverage
  - Changes involving multiple services or modules
  - Database or external-service interactions where integration testing is appropriate

- Untested public API changes:
  - New endpoints
  - Changed request or response behavior
  - Changed validation rules
  - Changed public interfaces

- External dependencies:
  - Missing mocks for external APIs when unit tests should isolate them
  - Missing database mocks or fixtures where appropriate
  - Tests making unintended real network calls
  - Tests depending on external services unnecessarily

Only report a missing test when the changed behavior is meaningful enough to
warrant additional coverage. Do not report a finding simply because every
possible edge case is not explicitly tested.
"""


async def run_testing_agent(
    pr_context: PRContext,
    rag_context_str: str,
) -> tuple[list[ReviewFindingDict], AgentMetricDict]:
    """
    Run the testing review agent.

    Args:
        pr_context: Pull request metadata, diff, files, and related information.
        rag_context_str: Relevant repository context retrieved by the RAG system.

    Returns:
        A tuple containing:
        - findings: Testing issues found by the agent.
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