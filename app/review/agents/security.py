"""
Security agent for detecting security vulnerabilities in pull requests.

This agent focuses on security issues that could be exploited by an attacker,
including:
- Injection vulnerabilities
- Cross-site scripting (XSS)
- Authentication bypass
- Authorization issues
- Hardcoded secrets
- Insecure deserialization
- Path traversal
- SSRF
- Insecure dependencies or configurations
"""

from app.review.agents.base import run_agent
from app.review.state import AgentMetricDict, ReviewFindingDict
from app.services.github_service import PRContext


AGENT_NAME = "security_agent"
DOMAIN = "security"
CATEGORY = "security"


DOMAIN_INSTRUCTIONS = """
Focus on the following types of security issues:

- SQL injection:
  - String concatenation in SQL queries
  - Unsanitized user input
  - Unsafe dynamic query construction

- Cross-site scripting (XSS):
  - Unescaped user input rendered in HTML
  - Unsafe DOM manipulation
  - Unsafe JavaScript execution

- Authentication bypass:
  - Missing authentication checks
  - Weak token validation
  - Incorrect session validation

- Authorization issues:
  - Insecure direct object references (IDOR)
  - Missing permission checks
  - Privilege escalation
  - Access to resources belonging to other users

- Secrets in code:
  - Hardcoded API keys
  - Passwords
  - Access tokens
  - Private keys
  - Other sensitive credentials

- Insecure deserialization:
  - Unsafe use of pickle
  - eval() or exec() with untrusted input
  - Other unsafe deserialization patterns

- Path traversal:
  - Unsanitized file paths
  - User-controlled paths accessing restricted files

- Insecure dependencies or configurations:
  - Unsafe dependency usage
  - Insecure default configurations
  - Disabled security controls

- SSRF (server-side request forgery):
  - User-controlled URLs used in server-side requests
  - Missing URL or host validation
"""


async def run_security_agent(
    pr_context: PRContext,
    rag_context_str: str,
) -> tuple[list[ReviewFindingDict], AgentMetricDict]:
    """
    Run the security review agent.

    Args:
        pr_context: Pull request metadata, diff, files, and related information.
        rag_context_str: Relevant repository context retrieved by the RAG system.

    Returns:
        A tuple containing:
        - findings: Security vulnerabilities found by the agent.
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