"""
LangGraph state definition for the multi-agent PR review workflow.

The state is shared between all agents in the review graph.

Parallel agents can append their findings and metrics without
overwriting the results produced by other agents.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict

from app.rag.context import ReviewContext
from app.services.github_service import PRContext


class ReviewFindingDict(TypedDict):
    """
    Represents a single finding produced by an AI review agent.

    A TypedDict is used instead of a dataclass because LangGraph
    state should remain simple and serializable for checkpointing,
    streaming, and persistence.
    """

    severity: str
    category: str
    title: str
    description: str
    file_path: str
    line_number: int | None
    evidence: str | None
    confidence: float
    agent: str


class AgentMetricDict(TypedDict):
    """
    Performance metrics produced by one review agent.

    These metrics can later be stored in the ReviewRun table
    for monitoring agent performance.
    """

    agent_name: str
    latency_ms: int
    tokens_used: int | None
    findings_count: int


class ReviewState(TypedDict):
    """
    Shared state used by the PRGuard LangGraph workflow.

    The workflow follows this general structure:

        PR Context
            +
        RAG Context
            ↓
        Review Agents
            ↓
        Findings + Metrics
            ↓
        Aggregator

    Input fields are populated when the graph starts.

    The findings and agent_metrics fields use operator.add
    as their reducer. This means that when multiple agents
    run in parallel, their results are appended to the existing
    lists instead of replacing them.
    """

    # ========================================================================
    # INPUT DATA
    # ========================================================================

    # Pull request information fetched from GitHub.
    pr_context: PRContext

    # Structured context retrieved from the RAG system.
    rag_context: ReviewContext

    # RAG context formatted specifically for LLM prompts.
    rag_context_str: str

    # Database ID of the current review.
    review_id: int

    # ========================================================================
    # AGENT OUTPUTS
    # ========================================================================

    # Findings from all agents.
    #
    # operator.add means:
    #
    # Agent A → [finding_1, finding_2]
    # Agent B → [finding_3]
    #
    # Final state:
    #
    # [finding_1, finding_2, finding_3]
    findings: Annotated[
        list[ReviewFindingDict],
        operator.add,
    ]

    # Performance metrics from all agents.
    #
    # Each agent appends its own metrics instead of
    # replacing metrics produced by another agent.
    agent_metrics: Annotated[
        list[AgentMetricDict],
        operator.add,
    ]

    # Deduplicated, merged and severity-ranked findings produced by the
    # aggregator node after all agents finish.
    #
    # This is a plain (last-write-wins) channel — the aggregator writes it
    # exactly once, so no reducer is needed. Consumers should prefer this
    # over the raw `findings` list.
    aggregated_findings: list[ReviewFindingDict]