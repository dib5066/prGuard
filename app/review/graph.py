"""
LangGraph state graph for the multi-agent PR review workflow.

Review pipeline:

1. Receive PRContext and RAG context.
2. Run five specialized review agents in parallel.
3. Aggregate and deduplicate all agent findings.
4. Return the final findings and agent metrics.

Graph topology:

    START
      │
      ├── correctness_agent
      ├── security_agent
      ├── error_handling_agent
      ├── quality_agent
      └── testing_agent
              │
              ▼
         aggregator
              │
              ▼
             END
"""

import logging

from langgraph.graph import END, START, StateGraph

from app.review.aggregator import aggregate_findings
from app.review.agents.correctness import run_correctness_agent
from app.review.agents.error_handling import run_error_handling_agent
from app.review.agents.quality import run_quality_agent
from app.review.agents.security import run_security_agent
from app.review.agents.testing import run_testing_agent
from app.review.checkpointer import get_checkpointer
from app.review.state import ReviewState


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent nodes
#
# Each LangGraph node:
# 1. Reads the required information from the state.
# 2. Runs one specialized review agent.
# 3. Returns findings and metrics.
#
# The findings and agent_metrics fields use operator.add reducers,
# so results from all five agents are accumulated automatically.
# ---------------------------------------------------------------------------

async def correctness_node(
    state: ReviewState,
) -> dict:
    """Run the correctness review agent."""

    findings, metrics = await run_correctness_agent(
        pr_context=state["pr_context"],
        rag_context_str=state.get("rag_context_str", ""),
    )

    return {
        "findings": findings,
        "agent_metrics": [metrics],
    }


async def security_node(
    state: ReviewState,
) -> dict:
    """Run the security review agent."""

    findings, metrics = await run_security_agent(
        pr_context=state["pr_context"],
        rag_context_str=state.get("rag_context_str", ""),
    )

    return {
        "findings": findings,
        "agent_metrics": [metrics],
    }

async def error_handling_node(
    state: ReviewState,
) -> dict:
    """Run the error handling review agent."""

    findings, metrics = await run_error_handling_agent(
        pr_context=state["pr_context"],
        rag_context_str=state.get("rag_context_str", ""),
    )

    return {
        "findings": findings,
        "agent_metrics": [metrics],
    }


async def quality_node(
    state: ReviewState,
) -> dict:
    """Run the code quality review agent."""

    findings, metrics = await run_quality_agent(
        pr_context=state["pr_context"],
        rag_context_str=state.get("rag_context_str", ""),
    )

    return {
        "findings": findings,
        "agent_metrics": [metrics],
    }


async def testing_node(
    state: ReviewState,
) -> dict:
    """Run the testing review agent."""

    findings, metrics = await run_testing_agent(
        pr_context=state["pr_context"],
        rag_context_str=state.get("rag_context_str", ""),
    )

    return {
        "findings": findings,
        "agent_metrics": [metrics],
    }


# ---------------------------------------------------------------------------
# Aggregator node
#
# This node runs after all five agents finish.
#
# The raw findings remain in `findings`.
# The deduplicated findings are stored separately in
# `aggregated_findings`.
# ---------------------------------------------------------------------------


def aggregator_node(
    state: ReviewState,
) -> dict:
    """
    Aggregate and deduplicate findings from all review agents.

    This is a synchronous node because it does not make any LLM or
    network calls.
    """

    aggregated_findings = aggregate_findings(
        state["findings"]
    )

    logger.info(
        "Aggregator produced %d final findings from %d raw findings",
        len(aggregated_findings),
        len(state["findings"]),
    )

    return {
        "aggregated_findings": aggregated_findings,
    }


# ---------------------------------------------------------------------------
# Build the LangGraph workflow
# ---------------------------------------------------------------------------


def build_review_graph():
    """
    Build and compile the LangGraph review workflow.

    Returns:
        A compiled LangGraph workflow that can be executed with
        `.ainvoke()`.
    """

    workflow = StateGraph(ReviewState)

    # ---------------------------------------------------------------
    # Add specialized agent nodes
    # ---------------------------------------------------------------

    workflow.add_node(
        "correctness_agent",
        correctness_node,
    )

    workflow.add_node(
        "security_agent",
        security_node,
    )

    workflow.add_node(
        "error_handling_agent",
        error_handling_node,
    )

    workflow.add_node(
        "quality_agent",
        quality_node,
    )

    workflow.add_node(
        "testing_agent",
        testing_node,
    )

    # Add aggregator node
    workflow.add_node(
        "aggregator",
        aggregator_node,
    )

    # ---------------------------------------------------------------
    # Parallel fan-out
    #
    # START triggers all five agents.
    # LangGraph can execute these independent nodes concurrently.
    # ---------------------------------------------------------------

    workflow.add_edge(
        START,
        "correctness_agent",
    )

    workflow.add_edge(
        START,
        "security_agent",
    )

    workflow.add_edge(
        START,
        "error_handling_agent",
    )

    workflow.add_edge(
        START,
        "quality_agent",
    )

    workflow.add_edge(
        START,
        "testing_agent",
    )

    # ---------------------------------------------------------------
    # Fan-in
    #
    # The aggregator runs after all five agent branches complete.
    # ---------------------------------------------------------------

    workflow.add_edge(
        "correctness_agent",
        "aggregator",
    )

    workflow.add_edge(
        "security_agent",
        "aggregator",
    )

    workflow.add_edge(
        "error_handling_agent",
        "aggregator",
    )

    workflow.add_edge(
        "quality_agent",
        "aggregator",
    )

    workflow.add_edge(
        "testing_agent",
        "aggregator",
    )

    # ---------------------------------------------------------------
    # Aggregator → END
    # ---------------------------------------------------------------

    workflow.add_edge(
        "aggregator",
        END,
    )

    # Compile the workflow. The checkpointer persists per-review state so a
    # retried review resumes instead of re-running every agent. It may be
    # None (setup not run / failed) — the graph then runs stateless.
    checkpointer = get_checkpointer()
    review_graph = workflow.compile(checkpointer=checkpointer)

    logger.info(
        "LangGraph review workflow compiled with 5 specialized agents "
        "(checkpointer=%s)",
        "on" if checkpointer is not None else "off",
    )

    return review_graph

# ---------------------------------------------------------------------------
# Reusable compiled graph
# ---------------------------------------------------------------------------

_review_graph = None


def get_review_graph():
    """
    Get the compiled review graph.

    The graph is compiled once and reused for subsequent reviews.
    """

    global _review_graph

    if _review_graph is None:
        _review_graph = build_review_graph()

    return _review_graph