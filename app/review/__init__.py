"""
Multi-agent review workflow for PRGuard.

This package implements the LangGraph-based multi-agent review system
that replaces the single-LLM baseline reviewer from Phase 5.

Components:
    - state: ReviewState TypedDict with LangGraph-compatible reducers
    - graph: StateGraph wiring 5 parallel agents + aggregator
    - aggregator: Finding deduplication, ranking, and merging
    - agents/: Specialized review agents (correctness, security, etc.)
"""
