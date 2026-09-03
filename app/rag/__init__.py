"""
RAG (Retrieval-Augmented Generation) pipeline for PRGuard.

This package provides the code indexing and retrieval system that gives
review agents access to the full repository context.

Components:
    - loader: LangChain document loading for code repositories
    - splitter: Language-aware text splitting for code
    - embeddings: HuggingFace API embeddings via LangChain
    - qdrant_store: LangChain Qdrant vector store management
    - context: Context builder for review enrichment
    - indexer: Repository indexing orchestrator
"""
