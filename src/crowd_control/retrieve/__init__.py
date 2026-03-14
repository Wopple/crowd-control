"""Retrieval and ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from crowd_control.config import RetrievalConfig
from crowd_control.embed.base import Embedder
from crowd_control.retrieve.rank import RankedResult, rank_results
from crowd_control.retrieve.search import (
    BaseResult,
    Scope,
    SearchResult,
    SearchResults,
    search_learnings,
    validate_scope,
)
from crowd_control.storage.db import LearningStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Outcome of a full retrieve-and-rank pipeline run."""

    ranked: list[RankedResult]
    search_results: SearchResults
    total_learnings: int


def retrieve_learnings(
    query: str,
    store: LearningStore,
    embedder: Embedder,
    retrieval_config: RetrievalConfig,
    scope: Scope = "project",
    current_project: str | None = None,
    category: str | None = None,
) -> RetrievalResult:
    """Run the full search-and-rank pipeline.

    Embeds the query, searches the store, ranks results, and increments
    active counts for returned learnings. This is the single entry point
    that the CLI, MCP server, and hooks should all call.
    """
    search_results = search_learnings(
        query=query,
        store=store,
        embedder=embedder,
        config=retrieval_config,
        current_project=current_project,
        scope=scope,
        category=category,
    )

    ranked = rank_results(
        search_results.results,
        retrieval_config,
        current_project=current_project,
        scope=scope,
    )

    logger.info("Search: %d candidates, %d after ranking", len(search_results.results), len(ranked))

    if ranked:
        store.increment_active_count([r.id for r in ranked])

    return RetrievalResult(
        ranked=ranked,
        search_results=search_results,
        total_learnings=store.count(),
    )


__all__ = [
    "BaseResult",
    "RankedResult",
    "RetrievalResult",
    "Scope",
    "SearchResult",
    "SearchResults",
    "rank_results",
    "retrieve_learnings",
    "search_learnings",
    "validate_scope",
]
