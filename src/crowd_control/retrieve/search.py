"""Vector search and metadata filtering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from crowd_control.config import RetrievalConfig
from crowd_control.embed.base import Embedder
from crowd_control.storage.db import LearningStore

logger = logging.getLogger(__name__)

Scope = Literal["project", "shared", "mixed"]

VALID_SCOPES: frozenset[str] = frozenset({"project", "shared", "mixed"})


def validate_scope(scope: str) -> Scope:
    """Validate and return a scope value, raising ValueError for invalid scopes."""
    if scope not in VALID_SCOPES:
        valid = ", ".join(sorted(VALID_SCOPES))
        raise ValueError(f"Invalid scope {scope!r}. Must be one of: {valid}")
    return scope  # type: ignore[return-value]


@dataclass
class BaseResult:
    """Fields common to all result types."""

    id: str
    text: str
    category: str
    tags: list[str]
    project: str
    similarity: float


@dataclass
class SearchResult(BaseResult):
    """A single search result with its raw similarity score."""

    session_id: str = ""
    timestamp: datetime | None = None
    confidence: float = 0.0
    active_count: int = 0


@dataclass
class SearchResults:
    """Container for search results with metadata about the search."""

    results: list[SearchResult]
    query_text: str


def search_learnings(
    query: str,
    store: LearningStore,
    embedder: Embedder,
    config: RetrievalConfig,
    current_project: str | None = None,
    scope: Scope = "project",
    category: str | None = None,
) -> SearchResults:
    """Search for learnings matching a query.

    Embeds the query, runs vector search, and returns raw results
    with similarity scores. Does not rank or deduplicate — that's
    the ranker's job.
    """
    validate_scope(scope)

    query_vector = embedder.embed([query])[0]

    # Over-fetch to give the ranker enough candidates after dedup/packing
    limit = max(config.max_results * 2, 30)

    raw_results = store.vector_search(
        query_vector=query_vector,
        limit=limit,
        min_similarity=config.min_similarity,
        category=category,
        scope=scope,
        current_project=current_project,
    )

    results = [
        SearchResult(
            id=row["id"],
            text=row["text"],
            category=row["category"],
            tags=row.get("tags", []),
            project=row["project"],
            similarity=row["_similarity"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            confidence=row.get("confidence", 0.0),
            active_count=row.get("active_count", 0),
        )
        for row in raw_results
    ]

    logger.debug(
        "search_learnings: query=%r scope=%s project=%s results=%d",
        query,
        scope,
        current_project,
        len(results),
    )

    return SearchResults(results=results, query_text=query)
