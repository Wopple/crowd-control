"""Recency decay, deduplication, and token packing."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher

from crowd_control.config import RetrievalConfig
from crowd_control.retrieve.search import BaseResult, Scope, SearchResult, validate_scope

logger = logging.getLogger(__name__)


@dataclass
class RankedResult(BaseResult):
    """A search result after ranking and scoring."""

    hotness: float = 0.0
    final_score: float = 0.0


def rank_results(
    results: list[SearchResult],
    config: RetrievalConfig,
    current_project: str | None = None,
    scope: Scope = "project",
    now: datetime | None = None,
) -> list[RankedResult]:
    """Rank search results by blending semantic similarity with hotness.

    Applies scoring, deduplication, and token packing. Returns results
    ordered by final_score descending, capped at config.max_results.
    """
    validate_scope(scope)

    if not results:
        return []

    scored = _score_results(results, config, current_project, scope, now)
    deduped = _deduplicate(scored)
    packed = _pack_to_budget(deduped, config.max_tokens, config.max_results)
    return packed


def _sigmoid(x: float) -> float:
    """Standard sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def _score_results(
    results: list[SearchResult],
    config: RetrievalConfig,
    current_project: str | None = None,
    scope: Scope = "project",
    now: datetime | None = None,
) -> list[RankedResult]:
    """Compute final scores for each result.

    Scoring formula (from OpenViking's memory_lifecycle.py):
    - recency = exp(-ln(2) / half_life_days * age_days)
    - hotness = sigmoid(log1p(active_count)) * recency
    - blended = (1 - hotness_weight) * similarity + hotness_weight * hotness
    - project boost applied for same-project matches in non-project scopes
    """
    if now is None:
        now = datetime.now(UTC)

    decay_constant = math.log(2) / config.recency_half_life_days

    scored: list[RankedResult] = []
    for result in results:
        # Recency: clamp age to >= 0 to handle clock skew
        age_days = max(0.0, (now - result.timestamp).total_seconds() / 86400.0)
        recency = math.exp(-decay_constant * age_days)

        # Hotness: usage-weighted recency
        hotness = _sigmoid(math.log1p(result.active_count)) * recency

        # Blend semantic similarity with hotness
        blended = (1 - config.hotness_weight) * result.similarity + config.hotness_weight * hotness

        # Project boost: only in non-project scopes for same-project results
        if scope != "project" and current_project and result.project == current_project:
            blended *= config.project_boost

        scored.append(
            RankedResult(
                id=result.id,
                text=result.text,
                category=result.category,
                tags=result.tags,
                project=result.project,
                similarity=result.similarity,
                hotness=hotness,
                final_score=blended,
            )
        )

    # Sort by final score descending
    scored.sort(key=lambda r: r.final_score, reverse=True)

    logger.debug(
        "rank_results: scored=%d min=%.3f max=%.3f",
        len(scored),
        scored[-1].final_score if scored else 0,
        scored[0].final_score if scored else 0,
    )

    return scored


def _deduplicate(results: list[RankedResult], threshold: float = 0.85) -> list[RankedResult]:
    """Remove near-duplicate results by text similarity.

    Results must already be sorted by final_score descending. For each
    result, if its text is >= threshold similar to any already-kept result,
    it is dropped (the higher-scored version was already kept).
    """
    kept: list[RankedResult] = []
    for result in results:
        is_dup = False
        for kept_result in kept:
            ratio = SequenceMatcher(None, result.text, kept_result.text).ratio()
            if ratio >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(result)

    if len(kept) < len(results):
        logger.debug(
            "dedup: removed %d near-duplicates from %d results",
            len(results) - len(kept), len(results),
        )

    return kept


def _pack_to_budget(
    results: list[RankedResult], max_tokens: int, max_results: int
) -> list[RankedResult]:
    """Pack results into a token budget, stopping at max_results.

    Uses len(text) / 4 as a rough token estimate. Walks the sorted results
    and includes as many as fit within both the token budget and result cap.
    """
    packed: list[RankedResult] = []
    tokens_used = 0

    for result in results:
        if len(packed) >= max_results:
            break
        est_tokens = len(result.text) / 4
        if tokens_used + est_tokens > max_tokens:
            break
        packed.append(result)
        tokens_used += est_tokens

    logger.debug(
        "pack_to_budget: %d/%d results, ~%d/%d tokens",
        len(packed),
        len(results),
        int(tokens_used),
        max_tokens,
    )

    return packed
