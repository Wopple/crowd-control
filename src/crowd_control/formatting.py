"""Shared formatting for retrieval results and status output.

Used by both the CLI and the MCP server to render search and status output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from crowd_control.retrieve import RetrievalResult
from crowd_control.retrieve.search import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class FormattedLearning:
    """A single learning formatted for display."""

    id: str
    rank: int
    score: float
    category: str
    text: str
    project: str
    active_count: int
    age_days: int


def extract_display_learnings(result: RetrievalResult) -> list[FormattedLearning]:
    """Extract display-ready learnings from a retrieval result.

    Joins ranked results with their search result metadata (timestamps,
    active counts). Logs a warning if a ranked result has no matching
    search result — this indicates a bug upstream.
    """
    sr_by_id: dict[str, SearchResult] = {sr.id: sr for sr in result.search_results.results}
    now = datetime.now(UTC)
    learnings: list[FormattedLearning] = []

    for i, r in enumerate(result.ranked, 1):
        sr = sr_by_id.get(r.id)
        if sr is None:
            logger.warning("Ranked result %s not found in search results lookup", r.id)

        age_days = 0
        if sr and sr.timestamp:
            age_days = (now - sr.timestamp).days

        active_count = sr.active_count if sr else 0

        learnings.append(
            FormattedLearning(
                id=r.id,
                rank=i,
                score=r.final_score,
                category=r.category,
                text=r.text,
                project=r.project,
                active_count=active_count,
                age_days=age_days,
            )
        )

    return learnings


@dataclass
class StatusCounts:
    """Formatted status lines for project-scoped output."""

    learnings_line: str
    tags_line: str
    all_tags_line: str | None


def format_status_counts(
    project_count: int,
    total_count: int,
    project_tags: list[str],
    all_tags: list[str],
) -> StatusCounts:
    """Format project-scoped status lines.

    When project counts equal total counts (single-project DB), the output
    is simplified to avoid redundant "(N total)" noise.
    """
    project_tag_str = ", ".join(project_tags) if project_tags else "(none)"
    all_tag_str = ", ".join(all_tags) if all_tags else "(none)"

    if project_count == total_count:
        return StatusCounts(
            learnings_line=f"Learnings: {project_count}",
            tags_line=f"Tags: {project_tag_str}",
            all_tags_line=None,
        )

    return StatusCounts(
        learnings_line=f"Learnings: {project_count} ({total_count} total)",
        tags_line=f"Tags: {project_tag_str}",
        all_tags_line=f"Tags (all): {all_tag_str}",
    )


def format_results_text(result: RetrievalResult) -> str:
    """Format a RetrievalResult as readable text.

    Used by the MCP server to return results to the agent.
    """
    if not result.ranked:
        return f"No matching learnings found (searched {result.total_learnings} learnings)."

    learnings = extract_display_learnings(result)
    lines = []

    for fl in learnings:
        lines.append(
            f"[{fl.rank}] (score={fl.score:.2f}) [{fl.category}] id={fl.id[:8]}\n"
            f"    {fl.text}\n"
            f"    project={fl.project}  retrieved={fl.active_count}x  "
            f"age={fl.age_days}d"
        )

    result_word = "result" if len(learnings) == 1 else "results"
    lines.append(f"\n{len(learnings)} {result_word} (searched {result.total_learnings} learnings)")
    return "\n\n".join(lines)
