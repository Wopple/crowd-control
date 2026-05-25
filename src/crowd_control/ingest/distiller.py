"""LLM-powered learning extraction (orchestration).

The actual LLM calls live in `crowd_control.ingest.llm.*`. This module owns
the prompt template, the JSON schema, segment filtering, and the concurrent
session-level orchestration — all provider-agnostic.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from crowd_control.ingest.llm.base import DistillationError, DistillerLLM
from crowd_control.project import resolve_project
from crowd_control.storage.models import (
    ConversationSegment,
    Learning,
    LearningCategory,
    MessageRole,
    Session,
    ThinkingBlock,
)

# Re-export so existing callers can `from crowd_control.ingest.distiller import DistillationError`.
__all__ = [
    "LEARNING_EXTRACTION_SCHEMA",
    "DistillationError",
    "build_distillation_prompt",
    "distill_segment",
    "distill_session",
    "is_segment_worth_distilling",
    "truncate_segment_text",
]

logger = logging.getLogger(__name__)


LEARNING_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "learnings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "A specific, self-contained technical insight or decision.",
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "architecture_decision",
                            "debugging_insight",
                            "pattern_discovery",
                            "tool_usage",
                            "codebase_convention",
                            "gotcha",
                        ],
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Languages, frameworks, libraries, or concepts involved.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "How significant and reliable this learning is.",
                    },
                },
                "required": ["text", "category", "tags", "confidence"],
            },
        }
    },
    "required": ["learnings"],
}


_PROMPT_TEMPLATE = """\
You are analyzing a coding session transcript to extract learnings. Extract as many
distinct learnings as you can find — most segments contain multiple insights.

A "learning" is a specific technical insight, decision, or discovery that would help
a future AI agent working on this codebase. Each learning should be:

- Self-contained: understandable without reading the full session
- Specific: references concrete code, patterns, or decisions (not generic advice)
- Actionable: a future agent could apply this insight directly
- Concise: each learning must be under {max_learning_chars} characters, but don't pad
  to fill that limit — most learnings should be a few sentences. If an insight is complex,
  break it into multiple smaller learnings rather than writing one large one.

Look for all learnings present in the segment — there may be several across different
categories. Consider: what decisions were made? What was discovered? What would a
future agent need to know?

Categories:
- architecture_decision: A deliberate choice about code structure, patterns, or organization
- debugging_insight: A bug's root cause and how it was identified or fixed
- pattern_discovery: A recurring pattern, idiom, or convention found in the codebase
- tool_usage: How a specific tool, library, or API should be used in this project
- codebase_convention: A naming, style, or structural convention specific to this project
- gotcha: A non-obvious pitfall, edge case, or surprising behavior

Do NOT extract:
- Generic programming knowledge (e.g., "use is instead of == for None comparison")
- General knowledge about tools, frameworks, or build systems (e.g., "uv uses \
dependency-groups for dev deps", "pytest discovers tests in test_ files")
- File contents or raw error logs
- Exploratory dead ends that produced no insight
- Information that's obvious from reading the code itself
- Anything already well-known about the language or framework
- Rejected alternatives — if the transcript discusses multiple options and picks one, \
only extract the chosen approach, not the ones that were considered and discarded

Confidence scoring — use the FULL range, not just high values:
- 1.0: a hard-won insight that would be very costly to rediscover (e.g., a subtle bug \
root cause, a non-obvious integration requirement)
- 0.8: a solid architectural decision or useful pattern with clear rationale
- 0.5: a useful convention or practice that saves some time but isn't critical
- 0.3: a minor observation that might be useful in narrow circumstances
- 0.1: barely worth recording
Most learnings should fall between 0.4 and 0.8. If you find yourself giving everything \
the same score, you are not discriminating enough.

If the segment contains no learnings worth extracting, return an empty list.

Project: {project_path}
Git branch: {git_branch}

--- Session transcript ---
{segment_text}
--- End transcript ---"""


def truncate_segment_text(text: str, max_chars: int = 30000) -> str:
    """Truncate segment text if it exceeds max_chars, keeping head and tail."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[segment truncated]...\n" + text[-half:]


def build_distillation_prompt(
    segment: ConversationSegment,
    project_path: str,
    git_branch: str,
    max_learning_chars: int,
) -> str:
    """Build the distillation prompt for a segment.

    Returns empty string if segment text is too short (< 50 chars after truncation).
    """
    segment_text = segment.to_prompt_text(include_thinking=False)
    segment_text = truncate_segment_text(segment_text)

    if len(segment_text) < 50:
        return ""

    return _PROMPT_TEMPLATE.format(
        max_learning_chars=max_learning_chars,
        project_path=project_path,
        git_branch=git_branch,
        segment_text=segment_text,
    )


def _get_git_sha(project_path: str) -> str | None:
    """Get the current git HEAD SHA for a project path."""
    if not project_path:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def distill_segment(
    segment: ConversationSegment,
    session: Session,
    llm: DistillerLLM,
    max_learning_chars: int = 2000,
    git_sha: str | None = None,
) -> list[Learning]:
    """Distill a single conversation segment into learnings."""
    filesystem_path = session.project_path if session.project_path else str(Path.cwd())
    project_id = resolve_project(Path(filesystem_path))
    git_branch = session.git_branch or "unknown"

    prompt = build_distillation_prompt(segment, filesystem_path, git_branch, max_learning_chars)
    if not prompt:
        return []

    logger.debug("Distill segment: prompt size=%d chars", len(prompt))
    response = llm.generate_structured(prompt, LEARNING_EXTRACTION_SCHEMA)

    raw_learnings = response.get("learnings", [])
    if git_sha is None:
        git_sha = _get_git_sha(filesystem_path)

    learnings: list[Learning] = []
    for raw in raw_learnings:
        try:
            learning = Learning(
                text=raw["text"],
                category=LearningCategory(raw["category"]),
                tags=raw.get("tags", []),
                project=project_id,
                session_id=session.session_id,
                git_sha=git_sha,
                confidence=raw["confidence"],
            )
            learnings.append(learning)
        except Exception as e:
            logger.warning("Skipping invalid learning: %s (error: %s)", raw, e)

    logger.debug(
        "Distill segment: %d learnings, categories=%s, confidences=%s",
        len(learnings),
        [lr.category.value for lr in learnings],
        [lr.confidence for lr in learnings],
    )

    return learnings


def is_segment_worth_distilling(segment: ConversationSegment) -> bool:
    """Check whether a segment has enough content to be worth distilling."""
    if len(segment.messages) < 2:
        return False

    assistant_msgs = [m for m in segment.messages if m.role == MessageRole.ASSISTANT]
    if not assistant_msgs:
        return False

    for msg in assistant_msgs:
        for block in msg.content:
            if not isinstance(block, ThinkingBlock) or block.thinking:
                return True
    return False


def distill_session(
    session: Session,
    llm: DistillerLLM,
    max_learnings: int = 20,
    max_learning_chars: int = 2000,
    max_workers: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Learning]:
    """Distill all qualifying segments in a session into learnings.

    If max_workers is None, uses the LLM's recommended_concurrency.
    """
    qualifying_segments = [seg for seg in session.segments if is_segment_worth_distilling(seg)]

    total = len(qualifying_segments)
    if total == 0:
        return []

    project_path = session.project_path if session.project_path else str(Path.cwd())
    git_sha = _get_git_sha(project_path)

    if max_workers is None:
        max_workers = llm.recommended_concurrency
    effective_workers = min(max_workers, total)

    logger.info(
        "distillation: provider=%s model=%s concurrency=%d segments=%d",
        llm.provider_name,
        llm.model_id,
        effective_workers,
        total,
    )

    indexed_learnings: list[tuple[int, list[Learning]]] = []
    lock = threading.Lock()
    completed = 0

    def _process_segment(index: int, seg: ConversationSegment) -> tuple[int, list[Learning]]:
        return index, distill_segment(
            seg,
            session,
            llm,
            max_learning_chars=max_learning_chars,
            git_sha=git_sha,
        )

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(_process_segment, i, seg): i
            for i, seg in enumerate(qualifying_segments)
        }

        for future in as_completed(futures):
            seg_index = futures[future]
            try:
                index, seg_learnings = future.result()
                with lock:
                    indexed_learnings.append((index, seg_learnings))
                logger.debug(
                    "Segment %d/%d completed with %d learnings",
                    seg_index + 1,
                    total,
                    len(seg_learnings),
                )
            except DistillationError as e:
                logger.warning("Segment %d/%d failed: %s", seg_index + 1, total, e)

            with lock:
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

    indexed_learnings.sort(key=lambda pair: pair[0])
    all_learnings: list[Learning] = []
    for _, seg_learnings in indexed_learnings:
        all_learnings.extend(seg_learnings)

    if len(all_learnings) > max_learnings:
        indexed = list(enumerate(all_learnings))
        indexed.sort(key=lambda pair: (-pair[1].confidence, pair[0]))
        kept_indices = sorted(pair[0] for pair in indexed[:max_learnings])
        all_learnings = [all_learnings[i] for i in kept_indices]

    return all_learnings
