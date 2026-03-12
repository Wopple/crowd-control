"""LLM-powered learning extraction via claude -p CLI."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from crowd_control.storage.models import (
    ConversationSegment,
    Learning,
    LearningCategory,
    Session,
    ThinkingBlock,
)

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
- File contents or raw error logs
- Exploratory dead ends that produced no insight
- Information that's obvious from reading the code itself
- Anything already well-known about the language or framework

If the segment contains no learnings worth extracting, return an empty list.

Project: {project_path}
Git branch: {git_branch}

--- Session transcript ---
{segment_text}
--- End transcript ---"""


class DistillationError(Exception):
    """Raised when distillation fails."""


def _extract_json(raw: str) -> dict:
    """Extract the first valid JSON object from raw output.

    Tries up to 4 positions starting from the first '{' character.
    """
    pos = 0
    attempts = 0
    while attempts < 4:
        idx = raw.find("{", pos)
        if idx == -1:
            break
        try:
            return json.loads(raw[idx:])
        except json.JSONDecodeError:
            pos = idx + 1
            attempts += 1

    raise DistillationError(f"Could not extract valid JSON from output (length={len(raw)})")


def truncate_segment_text(text: str, max_chars: int = 30000) -> str:
    """Truncate segment text if it exceeds max_chars, keeping head and tail."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[segment truncated]...\n" + text[-half:]


def build_distillation_prompt(
    segment: ConversationSegment,
    session: Session,
    max_learning_chars: int,
) -> str:
    """Build the distillation prompt for a segment.

    Returns empty string if segment text is too short (< 50 chars after truncation).
    """
    project_path = session.project_path if session.project_path else str(Path.cwd())
    git_branch = session.git_branch or "unknown"
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


def call_claude(
    prompt: str,
    json_schema: dict,
    model: str = "haiku",
    timeout: int = 120,
) -> dict:
    """Call claude -p CLI and return parsed structured output.

    Retries up to 2 times (3 total attempts) on retryable errors with backoff.
    """
    # Check if running inside Claude Code — non-retryable
    if os.environ.get("CLAUDECODE"):
        raise DistillationError(
            "Cannot call claude -p from inside Claude Code (CLAUDECODE env var is set)"
        )

    schema_json = json.dumps(json_schema, separators=(",", ":"))
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--json-schema",
        schema_json,
        "--no-session-persistence",
    ]

    max_retries = 2
    backoff = [2, 5]
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            raise DistillationError("claude CLI not found. Is it installed and on PATH?")
        except subprocess.TimeoutExpired as e:
            last_error = DistillationError(f"claude CLI timed out after {timeout}s")
            if attempt < max_retries:
                logger.warning(
                    "claude CLI timed out (attempt %d/%d), retrying in %ds",
                    attempt + 1,
                    max_retries + 1,
                    backoff[attempt],
                )
                time.sleep(backoff[attempt])
                continue
            raise last_error from e

        if result.returncode != 0:
            last_error = DistillationError(
                f"claude CLI exited with code {result.returncode}: {result.stderr[:200]}"
            )
            if attempt < max_retries:
                logger.warning(
                    "claude CLI failed with exit code %d (attempt %d/%d), retrying in %ds",
                    result.returncode,
                    attempt + 1,
                    max_retries + 1,
                    backoff[attempt],
                )
                time.sleep(backoff[attempt])
                continue
            raise last_error

        # Parse output — JSON parse failure is non-retryable
        try:
            parsed = _extract_json(result.stdout)
        except DistillationError:
            raise

        # Extract structured_output — missing is non-retryable
        if "structured_output" not in parsed:
            raise DistillationError("claude CLI response missing 'structured_output' key")

        return parsed["structured_output"]

    # Should not reach here, but just in case
    raise last_error or DistillationError("Unexpected retry exhaustion")


def _get_git_sha(project_path: str) -> str | None:
    """Get the current git HEAD SHA for a project path."""
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
    model: str = "haiku",
    max_learning_chars: int = 2000,
) -> list[Learning]:
    """Distill a single conversation segment into learnings.

    Returns a list of Learning objects extracted by the LLM.
    """
    project_path = session.project_path if session.project_path else str(Path.cwd())

    prompt = build_distillation_prompt(segment, session, max_learning_chars)
    if not prompt:
        return []

    response = call_claude(prompt, LEARNING_EXTRACTION_SCHEMA, model=model)

    raw_learnings = response.get("learnings", [])
    git_sha = _get_git_sha(project_path)

    learnings: list[Learning] = []
    for raw in raw_learnings:
        try:
            learning = Learning(
                text=raw["text"],
                category=LearningCategory(raw["category"]),
                tags=raw.get("tags", []),
                project=project_path,
                session_id=session.session_id,
                git_sha=git_sha,
                confidence=raw["confidence"],
            )
            learnings.append(learning)
        except Exception as e:
            logger.warning("Skipping invalid learning: %s (error: %s)", raw, e)

    return learnings


def distill_session(
    session: Session,
    model: str = "haiku",
    max_learnings: int = 20,
    max_learning_chars: int = 2000,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Learning]:
    """Distill all qualifying segments in a session into learnings.

    Filters out trivial segments (too few messages, no assistant content).
    Caps results at max_learnings by confidence descending (preserving order for ties).
    """
    qualifying_segments: list[ConversationSegment] = []
    for seg in session.segments:
        # Skip if fewer than 2 messages
        if len(seg.messages) < 2:
            continue

        # Skip if no assistant messages
        has_assistant = any(m.role.value == "assistant" for m in seg.messages)
        if not has_assistant:
            continue

        # Skip if all assistant content is empty thinking blocks
        all_empty_thinking = True
        for msg in seg.messages:
            if msg.role.value != "assistant":
                continue
            for block in msg.content:
                if not isinstance(block, ThinkingBlock):
                    all_empty_thinking = False
                    break
                elif block.thinking:
                    all_empty_thinking = False
                    break
            if not all_empty_thinking:
                break
        if all_empty_thinking:
            continue

        qualifying_segments.append(seg)

    total = len(qualifying_segments)
    all_learnings: list[Learning] = []

    for i, seg in enumerate(qualifying_segments):
        if progress_callback:
            progress_callback(i, total)

        try:
            seg_learnings = distill_segment(
                seg, session, model=model, max_learning_chars=max_learning_chars
            )
            all_learnings.extend(seg_learnings)
        except DistillationError as e:
            logger.warning("Segment %d/%d failed: %s", i + 1, total, e)
            continue

    # Cap at max_learnings by confidence descending, preserving order for ties
    if len(all_learnings) > max_learnings:
        # Stable sort by confidence descending — keeps original order for ties
        indexed = list(enumerate(all_learnings))
        indexed.sort(key=lambda pair: (-pair[1].confidence, pair[0]))
        kept_indices = sorted(pair[0] for pair in indexed[:max_learnings])
        all_learnings = [all_learnings[i] for i in kept_indices]

    return all_learnings
