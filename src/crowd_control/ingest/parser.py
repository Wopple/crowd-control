"""Parse Claude Code JSONL session transcripts into structured data."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from crowd_control.storage.models import (
    ContentBlock,
    ConversationSegment,
    Message,
    MessageRole,
    Session,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 500
_TRUNCATE_HEAD = 200
_TRUNCATE_TAIL = 200

_DATETIME_MIN_UTC = datetime.min.replace(tzinfo=UTC)

# JSONL line types that are noise and should be skipped entirely.
_SKIP_TYPES = frozenset({"file-history-snapshot", "progress", "queue-operation"})


def parse_session_file(path: Path) -> Session:
    """Parse a Claude Code session JSONL file into a Session object."""
    path = Path(path).expanduser().resolve()
    raw_lines: list[dict] = []
    for lineno, line in enumerate(path.open(encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw_lines.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON at %s:%d", path, lineno)

    metadata = extract_session_metadata(raw_lines)
    kept: list[dict] = [r for r in raw_lines if _should_keep(r)]

    messages: list[Message] = []
    for raw in kept:
        msg = parse_message(raw)
        if msg is not None:
            messages.append(msg)

    segments = segment_messages(messages)

    first_model = None
    for msg in messages:
        if msg.model:
            first_model = msg.model
            break

    all_timestamps = [m.timestamp for m in messages]
    start = min(all_timestamps) if all_timestamps else _DATETIME_MIN_UTC
    end = max(all_timestamps) if all_timestamps else _DATETIME_MIN_UTC

    return Session(
        session_id=metadata.get("session_id", path.stem),
        project_path=metadata.get("project_path", ""),
        git_branch=metadata.get("git_branch"),
        claude_version=metadata.get("claude_version"),
        segments=segments,
        start_time=start,
        end_time=end,
        message_count=len(messages),
        model=first_model,
    )


def extract_session_metadata(raw_lines: list[dict]) -> dict:
    """Extract session-level metadata from raw JSONL dicts."""
    result: dict = {}
    for raw in raw_lines[:20]:
        if "sessionId" in raw and "session_id" not in result:
            result["session_id"] = raw["sessionId"]
        if "cwd" in raw and "project_path" not in result:
            result["project_path"] = raw["cwd"]
        if "gitBranch" in raw and "git_branch" not in result:
            result["git_branch"] = raw["gitBranch"]
        if "version" in raw and "claude_version" not in result:
            result["claude_version"] = raw["version"]
        if len(result) >= 4:
            break
    return result


def parse_message(raw: dict) -> Message | None:
    """Convert a raw JSONL dict into a Message, or None if it should be skipped."""
    msg_type = raw.get("type")
    timestamp = _parse_timestamp(raw.get("timestamp", ""))
    uuid = raw.get("uuid", "")
    parent_uuid = raw.get("parentUuid")
    is_meta = raw.get("isMeta", False)

    if msg_type == "user":
        msg_data = raw.get("message", {})
        content_raw = msg_data.get("content", "")
        content = _parse_user_content(content_raw)
        if not content:
            return None
        return Message(
            role=MessageRole.USER,
            content=content,
            uuid=uuid,
            parent_uuid=parent_uuid,
            timestamp=timestamp,
            is_meta=is_meta,
        )

    if msg_type == "assistant":
        msg_data = raw.get("message", {})
        content_raw = msg_data.get("content", [])
        content = _parse_assistant_content(content_raw)
        if not content:
            return None
        return Message(
            role=MessageRole.ASSISTANT,
            content=content,
            uuid=uuid,
            parent_uuid=parent_uuid,
            timestamp=timestamp,
            model=msg_data.get("model"),
            usage=msg_data.get("usage"),
        )

    if msg_type == "system":
        subtype = raw.get("subtype", "")
        if subtype in ("local_command", "compact_boundary"):
            text = raw.get("content", "")
            return Message(
                role=MessageRole.SYSTEM,
                content=[TextBlock(text=text)],
                uuid=uuid,
                parent_uuid=parent_uuid,
                timestamp=timestamp,
            )
        return None

    return None


def segment_messages(messages: list[Message]) -> list[ConversationSegment]:
    """Group messages into conversation segments by user intent boundaries."""
    if not messages:
        return []

    segments: list[ConversationSegment] = []
    current: list[Message] = []
    preamble: list[Message] = []

    def _flush():
        nonlocal preamble
        if not current:
            return
        if _has_substantive_content(current):
            segments.append(_build_segment(preamble + current))
            preamble = []
        else:
            # No real content — carry forward as preamble for the next segment
            preamble.extend(current)

    for msg in messages:
        is_compact_boundary = msg.role == MessageRole.SYSTEM and any(
            isinstance(b, TextBlock) and b.text == "Conversation compacted" for b in msg.content
        )

        if is_compact_boundary:
            _flush()
            current = []
            preamble = []
            continue

        is_new_user_intent = (
            msg.role == MessageRole.USER and not msg.is_meta and not _is_tool_result_only(msg)
        )

        if is_new_user_intent and current:
            _flush()
            current = []

        current.append(msg)

    _flush()
    return segments


def find_sessions(project_path: str | None = None) -> list[Path]:
    """Find session JSONL files for a project, sorted by modification time (most recent first)."""
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.is_dir():
        return []

    if project_path is None:
        project_path = str(Path.cwd())

    encoded = encode_project_path(project_path)
    session_dir = projects_dir / encoded
    if not session_dir.is_dir():
        return []

    jsonl_files = list(session_dir.glob("*.jsonl"))
    jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonl_files


def encode_project_path(project_path: str) -> str:
    """Encode a project path the way Claude Code does: replace / with -."""
    return project_path.replace("/", "-")


# --- Internal helpers ---


def _should_keep(raw: dict) -> bool:
    """Return True if this JSONL line should be parsed into a message."""
    msg_type = raw.get("type", "")
    if msg_type in _SKIP_TYPES:
        return False
    if msg_type == "system" and raw.get("subtype") == "turn_duration":
        return False
    return True


def _parse_user_content(content_raw) -> list[ContentBlock]:
    """Parse user message content (string or list of blocks)."""
    if isinstance(content_raw, str):
        if not content_raw.strip():
            return []
        return [TextBlock(text=content_raw)]

    if isinstance(content_raw, list):
        blocks: list[ContentBlock] = []
        for item in content_raw:
            if not isinstance(item, dict):
                continue
            block_type = item.get("type")
            if block_type == "text":
                text = item.get("text", "")
                if text.strip():
                    blocks.append(TextBlock(text=text))
            elif block_type == "tool_result":
                raw_content = _extract_tool_result_content(item.get("content", ""))
                blocks.append(
                    _make_tool_result_block(
                        tool_use_id=item.get("tool_use_id", ""),
                        content=raw_content,
                    )
                )
        return blocks

    return []


def _parse_assistant_content(content_raw: list) -> list[ContentBlock]:
    """Parse assistant message content blocks."""
    blocks: list[ContentBlock] = []
    for item in content_raw:
        if not isinstance(item, dict):
            continue
        block_type = item.get("type")
        if block_type == "text":
            text = item.get("text", "")
            if text.strip():
                blocks.append(TextBlock(text=text))
        elif block_type == "tool_use":
            blocks.append(
                ToolUseBlock(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    input=item.get("input", {}),
                )
            )
        elif block_type == "thinking":
            thinking_text = item.get("thinking", "")
            blocks.append(ThinkingBlock(thinking=thinking_text))
    return blocks


def _extract_tool_result_content(content) -> str:
    """Extract text from tool_result content, which can be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        if parts:
            return "\n".join(parts)
    return str(content)


def _make_tool_result_block(tool_use_id: str, content: str) -> ToolResultBlock:
    """Create a ToolResultBlock, truncating content if too long."""
    original_length = len(content)
    if original_length > MAX_TOOL_RESULT_CHARS:
        truncated_chars = original_length - _TRUNCATE_HEAD - _TRUNCATE_TAIL
        content = (
            content[:_TRUNCATE_HEAD]
            + f"\n...[truncated {truncated_chars} chars]...\n"
            + content[-_TRUNCATE_TAIL:]
        )
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            original_length=original_length,
        )
    return ToolResultBlock(tool_use_id=tool_use_id, content=content)


def _has_substantive_content(messages: list[Message]) -> bool:
    """Return True if the message list contains at least one non-meta user or assistant message."""
    return any(
        (msg.role in (MessageRole.USER, MessageRole.ASSISTANT) and not msg.is_meta)
        for msg in messages
    )


def _is_tool_result_only(msg: Message) -> bool:
    """Return True if the message contains only tool_result blocks."""
    return bool(msg.content) and all(isinstance(b, ToolResultBlock) for b in msg.content)


def _build_segment(messages: list[Message]) -> ConversationSegment:
    """Build a ConversationSegment from a list of messages."""
    tool_names: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        for block in msg.content:
            if isinstance(block, ToolUseBlock) and block.name not in seen:
                tool_names.append(block.name)
                seen.add(block.name)

    timestamps = [m.timestamp for m in messages]
    return ConversationSegment(
        messages=messages,
        tool_names=tool_names,
        start_time=min(timestamps),
        end_time=max(timestamps),
    )


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp, falling back to datetime.min (UTC-aware)."""
    if not ts:
        return _DATETIME_MIN_UTC
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return _DATETIME_MIN_UTC
