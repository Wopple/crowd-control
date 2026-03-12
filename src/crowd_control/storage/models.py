"""Data models for parsed sessions and extracted learnings."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class LearningCategory(StrEnum):
    ARCHITECTURE_DECISION = "architecture_decision"
    DEBUGGING_INSIGHT = "debugging_insight"
    PATTERN_DISCOVERY = "pattern_discovery"
    TOOL_USAGE = "tool_usage"
    CODEBASE_CONVENTION = "codebase_convention"
    GOTCHA = "gotcha"


class KnowledgeScope(StrEnum):
    PROJECT = "project"
    SHARED = "shared"
    MIXED = "mixed"


# --- Content blocks (discriminated union on "type") ---


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    original_length: int | None = None


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock,
    Field(discriminator="type"),
]


# --- Parsed session structures ---


class Message(BaseModel):
    role: MessageRole
    content: list[ContentBlock]
    uuid: str
    parent_uuid: str | None = None
    timestamp: datetime
    model: str | None = None
    usage: dict | None = None
    is_meta: bool = False


class ConversationSegment(BaseModel):
    messages: list[Message]
    tool_names: list[str]
    start_time: datetime
    end_time: datetime

    def to_prompt_text(self) -> str:
        """Render this segment as a readable transcript for the distillation prompt."""
        lines: list[str] = []
        for msg in self.messages:
            if msg.is_meta:
                continue
            prefix = msg.role.value.capitalize()
            for block in msg.content:
                if isinstance(block, TextBlock):
                    lines.append(f"{prefix}: {block.text}")
                elif isinstance(block, ToolUseBlock):
                    lines.append(f"{prefix} [tool_use]: {block.name}({_summarize_input(block.input)})")
                elif isinstance(block, ToolResultBlock):
                    lines.append(f"{prefix} [tool_result]: {block.content}")
                elif isinstance(block, ThinkingBlock):
                    if block.thinking:
                        lines.append(f"{prefix} [thinking]: {block.thinking}")
        return "\n".join(lines)


class Session(BaseModel):
    session_id: str
    project_path: str
    git_branch: str | None = None
    claude_version: str | None = None
    segments: list[ConversationSegment]
    start_time: datetime
    end_time: datetime
    message_count: int
    model: str | None = None


# --- Learning (output of distillation, input to embedding/storage) ---


class Learning(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    category: LearningCategory
    tags: list[str] = Field(default_factory=list)
    project: str
    session_id: str
    git_sha: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(ge=0.0, le=1.0)
    stale: bool = False
    shared: bool = False


def _summarize_input(input_dict: dict) -> str:
    """Produce a short summary of tool input for prompt rendering."""
    parts = []
    for k, v in input_dict.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)
