"""Tests for data models."""

from datetime import datetime, timezone

from crowd_control.storage.models import (
    ContentBlock,
    ConversationSegment,
    KnowledgeScope,
    Learning,
    LearningCategory,
    Message,
    MessageRole,
    Session,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _ts(s: str = "2026-03-11T19:00:00+00:00") -> datetime:
    return datetime.fromisoformat(s)


class TestContentBlocks:
    def test_text_block(self):
        b = TextBlock(text="hello")
        assert b.type == "text"
        assert b.text == "hello"

    def test_tool_use_block(self):
        b = ToolUseBlock(id="t1", name="Read", input={"file_path": "/a.py"})
        assert b.type == "tool_use"
        assert b.name == "Read"

    def test_tool_result_block(self):
        b = ToolResultBlock(tool_use_id="t1", content="file contents")
        assert b.type == "tool_result"
        assert b.original_length is None

    def test_tool_result_block_with_truncation(self):
        b = ToolResultBlock(tool_use_id="t1", content="short", original_length=5000)
        assert b.original_length == 5000

    def test_thinking_block(self):
        b = ThinkingBlock(thinking="let me think")
        assert b.type == "thinking"

    def test_discriminated_union_roundtrip(self):
        """Content blocks can round-trip through JSON via the discriminated union."""
        msg = Message(
            role=MessageRole.USER,
            content=[
                TextBlock(text="hello"),
                ToolResultBlock(tool_use_id="t1", content="result"),
            ],
            uuid="u1",
            timestamp=_ts(),
        )
        json_str = msg.model_dump_json()
        restored = Message.model_validate_json(json_str)
        assert len(restored.content) == 2
        assert isinstance(restored.content[0], TextBlock)
        assert isinstance(restored.content[1], ToolResultBlock)


class TestMessage:
    def test_basic_user_message(self):
        msg = Message(
            role=MessageRole.USER,
            content=[TextBlock(text="hello")],
            uuid="u1",
            timestamp=_ts(),
        )
        assert msg.role == MessageRole.USER
        assert msg.model is None
        assert msg.is_meta is False

    def test_assistant_message_with_model(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content=[TextBlock(text="hi")],
            uuid="a1",
            timestamp=_ts(),
            model="claude-sonnet-4-6",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert msg.model == "claude-sonnet-4-6"
        assert msg.usage["input_tokens"] == 10

    def test_roundtrip(self):
        msg = Message(
            role=MessageRole.SYSTEM,
            content=[TextBlock(text="system msg")],
            uuid="s1",
            parent_uuid="p1",
            timestamp=_ts(),
            is_meta=True,
        )
        restored = Message.model_validate_json(msg.model_dump_json())
        assert restored.role == MessageRole.SYSTEM
        assert restored.parent_uuid == "p1"
        assert restored.is_meta is True


class TestConversationSegment:
    def _make_segment(self):
        return ConversationSegment(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=[TextBlock(text="Fix the bug")],
                    uuid="u1",
                    timestamp=_ts("2026-03-11T19:00:00+00:00"),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        ToolUseBlock(id="t1", name="Read", input={"file_path": "a.py"}),
                    ],
                    uuid="a1",
                    timestamp=_ts("2026-03-11T19:00:10+00:00"),
                ),
                Message(
                    role=MessageRole.USER,
                    content=[ToolResultBlock(tool_use_id="t1", content="contents")],
                    uuid="u2",
                    timestamp=_ts("2026-03-11T19:00:11+00:00"),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[TextBlock(text="Fixed it.")],
                    uuid="a2",
                    timestamp=_ts("2026-03-11T19:00:20+00:00"),
                ),
            ],
            tool_names=["Read"],
            start_time=_ts("2026-03-11T19:00:00+00:00"),
            end_time=_ts("2026-03-11T19:00:20+00:00"),
        )

    def test_to_prompt_text(self):
        seg = self._make_segment()
        text = seg.to_prompt_text()
        assert "User: Fix the bug" in text
        assert "Assistant [tool_use]: Read(" in text
        assert "User [tool_result]: contents" in text
        assert "Assistant: Fixed it." in text

    def test_roundtrip(self):
        seg = self._make_segment()
        restored = ConversationSegment.model_validate_json(seg.model_dump_json())
        assert len(restored.messages) == 4
        assert restored.tool_names == ["Read"]


class TestSession:
    def test_roundtrip(self):
        session = Session(
            session_id="s1",
            project_path="/test",
            git_branch="main",
            segments=[],
            start_time=_ts(),
            end_time=_ts(),
            message_count=0,
        )
        restored = Session.model_validate_json(session.model_dump_json())
        assert restored.session_id == "s1"
        assert restored.model is None


class TestLearning:
    def test_generates_uuid(self):
        l1 = Learning(
            text="test",
            category=LearningCategory.GOTCHA,
            project="/test",
            session_id="s1",
            confidence=0.8,
        )
        l2 = Learning(
            text="test",
            category=LearningCategory.GOTCHA,
            project="/test",
            session_id="s1",
            confidence=0.8,
        )
        assert l1.id != l2.id
        assert len(l1.id) == 32  # uuid4 hex

    def test_explicit_id(self):
        l = Learning(
            id="custom-id",
            text="test",
            category=LearningCategory.DEBUGGING_INSIGHT,
            project="/test",
            session_id="s1",
            confidence=0.5,
        )
        assert l.id == "custom-id"

    def test_defaults(self):
        l = Learning(
            text="test",
            category=LearningCategory.PATTERN_DISCOVERY,
            project="/test",
            session_id="s1",
            confidence=0.9,
        )
        assert l.stale is False
        assert l.shared is False
        assert l.tags == []
        assert l.git_sha is None

    def test_roundtrip(self):
        l = Learning(
            text="use is not None",
            category=LearningCategory.CODEBASE_CONVENTION,
            tags=["python", "style"],
            project="/test",
            session_id="s1",
            confidence=0.7,
            shared=True,
        )
        restored = Learning.model_validate_json(l.model_dump_json())
        assert restored.text == l.text
        assert restored.shared is True
        assert restored.tags == ["python", "style"]


class TestEnums:
    def test_knowledge_scope_values(self):
        assert KnowledgeScope.PROJECT == "project"
        assert KnowledgeScope.SHARED == "shared"
        assert KnowledgeScope.MIXED == "mixed"

    def test_learning_category_values(self):
        assert LearningCategory.ARCHITECTURE_DECISION == "architecture_decision"
        assert LearningCategory.DEBUGGING_INSIGHT == "debugging_insight"
        assert LearningCategory.PATTERN_DISCOVERY == "pattern_discovery"
        assert LearningCategory.TOOL_USAGE == "tool_usage"
        assert LearningCategory.CODEBASE_CONVENTION == "codebase_convention"
        assert LearningCategory.GOTCHA == "gotcha"

    def test_message_role_values(self):
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
