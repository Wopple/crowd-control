"""Tests for JSONL session parsing."""

from pathlib import Path

import pytest

from crowd_control.ingest.parser import (
    encode_project_path,
    find_sessions,
    parse_message,
    parse_session_file,
    segment_messages,
)
from crowd_control.storage.models import (
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseSessionFile:
    def test_sample_session(self):
        session = parse_session_file(FIXTURES / "sample_session.jsonl")
        assert session.session_id == "test-session-001"
        assert session.project_path == "/Users/test/project"
        assert session.git_branch == "main"
        assert session.model == "claude-sonnet-4-6"
        assert session.claude_version == "2.1.72"
        # message_count should reflect parsed messages, not raw JSONL lines.
        # The fixture has 15 raw lines but some are noise (file-history-snapshot,
        # progress, turn_duration) that never become Message objects.
        assert session.message_count < 15  # Bug: was counting raw lines
        # Should produce 2 segments (messages 4-9 and 12-15 from plan)
        assert len(session.segments) == 2

    def test_sample_session_segment_1(self):
        session = parse_session_file(FIXTURES / "sample_session.jsonl")
        seg1 = session.segments[0]
        # Segment 1 includes preamble (meta user, system local_command) merged with:
        # user text, assistant (thinking+tool_use), user (tool_result),
        # assistant (thinking+text+tool_use), user (tool_result), assistant (text)
        assert "Read" in seg1.tool_names
        assert "Edit" in seg1.tool_names

    def test_sample_session_segment_2(self):
        session = parse_session_file(FIXTURES / "sample_session.jsonl")
        assert len(session.segments) == 2
        seg2 = session.segments[1]
        assert "Bash" in seg2.tool_names

    def test_minimal_session(self):
        session = parse_session_file(FIXTURES / "minimal_session.jsonl")
        assert session.session_id == "minimal-session-001"
        assert session.project_path == "/Users/test/minimal"
        assert session.git_branch == "dev"
        # 3 raw lines, but turn_duration is filtered -> 2 parsed messages -> 1 segment
        assert session.message_count == 2
        assert len(session.segments) == 1

    def test_compact_session(self):
        session = parse_session_file(FIXTURES / "compact_session.jsonl")
        assert session.session_id == "compact-session-001"
        # compact_boundary should split into 2 segments
        assert len(session.segments) == 2
        # First segment: "Refactor the database module"
        seg1_text = session.segments[0].to_prompt_text()
        assert "Refactor" in seg1_text
        # Second segment: "Now add connection pooling"
        seg2_text = session.segments[1].to_prompt_text()
        assert "connection pooling" in seg2_text

    def test_noise_filtered(self):
        """progress, file-history-snapshot, turn_duration should not appear in messages."""
        session = parse_session_file(FIXTURES / "sample_session.jsonl")
        all_messages = [m for s in session.segments for m in s.messages]
        for msg in all_messages:
            # No system messages with turn_duration content should exist
            assert msg.role != MessageRole.SYSTEM or not any(
                isinstance(b, TextBlock) and "durationMs" in b.text
                for b in msg.content
            )


class TestParseMessage:
    def test_user_plain_text(self):
        raw = {
            "type": "user",
            "message": {"role": "user", "content": "hello world"},
            "uuid": "u1",
            "timestamp": "2026-03-11T19:00:00.000Z",
            "isMeta": False,
        }
        msg = parse_message(raw)
        assert msg is not None
        assert msg.role == MessageRole.USER
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextBlock)
        assert msg.content[0].text == "hello world"

    def test_user_tool_result(self):
        raw = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "file data"},
                ],
            },
            "uuid": "u2",
            "timestamp": "2026-03-11T19:00:00.000Z",
            "isMeta": False,
        }
        msg = parse_message(raw)
        assert msg is not None
        assert isinstance(msg.content[0], ToolResultBlock)
        assert msg.content[0].tool_use_id == "t1"

    def test_user_meta_message(self):
        raw = {
            "type": "user",
            "message": {"role": "user", "content": "caveat text"},
            "uuid": "u3",
            "timestamp": "2026-03-11T19:00:00.000Z",
            "isMeta": True,
        }
        msg = parse_message(raw)
        assert msg is not None
        assert msg.is_meta is True

    def test_assistant_with_all_block_types(self):
        raw = {
            "type": "assistant",
            "message": {
                "model": "claude-sonnet-4-6",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "here's what I found"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            "uuid": "a1",
            "timestamp": "2026-03-11T19:00:00.000Z",
        }
        msg = parse_message(raw)
        assert msg is not None
        assert msg.role == MessageRole.ASSISTANT
        assert msg.model == "claude-sonnet-4-6"
        assert len(msg.content) == 3
        assert isinstance(msg.content[0], ThinkingBlock)
        assert isinstance(msg.content[1], TextBlock)
        assert isinstance(msg.content[2], ToolUseBlock)

    def test_system_local_command(self):
        raw = {
            "type": "system",
            "subtype": "local_command",
            "content": "output here",
            "uuid": "s1",
            "timestamp": "2026-03-11T19:00:00.000Z",
        }
        msg = parse_message(raw)
        assert msg is not None
        assert msg.role == MessageRole.SYSTEM

    def test_system_compact_boundary(self):
        raw = {
            "type": "system",
            "subtype": "compact_boundary",
            "content": "Conversation compacted",
            "uuid": "s2",
            "timestamp": "2026-03-11T19:00:00.000Z",
        }
        msg = parse_message(raw)
        assert msg is not None
        assert msg.content[0].text == "Conversation compacted"

    def test_system_turn_duration_returns_none(self):
        raw = {
            "type": "system",
            "subtype": "turn_duration",
            "durationMs": 5000,
            "uuid": "s3",
            "timestamp": "2026-03-11T19:00:00.000Z",
        }
        # turn_duration is filtered in _should_keep, but if somehow passed to parse_message
        # it should return None because subtype isn't local_command or compact_boundary
        msg = parse_message(raw)
        assert msg is None

    def test_unknown_type_returns_none(self):
        raw = {
            "type": "some-future-type",
            "uuid": "x1",
            "timestamp": "2026-03-11T19:00:00.000Z",
        }
        assert parse_message(raw) is None


class TestToolResultListContent:
    """Bug: tool_result content can be a list of content blocks (MCP tools), not just a string."""

    def test_list_content_extracts_text(self):
        """When tool_result content is a list of text blocks, extract the text."""
        raw = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "First paragraph of results."},
                            {"type": "text", "text": "Second paragraph of results."},
                        ],
                    },
                ],
            },
            "uuid": "u1",
            "timestamp": "2026-03-11T19:00:00.000Z",
            "isMeta": False,
        }
        msg = parse_message(raw)
        block = msg.content[0]
        assert isinstance(block, ToolResultBlock)
        # Should contain the actual text, NOT Python repr like "[{'type': 'text', ..."
        assert "First paragraph" in block.content
        assert "Second paragraph" in block.content
        assert "{'type'" not in block.content

    def test_list_content_with_non_text_blocks(self):
        """List content with non-text blocks (e.g., tool_reference) should not crash."""
        raw = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "tool_reference", "tool_name": "some_tool"},
                            {"type": "text", "text": "Actual result text here."},
                        ],
                    },
                ],
            },
            "uuid": "u1",
            "timestamp": "2026-03-11T19:00:00.000Z",
            "isMeta": False,
        }
        msg = parse_message(raw)
        block = msg.content[0]
        assert isinstance(block, ToolResultBlock)
        assert "Actual result text here." in block.content


class TestToolResultTruncation:
    def test_short_content_not_truncated(self):
        raw = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "short"},
                ],
            },
            "uuid": "u1",
            "timestamp": "2026-03-11T19:00:00.000Z",
            "isMeta": False,
        }
        msg = parse_message(raw)
        block = msg.content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.content == "short"
        assert block.original_length is None

    def test_long_content_truncated(self):
        long_content = "A" * 1000
        raw = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": long_content},
                ],
            },
            "uuid": "u1",
            "timestamp": "2026-03-11T19:00:00.000Z",
            "isMeta": False,
        }
        msg = parse_message(raw)
        block = msg.content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.original_length == 1000
        assert len(block.content) < 1000
        assert "truncated" in block.content
        # Should contain head and tail
        assert block.content.startswith("A" * 200)
        assert block.content.endswith("A" * 200)


class TestSegmentation:
    def _msg(self, role, content_text, uuid, ts, is_meta=False, is_tool_result=False):
        if is_tool_result:
            content = [ToolResultBlock(tool_use_id="t1", content=content_text)]
        else:
            content = [TextBlock(text=content_text)]
        return Message(
            role=MessageRole(role),
            content=content,
            uuid=uuid,
            timestamp=ts,
            is_meta=is_meta,
        )

    def test_basic_segmentation(self):
        from datetime import datetime
        t1 = datetime.fromisoformat("2026-03-11T19:00:00+00:00")
        t2 = datetime.fromisoformat("2026-03-11T19:00:10+00:00")
        t3 = datetime.fromisoformat("2026-03-11T19:01:00+00:00")
        t4 = datetime.fromisoformat("2026-03-11T19:01:10+00:00")

        messages = [
            self._msg("user", "first task", "u1", t1),
            self._msg("assistant", "done", "a1", t2),
            self._msg("user", "second task", "u2", t3),
            self._msg("assistant", "done again", "a2", t4),
        ]
        segments = segment_messages(messages)
        assert len(segments) == 2
        assert len(segments[0].messages) == 2
        assert len(segments[1].messages) == 2

    def test_tool_results_dont_split(self):
        from datetime import datetime
        t1 = datetime.fromisoformat("2026-03-11T19:00:00+00:00")
        t2 = datetime.fromisoformat("2026-03-11T19:00:05+00:00")
        t3 = datetime.fromisoformat("2026-03-11T19:00:10+00:00")

        messages = [
            self._msg("user", "do something", "u1", t1),
            self._msg("assistant", "calling tool", "a1", t2),
            self._msg("user", "result data", "u2", t3, is_tool_result=True),
        ]
        segments = segment_messages(messages)
        assert len(segments) == 1
        assert len(segments[0].messages) == 3

    def test_meta_messages_dont_split(self):
        from datetime import datetime
        t1 = datetime.fromisoformat("2026-03-11T19:00:00+00:00")
        t2 = datetime.fromisoformat("2026-03-11T19:00:05+00:00")
        t3 = datetime.fromisoformat("2026-03-11T19:00:10+00:00")

        messages = [
            self._msg("user", "caveat", "u1", t1, is_meta=True),
            self._msg("user", "actual question", "u2", t2),
            self._msg("assistant", "answer", "a1", t3),
        ]
        segments = segment_messages(messages)
        # Meta preamble gets merged into the first real segment
        assert len(segments) == 1
        assert len(segments[0].messages) == 3

    def test_empty_messages(self):
        assert segment_messages([]) == []


class TestFileEncoding:
    def test_utf8_content_parsed_correctly(self, tmp_path):
        """Bug: parser opens files without encoding='utf-8'. Verify UTF-8 content works."""
        session_file = tmp_path / "utf8_session.jsonl"
        import json

        lines = [
            json.dumps(
                {
                    "type": "user",
                    "message": {"role": "user", "content": "Fix the café module — it's broken"},
                    "uuid": "u1",
                    "timestamp": "2026-03-11T19:00:00.000Z",
                    "isMeta": False,
                    "sessionId": "utf8-test",
                    "cwd": "/test",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-6",
                        "content": [{"type": "text", "text": "I'll fix the café module."}],
                    },
                    "uuid": "a1",
                    "timestamp": "2026-03-11T19:00:10.000Z",
                }
            ),
        ]
        session_file.write_text("\n".join(lines), encoding="utf-8")
        session = parse_session_file(session_file)
        seg_text = session.segments[0].to_prompt_text()
        assert "café" in seg_text


class TestEncodeProjectPath:
    def test_basic(self):
        assert encode_project_path("/Users/daniel/git/crowd-control") == "-Users-daniel-git-crowd-control"

    def test_root(self):
        assert encode_project_path("/") == "-"


class TestFindSessions:
    def test_nonexistent_project_returns_empty(self, tmp_path):
        # find_sessions relies on ~/.claude/projects which may not have our fake project
        result = find_sessions("/nonexistent/path/that/does/not/exist/anywhere")
        assert result == []
