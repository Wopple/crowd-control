"""Tests for LLM-powered distillation pipeline."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crowd_control.ingest.distiller import (
    LEARNING_EXTRACTION_SCHEMA,
    DistillationError,
    _get_git_sha,
    build_distillation_prompt,
    call_claude,
    distill_segment,
    distill_session,
    is_segment_worth_distilling,
)
from crowd_control.storage.models import (
    ConversationSegment,
    Learning,
    LearningCategory,
    Message,
    MessageRole,
    Session,
    TextBlock,
    ThinkingBlock,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _ts(s: str = "2026-03-11T19:00:00+00:00") -> datetime:
    return datetime.fromisoformat(s)


def _make_segment(
    messages=None,
    tool_names=None,
    start="2026-03-11T19:00:00+00:00",
    end="2026-03-11T19:01:00+00:00",
):
    if messages is None:
        messages = [
            Message(
                role=MessageRole.USER,
                content=[TextBlock(text="Fix the database connection pooling issue")],
                uuid="u1",
                timestamp=_ts(start),
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    TextBlock(text="I found the issue in db.py. The pool size was hardcoded to 1.")
                ],
                uuid="a1",
                timestamp=_ts(end),
            ),
        ]
    return ConversationSegment(
        messages=messages,
        tool_names=tool_names or [],
        start_time=_ts(start),
        end_time=_ts(end),
    )


def _make_session(segments=None, project_path="/test/project", git_branch="main"):
    if segments is None:
        segments = [_make_segment()]
    return Session(
        session_id="test-session-001",
        project_path=project_path,
        git_branch=git_branch,
        segments=segments,
        start_time=_ts("2026-03-11T19:00:00+00:00"),
        end_time=_ts("2026-03-11T19:05:00+00:00"),
        message_count=sum(len(s.messages) for s in segments),
    )


def _load_fixture_response():
    with open(FIXTURES / "distillation_response.json") as f:
        return json.load(f)


class TestBuildPrompt:
    def test_project_path(self):
        segment = _make_segment()
        prompt = build_distillation_prompt(
            segment, project_path="/my/project", git_branch="main", max_learning_chars=2000
        )
        assert "Project: /my/project" in prompt

    def test_segment_text(self):
        segment = _make_segment()
        prompt = build_distillation_prompt(
            segment, project_path="/test/project", git_branch="main", max_learning_chars=2000
        )
        assert "Fix the database connection pooling issue" in prompt
        assert "pool size was hardcoded" in prompt

    def test_git_branch(self):
        segment = _make_segment()
        prompt = build_distillation_prompt(
            segment, project_path="/test", git_branch="unknown", max_learning_chars=2000
        )
        assert "Git branch: unknown" in prompt

    def test_truncation(self):
        long_text = "A" * 40000
        messages = [
            Message(
                role=MessageRole.USER,
                content=[TextBlock(text=long_text)],
                uuid="u1",
                timestamp=_ts(),
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[TextBlock(text="ok")],
                uuid="a1",
                timestamp=_ts(),
            ),
        ]
        segment = _make_segment(messages=messages)
        prompt = build_distillation_prompt(
            segment, project_path="/test", git_branch="main", max_learning_chars=2000
        )
        assert "...[segment truncated]..." in prompt

    def test_max_learning_chars(self):
        segment = _make_segment()
        prompt = build_distillation_prompt(
            segment, project_path="/test", git_branch="main", max_learning_chars=500
        )
        assert "under 500 characters" in prompt

    def test_multiple_learnings(self):
        """Prompt should instruct extraction of multiple learnings."""
        segment = _make_segment()
        prompt = build_distillation_prompt(
            segment, project_path="/test", git_branch="main", max_learning_chars=2000
        )
        assert "Extract as many" in prompt

    def test_excludes_thinking(self):
        messages = [
            Message(
                role=MessageRole.USER,
                content=[TextBlock(text="Fix the thing in the codebase please")],
                uuid="u1",
                timestamp=_ts(),
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ThinkingBlock(thinking="secret internal reasoning that should be hidden"),
                    TextBlock(text="I'll fix it."),
                ],
                uuid="a1",
                timestamp=_ts(),
            ),
        ]
        segment = _make_segment(messages=messages)
        prompt = build_distillation_prompt(
            segment, project_path="/test", git_branch="main", max_learning_chars=2000
        )
        assert "secret internal reasoning" not in prompt
        assert "I'll fix it." in prompt


class TestCallClaude:
    def _mock_result(self, stdout="", returncode=0, stderr=""):
        mock = MagicMock()
        mock.stdout = stdout
        mock.returncode = returncode
        mock.stderr = stderr
        return mock

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.subprocess.run")
    def test_successful_call(self, mock_run):
        response = _load_fixture_response()
        mock_run.return_value = self._mock_result(stdout=json.dumps(response))

        result = call_claude("test prompt", LEARNING_EXTRACTION_SCHEMA)
        assert "learnings" in result
        assert len(result["learnings"]) == 3
        mock_run.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.subprocess.run")
    def test_claude_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("claude not found")

        with pytest.raises(DistillationError, match="claude CLI not found"):
            call_claude("test prompt", LEARNING_EXTRACTION_SCHEMA)

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.time.sleep")
    @patch("crowd_control.ingest.distiller.subprocess.run")
    def test_timeout(self, mock_run, mock_sleep):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)

        with pytest.raises(DistillationError, match="timed out"):
            call_claude("test prompt", LEARNING_EXTRACTION_SCHEMA)
        # Should have retried
        assert mock_run.call_count == 3

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.time.sleep")
    @patch("crowd_control.ingest.distiller.subprocess.run")
    def test_nonzero_exit(self, mock_run, mock_sleep):
        mock_run.return_value = self._mock_result(returncode=1, stderr="error msg")

        with pytest.raises(DistillationError, match="exited with code 1"):
            call_claude("test prompt", LEARNING_EXTRACTION_SCHEMA)
        assert mock_run.call_count == 3

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.subprocess.run")
    def test_invalid_json_output(self, mock_run):
        mock_run.return_value = self._mock_result(stdout="not json at all")

        with pytest.raises(DistillationError, match="claude CLI returned invalid JSON"):
            call_claude("test prompt", LEARNING_EXTRACTION_SCHEMA)
        # Non-retryable, so only 1 attempt
        assert mock_run.call_count == 1

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.subprocess.run")
    def test_missing_structured_output(self, mock_run):
        mock_run.return_value = self._mock_result(
            stdout=json.dumps({"result": "no structured_output key"})
        )

        with pytest.raises(DistillationError, match="missing 'structured_output'"):
            call_claude("test prompt", LEARNING_EXTRACTION_SCHEMA)
        assert mock_run.call_count == 1

    @patch.dict("os.environ", {"CLAUDECODE": "1"}, clear=False)
    def test_rejects_inside_claude_code(self):
        with pytest.raises(DistillationError, match="CLAUDECODE"):
            call_claude("test prompt", LEARNING_EXTRACTION_SCHEMA)

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.time.sleep")
    @patch("crowd_control.ingest.distiller.subprocess.run")
    def test_retry_succeeds_on_second_attempt(self, mock_run, mock_sleep):
        response = _load_fixture_response()
        mock_run.side_effect = [
            self._mock_result(returncode=1, stderr="temporary failure"),
            self._mock_result(stdout=json.dumps(response)),
        ]

        result = call_claude("test prompt", LEARNING_EXTRACTION_SCHEMA)
        assert "learnings" in result
        assert mock_run.call_count == 2
        mock_sleep.assert_called_once_with(2)

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.time.sleep")
    @patch("crowd_control.ingest.distiller.subprocess.run")
    def test_retry_exhausted(self, mock_run, mock_sleep):
        mock_run.return_value = self._mock_result(returncode=1, stderr="persistent failure")

        with pytest.raises(DistillationError, match="exited with code 1"):
            call_claude("test prompt", LEARNING_EXTRACTION_SCHEMA)
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2


class TestGetGitSha:
    def test_empty_project_path_returns_none(self):
        """Bug: _get_git_sha('') passes cwd='' to subprocess — undefined behavior."""
        result = _get_git_sha("")
        assert result is None


class TestDistillSegment:
    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller._get_git_sha", return_value="abc123")
    @patch("crowd_control.ingest.distiller.call_claude")
    def test_returns_learning_objects(self, mock_call, mock_sha):
        mock_call.return_value = {
            "learnings": [
                {
                    "text": "The pool size was hardcoded.",
                    "category": "debugging_insight",
                    "tags": ["database", "pooling"],
                    "confidence": 0.9,
                }
            ]
        }

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session)

        assert len(learnings) == 1
        assert isinstance(learnings[0], Learning)
        assert learnings[0].text == "The pool size was hardcoded."
        assert learnings[0].category == LearningCategory.DEBUGGING_INSIGHT

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller._get_git_sha", return_value="def456")
    @patch("crowd_control.ingest.distiller.call_claude")
    def test_populates_metadata(self, mock_call, mock_sha):
        mock_call.return_value = {
            "learnings": [
                {
                    "text": "A learning",
                    "category": "gotcha",
                    "tags": ["python"],
                    "confidence": 0.8,
                }
            ]
        }

        session = _make_session(project_path="/my/project")
        segment = _make_segment()
        learnings = distill_segment(segment, session)

        assert learnings[0].project == "/my/project"
        assert learnings[0].session_id == "test-session-001"
        assert learnings[0].git_sha == "def456"
        assert learnings[0].confidence == 0.8

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.call_claude")
    def test_skips_invalid_learnings(self, mock_call, mock_sha):
        mock_call.return_value = {
            "learnings": [
                {
                    "text": "Valid learning",
                    "category": "gotcha",
                    "tags": [],
                    "confidence": 0.7,
                },
                {
                    "text": "Invalid learning",
                    "category": "not_a_real_category",
                    "tags": [],
                    "confidence": 0.5,
                },
            ]
        }

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session)

        assert len(learnings) == 1
        assert learnings[0].text == "Valid learning"

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.call_claude")
    def test_empty_learnings_list(self, mock_call, mock_sha):
        mock_call.return_value = {"learnings": []}

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session)

        assert learnings == []

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.call_claude")
    def test_empty_prompt_text_skips_llm_call(self, mock_call):
        """Segment with very short text should skip LLM call."""
        messages = [
            Message(
                role=MessageRole.USER,
                content=[TextBlock(text="hi")],
                uuid="u1",
                timestamp=_ts(),
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[TextBlock(text="ok")],
                uuid="a1",
                timestamp=_ts(),
            ),
        ]
        segment = _make_segment(messages=messages)
        session = _make_session(segments=[segment])
        learnings = distill_segment(segment, session)

        assert learnings == []
        mock_call.assert_not_called()

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.call_claude")
    def test_empty_project_path_uses_cwd(self, mock_call, mock_sha):
        """When session.project_path is empty, distill_segment falls back to cwd."""
        mock_call.return_value = {
            "learnings": [
                {
                    "text": "A learning",
                    "category": "gotcha",
                    "tags": [],
                    "confidence": 0.7,
                }
            ]
        }

        session = _make_session(project_path="")
        segment = _make_segment()
        learnings = distill_segment(segment, session)

        assert len(learnings) == 1
        assert learnings[0].project != ""
        assert learnings[0].project == str(Path.cwd())

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.call_claude")
    def test_oversized_learning_skipped(self, mock_call, mock_sha):
        """Oversized learning text is skipped; valid learnings in the same batch are kept."""
        mock_call.return_value = {
            "learnings": [
                {
                    "text": "Valid short learning",
                    "category": "gotcha",
                    "tags": [],
                    "confidence": 0.7,
                },
                {
                    "text": "X" * 3000,
                    "category": "gotcha",
                    "tags": [],
                    "confidence": 0.9,
                },
            ]
        }

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session)

        assert len(learnings) == 1
        assert learnings[0].text == "Valid short learning"


class TestIsSegmentWorthDistilling:
    def test_too_few_messages(self):
        """Single-message segment is not worth distilling."""
        seg = ConversationSegment(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=[TextBlock(text="hello")],
                    uuid="u1",
                    timestamp=_ts(),
                ),
            ],
            tool_names=[],
            start_time=_ts(),
            end_time=_ts(),
        )
        assert is_segment_worth_distilling(seg) is False

    def test_no_assistant_messages(self):
        """Segment with only user messages is not worth distilling."""
        seg = ConversationSegment(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=[TextBlock(text="hello")],
                    uuid="u1",
                    timestamp=_ts(),
                ),
                Message(
                    role=MessageRole.USER,
                    content=[TextBlock(text="world")],
                    uuid="u2",
                    timestamp=_ts(),
                ),
            ],
            tool_names=[],
            start_time=_ts(),
            end_time=_ts(),
        )
        assert is_segment_worth_distilling(seg) is False

    def test_all_empty_thinking(self):
        """Segment where all assistant content is empty thinking blocks is not worth distilling."""
        seg = ConversationSegment(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=[TextBlock(text="hello")],
                    uuid="u1",
                    timestamp=_ts(),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[ThinkingBlock(thinking="")],
                    uuid="a1",
                    timestamp=_ts(),
                ),
            ],
            tool_names=[],
            start_time=_ts(),
            end_time=_ts(),
        )
        assert is_segment_worth_distilling(seg) is False

    def test_non_empty_thinking_qualifies(self):
        """Assistant with non-empty thinking content qualifies."""
        seg = ConversationSegment(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=[TextBlock(text="hello")],
                    uuid="u1",
                    timestamp=_ts(),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[ThinkingBlock(thinking="actual thought")],
                    uuid="a1",
                    timestamp=_ts(),
                ),
            ],
            tool_names=[],
            start_time=_ts(),
            end_time=_ts(),
        )
        assert is_segment_worth_distilling(seg) is True

    def test_normal_segment_qualifies(self):
        """Standard user + assistant with text qualifies."""
        seg = _make_segment()
        assert is_segment_worth_distilling(seg) is True


class TestDistillSession:

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_caps_at_max_learnings(self, mock_distill):
        def make_learning(confidence):
            return Learning(
                text=f"learning with confidence {confidence}",
                category=LearningCategory.GOTCHA,
                project="/test",
                session_id="s1",
                confidence=confidence,
            )

        # Return 3 learnings per segment
        mock_distill.side_effect = [
            [make_learning(0.9), make_learning(0.5), make_learning(0.7)],
            [make_learning(0.8), make_learning(0.6), make_learning(0.4)],
        ]

        segments = [_make_segment(), _make_segment()]
        session = _make_session(segments=segments)
        learnings = distill_session(session, max_learnings=3)

        assert len(learnings) == 3
        # Should keep highest confidence: 0.9, 0.8, 0.7
        confidences = [learning.confidence for learning in learnings]
        assert 0.9 in confidences
        assert 0.8 in confidences
        assert 0.7 in confidences

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_continues_on_segment_error(self, mock_distill):
        good_learning = Learning(
            text="good learning",
            category=LearningCategory.GOTCHA,
            project="/test",
            session_id="s1",
            confidence=0.8,
        )
        mock_distill.side_effect = [
            DistillationError("segment 1 failed"),
            [good_learning],
        ]

        segments = [_make_segment(), _make_segment()]
        session = _make_session(segments=segments)
        learnings = distill_session(session)

        assert len(learnings) == 1
        assert learnings[0].text == "good learning"

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_progress_callback_called(self, mock_distill):
        mock_distill.return_value = []

        segments = [_make_segment(), _make_segment()]
        session = _make_session(segments=segments)

        callback = MagicMock()
        distill_session(session, progress_callback=callback)

        assert callback.call_count == 2
        callback.assert_any_call(0, 2)
        callback.assert_any_call(1, 2)
