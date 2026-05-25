"""Tests for LLM-powered distillation pipeline."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crowd_control.hooks import INGEST_MARKER_ENV
from crowd_control.ingest.distiller import (
    LEARNING_EXTRACTION_SCHEMA,
    DistillationError,
    _get_git_sha,
    build_distillation_prompt,
    distill_segment,
    distill_session,
    is_segment_worth_distilling,
)
from crowd_control.ingest.llm.claude import ClaudeCLILLM
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


class FakeLLM:
    """Stub DistillerLLM for tests. Returns a canned schema-conformant dict."""

    def __init__(self, response=None, concurrency=8, provider="fake", model="fake-model"):
        self.response = response if response is not None else {"learnings": []}
        self._concurrency = concurrency
        self._provider = provider
        self._model = model
        self.calls: list[tuple[str, dict]] = []

    def generate_structured(self, prompt: str, schema: dict) -> dict:
        self.calls.append((prompt, schema))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    @property
    def recommended_concurrency(self) -> int:
        return self._concurrency

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def model_id(self) -> str:
        return self._model


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


class TestClaudeCLILLM:
    def _mock_result(self, stdout="", returncode=0, stderr=""):
        mock = MagicMock()
        mock.stdout = stdout
        mock.returncode = returncode
        mock.stderr = stderr
        return mock

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_successful_call(self, mock_run):
        response = _load_fixture_response()
        mock_run.return_value = self._mock_result(stdout=json.dumps(response))

        llm = ClaudeCLILLM(model="haiku")
        result = llm.generate_structured("test prompt", LEARNING_EXTRACTION_SCHEMA)
        assert "learnings" in result
        assert len(result["learnings"]) == 3
        mock_run.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_claude_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("claude not found")

        with pytest.raises(DistillationError, match="claude CLI not found"):
            ClaudeCLILLM(model="haiku").generate_structured(
                "test prompt", LEARNING_EXTRACTION_SCHEMA
            )

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.time.sleep")
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_timeout(self, mock_run, mock_sleep):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)

        with pytest.raises(DistillationError, match="timed out"):
            ClaudeCLILLM(model="haiku").generate_structured(
                "test prompt", LEARNING_EXTRACTION_SCHEMA
            )
        assert mock_run.call_count == 3

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.time.sleep")
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_nonzero_exit(self, mock_run, mock_sleep):
        mock_run.return_value = self._mock_result(returncode=1, stderr="error msg")

        with pytest.raises(DistillationError, match="exited with code 1"):
            ClaudeCLILLM(model="haiku").generate_structured(
                "test prompt", LEARNING_EXTRACTION_SCHEMA
            )
        assert mock_run.call_count == 3

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_invalid_json_output(self, mock_run):
        mock_run.return_value = self._mock_result(stdout="not json at all")

        with pytest.raises(DistillationError, match="claude CLI returned invalid JSON"):
            ClaudeCLILLM(model="haiku").generate_structured(
                "test prompt", LEARNING_EXTRACTION_SCHEMA
            )
        assert mock_run.call_count == 1

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_missing_structured_output(self, mock_run):
        mock_run.return_value = self._mock_result(
            stdout=json.dumps({"result": "no structured_output key"})
        )

        with pytest.raises(DistillationError, match="missing 'structured_output'"):
            ClaudeCLILLM(model="haiku").generate_structured(
                "test prompt", LEARNING_EXTRACTION_SCHEMA
            )
        assert mock_run.call_count == 1

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_streaming_json_array_format(self, mock_run):
        fixture = _load_fixture_response()
        streaming_output = [
            {"type": "system", "subtype": "init", "session_id": "test"},
            {"type": "assistant", "message": {"role": "assistant", "content": []}},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "Done",
                "structured_output": fixture["structured_output"],
            },
        ]
        mock_run.return_value = self._mock_result(stdout=json.dumps(streaming_output))

        result = ClaudeCLILLM(model="haiku").generate_structured(
            "test prompt", LEARNING_EXTRACTION_SCHEMA
        )
        assert "learnings" in result
        assert len(result["learnings"]) == 3

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_streaming_json_array_no_result_element(self, mock_run):
        streaming_output = [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"role": "assistant", "content": []}},
        ]
        mock_run.return_value = self._mock_result(stdout=json.dumps(streaming_output))

        with pytest.raises(DistillationError, match="missing 'structured_output'"):
            ClaudeCLILLM(model="haiku").generate_structured(
                "test prompt", LEARNING_EXTRACTION_SCHEMA
            )

    @patch.dict("os.environ", {"CLAUDECODE": "1"}, clear=False)
    def test_rejects_inside_claude_code(self):
        with pytest.raises(DistillationError, match="CLAUDECODE"):
            ClaudeCLILLM(model="haiku").generate_structured(
                "test prompt", LEARNING_EXTRACTION_SCHEMA
            )

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_sets_ingest_marker_on_subprocess(self, mock_run):
        """ClaudeCLILLM must pass INGEST_MARKER_ENV=1 to the claude subprocess.

        Producer half of the recursive-ingestion guard. Consumer half is in
        tests/test_hooks.py.
        """
        response = _load_fixture_response()
        mock_run.return_value = self._mock_result(stdout=json.dumps(response))

        ClaudeCLILLM(model="haiku").generate_structured("test prompt", LEARNING_EXTRACTION_SCHEMA)

        call_kwargs = mock_run.call_args.kwargs
        assert "env" in call_kwargs
        assert call_kwargs["env"].get(INGEST_MARKER_ENV) == "1"

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.time.sleep")
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_retry_succeeds_on_second_attempt(self, mock_run, mock_sleep):
        response = _load_fixture_response()
        mock_run.side_effect = [
            self._mock_result(returncode=1, stderr="temporary failure"),
            self._mock_result(stdout=json.dumps(response)),
        ]

        result = ClaudeCLILLM(model="haiku").generate_structured(
            "test prompt", LEARNING_EXTRACTION_SCHEMA
        )
        assert "learnings" in result
        assert mock_run.call_count == 2
        mock_sleep.assert_called_once_with(2)

    @patch.dict("os.environ", {}, clear=True)
    @patch("crowd_control.ingest.llm.claude.time.sleep")
    @patch("crowd_control.ingest.llm.claude.subprocess.run")
    def test_retry_exhausted(self, mock_run, mock_sleep):
        mock_run.return_value = self._mock_result(returncode=1, stderr="persistent failure")

        with pytest.raises(DistillationError, match="exited with code 1"):
            ClaudeCLILLM(model="haiku").generate_structured(
                "test prompt", LEARNING_EXTRACTION_SCHEMA
            )
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2

    def test_recommended_concurrency(self):
        assert ClaudeCLILLM(model="haiku").recommended_concurrency == 8

    def test_provider_and_model_properties(self):
        llm = ClaudeCLILLM(model="haiku")
        assert llm.provider_name == "claude-code"
        assert llm.model_id == "haiku"


class TestGetGitSha:
    def test_empty_project_path_returns_none(self):
        """Bug: _get_git_sha('') passes cwd='' to subprocess — undefined behavior."""
        result = _get_git_sha("")
        assert result is None


class TestDistillSegment:
    def _llm(self, response):
        return FakeLLM(response=response)

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value="abc123")
    def test_returns_learning_objects(self, mock_sha):
        llm = self._llm({
            "learnings": [
                {
                    "text": "The pool size was hardcoded.",
                    "category": "debugging_insight",
                    "tags": ["database", "pooling"],
                    "confidence": 0.9,
                }
            ]
        })

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm)

        assert len(learnings) == 1
        assert isinstance(learnings[0], Learning)
        assert learnings[0].text == "The pool size was hardcoded."
        assert learnings[0].category == LearningCategory.DEBUGGING_INSIGHT

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value="def456")
    def test_populates_metadata(self, mock_sha):
        llm = self._llm({
            "learnings": [
                {
                    "text": "A learning",
                    "category": "gotcha",
                    "tags": ["python"],
                    "confidence": 0.8,
                }
            ]
        })

        session = _make_session(project_path="/my/project")
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm)

        assert learnings[0].project == "/my/project"
        assert learnings[0].session_id == "test-session-001"
        assert learnings[0].git_sha == "def456"
        assert learnings[0].confidence == 0.8

    @patch("crowd_control.ingest.distiller._get_git_sha")
    def test_uses_provided_git_sha(self, mock_sha):
        """When git_sha is provided, it is used directly without calling _get_git_sha."""
        llm = self._llm({
            "learnings": [
                {
                    "text": "A learning",
                    "category": "gotcha",
                    "tags": [],
                    "confidence": 0.7,
                }
            ]
        })

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm, git_sha="provided-sha")

        assert learnings[0].git_sha == "provided-sha"
        mock_sha.assert_not_called()

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value="resolved-sha")
    def test_resolves_git_sha_when_not_provided(self, mock_sha):
        llm = self._llm({
            "learnings": [
                {
                    "text": "A learning",
                    "category": "gotcha",
                    "tags": [],
                    "confidence": 0.7,
                }
            ]
        })

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm)

        assert learnings[0].git_sha == "resolved-sha"
        mock_sha.assert_called_once()

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    def test_skips_invalid_learnings(self, mock_sha):
        llm = self._llm({
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
        })

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm)

        assert len(learnings) == 1
        assert learnings[0].text == "Valid learning"

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    def test_empty_learnings_list(self, mock_sha):
        llm = self._llm({"learnings": []})

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm)

        assert learnings == []

    def test_empty_prompt_text_skips_llm_call(self):
        """Segment with very short text should skip LLM call."""
        llm = self._llm({"learnings": []})
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
        learnings = distill_segment(segment, session, llm)

        assert learnings == []
        assert llm.calls == []

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    def test_empty_project_path_uses_cwd(self, mock_sha):
        """When session.project_path is empty, distill_segment falls back to cwd."""
        llm = self._llm({
            "learnings": [
                {
                    "text": "A learning",
                    "category": "gotcha",
                    "tags": [],
                    "confidence": 0.7,
                }
            ]
        })

        session = _make_session(project_path="")
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm)

        assert len(learnings) == 1
        assert learnings[0].project != ""
        assert learnings[0].project == str(Path.cwd())

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    def test_project_name_from_config_file(self, mock_sha, tmp_path):
        """When .crowd-control exists at session path, learning uses the project name."""
        (tmp_path / ".crowd-control").write_text('[project]\nname = "my-app"\n')
        llm = self._llm({
            "learnings": [
                {
                    "text": "A learning",
                    "category": "gotcha",
                    "tags": [],
                    "confidence": 0.7,
                }
            ]
        })

        session = _make_session(project_path=str(tmp_path))
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm)

        assert len(learnings) == 1
        assert learnings[0].project == "my-app"

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    def test_project_path_without_config_file(self, mock_sha, tmp_path):
        """Without .crowd-control, learning uses the filesystem path."""
        llm = self._llm({
            "learnings": [
                {
                    "text": "A learning",
                    "category": "gotcha",
                    "tags": [],
                    "confidence": 0.7,
                }
            ]
        })

        session = _make_session(project_path=str(tmp_path))
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm)

        assert len(learnings) == 1
        assert learnings[0].project == str(tmp_path.resolve())

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    def test_oversized_learning_skipped(self, mock_sha):
        """Oversized learning text is skipped; valid learnings in the same batch are kept."""
        llm = self._llm({
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
        })

        session = _make_session()
        segment = _make_segment()
        learnings = distill_segment(segment, session, llm)

        assert len(learnings) == 1
        assert learnings[0].text == "Valid short learning"


class TestIsSegmentWorthDistilling:
    def test_too_few_messages(self):
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
        seg = _make_segment()
        assert is_segment_worth_distilling(seg) is True


class TestDistillSession:
    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_caps_at_max_learnings(self, mock_distill, mock_sha):
        def make_learning(confidence):
            return Learning(
                text=f"learning with confidence {confidence}",
                category=LearningCategory.GOTCHA,
                project="/test",
                session_id="s1",
                confidence=confidence,
            )

        mock_distill.side_effect = [
            [make_learning(0.9), make_learning(0.5), make_learning(0.7)],
            [make_learning(0.8), make_learning(0.6), make_learning(0.4)],
        ]

        segments = [_make_segment(), _make_segment()]
        session = _make_session(segments=segments)
        learnings = distill_session(session, FakeLLM(), max_learnings=3)

        assert len(learnings) == 3
        confidences = [learning.confidence for learning in learnings]
        assert 0.9 in confidences
        assert 0.8 in confidences
        assert 0.7 in confidences

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_continues_on_segment_error(self, mock_distill, mock_sha):
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
        learnings = distill_session(session, FakeLLM())

        assert len(learnings) == 1
        assert learnings[0].text == "good learning"

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_progress_callback_called(self, mock_distill, mock_sha):
        mock_distill.return_value = []

        segments = [_make_segment(), _make_segment()]
        session = _make_session(segments=segments)

        callback = MagicMock()
        distill_session(session, FakeLLM(), progress_callback=callback)

        assert callback.call_count == 2
        callback.assert_any_call(1, 2)
        callback.assert_any_call(2, 2)

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_max_workers_limits_concurrency(self, mock_distill, mock_sha):
        mock_distill.return_value = []

        segments = [_make_segment() for _ in range(4)]
        session = _make_session(segments=segments)

        with patch(
            "crowd_control.ingest.distiller.ThreadPoolExecutor",
            wraps=ThreadPoolExecutor,
        ) as mock_pool:
            distill_session(session, FakeLLM(), max_workers=2)
            mock_pool.assert_called_once_with(max_workers=2)

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_max_workers_capped_at_segment_count(self, mock_distill, mock_sha):
        mock_distill.return_value = []

        segments = [_make_segment(), _make_segment()]
        session = _make_session(segments=segments)

        with patch(
            "crowd_control.ingest.distiller.ThreadPoolExecutor",
            wraps=ThreadPoolExecutor,
        ) as mock_pool:
            distill_session(session, FakeLLM(), max_workers=8)
            mock_pool.assert_called_once_with(max_workers=2)

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_max_workers_defaults_to_llm_recommendation(self, mock_distill, mock_sha):
        """When max_workers is None, distill_session uses llm.recommended_concurrency."""
        mock_distill.return_value = []

        segments = [_make_segment() for _ in range(5)]
        session = _make_session(segments=segments)

        with patch(
            "crowd_control.ingest.distiller.ThreadPoolExecutor",
            wraps=ThreadPoolExecutor,
        ) as mock_pool:
            distill_session(session, FakeLLM(concurrency=3))
            mock_pool.assert_called_once_with(max_workers=3)

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_parallel_results_ordered_by_segment(self, mock_distill, mock_sha):
        def make_learning(text):
            return Learning(
                text=text,
                category=LearningCategory.GOTCHA,
                project="/test",
                session_id="s1",
                confidence=0.7,
            )

        mock_distill.side_effect = [
            [make_learning("from segment 0")],
            [make_learning("from segment 1")],
            [make_learning("from segment 2")],
        ]

        segments = [_make_segment() for _ in range(3)]
        session = _make_session(segments=segments)
        learnings = distill_session(session, FakeLLM(concurrency=1))

        assert [learning.text for learning in learnings] == [
            "from segment 0",
            "from segment 1",
            "from segment 2",
        ]

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value="abc123")
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_git_sha_resolved_once(self, mock_distill, mock_sha):
        mock_distill.return_value = []

        segments = [_make_segment() for _ in range(3)]
        session = _make_session(segments=segments)
        distill_session(session, FakeLLM())

        mock_sha.assert_called_once()
        for call in mock_distill.call_args_list:
            assert call.kwargs.get("git_sha") == "abc123"

    @patch("crowd_control.ingest.distiller._get_git_sha", return_value=None)
    @patch("crowd_control.ingest.distiller.distill_segment")
    def test_logs_provider_and_model(self, mock_distill, mock_sha, caplog):
        """Headline INFO line must include external provider name AND model id."""
        import logging

        mock_distill.return_value = []
        segments = [_make_segment() for _ in range(2)]
        session = _make_session(segments=segments)

        with caplog.at_level(logging.INFO, logger="crowd_control.ingest.distiller"):
            distill_session(session, FakeLLM(provider="ollama", model="qwen3:8b"))

        info_lines = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            "provider=ollama" in line
            and "model=qwen3:8b" in line
            and "concurrency=" in line
            and "segments=2" in line
            for line in info_lines
        ), f"expected provider/model/concurrency/segments in INFO; got {info_lines!r}"
