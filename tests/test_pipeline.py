"""Tests for the ingestion pipeline."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import FakeEmbedder

from crowd_control.config import CrowdControlConfig
from crowd_control.ingest.pipeline import IngestResult, ingest_session
from crowd_control.storage.models import (
    ConversationSegment,
    Learning,
    Message,
    MessageRole,
    Session,
    TextBlock,
)


def _make_session() -> Session:
    """Create a minimal session for pipeline tests."""
    msg1 = Message(
        role=MessageRole.USER,
        content=[TextBlock(text="How do I fix this bug?")],
        uuid="u1",
        timestamp=datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
    )
    msg2 = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="The issue is in the parser. Here's the fix...")],
        uuid="u2",
        parent_uuid="u1",
        timestamp=datetime(2025, 1, 15, 10, 1, tzinfo=UTC),
    )
    segment = ConversationSegment(
        messages=[msg1, msg2],
        tool_names=["Read"],
        start_time=datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
        end_time=datetime(2025, 1, 15, 10, 1, tzinfo=UTC),
    )
    return Session(
        session_id="test-sess-001",
        project_path="/test/project",
        git_branch="main",
        segments=[segment],
        start_time=datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
        end_time=datetime(2025, 1, 15, 10, 1, tzinfo=UTC),
        message_count=2,
    )


def _make_learnings() -> list[Learning]:
    """Create sample learnings that distill_session would return."""
    return [
        Learning(
            id="learn-001",
            text="The parser splits on newlines but should split on double newlines",
            category="debugging_insight",
            tags=["parser", "python"],
            project="/test/project",
            session_id="test-sess-001",
            confidence=0.85,
        ),
        Learning(
            id="learn-002",
            text="Use frozen dataclasses for config to prevent mutation",
            category="architecture_decision",
            tags=["python", "config"],
            project="/test/project",
            session_id="test-sess-001",
            confidence=0.7,
        ),
    ]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder(dimensions=8)


@pytest.fixture
def pipeline_config(tmp_path):
    # Use the Claude provider in tests so the factory does not require the
    # optional `ollama` package. distill_session is mocked, so no real LLM
    # call ever happens.
    from crowd_control.config import DistillationConfig

    return CrowdControlConfig(
        storage_dir=str(tmp_path / "cc"),
        distillation=DistillationConfig(model="claude-code:haiku"),
    )


class TestIngestEndToEnd:
    @patch("crowd_control.ingest.pipeline.create_embedder")
    @patch("crowd_control.ingest.pipeline.distill_session")
    @patch("crowd_control.ingest.pipeline.parse_session_file")
    def test_full_pipeline(
        self, mock_parse, mock_distill, mock_create_embedder, fake_embedder, pipeline_config
    ):
        mock_parse.return_value = _make_session()
        mock_distill.return_value = _make_learnings()
        mock_create_embedder.return_value = fake_embedder

        result = ingest_session(Path("/fake/session.jsonl"), pipeline_config)

        assert isinstance(result, IngestResult)
        assert result.session_id == "test-sess-001"
        assert result.segments_processed == 1
        assert result.learnings_distilled == 2
        assert result.learnings_stored == 2
        assert result.learnings_deduplicated == 0

    @patch("crowd_control.ingest.pipeline.create_embedder")
    @patch("crowd_control.ingest.pipeline.distill_session")
    @patch("crowd_control.ingest.pipeline.parse_session_file")
    def test_dedup_across_runs(
        self, mock_parse, mock_distill, mock_create_embedder, fake_embedder, pipeline_config
    ):
        mock_parse.return_value = _make_session()
        mock_distill.return_value = _make_learnings()
        mock_create_embedder.return_value = fake_embedder

        result1 = ingest_session(Path("/fake/session.jsonl"), pipeline_config)
        assert result1.learnings_stored == 2

        result2 = ingest_session(Path("/fake/session.jsonl"), pipeline_config)
        assert result2.learnings_stored == 0
        assert result2.learnings_deduplicated == 2

    @patch("crowd_control.ingest.pipeline.create_embedder")
    @patch("crowd_control.ingest.pipeline.distill_session")
    @patch("crowd_control.ingest.pipeline.parse_session_file")
    def test_empty_session(
        self, mock_parse, mock_distill, mock_create_embedder, fake_embedder, pipeline_config
    ):
        mock_parse.return_value = _make_session()
        mock_distill.return_value = []
        mock_create_embedder.return_value = fake_embedder

        result = ingest_session(Path("/fake/session.jsonl"), pipeline_config)
        assert result.learnings_distilled == 0
        assert result.learnings_stored == 0
        # create_embedder should not be called when there are no learnings
        mock_create_embedder.assert_not_called()

    @patch("crowd_control.ingest.pipeline.create_embedder")
    @patch("crowd_control.ingest.pipeline.distill_session")
    @patch("crowd_control.ingest.pipeline.parse_session_file")
    def test_result_counts(
        self, mock_parse, mock_distill, mock_create_embedder, fake_embedder, pipeline_config
    ):
        mock_parse.return_value = _make_session()
        mock_distill.return_value = _make_learnings()
        mock_create_embedder.return_value = fake_embedder

        result = ingest_session(Path("/fake/session.jsonl"), pipeline_config)
        assert result.learnings_distilled == result.learnings_stored + result.learnings_deduplicated

    @patch("crowd_control.ingest.pipeline.create_embedder")
    @patch("crowd_control.ingest.pipeline.distill_session")
    @patch("crowd_control.ingest.pipeline.parse_session_file")
    def test_embedding_failure(
        self, mock_parse, mock_distill, mock_create_embedder, pipeline_config
    ):
        from crowd_control.embed.base import EmbeddingError

        mock_parse.return_value = _make_session()
        mock_distill.return_value = _make_learnings()
        bad_embedder = FakeEmbedder()
        bad_embedder.embed = lambda texts: (_ for _ in ()).throw(
            EmbeddingError("Connection refused")
        )
        mock_create_embedder.return_value = bad_embedder

        with pytest.raises(EmbeddingError, match="Connection refused"):
            ingest_session(Path("/fake/session.jsonl"), pipeline_config)
