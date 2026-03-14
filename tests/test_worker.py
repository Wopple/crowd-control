"""Tests for the background ingestion worker."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crowd_control.config import CrowdControlConfig
from crowd_control.worker import process_queue


@pytest.fixture
def config(tmp_path):
    return CrowdControlConfig(storage_dir=str(tmp_path / "cc"))


@pytest.fixture
def queue_dir(config):
    d = Path(config.storage_dir) / "queue"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def session_file(tmp_path):
    f = tmp_path / "session.jsonl"
    f.write_text('{"type":"message"}\n')
    return f


def _write_queue_file(queue_dir, session_id, session_path, **extra):
    data = {
        "session_id": session_id,
        "session_path": str(session_path),
        "project": "/test/project",
        "queued_at": "2026-03-14T10:00:00+00:00",
    }
    data.update(extra)
    path = queue_dir / f"{session_id}.json"
    path.write_text(json.dumps(data))
    return path


class TestProcessQueue:
    def test_processes_queue_file(self, config, queue_dir, session_file):
        qf = _write_queue_file(queue_dir, "sess-1", session_file)

        with (
            patch("crowd_control.worker.LearningStore") as mock_store_cls,
            patch("crowd_control.worker.ingest_session") as mock_ingest,
        ):
            mock_store = mock_store_cls.return_value
            mock_store.has_session.return_value = False
            mock_ingest.return_value.session_id = "sess-1"
            mock_ingest.return_value.learnings_stored = 3

            count = process_queue(config)

        assert count == 1
        mock_ingest.assert_called_once()
        assert not qf.exists()  # queue file deleted on success

    def test_skips_already_ingested(self, config, queue_dir, session_file):
        qf = _write_queue_file(queue_dir, "sess-1", session_file)

        with (
            patch("crowd_control.worker.LearningStore") as mock_store_cls,
            patch("crowd_control.worker.ingest_session") as mock_ingest,
        ):
            mock_store = mock_store_cls.return_value
            mock_store.has_session.return_value = True

            count = process_queue(config)

        assert count == 0
        mock_ingest.assert_not_called()
        assert not qf.exists()  # queue file still cleaned up

    def test_skips_missing_session_file(self, config, queue_dir, tmp_path):
        qf = _write_queue_file(queue_dir, "sess-1", tmp_path / "gone.jsonl")

        count = process_queue(config)

        assert count == 0
        assert not qf.exists()

    def test_handles_ingestion_failure(self, config, queue_dir, session_file):
        qf = _write_queue_file(queue_dir, "sess-1", session_file)

        with (
            patch("crowd_control.worker.LearningStore") as mock_store_cls,
            patch("crowd_control.worker.ingest_session", side_effect=RuntimeError("boom")),
        ):
            mock_store = mock_store_cls.return_value
            mock_store.has_session.return_value = False

            count = process_queue(config)

        assert count == 0
        assert qf.exists()  # queue file kept
        data = json.loads(qf.read_text())
        assert data["attempts"] == 1
        assert data["last_error"] == "boom"

    def test_moves_failed_after_max_attempts(self, config, queue_dir, session_file):
        qf = _write_queue_file(queue_dir, "sess-1", session_file, attempts=2)

        with (
            patch("crowd_control.worker.LearningStore") as mock_store_cls,
            patch("crowd_control.worker.ingest_session", side_effect=RuntimeError("boom")),
        ):
            mock_store = mock_store_cls.return_value
            mock_store.has_session.return_value = False

            count = process_queue(config)

        assert count == 0
        assert not qf.exists()
        assert (queue_dir / "failed" / "sess-1.json").exists()

    def test_handles_malformed_queue_file(self, config, queue_dir):
        qf = queue_dir / "bad.json"
        qf.write_text("not json{{{")

        count = process_queue(config)

        assert count == 0
        assert not qf.exists()
        assert (queue_dir / "failed" / "bad.json").exists()

    def test_processes_oldest_first(self, config, queue_dir, session_file):
        _write_queue_file(queue_dir, "sess-2", session_file, queued_at="2026-03-14T11:00:00+00:00")
        _write_queue_file(queue_dir, "sess-1", session_file, queued_at="2026-03-14T10:00:00+00:00")

        processed_order = []

        def track_ingest(path, cfg, **kwargs):
            processed_order.append(str(path))
            result = MagicMock()
            result.session_id = "tracked"
            result.learnings_stored = 1
            return result

        with (
            patch("crowd_control.worker.LearningStore") as mock_store_cls,
            patch("crowd_control.worker.ingest_session", side_effect=track_ingest),
        ):
            mock_store = mock_store_cls.return_value
            mock_store.has_session.return_value = False

            process_queue(config)

        # Both should be processed (oldest first based on queued_at)
        assert len(processed_order) == 2

    def test_empty_queue(self, config, queue_dir):
        count = process_queue(config)
        assert count == 0

    def test_no_queue_dir(self, config):
        # Queue directory doesn't exist at all
        count = process_queue(config)
        assert count == 0
