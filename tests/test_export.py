"""Tests for the export_learnings functionality."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import FakeEmbedder, insert_learning

from crowd_control.storage.db import LearningStore


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, embedder):
    db_path = str(tmp_path / "test_db")
    return LearningStore(db_path, vector_dimensions=embedder.dimensions)


class TestExportLearnings:
    def test_empty_db_returns_empty_list(self, store):
        result = store.export_learnings()
        assert result == []

    def test_export_returns_all_learnings(self, store, embedder):
        insert_learning(store, embedder, "First learning", id="id-1")
        insert_learning(store, embedder, "Second learning", id="id-2")

        result = store.export_learnings()

        assert len(result) == 2
        texts = {r["text"] for r in result}
        assert texts == {"First learning", "Second learning"}

    def test_exported_learnings_contain_expected_fields(self, store, embedder):
        insert_learning(store, embedder, "A learning about testing")

        result = store.export_learnings()

        assert len(result) == 1
        record = result[0]
        expected_fields = {
            "id", "text", "category", "tags", "project",
            "session_id", "git_sha", "timestamp", "confidence",
            "active_count", "stale", "shared",
        }
        assert expected_fields.issubset(record.keys())

    def test_vectors_not_in_output(self, store, embedder):
        insert_learning(store, embedder, "Learning one", id="id-1")
        insert_learning(store, embedder, "Learning two", id="id-2")

        result = store.export_learnings()

        for record in result:
            assert "vector" not in record
            assert "_rowid" not in record
            assert "_distance" not in record

    def test_project_filter(self, store, embedder):
        insert_learning(store, embedder, "Project A learning", id="id-a", project="/proj/a")
        insert_learning(store, embedder, "Project B learning", id="id-b", project="/proj/b")

        result = store.export_learnings(project="/proj/a")

        assert len(result) == 1
        assert result[0]["text"] == "Project A learning"

    def test_category_filter(self, store, embedder):
        insert_learning(store, embedder, "A gotcha", id="id-g", category="gotcha")
        insert_learning(
            store, embedder, "A debugging insight", id="id-d", category="debugging_insight"
        )

        result = store.export_learnings(category="gotcha")

        assert len(result) == 1
        assert result[0]["text"] == "A gotcha"

    def test_timestamps_are_iso_strings(self, store, embedder):
        ts = datetime(2025, 6, 15, 12, 30, 0, tzinfo=UTC)
        insert_learning(store, embedder, "Timestamped learning", timestamp=ts)

        result = store.export_learnings()

        assert len(result) == 1
        timestamp_value = result[0]["timestamp"]
        assert isinstance(timestamp_value, str)
        # Verify it parses as a valid ISO timestamp
        parsed = datetime.fromisoformat(timestamp_value)
        assert parsed.year == 2025
        assert parsed.month == 6
        assert parsed.day == 15
