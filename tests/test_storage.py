"""Tests for LanceDB storage operations."""

from datetime import UTC, datetime

import pytest
from conftest import FakeEmbedder

from crowd_control.storage.db import LearningStore


@pytest.fixture
def embedder():
    return FakeEmbedder(dimensions=8)


@pytest.fixture
def store(tmp_path):
    return LearningStore(str(tmp_path / "test_db"), vector_dimensions=8)


def _make_learning(embedder, text="Test learning", **overrides):
    """Create a learning dict ready for storage."""
    vector = embedder.embed([text])[0]
    record = {
        "id": overrides.pop("id", f"id-{hash(text) % 10000}"),
        "vector": vector,
        "text": text,
        "category": overrides.pop("category", "debugging_insight"),
        "tags": overrides.pop("tags", ["python"]),
        "project": overrides.pop("project", "/test/project"),
        "session_id": overrides.pop("session_id", "sess-001"),
        "git_sha": overrides.pop("git_sha", "abc123"),
        "timestamp": overrides.pop("timestamp", datetime(2025, 1, 15, 10, 0, tzinfo=UTC)),
        "confidence": overrides.pop("confidence", 0.8),
        "active_count": overrides.pop("active_count", 0),
        "stale": overrides.pop("stale", False),
        "shared": overrides.pop("shared", False),
    }
    record.update(overrides)
    return record


class TestAddAndGet:
    def test_add_and_get(self, store, embedder):
        record = _make_learning(embedder, id="learn-1")
        store.add([record])
        result = store.get("learn-1")
        assert result is not None
        assert result["id"] == "learn-1"
        assert result["text"] == "Test learning"
        assert result["category"] == "debugging_insight"
        assert result["confidence"] == pytest.approx(0.8, abs=0.01)

    def test_add_multiple(self, store, embedder):
        records = [_make_learning(embedder, text=f"Learning {i}", id=f"id-{i}") for i in range(5)]
        count = store.add(records)
        assert count == 5
        assert store.count() == 5

    def test_add_empty_list(self, store):
        assert store.add([]) == 0

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None


class TestList:
    def test_list_all(self, store, embedder):
        records = [
            _make_learning(
                embedder,
                text=f"Learning {i}",
                id=f"id-{i}",
                timestamp=datetime(2025, 1, i + 1, tzinfo=UTC),
            )
            for i in range(3)
        ]
        store.add(records)
        results = store.list_learnings()
        assert len(results) == 3
        # Ordered by timestamp descending
        timestamps = [r["timestamp"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_filter_by_project(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="A", id="a", project="/proj/a"),
                _make_learning(embedder, text="B", id="b", project="/proj/b"),
            ]
        )
        results = store.list_learnings(project="/proj/a")
        assert len(results) == 1
        assert results[0]["project"] == "/proj/a"

    def test_list_filter_by_category(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="A", id="a", category="gotcha"),
                _make_learning(embedder, text="B", id="b", category="debugging_insight"),
            ]
        )
        results = store.list_learnings(category="gotcha")
        assert len(results) == 1
        assert results[0]["category"] == "gotcha"


class TestDelete:
    def test_delete(self, store, embedder):
        store.add([_make_learning(embedder, id="del-1")])
        assert store.delete("del-1") is True
        assert store.get("del-1") is None

    def test_delete_nonexistent(self, store):
        assert store.delete("nonexistent") is False


class TestCount:
    def test_count(self, store, embedder):
        assert store.count() == 0
        store.add([_make_learning(embedder, text="one", id="1")])
        assert store.count() == 1
        store.add([_make_learning(embedder, text="two", id="2")])
        assert store.count() == 2
        store.delete("1")
        assert store.count() == 1


class TestDedup:
    def test_exact_text(self, store, embedder):
        record1 = _make_learning(embedder, text="exact same text", id="first")
        record2 = _make_learning(embedder, text="exact same text", id="second")
        store.add([record1])
        count = store.add([record2])
        assert count == 0
        assert store.count() == 1

    def test_exact_text_with_quotes(self, store, embedder):
        text = "Use 'is not None' instead of '!= None'"
        record1 = _make_learning(embedder, text=text, id="q1")
        record2 = _make_learning(embedder, text=text, id="q2")
        store.add([record1])
        count = store.add([record2])
        assert count == 0

    def test_near_duplicate(self, store, embedder):
        record1 = _make_learning(embedder, text="test text", id="near-1")
        # Same vector (same text → same hash) but different id
        record2 = _make_learning(embedder, text="test text", id="near-2")
        store.add([record1])
        count = store.add([record2])
        assert count == 0

    def test_allows_dissimilar(self, store, embedder):
        record1 = _make_learning(
            embedder, text="completely different topic about databases", id="d1"
        )
        record2 = _make_learning(
            embedder, text="unrelated subject regarding frontend rendering", id="d2"
        )
        store.add([record1])
        count = store.add([record2])
        assert count == 1
        assert store.count() == 2

    def test_skipped_on_empty_table(self, store, embedder):
        records = [_make_learning(embedder, text="same text", id=f"empty-{i}") for i in range(2)]
        # All inserted because table was empty at start of add()
        count = store.add(records)
        assert count == 2


class TestDimensionHandling:
    def test_dimension_mismatch_raises(self, tmp_path):
        db_path = str(tmp_path / "dim_test")
        LearningStore(db_path, vector_dimensions=8)
        with pytest.raises(ValueError, match="dimension mismatch"):
            LearningStore(db_path, vector_dimensions=16)

    def test_creates_table_on_first_use(self, tmp_path):
        store = LearningStore(str(tmp_path / "new_db"), vector_dimensions=8)
        assert store.count() == 0

    def test_opens_existing_table(self, tmp_path, embedder):
        db_path = str(tmp_path / "reopen_db")
        store1 = LearningStore(db_path, vector_dimensions=8)
        store1.add([_make_learning(embedder, id="persist")])
        store2 = LearningStore(db_path)
        assert store2.count() == 1
        assert store2.get("persist") is not None

    def test_no_dimensions_no_table_raises(self, tmp_path):
        with pytest.raises(ValueError, match="vector_dimensions is required"):
            LearningStore(str(tmp_path / "empty_db"))


class TestActiveCountMigration:
    def test_migrate_adds_active_count_column(self, tmp_path, embedder):
        """Opening an old table without active_count auto-migrates it."""
        import pyarrow as pa

        db_path = str(tmp_path / "migrate_db")
        db = __import__("lancedb").connect(db_path)
        # Create a table with the old schema (no active_count)
        old_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 8)),
                pa.field("text", pa.string()),
                pa.field("category", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("project", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("git_sha", pa.string()),
                pa.field("timestamp", pa.timestamp("us", tz="UTC")),
                pa.field("confidence", pa.float32()),
                pa.field("stale", pa.bool_()),
                pa.field("shared", pa.bool_()),
            ]
        )
        table = db.create_table("learnings", schema=old_schema)
        record = _make_learning(embedder, id="old-1")
        del record["active_count"]
        table.add([record])

        # Reopen with new code — should auto-migrate
        store = LearningStore(db_path)
        result = store.get("old-1")
        assert result is not None
        assert result["active_count"] == 0


class TestIncrementActiveCount:
    def test_single(self, store, embedder):
        store.add([_make_learning(embedder, id="inc-1")])
        store.increment_active_count(["inc-1"])
        result = store.get("inc-1")
        assert result["active_count"] == 1

    def test_batch(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="A", id="a"),
                _make_learning(embedder, text="B", id="b"),
                _make_learning(embedder, text="C", id="c"),
            ]
        )
        store.increment_active_count(["a", "b"])
        assert store.get("a")["active_count"] == 1
        assert store.get("b")["active_count"] == 1
        assert store.get("c")["active_count"] == 0

    def test_repeated(self, store, embedder):
        store.add([_make_learning(embedder, id="rep-1")])
        store.increment_active_count(["rep-1"])
        store.increment_active_count(["rep-1"])
        store.increment_active_count(["rep-1"])
        assert store.get("rep-1")["active_count"] == 3

    def test_missing_id(self, store):
        # Should not raise
        store.increment_active_count(["nonexistent"])

    def test_empty_list(self, store):
        # Should not raise or touch DB
        store.increment_active_count([])


class TestVectorSearch:
    def test_basic(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="python asyncio concurrency", id="v1"),
                _make_learning(embedder, text="javascript react frontend", id="v2"),
                _make_learning(embedder, text="database postgresql indexing", id="v3"),
            ]
        )
        query_vec = embedder.embed(["python asyncio concurrency"])[0]
        results = store.vector_search(query_vec, limit=3, min_similarity=0.0)
        assert len(results) > 0
        # The exact match should be first with highest similarity
        assert results[0]["id"] == "v1"
        assert results[0]["_similarity"] > 0.9

    def test_min_similarity_filter(self, store, embedder):
        store.add([_make_learning(embedder, text="unique topic xyz", id="ms-1")])
        query_vec = embedder.embed(["completely unrelated abcdef"])[0]
        # With a very high threshold, dissimilar results should be filtered
        results = store.vector_search(query_vec, limit=5, min_similarity=0.99)
        assert len(results) == 0

    def test_project_scope(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="proj A learning", id="pa", project="/proj/a"),
                _make_learning(embedder, text="proj B learning", id="pb", project="/proj/b"),
            ]
        )
        query_vec = embedder.embed(["proj A learning"])[0]
        results = store.vector_search(
            query_vec, limit=5, min_similarity=0.0, scope="project", current_project="/proj/a"
        )
        projects = {r["project"] for r in results}
        assert "/proj/b" not in projects

    def test_shared_scope(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="proj A thing", id="sa", project="/proj/a"),
                _make_learning(embedder, text="proj B thing", id="sb", project="/proj/b"),
            ]
        )
        query_vec = embedder.embed(["proj"])[0]
        results = store.vector_search(query_vec, limit=5, min_similarity=0.0, scope="shared")
        assert len(results) == 2

    def test_mixed_scope(self, store, embedder):
        store.add(
            [
                _make_learning(
                    embedder, text="proj A specific", id="mx-a", project="/proj/a", shared=False
                ),
                _make_learning(
                    embedder, text="shared learning", id="mx-s", project="/proj/b", shared=True
                ),
                _make_learning(
                    embedder, text="proj B only", id="mx-b", project="/proj/b", shared=False
                ),
            ]
        )
        query_vec = embedder.embed(["learning"])[0]
        results = store.vector_search(
            query_vec, limit=5, min_similarity=0.0, scope="mixed", current_project="/proj/a"
        )
        ids = {r["id"] for r in results}
        # Should include proj A's and shared, but not proj B only
        assert "mx-a" in ids
        assert "mx-s" in ids
        assert "mx-b" not in ids

    def test_excludes_stale(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="fresh learning", id="es-f", stale=False),
                _make_learning(embedder, text="stale learning", id="es-s", stale=True),
            ]
        )
        query_vec = embedder.embed(["learning"])[0]
        results = store.vector_search(query_vec, limit=5, min_similarity=0.0)
        ids = {r["id"] for r in results}
        assert "es-f" in ids
        assert "es-s" not in ids

    def test_includes_stale_when_requested(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="stale but wanted", id="is-s", stale=True),
            ]
        )
        query_vec = embedder.embed(["stale but wanted"])[0]
        results = store.vector_search(query_vec, limit=5, min_similarity=0.0, exclude_stale=False)
        assert len(results) == 1
        assert results[0]["id"] == "is-s"

    def test_empty_table(self, store):
        query_vec = [0.1] * 8
        results = store.vector_search(query_vec, limit=5)
        assert results == []
