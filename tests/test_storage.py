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
        texts = [
            "python asyncio event loops",
            "javascript react component lifecycle",
            "database postgresql query optimization",
            "rust ownership and borrowing rules",
            "docker container networking setup",
        ]
        records = [_make_learning(embedder, text=t, id=f"id-{i}") for i, t in enumerate(texts)]
        result = store.add(records)
        assert result.stored == 5
        assert store.count() == 5

    def test_add_empty_list(self, store):
        assert store.add([]).stored == 0

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None


class TestList:
    def test_list_all(self, store, embedder):
        texts = [
            "python asyncio concurrency patterns",
            "javascript frontend rendering pipeline",
            "database indexing and query plans",
        ]
        records = [
            _make_learning(
                embedder,
                text=texts[i],
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
        result = store.add([record2])
        assert result.stored == 0
        assert store.count() == 1

    def test_exact_text_with_quotes(self, store, embedder):
        text = "Use 'is not None' instead of '!= None'"
        record1 = _make_learning(embedder, text=text, id="q1")
        record2 = _make_learning(embedder, text=text, id="q2")
        store.add([record1])
        result = store.add([record2])
        assert result.stored == 0

    def test_near_duplicate(self, store, embedder):
        record1 = _make_learning(embedder, text="test text", id="near-1")
        # Same vector (same text → same hash) but different id
        record2 = _make_learning(embedder, text="test text", id="near-2")
        store.add([record1])
        result = store.add([record2])
        assert result.stored == 0

    def test_near_duplicate_reports_match(self, store, embedder):
        record1 = _make_learning(embedder, text="test text original", id="orig")
        record2 = _make_learning(embedder, text="test text original", id="dup")
        store.add([record1])
        result = store.add([record2])
        assert result.stored == 0
        # Exact text match won't produce DuplicateInfo (it's caught before vector check)
        # but near-vector match will. Use same vector, different text to test:

    def test_near_duplicate_includes_matched_text(self, store, embedder):
        """Near-duplicate rejection includes the matched learning's text."""
        record1 = _make_learning(embedder, text="unique insight alpha", id="match-1")
        store.add([record1])
        # Same embedding (same text → same hash) but slightly different text
        # We need same vector but different text to trigger _find_near_duplicate
        vec = embedder.embed(["unique insight alpha"])[0]
        record2 = _make_learning(embedder, text="different text", id="match-2")
        record2["vector"] = vec  # Force same vector
        result = store.add([record2])
        assert result.stored == 0
        assert len(result.duplicates) == 1
        assert result.duplicates[0].matched_text == "unique insight alpha"
        assert result.duplicates[0].similarity > 0.99

    def test_allows_dissimilar(self, store, embedder):
        record1 = _make_learning(
            embedder, text="completely different topic about databases", id="d1"
        )
        record2 = _make_learning(
            embedder, text="unrelated subject regarding frontend rendering", id="d2"
        )
        store.add([record1])
        result = store.add([record2])
        assert result.stored == 1
        assert store.count() == 2

    def test_within_batch_exact_text_dedup(self, store, embedder):
        records = [_make_learning(embedder, text="same text", id=f"empty-{i}") for i in range(2)]
        result = store.add(records)
        assert result.stored == 1

    def test_within_batch_vector_dedup_on_empty_table(self, store, embedder):
        """Near-duplicate vectors within a batch are caught even on empty table."""
        vec = embedder.embed(["shared topic"])[0]
        record1 = _make_learning(embedder, text="first version", id="batch-1")
        record1["vector"] = vec
        record2 = _make_learning(embedder, text="second version", id="batch-2")
        record2["vector"] = vec  # Identical vector, different text
        result = store.add([record1, record2])
        assert result.stored == 1
        assert len(result.duplicates) == 1
        assert result.duplicates[0].new_text == "second version"
        assert result.duplicates[0].matched_text == "first version"


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


class TestPrune:
    def test_deletes_old_inactive_learnings(self, store, embedder):
        """Old learning with 0 retrievals is pruned; old with enough is kept."""
        now = datetime(2025, 6, 1, tzinfo=UTC)
        # 151 days old: needs ceil(151/30) = 6 retrievals
        store.add(
            [
                _make_learning(
                    embedder, text="old inactive", id="old-0",
                    timestamp=datetime(2025, 1, 1, tzinfo=UTC), active_count=0,
                ),
                _make_learning(
                    embedder, text="old well retrieved", id="old-6",
                    timestamp=datetime(2025, 1, 1, tzinfo=UTC), active_count=6,
                ),
                _make_learning(
                    embedder, text="recent inactive", id="new-0",
                    timestamp=datetime(2025, 5, 20, tzinfo=UTC), active_count=0,
                ),
            ]
        )
        # interval=30: 151-day-old needs 6, so active_count=6 survives, 0 doesn't
        pruned = store.prune(max_age_days=90, retrieval_interval_days=30, now=now)
        assert pruned == 1
        assert store.count() == 2
        assert store.get("old-0") is None
        assert store.get("old-6") is not None
        assert store.get("new-0") is not None

    def test_required_count_scales_with_age(self, store, embedder):
        """Older learnings need proportionally more retrievals."""
        now = datetime(2025, 7, 1, tzinfo=UTC)
        store.add(
            [
                # 91 days old: needs ceil(91/30) = 4, has 3 → pruned
                _make_learning(
                    embedder, text="barely old learning", id="age-91",
                    timestamp=datetime(2025, 4, 1, tzinfo=UTC), active_count=3,
                ),
                # 181 days old: needs ceil(181/30) = 7, has 5 → pruned
                _make_learning(
                    embedder, text="very old learning", id="age-181",
                    timestamp=datetime(2025, 1, 1, tzinfo=UTC), active_count=5,
                ),
                # 181 days old: needs 7, has 7 → survives
                _make_learning(
                    embedder, text="very old active learning", id="age-181-ok",
                    timestamp=datetime(2025, 1, 1, tzinfo=UTC), active_count=7,
                ),
            ]
        )
        pruned = store.prune(max_age_days=90, retrieval_interval_days=30, now=now)
        assert pruned == 2
        assert store.get("age-91") is None
        assert store.get("age-181") is None
        assert store.get("age-181-ok") is not None

    def test_zero_max_age_disables_pruning(self, store, embedder):
        store.add(
            [
                _make_learning(
                    embedder, text="ancient learning", id="ancient",
                    timestamp=datetime(2020, 1, 1, tzinfo=UTC), active_count=0,
                ),
            ]
        )
        pruned = store.prune(max_age_days=0, retrieval_interval_days=30)
        assert pruned == 0
        assert store.count() == 1

    def test_empty_table(self, store):
        pruned = store.prune(max_age_days=90, retrieval_interval_days=30)
        assert pruned == 0

    def test_nothing_to_prune(self, store, embedder):
        now = datetime(2025, 6, 1, tzinfo=UTC)
        store.add(
            [
                _make_learning(
                    embedder, text="recent learning", id="recent",
                    timestamp=datetime(2025, 5, 1, tzinfo=UTC), active_count=0,
                ),
            ]
        )
        pruned = store.prune(max_age_days=90, retrieval_interval_days=30, now=now)
        assert pruned == 0
        assert store.count() == 1


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


class TestDistinctTags:
    def test_returns_sorted_unique_tags(self, store, embedder):
        """Tags from multiple learnings are deduplicated and sorted."""
        store.add([
            _make_learning(embedder, text="python async patterns", id="dt-1",
                           tags=["python", "async"]),
            _make_learning(embedder, text="react component tips", id="dt-2",
                           tags=["javascript", "react"]),
            _make_learning(embedder, text="python testing strategies", id="dt-3",
                           tags=["python", "testing"]),
        ])
        tags = store.distinct_tags()
        assert tags == ["async", "javascript", "python", "react", "testing"]

    def test_empty_table(self, store):
        """Empty table returns empty list."""
        assert store.distinct_tags() == []

    def test_learnings_with_no_tags(self, store, embedder):
        """Learnings with empty tag lists don't contribute to distinct tags."""
        store.add([
            _make_learning(embedder, text="untagged insight", id="dt-4", tags=[]),
        ])
        assert store.distinct_tags() == []


class TestHasSession:
    def test_has_session_true(self, store, embedder):
        store.add([_make_learning(embedder, id="hs-1", session_id="sess-existing")])
        assert store.has_session("sess-existing") is True

    def test_has_session_false(self, store, embedder):
        store.add([_make_learning(embedder, id="hs-1", session_id="sess-other")])
        assert store.has_session("sess-missing") is False

    def test_has_session_empty_table(self, store):
        assert store.has_session("any-session") is False


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

    def test_tag_filter_single(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="python topic", id="tf-py", tags=["python"]),
                _make_learning(embedder, text="javascript topic", id="tf-js", tags=["javascript"]),
            ]
        )
        query_vec = embedder.embed(["topic"])[0]
        results = store.vector_search(
            query_vec, limit=5, min_similarity=0.0, scope="shared", tags=["python"]
        )
        ids = {r["id"] for r in results}
        assert "tf-py" in ids
        assert "tf-js" not in ids

    def test_tag_filter_match_any(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="python topic", id="ma-py", tags=["python"]),
                _make_learning(embedder, text="rust topic", id="ma-rs", tags=["rust"]),
                _make_learning(embedder, text="go topic", id="ma-go", tags=["go"]),
            ]
        )
        query_vec = embedder.embed(["topic"])[0]
        results = store.vector_search(
            query_vec, limit=5, min_similarity=0.0, scope="shared", tags=["python", "rust"]
        )
        ids = {r["id"] for r in results}
        assert "ma-py" in ids
        assert "ma-rs" in ids
        assert "ma-go" not in ids

    def test_tag_filter_case_insensitive(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="collision detection", id="ci-1", tags=["collision"]),
            ]
        )
        query_vec = embedder.embed(["collision detection"])[0]
        results = store.vector_search(
            query_vec, limit=5, min_similarity=0.0, scope="shared", tags=["Collision"]
        )
        assert len(results) == 1
        assert results[0]["id"] == "ci-1"

    def test_tag_filter_none_returns_all(self, store, embedder):
        store.add(
            [
                _make_learning(embedder, text="tagged item", id="tn-1", tags=["python"]),
                _make_learning(embedder, text="untagged item", id="tn-2", tags=[]),
            ]
        )
        query_vec = embedder.embed(["item"])[0]
        results = store.vector_search(
            query_vec, limit=5, min_similarity=0.0, scope="shared", tags=None
        )
        assert len(results) == 2

    def test_empty_table(self, store):
        query_vec = [0.1] * 8
        results = store.vector_search(query_vec, limit=5)
        assert results == []
