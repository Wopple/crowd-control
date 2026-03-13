"""Tests for the search module."""

import pytest
from conftest import FakeEmbedder, insert_learning

from crowd_control.config import RetrievalConfig
from crowd_control.retrieve.search import SearchResults, search_learnings, validate_scope
from crowd_control.storage.db import LearningStore


@pytest.fixture
def embedder():
    return FakeEmbedder(dimensions=8)


@pytest.fixture
def store(tmp_path):
    return LearningStore(str(tmp_path / "search_db"), vector_dimensions=8)


class TestSearchLearnings:
    def test_returns_results(self, store, embedder):
        insert_learning(store, embedder, "asyncio concurrency patterns", id="s1")
        insert_learning(store, embedder, "react component lifecycle", id="s2")

        config = RetrievalConfig(min_similarity=0.0)
        results = search_learnings(
            query="asyncio concurrency patterns",
            store=store,
            embedder=embedder,
            config=config,
            scope="shared",
        )

        assert isinstance(results, SearchResults)
        assert len(results.results) > 0
        assert results.query_text == "asyncio concurrency patterns"
        # The exact-match learning should have the highest similarity
        assert results.results[0].id == "s1"
        assert results.results[0].similarity > 0.9

    def test_empty_db(self, store, embedder):
        config = RetrievalConfig()
        results = search_learnings(
            query="anything",
            store=store,
            embedder=embedder,
            config=config,
            scope="shared",
        )
        assert results.results == []

    def test_respects_min_similarity(self, store, embedder):
        insert_learning(store, embedder, "very specific topic xyz", id="ms1")

        config = RetrievalConfig(min_similarity=0.99)
        results = search_learnings(
            query="completely unrelated abcdef",
            store=store,
            embedder=embedder,
            config=config,
            scope="shared",
        )
        assert len(results.results) == 0

    def test_project_filtering(self, store, embedder):
        insert_learning(store, embedder, "project A learning", id="pf-a", project="/proj/a")
        insert_learning(store, embedder, "project B learning", id="pf-b", project="/proj/b")

        config = RetrievalConfig(min_similarity=0.0)
        results = search_learnings(
            query="project learning",
            store=store,
            embedder=embedder,
            config=config,
            current_project="/proj/a",
            scope="project",
        )
        projects = {r.project for r in results.results}
        assert "/proj/b" not in projects

    def test_category_filtering(self, store, embedder):
        insert_learning(store, embedder, "gotcha about auth", id="cf-g", category="gotcha")
        insert_learning(
            store,
            embedder,
            "debug insight about auth",
            id="cf-d",
            category="debugging_insight",
        )

        config = RetrievalConfig(min_similarity=0.0)
        results = search_learnings(
            query="auth",
            store=store,
            embedder=embedder,
            config=config,
            scope="shared",
            category="gotcha",
        )
        categories = {r.category for r in results.results}
        assert categories == {"gotcha"}

    def test_search_result_fields(self, store, embedder):
        """Verify all fields are populated on SearchResult."""
        insert_learning(
            store,
            embedder,
            "test field population",
            id="fields-1",
            category="gotcha",
            tags=["test"],
            project="/proj/x",
            session_id="sess-99",
            confidence=0.9,
            active_count=5,
        )

        config = RetrievalConfig(min_similarity=0.0)
        results = search_learnings(
            query="test field population",
            store=store,
            embedder=embedder,
            config=config,
            scope="shared",
        )
        r = results.results[0]
        assert r.id == "fields-1"
        assert r.text == "test field population"
        assert r.category == "gotcha"
        assert r.project == "/proj/x"
        assert r.session_id == "sess-99"
        assert r.confidence == pytest.approx(0.9, abs=0.01)
        assert r.active_count == 5
        assert r.similarity > 0.0


class TestScopeValidation:
    def test_valid_scopes(self):
        for scope in ("project", "shared", "mixed"):
            assert validate_scope(scope) == scope

    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="Invalid scope"):
            validate_scope("typo")

    def test_invalid_scope_in_search(self, store, embedder):
        config = RetrievalConfig()
        with pytest.raises(ValueError, match="Invalid scope"):
            search_learnings(
                query="test",
                store=store,
                embedder=embedder,
                config=config,
                scope="invalid",
            )
