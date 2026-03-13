"""End-to-end retrieval pipeline integration tests."""

from datetime import UTC, datetime, timedelta

import pytest
from conftest import FakeEmbedder, insert_learning

from crowd_control.config import RetrievalConfig
from crowd_control.retrieve.rank import rank_results
from crowd_control.retrieve.search import search_learnings
from crowd_control.storage.db import LearningStore


@pytest.fixture
def embedder():
    return FakeEmbedder(dimensions=8)


@pytest.fixture
def store(tmp_path):
    return LearningStore(str(tmp_path / "integ_db"), vector_dimensions=8)


class TestFullPipeline:
    def test_search_and_rank(self, store, embedder):
        """Full pipeline: insert → search → rank → verify ordering."""
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        insert_learning(
            store,
            embedder,
            "python asyncio concurrency patterns",
            id="async",
            timestamp=now,
            active_count=5,
        )
        insert_learning(
            store,
            embedder,
            "javascript react component lifecycle",
            id="react",
            timestamp=now - timedelta(days=14),
            active_count=0,
        )
        insert_learning(
            store,
            embedder,
            "database postgresql index optimization",
            id="db",
            timestamp=now - timedelta(days=3),
            active_count=2,
        )

        config = RetrievalConfig(min_similarity=0.0, max_results=10)

        results = search_learnings(
            query="python asyncio concurrency patterns",
            store=store,
            embedder=embedder,
            config=config,
            scope="shared",
        )
        assert len(results.results) == 3

        ranked = rank_results(results.results, config, now=now)
        assert len(ranked) > 0
        # Results should be ordered by final_score descending
        scores = [r.final_score for r in ranked]
        assert scores == sorted(scores, reverse=True)
        # The exact-match learning should rank first
        assert ranked[0].id == "async"

    def test_project_filtering_end_to_end(self, store, embedder):
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        insert_learning(
            store,
            embedder,
            "auth system JWT tokens",
            id="proj-a",
            project="/proj/a",
            timestamp=now,
        )
        insert_learning(
            store,
            embedder,
            "auth middleware CORS",
            id="proj-b",
            project="/proj/b",
            timestamp=now,
        )

        config = RetrievalConfig(min_similarity=0.0)
        results = search_learnings(
            query="auth",
            store=store,
            embedder=embedder,
            config=config,
            current_project="/proj/a",
            scope="project",
        )
        ranked = rank_results(
            results.results,
            config,
            current_project="/proj/a",
            scope="project",
            now=now,
        )
        projects = {r.project for r in ranked}
        assert "/proj/b" not in projects

    def test_token_packing_limits_output(self, store, embedder):
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        # Insert learnings with long text
        for i in range(10):
            insert_learning(
                store,
                embedder,
                f"learning number {i} " + "x" * 200,
                id=f"pack-{i}",
                timestamp=now,
            )

        # Very small token budget
        config = RetrievalConfig(min_similarity=0.0, max_tokens=100, max_results=10)
        results = search_learnings(
            query="learning",
            store=store,
            embedder=embedder,
            config=config,
            scope="shared",
        )
        ranked = rank_results(results.results, config, now=now)
        # Should be limited by token budget
        assert len(ranked) < 10

    def test_active_count_increment_feedback(self, store, embedder):
        """Active count increment changes hotness on re-search."""
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        insert_learning(store, embedder, "test active count", id="active-test", timestamp=now)

        config = RetrievalConfig(min_similarity=0.0, hotness_weight=0.5)

        # First search
        results1 = search_learnings(
            query="test active count",
            store=store,
            embedder=embedder,
            config=config,
            scope="shared",
        )
        ranked1 = rank_results(results1.results, config, now=now)
        score_before = ranked1[0].final_score

        # Increment active count (simulating what the CLI does)
        store.increment_active_count([ranked1[0].id])

        # Second search — hotness should be higher
        results2 = search_learnings(
            query="test active count",
            store=store,
            embedder=embedder,
            config=config,
            scope="shared",
        )
        ranked2 = rank_results(results2.results, config, now=now)
        score_after = ranked2[0].final_score

        assert score_after > score_before
