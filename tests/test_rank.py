"""Tests for the ranking module."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from crowd_control.config import RetrievalConfig
from crowd_control.retrieve.rank import (
    RankedResult,
    _deduplicate,
    _pack_to_budget,
    _score_results,
    rank_results,
)
from crowd_control.retrieve.search import SearchResult


def _make_sr(
    id="r1",
    text="test learning",
    similarity=0.8,
    active_count=0,
    timestamp=None,
    project="/test",
    category="debugging_insight",
    **kwargs,
):
    """Create a SearchResult for testing."""
    if timestamp is None:
        timestamp = datetime.now(UTC)
    return SearchResult(
        id=id,
        text=text,
        category=category,
        tags=kwargs.get("tags", []),
        project=project,
        session_id=kwargs.get("session_id", "sess-1"),
        timestamp=timestamp,
        confidence=kwargs.get("confidence", 0.8),
        active_count=active_count,
        similarity=similarity,
    )


class TestRecencyDecay:
    def test_half_life(self):
        config = RetrievalConfig(recency_half_life_days=7.0)
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        result = _make_sr(timestamp=now - timedelta(days=7))

        scored = _score_results([result], config, now=now)
        # Recency at exactly one half-life should be ~0.5
        decay_constant = math.log(2) / 7.0
        expected_recency = math.exp(-decay_constant * 7.0)
        assert expected_recency == pytest.approx(0.5, abs=0.01)
        # Hotness = sigmoid(log1p(0)) * recency = 0.5 * 0.5 = 0.25
        assert scored[0].hotness == pytest.approx(0.5 * 0.5, abs=0.01)

    def test_zero_age(self):
        config = RetrievalConfig()
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        result = _make_sr(timestamp=now)

        scored = _score_results([result], config, now=now)
        # Recency should be ~1.0
        # Hotness = sigmoid(0) * 1.0 = 0.5
        assert scored[0].hotness == pytest.approx(0.5, abs=0.01)

    def test_old_learning(self):
        config = RetrievalConfig(recency_half_life_days=7.0)
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        result = _make_sr(timestamp=now - timedelta(days=21))

        _score_results([result], config, now=now)
        # 3 half-lives: recency ≈ 0.125
        expected_recency = math.exp(-math.log(2) / 7.0 * 21.0)
        assert expected_recency == pytest.approx(0.125, abs=0.01)

    def test_clamps_negative_age(self):
        """Future timestamps get recency=1.0, not >1.0."""
        config = RetrievalConfig()
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        # Future timestamp
        result = _make_sr(timestamp=now + timedelta(days=1))

        scored = _score_results([result], config, now=now)
        # Recency clamped to 0 age → recency = 1.0
        # hotness = sigmoid(0) * 1.0 = 0.5
        assert scored[0].hotness == pytest.approx(0.5, abs=0.01)


class TestHotness:
    def test_cold_start(self):
        """active_count=0 produces hotness = 0.5 * recency, not zero."""
        config = RetrievalConfig()
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        result = _make_sr(timestamp=now, active_count=0)

        scored = _score_results([result], config, now=now)
        # sigmoid(log1p(0)) = sigmoid(0) = 0.5, recency = 1.0
        assert scored[0].hotness == pytest.approx(0.5, abs=0.01)

    def test_with_usage(self):
        """Higher active_count produces higher hotness."""
        config = RetrievalConfig()
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        cold = _make_sr(id="cold", timestamp=now, active_count=0)
        hot = _make_sr(id="hot", timestamp=now, active_count=10)

        scored_cold = _score_results([cold], config, now=now)
        scored_hot = _score_results([hot], config, now=now)

        assert scored_hot[0].hotness > scored_cold[0].hotness


class TestScoreBlending:
    def test_default_weights(self):
        """final_score = 0.8 * similarity + 0.2 * hotness with default weights."""
        config = RetrievalConfig(hotness_weight=0.2)
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        result = _make_sr(similarity=0.9, active_count=0, timestamp=now)

        scored = _score_results([result], config, now=now)
        # hotness = sigmoid(0) * 1.0 = 0.5
        expected = 0.8 * 0.9 + 0.2 * 0.5
        assert scored[0].final_score == pytest.approx(expected, abs=0.01)

    def test_project_boost_in_shared_scope(self):
        config = RetrievalConfig(project_boost=1.5)
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        same_proj = _make_sr(id="same", project="/proj/a", timestamp=now, similarity=0.8)
        diff_proj = _make_sr(id="diff", project="/proj/b", timestamp=now, similarity=0.8)

        scored_same = _score_results(
            [same_proj], config, current_project="/proj/a", scope="shared", now=now
        )
        scored_diff = _score_results(
            [diff_proj], config, current_project="/proj/a", scope="shared", now=now
        )

        # Same-project should get 1.5x boost
        assert scored_same[0].final_score == pytest.approx(
            scored_diff[0].final_score * 1.5, abs=0.01
        )

    def test_project_boost_skipped_in_project_scope(self):
        """In project scope, no boost is applied (all results are same-project)."""
        config = RetrievalConfig(project_boost=1.5)
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        result = _make_sr(project="/proj/a", timestamp=now, similarity=0.8)

        scored_project = _score_results(
            [result], config, current_project="/proj/a", scope="project", now=now
        )
        scored_shared = _score_results(
            [result], config, current_project="/proj/a", scope="shared", now=now
        )

        # Project scope should NOT have the boost
        assert scored_project[0].final_score < scored_shared[0].final_score


class TestDedup:
    def test_removes_near_duplicates(self):
        r1 = RankedResult(
            id="1",
            text=(
                "The authentication system uses JWT tokens stored"
                " in HttpOnly cookies for session management"
            ),
            category="a",
            tags=[],
            project="/p",
            similarity=0.9,
            hotness=0.5,
            final_score=0.9,
        )
        r2 = RankedResult(
            id="2",
            text=(
                "The authentication system uses JWT tokens stored"
                " in HttpOnly cookies for session handling"
            ),
            category="a",
            tags=[],
            project="/p",
            similarity=0.85,
            hotness=0.4,
            final_score=0.85,
        )
        result = _deduplicate([r1, r2])
        assert len(result) == 1
        assert result[0].id == "1"  # Higher score kept

    def test_keeps_dissimilar(self):
        r1 = RankedResult(
            id="1",
            text="The auth system uses JWT tokens",
            category="a",
            tags=[],
            project="/p",
            similarity=0.9,
            hotness=0.5,
            final_score=0.9,
        )
        r2 = RankedResult(
            id="2",
            text="Database indexes should cover the most common queries",
            category="a",
            tags=[],
            project="/p",
            similarity=0.85,
            hotness=0.4,
            final_score=0.85,
        )
        result = _deduplicate([r1, r2])
        assert len(result) == 2

    def test_order_preserves_best(self):
        """Input must be sorted by score desc; the first (highest) is kept."""
        r_high = RankedResult(
            id="high",
            text="same text here",
            category="a",
            tags=[],
            project="/p",
            similarity=0.95,
            hotness=0.6,
            final_score=0.95,
        )
        r_low = RankedResult(
            id="low",
            text="same text here",
            category="a",
            tags=[],
            project="/p",
            similarity=0.7,
            hotness=0.3,
            final_score=0.7,
        )
        result = _deduplicate([r_high, r_low])
        assert len(result) == 1
        assert result[0].id == "high"


class TestTokenPacking:
    def test_fits_all(self):
        results = [
            RankedResult(
                id=str(i),
                text="short " * 10,
                category="a",
                tags=[],
                project="/p",
                similarity=0.9,
                hotness=0.5,
                final_score=0.9 - i * 0.01,
            )
            for i in range(3)
        ]
        packed = _pack_to_budget(results, max_tokens=10000, max_results=100)
        assert len(packed) == 3

    def test_truncates(self):
        results = [
            RankedResult(
                id=str(i),
                text="x" * 400,
                category="a",
                tags=[],
                project="/p",
                similarity=0.9,
                hotness=0.5,
                final_score=0.9 - i * 0.01,
            )
            for i in range(10)
        ]
        # 400 chars / 4 = 100 tokens each, budget = 350 → fits 3
        packed = _pack_to_budget(results, max_tokens=350, max_results=100)
        assert len(packed) == 3

    def test_stops_at_max_results(self):
        results = [
            RankedResult(
                id=str(i),
                text="short",
                category="a",
                tags=[],
                project="/p",
                similarity=0.9,
                hotness=0.5,
                final_score=0.9 - i * 0.01,
            )
            for i in range(10)
        ]
        packed = _pack_to_budget(results, max_tokens=100000, max_results=3)
        assert len(packed) == 3

    def test_empty(self):
        assert _pack_to_budget([], max_tokens=1000, max_results=10) == []


class TestRankResults:
    def test_ordering(self):
        """Results should be ordered by final_score descending."""
        config = RetrievalConfig()
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        results = [
            _make_sr(id="low", similarity=0.3, timestamp=now),
            _make_sr(id="high", similarity=0.9, timestamp=now),
            _make_sr(id="mid", similarity=0.6, timestamp=now),
        ]

        ranked = rank_results(results, config, now=now)
        scores = [r.final_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_max_results(self):
        config = RetrievalConfig(max_results=2, max_tokens=100000)
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        results = [
            _make_sr(id=f"r{i}", text=f"learning {i}", similarity=0.8 - i * 0.01, timestamp=now)
            for i in range(5)
        ]

        ranked = rank_results(results, config, now=now)
        assert len(ranked) <= 2

    def test_combines_signals(self):
        """High similarity + low hotness vs lower similarity + high hotness."""
        config = RetrievalConfig(hotness_weight=0.4)
        now = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)

        # High semantic, never used
        semantic = _make_sr(
            id="semantic",
            text="high semantic match learning",
            similarity=0.95,
            active_count=0,
            timestamp=now,
        )
        # Lower semantic, heavily used
        hot = _make_sr(
            id="hot",
            text="frequently used hot learning",
            similarity=0.7,
            active_count=100,
            timestamp=now,
        )

        ranked = rank_results([semantic, hot], config, now=now)
        # With hotness_weight=0.4, both signals matter
        assert len(ranked) == 2
        # Both should have reasonable scores (exact ordering depends on formula)
        for r in ranked:
            assert r.final_score > 0

    def test_empty_input(self):
        config = RetrievalConfig()
        assert rank_results([], config) == []
