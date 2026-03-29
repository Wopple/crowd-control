"""Tests for the MCP server tools and helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import FakeEmbedder, insert_learning

from crowd_control.config import CrowdControlConfig, IngestionConfig
from crowd_control.formatting import format_results_text, format_status_counts
from crowd_control.retrieve import RetrievalResult
from crowd_control.retrieve.rank import RankedResult
from crowd_control.retrieve.search import SearchResult, SearchResults
from crowd_control.server import (
    ServerDeps,
    create_server,
    handle_add_learning,
    handle_delete_learning,
    handle_ingest_session,
    handle_search_learnings,
    handle_status,
)
from crowd_control.storage.db import LearningStore
from crowd_control.storage.models import Learning, LearningCategory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deps(tmp_path, dimensions=8, project_id="/test/project"):
    """Build ServerDeps with a FakeEmbedder and real LanceDB for testing."""
    embedder = FakeEmbedder(dimensions=dimensions)
    config = CrowdControlConfig(storage_dir=str(tmp_path))
    store = LearningStore(config.db_path, vector_dimensions=dimensions)
    return ServerDeps(config=config, store=store, embedder=embedder, project_id=project_id)


# ---------------------------------------------------------------------------
# T1: format_results_text (pure function, in formatting.py)
# ---------------------------------------------------------------------------


class TestFormatResultsText:
    def test_empty_results(self):
        result = RetrievalResult(
            ranked=[],
            search_results=SearchResults(results=[], query_text="test"),
            total_learnings=42,
        )
        output = format_results_text(result)
        assert "No matching learnings found" in output
        assert "42" in output

    def test_single_result_grammar(self):
        ts = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
        sr = SearchResult(
            id="abc",
            text="Use pytest fixtures for reusable setup",
            category="pattern_discovery",
            tags=["testing"],
            project="/my/project",
            similarity=0.85,
            session_id="sess-1",
            timestamp=ts,
            confidence=0.9,
            active_count=3,
        )
        rr = RankedResult(
            id="abc",
            text="Use pytest fixtures for reusable setup",
            category="pattern_discovery",
            tags=["testing"],
            project="/my/project",
            similarity=0.85,
            hotness=0.4,
            final_score=0.78,
        )
        result = RetrievalResult(
            ranked=[rr],
            search_results=SearchResults(results=[sr], query_text="testing tips"),
            total_learnings=100,
        )
        output = format_results_text(result)
        assert "[1]" in output
        assert "score=0.78" in output
        assert "pattern_discovery" in output
        assert "pytest fixtures" in output
        assert "id=abc" in output
        assert "retrieved=3x" in output
        assert "1 result " in output  # singular, not "1 results"
        assert "100 learnings" in output

    def test_id_prefix_truncation(self):
        """Full-length IDs are truncated to 8 chars in output."""
        full_id = "a3f8c012beef456789abcdef01234567"
        ts = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
        sr = SearchResult(
            id=full_id,
            text="Some learning with a long ID",
            category="gotcha",
            tags=[],
            project="/proj",
            similarity=0.7,
            session_id="sess-1",
            timestamp=ts,
            confidence=0.9,
            active_count=0,
        )
        rr = RankedResult(
            id=full_id,
            text="Some learning with a long ID",
            category="gotcha",
            tags=[],
            project="/proj",
            similarity=0.7,
            final_score=0.65,
        )
        result = RetrievalResult(
            ranked=[rr],
            search_results=SearchResults(results=[sr], query_text="q"),
            total_learnings=5,
        )
        output = format_results_text(result)
        assert "id=a3f8c012" in output
        assert "beef4567" not in output

    def test_none_timestamp_no_crash(self):
        sr = SearchResult(
            id="xyz",
            text="Some learning",
            category="gotcha",
            tags=[],
            project="/proj",
            similarity=0.7,
            timestamp=None,
            active_count=0,
        )
        rr = RankedResult(
            id="xyz",
            text="Some learning",
            category="gotcha",
            tags=[],
            project="/proj",
            similarity=0.7,
            final_score=0.65,
        )
        result = RetrievalResult(
            ranked=[rr],
            search_results=SearchResults(results=[sr], query_text="q"),
            total_learnings=5,
        )
        output = format_results_text(result)
        assert "age=0d" in output


# ---------------------------------------------------------------------------
# T1b: format_status_counts (pure function, in formatting.py)
# ---------------------------------------------------------------------------


class TestFormatStatusCounts:
    def test_multi_project(self):
        result = format_status_counts(
            42, 312, ["async", "python"], ["async", "javascript", "python"]
        )
        assert result.learnings_line == "Learnings: 42 (312 total)"
        assert result.tags_line == "Tags: async, python"
        assert result.all_tags_line == "Tags (all): async, javascript, python"

    def test_single_project(self):
        result = format_status_counts(42, 42, ["python"], ["python"])
        assert result.learnings_line == "Learnings: 42"
        assert result.tags_line == "Tags: python"
        assert result.all_tags_line is None

    def test_empty_project(self):
        result = format_status_counts(0, 312, [], ["python", "javascript"])
        assert result.learnings_line == "Learnings: 0 (312 total)"
        assert result.tags_line == "Tags: (none)"
        assert result.all_tags_line == "Tags (all): python, javascript"

    def test_no_tags_anywhere(self):
        result = format_status_counts(5, 5, [], [])
        assert result.learnings_line == "Learnings: 5"
        assert result.tags_line == "Tags: (none)"
        assert result.all_tags_line is None


# ---------------------------------------------------------------------------
# T2: add_learning validation logic
# ---------------------------------------------------------------------------


class TestAddLearningValidation:
    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="not a valid"):
            LearningCategory("invalid_category")

    def test_text_too_long_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="exceeds 2000"):
            Learning(
                text="x" * 2001,
                category=LearningCategory.DEBUGGING_INSIGHT,
                tags=[],
                project="/test",
                session_id="s1",
                confidence=0.9,
            )

    def test_valid_learning_creates(self):
        learning = Learning(
            text="Short insight",
            category=LearningCategory.GOTCHA,
            tags=["python"],
            project="/test",
            session_id="manual",
            confidence=1.0,
        )
        assert learning.text == "Short insight"
        assert learning.category == LearningCategory.GOTCHA
        assert learning.id  # auto-generated


# ---------------------------------------------------------------------------
# T3: add_learning storage (FakeEmbedder + real LanceDB)
# ---------------------------------------------------------------------------


class TestAddLearningStorage:
    def test_embed_and_store(self, tmp_path):
        embedder = FakeEmbedder(dimensions=8)
        store = LearningStore(str(tmp_path / "db"), vector_dimensions=8)

        learning = Learning(
            text="Config files live in ~/.crowd-control/",
            category=LearningCategory.CODEBASE_CONVENTION,
            tags=["config"],
            project="/test",
            session_id="manual",
            confidence=1.0,
        )
        vectors = embedder.embed([learning.text])
        record = learning.model_dump(mode="python")
        record["vector"] = vectors[0]

        result = store.add([record])
        assert result.stored == 1
        assert store.count() == 1

    def test_duplicate_rejected(self, tmp_path):
        embedder = FakeEmbedder(dimensions=8)
        store = LearningStore(str(tmp_path / "db"), vector_dimensions=8)

        text = "Unique insight about the system"
        vectors = embedder.embed([text])
        learning = Learning(
            text=text,
            category=LearningCategory.DEBUGGING_INSIGHT,
            tags=[],
            project="/test",
            session_id="manual",
            confidence=1.0,
        )
        record = learning.model_dump(mode="python")
        record["vector"] = vectors[0]

        assert store.add([record]).stored == 1

        # Same text again — should be rejected
        learning2 = Learning(
            text=text,
            category=LearningCategory.DEBUGGING_INSIGHT,
            tags=[],
            project="/test",
            session_id="manual",
            confidence=1.0,
        )
        record2 = learning2.model_dump(mode="python")
        record2["vector"] = vectors[0]
        assert store.add([record2]).stored == 0


# ---------------------------------------------------------------------------
# T4: status output format
# ---------------------------------------------------------------------------


class TestStatusFormat:
    def test_status_output(self, tmp_path):
        config = CrowdControlConfig(storage_dir=str(tmp_path))
        store = LearningStore(config.db_path, vector_dimensions=8)

        # Replicate the format string from the status tool
        count = store.count()
        output = (
            f"Database: {config.db_path}\n"
            f"Learnings: {count}\n"
            f"Embedding: {config.embedding.provider}/{config.embedding.model}\n"
            f"Scope: {config.knowledge.scope}\n"
            f"Retrieval: max_results={config.retrieval.max_results}, "
            f"max_tokens={config.retrieval.max_tokens}"
        )
        assert "Learnings: 0" in output
        assert "ollama/nomic-embed-text" in output
        assert "max_results=15" in output


# ---------------------------------------------------------------------------
# T5: ingest_session path resolution
# ---------------------------------------------------------------------------


class TestIngestSessionPathResolution:
    def test_nonexistent_path(self, tmp_path):
        fake_path = tmp_path / "nonexistent.jsonl"
        assert not fake_path.exists()
        from pathlib import Path

        resolved = Path(str(fake_path)).expanduser().resolve()
        assert not resolved.exists()


# ---------------------------------------------------------------------------
# T6: Tool behavior via handle_* functions
# ---------------------------------------------------------------------------


@pytest.fixture
def server_deps(tmp_path):
    """Build ServerDeps without needing a real embedding provider."""
    return _make_deps(tmp_path)


@pytest.mark.anyio
async def test_status_tool(server_deps):
    """Call status handler and verify output format."""
    text = await handle_status(server_deps)
    assert "Learnings: 0" in text
    assert "ollama/nomic-embed-text" in text


@pytest.mark.anyio
async def test_status_includes_tags(server_deps):
    """Status output includes Tags line with available tags."""
    await handle_add_learning(
        server_deps,
        text="Python asyncio event loop behavior",
        category="debugging_insight",
        tags=["python", "async"],
    )
    await handle_add_learning(
        server_deps,
        text="React component lifecycle hooks",
        category="pattern_discovery",
        tags=["javascript", "react"],
    )
    text = await handle_status(server_deps)
    assert "Tags:" in text
    assert "python" in text
    assert "javascript" in text


@pytest.mark.anyio
async def test_status_empty_tags(server_deps):
    """Status shows (none) when no tags exist."""
    text = await handle_status(server_deps)
    assert "Tags: (none)" in text


@pytest.mark.anyio
async def test_status_shows_project_scoped_count(tmp_path):
    """Status shows project-scoped count alongside total."""
    deps = _make_deps(tmp_path, project_id="/proj/a")
    insert_learning(
        deps.store,
        deps.embedder,
        "insight for proj a",
        id="sc-a1",
        tags=["python"],
        project="/proj/a",
    )
    insert_learning(
        deps.store,
        deps.embedder,
        "another for proj a",
        id="sc-a2",
        tags=["async"],
        project="/proj/a",
    )
    insert_learning(
        deps.store,
        deps.embedder,
        "insight for proj b",
        id="sc-b1",
        tags=["javascript"],
        project="/proj/b",
    )

    text = await handle_status(deps)

    assert "Learnings: 2 (3 total)" in text
    assert "Project: /proj/a" in text


@pytest.mark.anyio
async def test_status_shows_project_scoped_tags(tmp_path):
    """Status shows project tags and all tags separately."""
    deps = _make_deps(tmp_path, project_id="/proj/a")
    insert_learning(
        deps.store, deps.embedder, "python insight", id="st-a1", tags=["python"], project="/proj/a"
    )
    insert_learning(
        deps.store, deps.embedder, "js insight", id="st-b1", tags=["javascript"], project="/proj/b"
    )

    text = await handle_status(deps)

    assert "Tags: python" in text
    assert "Tags (all): javascript, python" in text


@pytest.mark.anyio
async def test_status_single_project_no_total_suffix(tmp_path):
    """When all learnings are in one project, no parenthetical total."""
    deps = _make_deps(tmp_path, project_id="/proj/a")
    insert_learning(
        deps.store, deps.embedder, "only insight", id="sn-a1", tags=["python"], project="/proj/a"
    )

    text = await handle_status(deps)

    assert "Learnings: 1" in text
    assert "total" not in text


@pytest.mark.anyio
async def test_status_uses_project_id(tmp_path):
    """Status uses deps.project_id to determine the current project."""
    deps = _make_deps(tmp_path, project_id="/proj/b")
    insert_learning(deps.store, deps.embedder, "insight b", id="su-b1", project="/proj/b")

    text = await handle_status(deps)

    assert "Project: /proj/b" in text
    assert "Learnings: 1" in text


@pytest.mark.anyio
async def test_status_empty_project(tmp_path):
    """Project with no learnings shows 0 with total."""
    deps = _make_deps(tmp_path, project_id="/proj/empty")
    insert_learning(
        deps.store, deps.embedder, "insight a", id="se-a1", tags=["python"], project="/proj/a"
    )

    text = await handle_status(deps)

    assert "Learnings: 0 (1 total)" in text
    assert "Tags: (none)" in text


@pytest.mark.anyio
async def test_add_learning_uses_project_id(server_deps):
    """add_learning uses deps.project_id for the project."""
    await handle_add_learning(server_deps, text="project id insight")
    learnings = server_deps.store.list_learnings(project=server_deps.project_id)
    assert len(learnings) == 1


@pytest.mark.anyio
async def test_add_and_search_learning(server_deps):
    """Add a learning then search for it."""
    add_result = await handle_add_learning(
        server_deps,
        text="LanceDB stores vectors as Arrow arrays for zero-copy access",
        category="architecture_decision",
        tags=["lancedb", "storage"],
    )
    assert "stored successfully" in add_result

    search_result = await handle_search_learnings(
        server_deps,
        query="how does LanceDB store vectors",
    )
    assert "Arrow arrays" in search_result
    assert "architecture_decision" in search_result


@pytest.mark.anyio
async def test_search_with_tag_filter(server_deps):
    """Tags filter narrows search results."""
    await handle_add_learning(
        server_deps,
        text="Python asyncio event loop internals",
        category="debugging_insight",
        tags=["python", "async"],
    )
    await handle_add_learning(
        server_deps,
        text="React component lifecycle hooks",
        category="pattern_discovery",
        tags=["javascript", "react"],
    )

    result = await handle_search_learnings(
        server_deps, query="programming patterns", tags=["python"]
    )
    assert "asyncio" in result
    assert "React" not in result


@pytest.mark.anyio
async def test_add_learning_normalizes_tags(server_deps):
    """Tags are lowercased on storage."""
    result = await handle_add_learning(
        server_deps,
        text="LanceDB uses Arrow format internally",
        category="architecture_decision",
        tags=["LanceDB", "Arrow"],
    )
    assert "stored successfully" in result

    # Search with lowercase tag should match
    search_result = await handle_search_learnings(
        server_deps, query="storage format", tags=["lancedb"]
    )
    assert "Arrow" in search_result


@pytest.mark.anyio
async def test_add_learning_invalid_category(server_deps):
    """Invalid category returns error, not exception."""
    result = await handle_add_learning(
        server_deps,
        text="some insight",
        category="bogus",
    )
    assert "Invalid category" in result


@pytest.mark.anyio
async def test_add_learning_duplicate(server_deps):
    """Duplicate learning is detected."""
    text = "Always check for None before accessing .timestamp"
    r1 = await handle_add_learning(server_deps, text=text, category="gotcha")
    assert "stored successfully" in r1

    r2 = await handle_add_learning(server_deps, text=text, category="gotcha")
    assert "duplicate" in r2.lower()


@pytest.mark.anyio
async def test_ingest_session_file_not_found(server_deps):
    """Ingest with nonexistent path returns error."""
    result = await handle_ingest_session(
        server_deps,
        session_path="/nonexistent/session.jsonl",
    )
    assert "File not found" in result


@pytest.mark.anyio
async def test_search_empty_db(server_deps):
    """Search on empty DB returns no results message."""
    result = await handle_search_learnings(
        server_deps,
        query="anything",
    )
    assert "No matching learnings found" in result


@pytest.mark.anyio
async def test_ingest_session_blocked_by_agent_ingest(tmp_path):
    """Ingest tool returns early message when agent_ingest is disabled."""
    from crowd_control.config import IngestionConfig

    config = CrowdControlConfig(
        storage_dir=str(tmp_path),
        ingestion=IngestionConfig(agent_ingest=False),
    )
    embedder = FakeEmbedder(dimensions=8)
    store = LearningStore(config.db_path, vector_dimensions=8)
    deps = ServerDeps(config=config, store=store, embedder=embedder)

    result = await handle_ingest_session(deps, session_path="/some/file.jsonl")
    assert "disabled" in result
    assert "add_learning" in result


@pytest.mark.anyio
async def test_status_includes_ingestion_flags(tmp_path):
    """Status output shows auto_ingest and agent_ingest settings."""
    from crowd_control.config import IngestionConfig

    config = CrowdControlConfig(
        storage_dir=str(tmp_path),
        ingestion=IngestionConfig(auto_ingest=False, agent_ingest=False),
    )
    embedder = FakeEmbedder(dimensions=8)
    store = LearningStore(config.db_path, vector_dimensions=8)
    deps = ServerDeps(config=config, store=store, embedder=embedder)

    text = await handle_status(deps)
    assert "Auto-ingest: disabled" in text
    assert "Agent ingest: disabled" in text


@pytest.mark.anyio
async def test_registered_tools():
    """Verify all expected tools are registered on a created server."""
    server = create_server()
    tools = await server.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "search_learnings",
        "add_learning",
        "ingest_session",
        "status",
        "delete_learning",
    }


# ---------------------------------------------------------------------------
# T7: delete_learning tool
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_learning_success(tmp_path):
    """Delete a learning by ID prefix."""
    deps = _make_deps(tmp_path, project_id="/my/project")
    insert_learning(
        deps.store, deps.embedder, "outdated insight", id="dead1234beef5678", project="/my/project"
    )

    result = await handle_delete_learning(deps, ["dead1234"])

    assert "Deleted 1" in result
    assert deps.store.count() == 0


@pytest.mark.anyio
async def test_delete_learning_blocked_by_config(tmp_path):
    """Deletion is blocked when agent_delete is disabled."""
    config = CrowdControlConfig(
        storage_dir=str(tmp_path),
        ingestion=IngestionConfig(agent_delete=False),
    )
    embedder = FakeEmbedder(dimensions=8)
    store = LearningStore(config.db_path, vector_dimensions=8)
    deps = ServerDeps(config=config, store=store, embedder=embedder)

    result = await handle_delete_learning(deps, ["abcd1234"])
    assert "disabled" in result
    assert "agent_delete" in result


@pytest.mark.anyio
async def test_delete_learning_prefix_too_short(server_deps):
    """Short prefixes are rejected."""
    result = await handle_delete_learning(server_deps, ["abc"])
    assert "too short" in result


@pytest.mark.anyio
async def test_delete_learning_non_hex_prefix(server_deps):
    """Non-hex prefixes are rejected."""
    result = await handle_delete_learning(server_deps, ["zzzzzzzz"])
    assert "invalid characters" in result


@pytest.mark.anyio
async def test_delete_learning_not_found(server_deps):
    """Nonexistent prefix returns not found."""
    result = await handle_delete_learning(server_deps, ["00000000"])
    assert "not found" in result


@pytest.mark.anyio
async def test_delete_learning_cross_project_rejected(tmp_path):
    """Learnings from other projects cannot be deleted."""
    deps = _make_deps(tmp_path, project_id="/my/project")
    insert_learning(
        deps.store,
        deps.embedder,
        "other project insight",
        id="c0051234c0055678",
        project="/other/project",
    )

    result = await handle_delete_learning(deps, ["c0051234"])

    assert "different project" in result
    assert deps.store.count() == 1


@pytest.mark.anyio
async def test_delete_learning_ambiguous_prefix(tmp_path):
    """Ambiguous prefix matching multiple learnings is rejected."""
    deps = _make_deps(tmp_path, project_id="/test/project")
    insert_learning(
        deps.store,
        deps.embedder,
        "python asyncio concurrency patterns",
        id="abcdabcd00000001",
        project="/test/project",
    )
    insert_learning(
        deps.store,
        deps.embedder,
        "javascript react component lifecycle",
        id="abcdabcd00000002",
        project="/test/project",
    )

    result = await handle_delete_learning(deps, ["abcdabcd"])

    assert "ambiguous" in result
    assert deps.store.count() == 2


@pytest.mark.anyio
async def test_delete_learning_mixed_batch(tmp_path):
    """Batch with valid, not-found, and cross-project IDs."""
    deps = _make_deps(tmp_path, project_id="/my/proj")
    insert_learning(
        deps.store, deps.embedder, "valid one", id="aaaa1111aaaa1111", project="/my/proj"
    )
    insert_learning(
        deps.store, deps.embedder, "other proj", id="bbbb2222bbbb2222", project="/other/proj"
    )

    result = await handle_delete_learning(deps, ["aaaa1111", "bbbb2222", "cccc3333"])

    assert "Deleted 1" in result
    assert "different project" in result
    assert "not found" in result


@pytest.mark.anyio
async def test_delete_learning_empty_list(server_deps):
    """Empty ID list returns appropriate message."""
    result = await handle_delete_learning(server_deps, [])
    assert "No IDs provided" in result


@pytest.mark.anyio
async def test_delete_learning_no_store(tmp_path):
    """Deletion with no store returns error."""
    config = CrowdControlConfig(storage_dir=str(tmp_path))
    deps = ServerDeps(config=config, store=None, embedder=None)
    result = await handle_delete_learning(deps, ["abcd1234"])
    assert "not available" in result
