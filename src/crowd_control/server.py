"""MCP server definition (FastMCP)."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

from crowd_control.config import CrowdControlConfig, load_config
from crowd_control.embed.base import Embedder, create_embedder
from crowd_control.formatting import format_results_text
from crowd_control.storage.db import LearningStore

logger = logging.getLogger(__name__)


@dataclass
class ServerDeps:
    """Shared resources available to all tool calls."""

    config: CrowdControlConfig
    store: LearningStore
    embedder: Embedder


@asynccontextmanager
async def _default_lifespan(server: FastMCP) -> AsyncIterator[ServerDeps]:
    """Initialize shared resources for the server lifetime."""
    config = load_config()
    embedder = await asyncio.to_thread(create_embedder, config.embedding)
    store = await asyncio.to_thread(LearningStore, config.db_path, embedder.dimensions)
    logger.info("Lifespan: store and embedder initialized")
    yield ServerDeps(config=config, store=store, embedder=embedder)


def create_server(lifespan=None) -> FastMCP:
    """Create and return a configured MCP server.

    Args:
        lifespan: Optional lifespan override. If not provided, uses the
                  default lifespan that loads config and initializes
                  embedder + store from disk. Pass a custom lifespan
                  for testing.
    """
    server = FastMCP(
        name="crowd-control",
        instructions=(
            "Crowd Control gives you access to learnings from past coding sessions. "
            "Use search_learnings to find relevant insights before tackling unfamiliar "
            "code. Use add_learning to store important discoveries for future sessions."
        ),
        lifespan=lifespan or _default_lifespan,
    )
    _register_tools(server)
    return server


def _get_deps(ctx: Context) -> ServerDeps:
    """Extract the lifespan dependencies from a tool context."""
    return ctx.request_context.lifespan_context


# ---------------------------------------------------------------------------
# Tool logic — standalone async functions for testability
# ---------------------------------------------------------------------------


async def handle_search_learnings(
    deps: ServerDeps,
    query: str,
    project: str | None = None,
    category: str | None = None,
    limit: int | None = None,
) -> str:
    """Search past session learnings by semantic similarity."""
    import dataclasses

    retrieval_config = deps.config.retrieval
    if limit is not None:
        retrieval_config = dataclasses.replace(retrieval_config, max_results=limit)

    current_project = project or os.getcwd()

    from crowd_control.retrieve import retrieve_learnings

    result = await asyncio.to_thread(
        retrieve_learnings,
        query=query,
        store=deps.store,
        embedder=deps.embedder,
        retrieval_config=retrieval_config,
        scope=deps.config.knowledge.scope,
        current_project=current_project,
        category=category,
    )

    logger.info("search_learnings: query=%r, results=%d", query, len(result.ranked))
    return format_results_text(result)


async def handle_add_learning(
    deps: ServerDeps,
    text: str,
    category: str = "pattern_discovery",
    tags: list[str] | None = None,
) -> str:
    """Manually store a learning for future sessions."""
    from pydantic import ValidationError

    from crowd_control.storage.models import Learning, LearningCategory

    try:
        validated_category = LearningCategory(category)
    except ValueError:
        valid = [c.value for c in LearningCategory]
        return f"Invalid category '{category}'. Must be one of: {', '.join(valid)}"

    try:
        learning = Learning(
            text=text,
            category=validated_category,
            tags=tags or [],
            project=os.getcwd(),
            session_id="manual",
            confidence=1.0,
        )
    except ValidationError as e:
        return f"Invalid learning: {e}"

    vectors = await asyncio.to_thread(deps.embedder.embed, [learning.text])

    record = learning.model_dump(mode="python")
    record["vector"] = vectors[0]

    stored = await asyncio.to_thread(deps.store.add, [record])

    if stored == 0:
        return "Learning was not stored (duplicate detected)."

    logger.info("add_learning: stored id=%s", learning.id)
    return f"Learning stored successfully (id={learning.id})."


async def handle_ingest_session(
    deps: ServerDeps,
    session_path: str | None = None,
) -> str:
    """Ingest a Claude Code session transcript to extract and store learnings."""
    from pathlib import Path

    from crowd_control.ingest.parser import find_sessions

    if session_path:
        resolved = Path(session_path).expanduser().resolve()
        if not resolved.exists():
            return f"File not found: {resolved}"
    else:
        sessions = await asyncio.to_thread(find_sessions)
        if not sessions:
            return "No session files found for the current project."
        resolved = sessions[0]

    from crowd_control.ingest.pipeline import ingest_session as run_ingest

    logger.info("ingest_session: path=%s", resolved)

    try:
        result = await asyncio.to_thread(run_ingest, resolved, deps.config)
    except (OSError, ValueError) as e:
        return f"Ingestion failed: {e}"
    except Exception:
        logger.exception("Unexpected error during ingestion of %s", resolved)
        return "Ingestion failed due to an unexpected error. Check logs for details."

    return (
        f"Ingested session {result.session_id}:\n"
        f"  Segments processed: {result.segments_processed}\n"
        f"  Learnings distilled: {result.learnings_distilled}\n"
        f"  Learnings stored: {result.learnings_stored}\n"
        f"  Duplicates skipped: {result.learnings_deduplicated}"
    )


async def handle_status(deps: ServerDeps) -> str:
    """Show the learnings database status and configuration."""
    count = await asyncio.to_thread(deps.store.count)

    return (
        f"Database: {deps.config.db_path}\n"
        f"Learnings: {count}\n"
        f"Embedding: {deps.config.embedding.provider}/{deps.config.embedding.model}\n"
        f"Scope: {deps.config.knowledge.scope}\n"
        f"Retrieval: max_results={deps.config.retrieval.max_results}, "
        f"max_tokens={deps.config.retrieval.max_tokens}"
    )


# ---------------------------------------------------------------------------
# MCP tool registration — thin wrappers that extract deps from context
# ---------------------------------------------------------------------------


def _register_tools(server: FastMCP) -> None:
    """Register all MCP tools on the given server instance."""

    @server.tool()
    async def search_learnings(
        query: str,
        project: str | None = None,
        category: str | None = None,
        limit: int | None = None,
        ctx: Context = None,
    ) -> str:
        """Search past session learnings by semantic similarity.

        Use this to find relevant insights, architecture decisions, debugging tips,
        and gotchas from previous coding sessions. Returns ranked results.

        Args:
            query: What to search for (natural language).
            project: Filter to a specific project path. Defaults to current project.
            category: Filter by learning category (e.g. 'debugging_insight',
                      'architecture_decision', 'gotcha', 'pattern_discovery',
                      'tool_usage', 'codebase_convention').
            limit: Maximum number of results (default: from config, typically 15).
        """
        return await handle_search_learnings(
            _get_deps(ctx), query, project, category, limit
        )

    @server.tool()
    async def add_learning(
        text: str,
        category: str = "pattern_discovery",
        tags: list[str] | None = None,
        ctx: Context = None,
    ) -> str:
        """Manually store a learning for future sessions.

        Use this when you discover something important during a session that should
        be remembered -- a debugging insight, architecture decision, gotcha, or pattern.

        Args:
            text: The learning content. Should be a single, self-contained insight.
            category: One of: architecture_decision, debugging_insight,
                      pattern_discovery, tool_usage, codebase_convention, gotcha.
            tags: Optional list of relevant tags (languages, frameworks, concepts).
        """
        return await handle_add_learning(_get_deps(ctx), text, category, tags)

    @server.tool()
    async def ingest_session(
        session_path: str | None = None,
        ctx: Context = None,
    ) -> str:
        """Ingest a Claude Code session transcript to extract and store learnings.

        Parses the session, uses LLM distillation to extract insights, embeds them,
        and stores in the learnings database. This can take a minute or more.

        Args:
            session_path: Path to the session JSONL file. If not provided, uses
                          the most recent session for the current project.
        """
        return await handle_ingest_session(_get_deps(ctx), session_path)

    @server.tool()
    async def status(ctx: Context = None) -> str:
        """Show the learnings database status and configuration.

        Returns the number of stored learnings, database path, and current
        embedding configuration.
        """
        return await handle_status(_get_deps(ctx))


def run_server() -> None:
    """Run the MCP server with stdio transport.

    Called by the CLI `serve` command.
    """
    logger.info("MCP server starting")
    server = create_server()
    server.run(transport="stdio")
