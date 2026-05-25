"""MCP server definition (FastMCP)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

from crowd_control.config import CrowdControlConfig, load_config
from crowd_control.embed.base import Embedder, EmbeddingError, create_embedder
from crowd_control.formatting import format_results_text
from crowd_control.project import resolve_project
from crowd_control.storage.db import LearningStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerDeps:
    """Shared resources available to all tool calls."""

    config: CrowdControlConfig
    store: LearningStore | None
    embedder: Embedder | None
    project_id: str = ""


@asynccontextmanager
async def _default_lifespan(server: FastMCP) -> AsyncIterator[ServerDeps]:
    """Initialize shared resources for the server lifetime."""
    config = load_config()

    embedder: Embedder | None = None
    try:
        embedder = await asyncio.to_thread(create_embedder, config.embedding)
    except EmbeddingError as e:
        logger.warning("Embedder unavailable: %s", e)

    store: LearningStore | None = None
    try:
        dims = embedder.dimensions if embedder else None
        store = await asyncio.to_thread(
            LearningStore, config.db_path, dims, config.ingestion.dedup_threshold
        )
    except ValueError:
        # New table needs dimensions from embedder; existing table works without.
        # If we get here, there's no existing table and no embedder to provide dims.
        logger.warning("LearningStore unavailable: no existing DB and no embedder")

    project_id = resolve_project()
    logger.info(
        "Lifespan: store=%s, embedder=%s, project_id=%s",
        store is not None,
        embedder is not None,
        project_id,
    )

    if store is not None and config.ingestion.max_age_days > 0:
        pruned = await asyncio.to_thread(
            store.prune,
            config.ingestion.max_age_days,
            config.ingestion.retention_retrieval_interval_days,
        )
        if pruned > 0:
            logger.info("Startup prune: removed %d old learnings", pruned)

    yield ServerDeps(config=config, store=store, embedder=embedder, project_id=project_id)


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
            "Crowd Control stores insights from past coding sessions — architecture "
            "decisions, debugging discoveries, gotchas, conventions, and patterns "
            "specific to this codebase. These are things learned the hard way in "
            "previous sessions.\n\n"
            "When to search (search_learnings):\n"
            "- When you receive a new prompt: search for learnings related to the "
            "task, the files involved, or the area of the codebase.\n"
            "- Before making architecture or design decisions: check if past "
            "sessions established relevant conventions or made related decisions.\n"
            "- When debugging: search for known gotchas, past debugging insights, "
            "or issues related to the error or behavior you're investigating.\n"
            "- When building a plan: search for learnings that might inform your "
            "approach — past attempts, constraints discovered, or patterns "
            "that worked.\n"
            "- When working with unfamiliar code: search for insights about how "
            "that part of the codebase works.\n\n"
            "Search tips:\n"
            "- Be concise. A short phrase or single sentence works best — the "
            "embedding model matches more precisely with focused text. Do not "
            "pad your query with extra context or preamble.\n"
            "- Keep each query focused on one topic. Multiple search calls for "
            "different aspects of a task work better than one broad query.\n"
            "- Queries are limited to a few hundred characters. If your query is "
            "that long, split it into multiple focused searches.\n\n"
            "Query effectiveness:\n"
            "- Results are ranked by a blend of semantic similarity, recency, "
            "and usage frequency. The most important factor is how well your "
            "query matches the learning text semantically.\n"
            "- Use the `tags` parameter to narrow by domain area — this is more "
            "effective than adding domain words to the query, which can dilute "
            "the semantic match. Call `status` to see available tags before "
            "filtering.\n"
            "- Use the `category` parameter to filter by type (e.g., only "
            "'gotcha' or only 'architecture_decision') when you know what kind "
            "of insight you need.\n\n"
            "When to store (add_learning):\n"
            "- When you discover something non-obvious that would save time in a "
            "future session — a gotcha, a pattern, an architectural constraint, "
            "or a debugging technique specific to this codebase.\n"
            "- Do not store generic programming knowledge that any developer would "
            "know. Only store insights specific to this project or its particular "
            "combination of tools and patterns.\n\n"
            "When to delete (delete_learning):\n"
            "- When a search result is clearly contradicted by the current codebase. "
            "Verify against actual code before deleting.\n"
            "- Do not delete learnings that could serve as useful context for what "
            "does not work, even if the referenced code has changed."
        ),
        lifespan=lifespan or _default_lifespan,
    )
    _register_tools(server)
    return server


def _get_deps(ctx: Context) -> ServerDeps:
    """Extract the lifespan dependencies from a tool context."""
    return ctx.request_context.lifespan_context


def _require_embedder(deps: ServerDeps) -> Embedder:
    """Return the embedder or raise a user-friendly error."""
    if deps.embedder is None:
        raise EmbeddingError(
            "Embedding provider not available. "
            f"Is {deps.config.embedding.provider} running? "
            "Check `crowd-control status` for details."
        )
    return deps.embedder


def _require_store(deps: ServerDeps) -> LearningStore:
    """Return the store or raise a ValueError."""
    if deps.store is None:
        raise ValueError(
            "Learnings database not available. Run `crowd-control ingest` to initialize."
        )
    return deps.store


# ---------------------------------------------------------------------------
# Tool logic — standalone async functions for testability
# ---------------------------------------------------------------------------


async def handle_search_learnings(
    deps: ServerDeps,
    query: str,
    category: str | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> str:
    """Search past session learnings by semantic similarity."""
    import dataclasses

    try:
        embedder = _require_embedder(deps)
        store = _require_store(deps)
    except (EmbeddingError, ValueError) as e:
        return str(e)

    retrieval_config = deps.config.retrieval
    if limit is not None:
        retrieval_config = dataclasses.replace(retrieval_config, max_results=limit)

    current_project = deps.project_id

    from crowd_control.retrieve import retrieve_learnings

    try:
        result = await asyncio.to_thread(
            retrieve_learnings,
            query=query,
            store=store,
            embedder=embedder,
            retrieval_config=retrieval_config,
            scope=deps.config.knowledge.scope,
            current_project=current_project,
            category=category,
            tags=tags,
        )
    except EmbeddingError as e:
        return f"Embedding error during search: {e}"

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
        embedder = _require_embedder(deps)
        store = _require_store(deps)
    except (EmbeddingError, ValueError) as e:
        return str(e)

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
            project=deps.project_id,
            session_id="manual",
            confidence=1.0,
        )
    except ValidationError as e:
        return f"Invalid learning: {e}"

    try:
        vectors = await asyncio.to_thread(embedder.embed, [learning.text])
    except EmbeddingError as e:
        return f"Embedding error: {e}"

    record = learning.model_dump(mode="python")
    record["vector"] = vectors[0]

    add_result = await asyncio.to_thread(store.add, [record])

    if add_result.stored == 0:
        if add_result.duplicates:
            dup = add_result.duplicates[0]
            return (
                f"Learning was not stored (duplicate detected, "
                f"similarity={dup.similarity:.2f}).\n"
                f"Existing: {dup.matched_text}"
            )
        return "Learning was not stored (duplicate detected)."

    logger.info("add_learning: stored id=%s", learning.id)
    return f"Learning stored successfully (id={learning.id})."


async def handle_ingest_session(
    deps: ServerDeps,
    session_path: str | None = None,
) -> str:
    """Ingest a Claude Code session transcript to extract and store learnings."""
    if not deps.config.ingestion.agent_ingest:
        logger.info("ingest_session: blocked by agent_ingest=false")
        return (
            "Agent-initiated ingestion is disabled. Use `add_learning` to store "
            "learnings manually, or set `agent_ingest = true` in "
            "~/.crowd-control/config.toml to enable."
        )

    from pathlib import Path

    from crowd_control.ingest.parser import find_sessions

    try:
        _require_embedder(deps)
    except EmbeddingError as e:
        return str(e)

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
    from crowd_control.formatting import format_status_counts
    from crowd_control.ingest.llm.status import check_distillation_status

    current_project = deps.project_id
    auto_ingest = "enabled" if deps.config.ingestion.auto_ingest else "disabled"
    agent_ingest = "enabled" if deps.config.ingestion.agent_ingest else "disabled"

    embedder_status = (
        f"{deps.config.embedding.provider}/{deps.config.embedding.model}"
        if deps.embedder is not None
        else f"{deps.config.embedding.provider}/{deps.config.embedding.model} (unavailable)"
    )

    distillation_status = await asyncio.to_thread(
        check_distillation_status, deps.config.distillation
    )
    distillation_lines = [
        "Distillation:",
        f"  provider: {distillation_status.provider}",
        f"  model:    {distillation_status.model}",
        f"  ready:    {'yes' if distillation_status.ready else 'no'}",
    ]
    if distillation_status.hint:
        distillation_lines.append(f"  hint:     {distillation_status.hint}")

    if deps.store is None:
        return "\n".join(
            [
                f"Database: {deps.config.db_path} (not initialized)",
                f"Project: {current_project}",
                "Learnings: 0",
                "Tags: (none)",
                f"Embedding: {embedder_status}",
                *distillation_lines,
                f"Scope: {deps.config.knowledge.scope}",
                f"Auto-ingest: {auto_ingest}",
                f"Agent ingest: {agent_ingest}",
                f"Retrieval: max_results={deps.config.retrieval.max_results}, "
                f"max_tokens={deps.config.retrieval.max_tokens}",
            ]
        )

    def _get_status_data(
        store: LearningStore,
        proj: str,
    ) -> tuple[int, int, list[str], list[str]]:
        return (
            store.count(project=proj),
            store.count(),
            store.distinct_tags(project=proj),
            store.distinct_tags(),
        )

    proj_count, total_count, proj_tags, all_tags = await asyncio.to_thread(
        _get_status_data,
        deps.store,
        current_project,
    )

    logger.info("status: project=%s, count=%d/%d", current_project, proj_count, total_count)

    sc = format_status_counts(proj_count, total_count, proj_tags, all_tags)

    lines = [
        f"Database: {deps.config.db_path}",
        f"Project: {current_project}",
        sc.learnings_line,
        sc.tags_line,
    ]
    if sc.all_tags_line is not None:
        lines.append(sc.all_tags_line)
    lines.extend(
        [
            f"Embedding: {embedder_status}",
            *distillation_lines,
            f"Scope: {deps.config.knowledge.scope}",
            f"Auto-ingest: {auto_ingest}",
            f"Agent ingest: {agent_ingest}",
            f"Retrieval: max_results={deps.config.retrieval.max_results}, "
            f"max_tokens={deps.config.retrieval.max_tokens}",
        ]
    )

    return "\n".join(lines)


async def handle_delete_learning(
    deps: ServerDeps,
    ids: list[str],
) -> str:
    """Delete learnings by ID prefix."""
    if not deps.config.ingestion.agent_delete:
        logger.info("delete_learning: blocked by agent_delete=false")
        return (
            "Agent-initiated deletion is disabled. Set `agent_delete = true` "
            "in ~/.crowd-control/config.toml under [ingestion] to enable."
        )

    try:
        store = _require_store(deps)
    except ValueError as e:
        return str(e)

    current_project = deps.project_id
    min_prefix_len = 8

    deleted: list[str] = []
    errors: list[str] = []

    for prefix in ids:
        if len(prefix) < min_prefix_len:
            errors.append(f"id={prefix}: too short (minimum {min_prefix_len} characters)")
            continue

        if not all(c in "0123456789abcdef" for c in prefix.lower()):
            errors.append(f"id={prefix}: invalid characters (must be hex)")
            continue

        matches = await asyncio.to_thread(store.find_by_prefix, prefix)

        if len(matches) == 0:
            errors.append(f"id={prefix}: not found")
            continue

        if len(matches) > 1:
            found = ", ".join(m["id"][:8] for m in matches)
            errors.append(f"id={prefix}: ambiguous, matches: {found}")
            continue

        match = matches[0]
        if match["project"] != current_project:
            errors.append(f"id={prefix}: belongs to a different project, skipping")
            continue

        full_id = match["id"]
        await asyncio.to_thread(store.delete, full_id)

        snippet = match["text"][:80]
        deleted.append(f"  Deleted id={full_id[:8]} [{match['category']}]: {snippet}")

    lines: list[str] = []
    if deleted:
        lines.append(f"Deleted {len(deleted)} learning(s):")
        lines.extend(deleted)
    if errors:
        if deleted:
            lines.append("")
        lines.append(f"Skipped {len(errors)} ID(s):")
        lines.extend(f"  {e}" for e in errors)
    if not deleted and not errors:
        lines.append("No IDs provided.")

    logger.info(
        "delete_learning: deleted=%d, errors=%d, project=%s",
        len(deleted),
        len(errors),
        current_project,
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP tool registration — thin wrappers that extract deps from context
# ---------------------------------------------------------------------------


def _register_tools(server: FastMCP) -> None:
    """Register all MCP tools on the given server instance."""

    @server.tool()
    async def search_learnings(
        query: str,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        ctx: Context = None,
    ) -> str:
        """Search past session learnings by semantic similarity.

        Use this to find relevant insights, architecture decisions, debugging tips,
        and gotchas from previous coding sessions. Returns ranked results.

        Args:
            query: What to search for. Matched via semantic similarity against
                   learning text. Use domain-specific terms, not generic project
                   vocabulary. Examples:
                   - Good: "LanceDB dedup threshold false positives"
                   - Bad: "how does the system work" (too broad, matches everything)
                   - Good: "collision detection AABB broadphase"
                   - Bad: "collision rework phase 1" ("phase"/"rework" match noise)
                   If results seem noisy, narrow with tags or category before
                   rephrasing the query.
            category: Filter by learning category (e.g. 'debugging_insight',
                      'architecture_decision', 'gotcha', 'pattern_discovery',
                      'tool_usage', 'codebase_convention').
            tags: Filter to learnings with any of the given tags (match-any).
                  Tags are case-insensitive. This is the most effective way to
                  narrow results to a specific domain area. Call the `status`
                  tool to see available tags.
            limit: Maximum number of results (default: from config, typically 15).
        """
        return await handle_search_learnings(_get_deps(ctx), query, category, tags, limit)

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
    async def status(
        ctx: Context = None,
    ) -> str:
        """Show the learnings database status and configuration.

        Returns the number of stored learnings, available tags, database
        path, and current embedding configuration. Use this to discover
        valid tag values before filtering with search_learnings.
        """
        return await handle_status(_get_deps(ctx))

    @server.tool()
    async def delete_learning(
        ids: list[str],
        ctx: Context = None,
    ) -> str:
        """Delete outdated learnings by ID prefix.

        Use this to remove learnings that are clearly contradicted by the current
        state of the codebase. Each search result includes an `id=XXXXXXXX` prefix
        that can be passed to this tool.

        When to delete:
        - ONLY delete a learning when it is clearly contradicted by the current
          state of the codebase AND it is not useful context of what has been
          tried and did not work. Before deleting, you MUST verify against the
          actual code that the learning's claims are no longer true.
        - Examples: a learning says "function X uses pattern Y" but X was
          rewritten to use pattern Z; a learning references a file or module
          that no longer exists; a learning describes a bug workaround for
          a bug that has since been fixed.

        When NOT to delete:
        - Do NOT delete learnings about general architecture decisions,
          conventions, or debugging techniques that are still conceptually
          valid, even if the specific code they reference has changed.
        - Do NOT delete a learning just because it is old or has a low
          retrieval count — the TTL pruning system handles that automatically.
        - Do NOT delete a learning when it could serve as useful context for
          what does not work.
        - When in doubt, leave the learning. False deletions lose knowledge
          that cannot be recovered.

        Args:
            ids: List of learning ID prefixes (minimum 8 characters each).
                 These are shown as `id=XXXXXXXX` in search results.
        """
        return await handle_delete_learning(_get_deps(ctx), ids)


def run_server() -> None:
    """Run the MCP server with stdio transport.

    Called by the CLI `serve` command.
    """
    logger.info("MCP server starting")
    server = create_server()
    server.run(transport="stdio")
