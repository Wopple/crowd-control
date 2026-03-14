import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import click

from crowd_control.config import ConfigError, CrowdControlConfig, load_config
from crowd_control.ingest.parser import find_sessions, parse_session_file
from crowd_control.storage.models import TextBlock

logger = logging.getLogger(__name__)


def _load_config_safe() -> CrowdControlConfig:
    """Load config, falling back to defaults on error.

    Returns (config, error). If error is not None, config is the default
    and the error should be reported by interactive commands.
    """
    try:
        return load_config(), None
    except ConfigError as e:
        return CrowdControlConfig(), e


@click.group()
@click.version_option(package_name="crowd-control")
@click.option("--verbose", "-v", is_flag=True, help="Show debug output on stderr.")
@click.pass_context
def main(ctx, verbose):
    """Crowd Control — learnings retention system for Claude Code."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    config, config_error = _load_config_safe()
    ctx.obj["config"] = config
    ctx.obj["config_error"] = config_error

    from crowd_control.logging_config import configure_logging

    configure_logging(config, interactive=True, verbose=verbose)


def _get_config_or_exit(ctx) -> CrowdControlConfig:
    """Get config from context, exiting on error. For interactive commands."""
    config_error = ctx.obj.get("config_error")
    if config_error is not None:
        click.echo(str(config_error), err=True)
        sys.exit(1)
    return ctx.obj["config"]


@main.command()
@click.pass_context
def status(ctx):
    """Show system status and database stats."""
    config = _get_config_or_exit(ctx)
    try:
        from crowd_control.storage.db import LearningStore

        store = LearningStore(config.db_path)
        count = store.count()
        click.echo(f"Database: {config.db_path}")
        click.echo(f"Learnings: {count}")
        click.echo(f"Embedding: {config.embedding.provider}/{config.embedding.model}")
    except Exception as e:
        click.echo(f"Database not initialized: {e}")


@main.group()
def hook():
    """Hook handlers (called by Claude Code, not directly by users)."""
    pass


@hook.command(name="session-end")
@click.pass_context
def hook_session_end(ctx):
    """Handle SessionEnd hook event from Claude Code."""
    # Hooks must always exit 0 to avoid blocking Claude Code.
    try:
        config = ctx.obj["config"]

        from crowd_control.logging_config import configure_logging

        configure_logging(config, interactive=False)

        from crowd_control.hooks import handle_session_end_hook

        try:
            raw = sys.stdin.read()
            event = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            click.echo("Invalid JSON on stdin", err=True)
            return

        result = handle_session_end_hook(event, config)

        if result.skipped_reason:
            click.echo(f"Skipped: {result.skipped_reason}", err=True)
    except Exception as e:
        logger.debug("hook session-end failed: %s", e)


@main.command()
@click.pass_context
def worker(ctx):
    """Process queued ingestion jobs."""
    config = ctx.obj["config"]

    from crowd_control.logging_config import configure_logging

    configure_logging(config, interactive=False)

    from crowd_control.worker import process_queue

    process_queue(config)


@main.command()
@click.option("--project", "project_scope", is_flag=True, help="Configure for current project.")
@click.pass_context
def setup(ctx, project_scope):
    """Configure hooks and MCP server in Claude Code."""
    from crowd_control.setup import run_setup

    config = _get_config_or_exit(ctx)
    result = run_setup(config, project_scope=project_scope)

    if result.issues:
        for issue in result.issues:
            click.echo(f"  ! {issue}", err=True)
        sys.exit(1)

    click.echo(f"Crowd Control configured successfully ({result.scope_label}).")
    click.echo()
    click.echo(f"MCP server: {result.mcp_path} (crowd-control serve)")
    click.echo("Hook:")
    click.echo("  SessionEnd -> queues ingestion + spawns background worker")
    click.echo()
    click.echo(f"Storage: {result.storage_dir}")
    click.echo(f"Embedding: {result.embedding_label}")
    click.echo()
    click.echo("Everything is automatic. When you end a session, learnings are")
    click.echo("extracted in the background. The agent uses search_learnings to")
    click.echo("find relevant insights during sessions.")
    click.echo()
    click.echo("Manual commands:")
    click.echo('  crowd-control search "query"   # Search from terminal')
    click.echo("  crowd-control worker           # Retry failed ingestions")
    click.echo("  crowd-control status           # Database stats")


@main.command()
@click.argument("path", required=False)
@click.option("--dry-run", is_flag=True, help="Parse and show structure without storing.")
@click.option(
    "--concurrency",
    default=8,
    type=int,
    show_default=True,
    help="Max parallel distillation requests.",
)
@click.pass_context
def ingest(ctx, path, dry_run, concurrency):
    """Ingest a session transcript."""
    resolved = _resolve_session_path(path)
    if resolved is None:
        sys.exit(1)

    if dry_run:
        try:
            session = parse_session_file(resolved)
        except Exception as e:
            click.echo(f"Error parsing {resolved}: {e}", err=True)
            sys.exit(1)
        _print_dry_run(session)
        return

    config = _get_config_or_exit(ctx)

    def _cli_progress(stage: str, completed: int, total: int) -> None:
        if completed == 1:
            click.echo(f"{stage.capitalize()} {total} segments ({concurrency} workers)...")
        click.echo(f"  Completed {completed}/{total}")

    try:
        from crowd_control.ingest.pipeline import ingest_session

        result = ingest_session(
            resolved, config, max_workers=concurrency, progress_callback=_cli_progress
        )
    except Exception as e:
        click.echo(f"Ingestion failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"\nIngested session {result.session_id}:")
    click.echo(f"  Segments processed: {result.segments_processed}")
    click.echo(f"  Learnings distilled: {result.learnings_distilled}")
    click.echo(f"  Learnings stored: {result.learnings_stored}")
    click.echo(f"  Duplicates skipped: {result.learnings_deduplicated}")


@main.command(name="list")
@click.option("--project", default=None, help="Filter by project path.")
@click.option("--category", default=None, help="Filter by category.")
@click.option("--limit", default=50, type=int, show_default=True)
@click.pass_context
def list_cmd(ctx, project, category, limit):
    """List stored learnings."""
    config = _get_config_or_exit(ctx)
    try:
        from crowd_control.storage.db import LearningStore

        store = LearningStore(config.db_path)
        learnings = store.list_learnings(project=project, category=category, limit=limit)
    except Exception as e:
        click.echo(f"Database not available: {e}", err=True)
        sys.exit(1)

    if not learnings:
        click.echo("No learnings found.")
        return

    for i, learning in enumerate(learnings, 1):
        click.echo(f"  [{i}] ({learning['category']}) [confidence={learning['confidence']:.2f}]")
        click.echo(f"      {learning['text']}")


@main.command()
@click.argument("query")
@click.option("--limit", default=None, type=int, help="Override max results.")
@click.option("--project", default=None, help="Filter by project path.")
@click.option("--category", default=None, help="Filter by category.")
@click.pass_context
def search(ctx, query, limit, project, category):
    """Search learnings for a query."""
    import dataclasses

    config = _get_config_or_exit(ctx)
    retrieval_config = config.retrieval
    if limit is not None:
        retrieval_config = dataclasses.replace(retrieval_config, max_results=limit)

    try:
        from crowd_control.embed.base import create_embedder

        embedder = create_embedder(config.embedding)
    except Exception as e:
        click.echo(f"Embedding provider error: {e}", err=True)
        click.echo(
            f"Is your embedding provider ({config.embedding.provider}) running?",
            err=True,
        )
        sys.exit(1)

    try:
        from crowd_control.storage.db import LearningStore

        store = LearningStore(config.db_path)
    except ValueError as e:
        if "vector_dimensions is required" in str(e):
            click.echo(
                "No learnings database found. Run `crowd-control ingest` first.",
                err=True,
            )
        else:
            click.echo(f"Database error: {e}", err=True)
        sys.exit(1)

    from crowd_control.retrieve import retrieve_learnings

    current_project = project or _detect_project()

    result = retrieve_learnings(
        query=query,
        store=store,
        embedder=embedder,
        retrieval_config=retrieval_config,
        scope=config.knowledge.scope,
        current_project=current_project,
        category=category,
    )

    _print_search_results(result, query)


@main.command()
@click.option("--output", "-o", default=None, type=click.Path(), help="Output file path.")
@click.option("--project", default=None, help="Filter by project path.")
@click.option("--category", default=None, help="Filter by category.")
@click.pass_context
def export(ctx, output, project, category):
    """Export learnings as JSON."""
    config = _get_config_or_exit(ctx)

    try:
        from crowd_control.storage.db import LearningStore

        store = LearningStore(config.db_path)
    except Exception as e:
        click.echo(f"Database not available: {e}", err=True)
        sys.exit(1)

    learnings = store.export_learnings(project=project, category=category)

    export_data = {
        "version": "1",
        "exported_at": datetime.now(UTC).isoformat(),
        "count": len(learnings),
        "learnings": learnings,
    }

    json_output = json.dumps(export_data, indent=2, default=str)

    if output:
        Path(output).write_text(json_output + "\n")
        click.echo(f"Exported {len(learnings)} learnings to {output}", err=True)
    else:
        click.echo(json_output)
        click.echo(f"Exported {len(learnings)} learnings", err=True)


@main.command()
def serve():
    """Run the MCP server (stdio transport)."""
    from crowd_control.server import run_server

    run_server()


def _print_search_results(result, query: str) -> None:
    """Format and display retrieval results."""
    from crowd_control.formatting import extract_display_learnings

    click.echo(f'Searching for: "{query}"')
    click.echo()

    if not result.ranked:
        click.echo("No matching learnings found.")
        return

    learnings = extract_display_learnings(result)

    for fl in learnings:
        age_str = f"{fl.age_days}d" if fl.age_days > 0 else "0s"
        click.echo(f"  [{fl.rank}] (score={fl.score:.2f}) [{fl.category}]")
        click.echo(f"      {fl.text}")
        click.echo(f"      project={fl.project}  retrieved={fl.active_count}x  age={age_str}")
        click.echo()

    result_word = "result" if len(learnings) == 1 else "results"
    click.echo(f"{len(learnings)} {result_word} (searched {result.total_learnings} learnings)")


def _print_dry_run(session) -> None:
    """Print session structure without distilling."""
    filtered_count = sum(len(s.messages) for s in session.segments)
    click.echo(f"Session: {session.session_id}")
    click.echo(f"Project: {session.project_path}")
    click.echo(f"Branch:  {session.git_branch or '(none)'}")
    click.echo(f"Model:   {session.model or '(unknown)'}")
    click.echo(f"Period:  {_fmt_time(session.start_time)} → {_fmt_time(session.end_time)}")
    click.echo(f"Messages: {session.message_count} parsed, {filtered_count} in segments")
    click.echo()

    if not session.segments:
        click.echo("No conversation segments found.")
        return

    click.echo(f"Segments ({len(session.segments)}):")
    for i, seg in enumerate(session.segments, 1):
        tools = ", ".join(seg.tool_names) if seg.tool_names else "(none)"
        click.echo(
            f"  [{i}] {_fmt_hms(seg.start_time)} — {_fmt_hms(seg.end_time)}"
            f"  ({len(seg.messages)} messages, tools: {tools})"
        )
        preview = _get_user_preview(seg)
        if preview:
            click.echo(f'      User: "{preview}"')


def _resolve_session_path(path: str | None) -> Path | None:
    """Resolve a session path argument, or find the most recent session for cwd."""
    if path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            click.echo(f"File not found: {resolved}", err=True)
            return None
        return resolved

    sessions = find_sessions()
    if not sessions:
        click.echo("No session files found for the current project.", err=True)
        return None
    return sessions[0]


def _get_user_preview(seg) -> str:
    """Get a preview of the first user text in a segment."""
    for msg in seg.messages:
        if msg.role.value == "user" and not msg.is_meta:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text = block.text.strip().replace("\n", " ")
                    if len(text) > 70:
                        return text[:67] + "..."
                    return text
    return ""


def _fmt_time(dt) -> str:
    try:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, AttributeError):
        return "(unknown)"


def _fmt_hms(dt) -> str:
    try:
        return dt.strftime("%H:%M:%S")
    except (ValueError, AttributeError):
        return "??:??:??"


def _detect_project() -> str:
    """Return the current working directory as the project path."""
    return os.getcwd()
