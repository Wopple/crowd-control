import sys
from pathlib import Path

import click

from crowd_control.ingest.parser import find_sessions, parse_session_file
from crowd_control.storage.models import TextBlock


@click.group()
@click.version_option(package_name="crowd-control")
def main():
    """Crowd Control — learnings retention system for Claude Code."""
    pass


@main.command()
def status():
    """Show system status and database stats."""
    click.echo("crowd-control is installed. No database configured yet.")


@main.command()
def setup():
    """Configure hooks and MCP server in Claude Code."""
    click.echo("Setup not yet implemented.")


@main.command()
@click.argument("path", required=False)
@click.option("--dry-run", is_flag=True, help="Parse and show structure without storing.")
def ingest(path, dry_run):
    """Ingest a session transcript."""
    if not dry_run:
        click.echo("Ingestion not yet implemented. Use --dry-run to preview parsing.")
        return

    resolved = _resolve_session_path(path)
    if resolved is None:
        sys.exit(1)

    try:
        session = parse_session_file(resolved)
    except Exception as e:
        click.echo(f"Error parsing {resolved}: {e}", err=True)
        sys.exit(1)

    filtered_count = sum(len(s.messages) for s in session.segments)
    click.echo(f"Session: {session.session_id}")
    click.echo(f"Project: {session.project_path}")
    click.echo(f"Branch:  {session.git_branch or '(none)'}")
    click.echo(f"Model:   {session.model or '(unknown)'}")
    click.echo(
        f"Period:  {_fmt_time(session.start_time)} → {_fmt_time(session.end_time)}"
    )
    click.echo(f"Messages: {session.message_count} total, {filtered_count} after filtering")
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


@main.command()
@click.argument("query")
def search(query):
    """Search learnings for a query."""
    click.echo("Search not yet implemented.")


@main.command()
def serve():
    """Run the MCP server (stdio transport)."""
    click.echo("MCP server not yet implemented.")


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
