import click


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
    click.echo("Ingestion not yet implemented.")


@main.command()
@click.argument("query")
def search(query):
    """Search learnings for a query."""
    click.echo("Search not yet implemented.")


@main.command()
def serve():
    """Run the MCP server (stdio transport)."""
    click.echo("MCP server not yet implemented.")
