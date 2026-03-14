# Setup Guide

## Prerequisites

- **Python 3.11+** — Crowd Control uses `tomllib` (stdlib in 3.11+) and modern Python features.
- **Ollama** — For default local embeddings. Install from [ollama.ai](https://ollama.ai).
- **Claude Code CLI** — Installed and authenticated. Crowd Control uses `claude -p` for distillation.

Pull the default embedding model:

```bash
ollama pull nomic-embed-text
```

## Installation

### From PyPI (end user)

```bash
pip install crowd-control
```

### With uv

```bash
uv pip install crowd-control
```

### From source (development)

```bash
git clone https://github.com/daniel/crowd-control.git
cd crowd-control
uv sync
```

When developing from source, use `uv run crowd-control` instead of `crowd-control`.

## Running Setup

```bash
crowd-control setup
```

This configures everything automatically:

1. Creates `~/.crowd-control/` directory
2. Writes a default `config.toml`
3. Adds the MCP server to Claude Code's config (`~/.claude.json`)
4. Adds the SessionEnd hook to Claude Code's settings (`~/.claude/settings.json`)

Expected output:

```
Crowd Control configured successfully (global).

MCP server: /Users/you/.claude.json (crowd-control serve)
Hook:
  SessionEnd -> queues ingestion + spawns background worker

Storage: /Users/you/.crowd-control
Embedding: ollama/nomic-embed-text

Everything is automatic. When you end a session, learnings are
extracted in the background. The agent uses search_learnings to
find relevant insights during sessions.
```

### Project-scoped setup

To configure for a specific project only (instead of globally):

```bash
cd /path/to/project
crowd-control setup --project
```

This writes to `.mcp.json` and `.claude/settings.json` in the project directory instead
of the home directory.

### Running setup again

Setup is idempotent. Running it again updates existing entries without creating duplicates.

## Verifying the Installation

### 1. Check the CLI works

```bash
crowd-control --version
crowd-control --help
```

### 2. Check status

```bash
crowd-control status
```

If the database hasn't been initialized yet, you'll see "Database not initialized" — this
is normal before the first ingestion.

### 3. Manual ingestion (optional)

If you have an existing Claude Code session, you can ingest it manually:

```bash
crowd-control ingest            # ingest most recent session for current project
crowd-control ingest /path/to/session.jsonl
crowd-control ingest --dry-run  # preview without storing
```

### 4. Search

After ingesting at least one session:

```bash
crowd-control search "how does authentication work"
```

### 5. Verify MCP server

Start a new Claude Code session. The MCP server should connect automatically. Ask Claude
to search for learnings to verify the tool works.

## Troubleshooting

### `crowd-control: command not found`

The package isn't on your PATH. Common fixes:

- If installed with `pip install --user`, add `~/.local/bin` to your PATH
- If installed in a virtualenv, activate it first
- If using `uv`, use `uv run crowd-control` or `uvx crowd-control`

### Ollama not running

```
Embedding provider error: ...
Is your embedding provider (ollama) running?
```

Start Ollama:

```bash
ollama serve
```

Or if using the Ollama desktop app, make sure it's running.

### Claude Code not finding the MCP server

Check that the MCP config exists:

```bash
cat ~/.claude.json
```

It should contain a `crowd-control` entry under `mcpServers`. If not, run
`crowd-control setup` again.

### Dimension mismatch after switching embedding models

```
Embedding dimension mismatch. Table has 768-dim vectors but embedder produces 1024-dim.
```

Switching embedding models requires re-creating the database:

```bash
cp -r ~/.crowd-control/db ~/.crowd-control/db.bak
rm -rf ~/.crowd-control/db
# Re-ingest your sessions
```

### Invalid config.toml

```
Invalid TOML in /Users/you/.crowd-control/config.toml: ...
```

Fix the syntax error in your config file, or delete it and run `crowd-control setup` to
regenerate the defaults.
