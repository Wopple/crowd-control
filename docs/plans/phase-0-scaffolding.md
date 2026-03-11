# Phase 0: Project Scaffolding — Detailed Plan

**Goal:** An installable Python package where `crowd-control --help` works.

---

## Step 1: Initialize `uv` project

```bash
uv init --lib --package crowd-control
```

This creates the basic `pyproject.toml` and `src/crowd_control/__init__.py`. We'll
overwrite most of it in the next steps, but it sets up the `uv` workspace correctly.

### Verify

```bash
uv run python -c "import crowd_control; print('ok')"
```

---

## Step 2: Configure `pyproject.toml`

Replace the generated `pyproject.toml` with the full project config:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "crowd-control"
version = "0.1.0"
description = "Learnings retention system for Claude Code"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [
    { name = "Daniel" },
]

dependencies = [
    "anthropic",
    "click",
    "lancedb",
    "mcp[cli]",
    "ollama",
    "pydantic",
]

[project.optional-dependencies]
voyage = ["voyageai"]
openai = ["openai"]
dev = [
    "pytest",
    "ruff",
]

[project.scripts]
crowd-control = "crowd_control.cli:main"
```

### Verify

```bash
uv sync
uv sync --extra dev
```

---

## Step 3: Create package directory structure

Create all directories and `__init__.py` files. Every subdirectory under
`src/crowd_control/` needs an `__init__.py` to be a proper Python package.

### Files to create

```
src/crowd_control/__init__.py       # Already exists from uv init
src/crowd_control/cli.py
src/crowd_control/config.py
src/crowd_control/server.py
src/crowd_control/hooks.py
src/crowd_control/ingest/__init__.py
src/crowd_control/ingest/parser.py
src/crowd_control/ingest/distiller.py
src/crowd_control/ingest/pipeline.py
src/crowd_control/embed/__init__.py
src/crowd_control/embed/base.py
src/crowd_control/embed/ollama.py
src/crowd_control/embed/voyage.py
src/crowd_control/embed/openai.py
src/crowd_control/storage/__init__.py
src/crowd_control/storage/db.py
src/crowd_control/storage/models.py
src/crowd_control/retrieve/__init__.py
src/crowd_control/retrieve/search.py
src/crowd_control/retrieve/rank.py
tests/conftest.py
tests/test_cli.py
```

### File contents

**`src/crowd_control/__init__.py`**
```python
"""Crowd Control — retrieval-augmented context for Claude Code."""

__version__ = "0.1.0"
```

**`src/crowd_control/cli.py`** — minimal click skeleton with `--help`:
```python
import click

@click.group()
@click.version_option(package_name="crowd-control")
def main():
    """Crowd Control — retrieval-augmented context for Claude Code."""
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
```

**`src/crowd_control/config.py`** — stub:
```python
"""Configuration loading and defaults."""
```

All other module files — empty stubs with a single docstring:
```python
"""<Module description from project-structure.md>."""
```

**`tests/conftest.py`**:
```python
"""Shared test fixtures for crowd-control."""
```

**`tests/test_cli.py`** — basic smoke test:
```python
from click.testing import CliRunner
from crowd_control.cli import main

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Crowd Control" in result.output

def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0

def test_status_command():
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
```

### Verify

```bash
uv run crowd-control --help
uv run crowd-control --version
uv run crowd-control status
```

---

## Step 4: Configure ruff

Add to `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

### Verify

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

---

## Step 5: Configure pytest

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Verify

```bash
uv run pytest
```

All 3 tests in `test_cli.py` should pass.

---

## Step 6: Add default config template

Create `src/crowd_control/default_config.toml` — the template that `crowd-control setup`
will copy to `~/.crowd-control/config.toml`:

```toml
[general]
storage_dir = "~/.crowd-control"
log_level = "info"

[embedding]
provider = "ollama"
model = "nomic-embed-text"

# [embedding.api]
# key_env = "VOYAGE_API_KEY"

[distillation]
model = "claude-haiku-4-5-20251001"
max_learnings_per_session = 20

[retrieval]
max_results = 15
max_tokens = 4000
min_similarity = 0.3
recency_decay = 0.95
project_boost = 1.5

[ingestion]
auto_ingest = true
batch_size = 5
```

---

## Checklist

| # | Task | Verify command |
|---|------|----------------|
| 1 | `uv init --lib --package crowd-control` | `uv run python -c "import crowd_control"` |
| 2 | Write full `pyproject.toml` | `uv sync --extra dev` |
| 3 | Create all package dirs + stubs | `uv run crowd-control --help` |
| 4 | Configure ruff | `uv run ruff check src/ tests/` |
| 5 | Configure pytest | `uv run pytest` (3 pass) |
| 6 | Add default config template | File exists |

**Phase 0 is complete when:** `uv run crowd-control --help` prints the help text,
`uv run pytest` passes all tests, and `uv run ruff check` reports no issues.
