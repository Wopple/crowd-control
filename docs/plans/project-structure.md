# Crowd Control - Project Structure

## Package Layout

```
crowd-control/
├── pyproject.toml                 # Package config, dependencies, entry points
├── LICENSE                        # MIT
├── README.md
├── CLAUDE.md                      # Instructions for Claude Code
├── docs/
│   ├── conversations/
│   │   └── initial.txt
│   └── plans/
│       ├── architecture.md
│       ├── decisions.md
│       ├── project-structure.md
│       └── implementation-phases.md
├── src/
│   └── crowd_control/
│       ├── __init__.py            # Package version, public API
│       ├── cli.py                 # CLI entry point (click or argparse)
│       ├── config.py              # Configuration loading and defaults
│       ├── server.py              # MCP server definition (FastMCP)
│       ├── hooks.py               # Hook handler logic
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── parser.py          # Parse JSONL session transcripts
│       │   ├── distiller.py       # LLM-powered learning extraction
│       │   └── pipeline.py        # End-to-end ingestion pipeline
│       ├── embed/
│       │   ├── __init__.py
│       │   ├── base.py            # Embedder protocol
│       │   ├── ollama.py          # Ollama embedding provider
│       │   ├── voyage.py          # Voyage AI provider
│       │   └── openai.py          # OpenAI provider
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── db.py              # LanceDB operations
│       │   └── models.py          # Data models (Learning, Session, etc.)
│       └── retrieve/
│           ├── __init__.py
│           ├── search.py          # Vector search + metadata filtering
│           └── rank.py            # Recency decay, dedup, token packing
└── tests/
    ├── conftest.py                # Shared fixtures
    ├── test_parser.py
    ├── test_distiller.py
    ├── test_pipeline.py
    ├── test_embedder.py
    ├── test_storage.py
    ├── test_search.py
    ├── test_server.py
    ├── test_hooks.py
    └── test_cli.py
```

## Dependencies

### Core

| Package     | Purpose                            | Notes                    |
|-------------|------------------------------------|--------------------------|
| `mcp`       | MCP server framework (FastMCP)     | `mcp[cli]` for dev tools |
| `lancedb`   | Vector database                    | Embedded, Rust-backed    |
| `ollama`    | Local embedding (default provider) | Optional if using API    |
| `click`     | CLI framework                      | Clean, composable CLI    |
| `pydantic`  | Data validation and models         | Already a dep of `mcp`   |

Note: distillation uses the Claude Code CLI (`claude -p`), not the `anthropic` Python SDK.
This avoids requiring a separate API key — the user's existing Claude Code subscription is
used.

### Optional (for alternative embedding providers)

| Package    | Purpose              |
|------------|----------------------|
| `voyageai` | Voyage AI embeddings |
| `openai`   | OpenAI embeddings    |

### Development

| Package  | Purpose                |
|----------|------------------------|
| `pytest` | Testing                |
| `ruff`   | Linting and formatting |

## Entry Points

Defined in `pyproject.toml`:

```toml
[project.scripts]
crowd-control = "crowd_control.cli:main"
```

This gives users a `crowd-control` command after `pip install`.

## Configuration File

Default location: `~/.crowd-control/config.toml`

```toml
[general]
storage_dir = "~/.crowd-control"
log_level = "off"                      # "off", "error", "warning", "info", "debug"

[knowledge]
scope = "project"                      # "project", "shared", or "mixed" (v0.2+)

[embedding]
provider = "ollama"                # "ollama", "voyage", "openai"
model = "nomic-embed-text"

[embedding.api]
# key_env = "VOYAGE_API_KEY"      # Uncomment for API providers

[distillation]
model = "haiku"                        # Claude Code model alias (passed to claude -p --model)
max_learnings_per_session = 20

[retrieval]
max_results = 15
max_tokens = 4000
min_similarity = 0.3               # Minimum cosine similarity to include
recency_decay = 0.95               # Multiplied per week of age
project_boost = 1.5                 # Boost for same-project matches

[ingestion]
auto_ingest = true                  # Whether Stop hook triggers ingestion
batch_size = 5                      # Sessions to process per batch
```
