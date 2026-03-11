# Project Structure

```
crowd-control/
├── pyproject.toml                 # Package config, dependencies, entry points, tool config
├── LICENSE                        # MIT
├── README.md
├── CLAUDE.md                      # Instructions for Claude Code
├── structure.md                   # This file — keep up-to-date
├── docs/
│   └── plans/
│       ├── architecture.md        # Component architecture and data flow
│       ├── decisions.md           # Design decisions with rationale
│       ├── implementation-phases.md  # High-level phase overview
│       ├── phase-0-scaffolding.md    # Detailed Phase 0 steps
│       └── project-structure.md   # Dependencies, config schema
├── src/
│   └── crowd_control/
│       ├── __init__.py            # Package version
│       ├── cli.py                 # CLI entry point (click)
│       ├── config.py              # Configuration loading and defaults
│       ├── server.py              # MCP server definition (FastMCP)
│       ├── hooks.py               # Hook handler logic
│       ├── default_config.toml    # Default config template
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
    └── test_cli.py                # CLI smoke tests
```

## Status

| Module | Status |
|--------|--------|
| `cli.py` | Skeleton — all commands stubbed |
| `config.py` | Stub |
| `server.py` | Stub |
| `hooks.py` | Stub |
| `ingest/*` | Stubs |
| `embed/*` | Stubs |
| `storage/*` | Stubs |
| `retrieve/*` | Stubs |
