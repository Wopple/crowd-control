# Project Structure

```
crowd-control/
├── pyproject.toml                 # Package config, dependencies, entry points, tool config
├── LICENSE                        # MIT
├── README.md
├── CLAUDE.md                      # Instructions for Claude Code
├── structure.md                   # This file — keep up-to-date
├── docs/
│   ├── distillation.md           # How the distillation pipeline works
│   └── plans/
│       ├── architecture.md        # Component architecture and data flow
│       ├── code-quality-fixes.md  # 5 targeted code quality improvements
│       ├── code-quality-phase-1.md # Detailed plan: deduplicate project path fallback
│       ├── code-quality-phase-2.md # Detailed plan: remove JSON extraction heuristics
│       ├── decisions.md           # Design decisions with rationale
│       ├── implementation-phases.md  # High-level phase overview
│       ├── phase-2-distillation.md       # Detailed Phase 2 steps
│       └── project-structure.md   # Dependencies, config schema
├── src/
│   └── crowd_control/
│       ├── __init__.py            # Package version
│       ├── cli.py                 # CLI entry point (click) — ingest with distillation, --dry-run
│       ├── config.py              # Configuration loading and defaults (stub)
│       ├── server.py              # MCP server definition (stub)
│       ├── hooks.py               # Hook handler logic (stub)
│       ├── default_config.toml    # Default config template
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── parser.py          # JSONL parsing, segmentation, session discovery
│       │   ├── distiller.py       # LLM-powered learning extraction via claude -p
│       │   └── pipeline.py        # End-to-end ingestion pipeline (stub)
│       ├── embed/
│       │   ├── __init__.py
│       │   ├── base.py            # Embedder protocol (stub)
│       │   ├── ollama.py          # Ollama embedding provider (stub)
│       │   ├── voyage.py          # Voyage AI provider (stub)
│       │   └── openai.py          # OpenAI provider (stub)
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── db.py              # LanceDB operations (stub)
│       │   └── models.py          # Data models (all Phase 1 models implemented)
│       └── retrieve/
│           ├── __init__.py
│           ├── search.py          # Vector search + metadata filtering (stub)
│           └── rank.py            # Recency decay, dedup, token packing (stub)
└── tests/
    ├── conftest.py                # Shared fixtures
    ├── test_cli.py                # CLI smoke tests
    ├── test_models.py             # Data model construction and serialization tests
    ├── test_parser.py             # JSONL parser and segmentation tests
    ├── test_distiller.py          # Distillation pipeline tests (mocked subprocess)
    └── fixtures/
        ├── sample_session.jsonl   # Multi-segment session with tool calls
        ├── minimal_session.jsonl  # Minimal 1-segment session
        ├── compact_session.jsonl  # Session with compact_boundary split
        └── distillation_response.json  # Canned claude -p response for tests
```

## Status

| Module | Status |
|--------|--------|
| `cli.py` | `ingest` with distillation, `--dry-run` preview, other commands stubbed |
| `config.py` | Stub |
| `server.py` | Stub |
| `hooks.py` | Stub |
| `ingest/parser.py` | Implemented — parsing, segmentation, discovery |
| `ingest/distiller.py` | Implemented — prompt building, claude -p invocation, learning extraction |
| `ingest/pipeline.py` | Stub |
| `storage/models.py` | Implemented — all data models |
| `embed/*` | Stubs |
| `storage/db.py` | Stub |
| `retrieve/*` | Stubs |
