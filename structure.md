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
│   ├── embedding-and-storage.md  # Embedding providers, LanceDB storage, dedup, pipeline
│   ├── hooks.md                  # Hooks, automation, queue/worker pipeline, setup command
│   ├── mcp-server.md             # MCP server tools, lifespan, architecture, instructions
│   ├── retrieval.md              # Retrieval and ranking system (search, scoring, packing)
│   └── plans/
│       ├── architecture.md        # Component architecture and data flow
│       ├── decisions.md           # Design decisions with rationale (12 decisions)
│       ├── implementation-phases.md  # Phase overview (0-6 complete, 7 planned)
│       ├── learning-deduplication.md # Within-session text-based dedup plan
│       ├── openviking-learnings.md   # Algorithms adopted from OpenViking for Phase 4
│       ├── phase6-hooks-and-automation.md  # Phase 6 implementation plan
│       └── project-structure.md   # Dependencies, config schema
├── src/
│   └── crowd_control/
│       ├── __init__.py            # Package version
│       ├── cli.py                 # CLI entry point (click) — ingest, list, status, search, serve, setup, hook, worker
│       ├── config.py              # Configuration loading from TOML with dataclass schema
│       ├── formatting.py          # Shared result formatting for CLI and MCP server
│       ├── server.py              # MCP server (FastMCP) — tools, lifespan, factory, agent instructions
│       ├── hooks.py               # Hook handler logic — SessionEnd queue + worker spawning
│       ├── worker.py              # Background ingestion worker — queue processing, retry, failure handling
│       ├── setup.py               # Setup logic — MCP config, hook config, prerequisites, JSON merging
│       ├── default_config.toml    # Default config template
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── parser.py          # JSONL parsing, segmentation, session discovery
│       │   ├── distiller.py       # LLM-powered learning extraction via claude -p
│       │   └── pipeline.py        # End-to-end ingestion: parse → distill → embed → store
│       ├── embed/
│       │   ├── __init__.py
│       │   ├── base.py            # Embedder protocol, factory, EmbeddingError
│       │   ├── ollama.py          # Ollama embedding provider
│       │   ├── voyage.py          # Voyage AI embedding provider
│       │   └── openai.py          # OpenAI embedding provider
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── db.py              # LanceDB operations — LearningStore with CRUD, dedup, has_session
│       │   └── models.py          # Data models (all Phase 1 models implemented)
│       └── retrieve/
│           ├── __init__.py        # Public exports + retrieve_learnings orchestrator
│           ├── search.py          # Vector search, metadata filtering, scope validation, BaseResult
│           └── rank.py            # Scoring, dedup, token packing
└── tests/
    ├── conftest.py                # Shared fixtures (FakeEmbedder)
    ├── test_cli.py                # CLI smoke tests
    ├── test_cli_hooks.py          # CLI hook subcommand integration tests
    ├── test_config.py             # Configuration loading tests
    ├── test_models.py             # Data model construction and serialization tests
    ├── test_parser.py             # JSONL parser and segmentation tests
    ├── test_distiller.py          # Distillation pipeline tests (mocked subprocess)
    ├── test_embedder.py           # Embedding protocol and provider tests
    ├── test_hooks.py              # Hook handler logic tests (SessionEnd, spawn_worker)
    ├── test_storage.py            # LanceDB storage, vector search, active count, has_session tests
    ├── test_pipeline.py           # End-to-end pipeline tests (mocked distiller + fake embedder)
    ├── test_search.py             # Search module tests (FakeEmbedder + real LanceDB)
    ├── test_rank.py               # Ranking module tests (pure function, no DB)
    ├── test_retrieval_integration.py  # End-to-end retrieval pipeline tests
    ├── test_server.py               # MCP server tools, formatting, integration tests
    ├── test_setup.py              # Setup command logic tests (MCP config, hooks, merge)
    ├── test_worker.py             # Background worker tests (queue processing, retry, failures)
    └── fixtures/
        ├── sample_session.jsonl   # Multi-segment session with tool calls
        ├── minimal_session.jsonl  # Minimal 1-segment session
        ├── compact_session.jsonl  # Session with compact_boundary split
        └── distillation_response.json  # Canned claude -p response for tests
```

## Status

| Module | Status |
|--------|--------|
| `cli.py` | `ingest` with full pipeline, `list`, `status` with DB stats, `search` with retrieval pipeline, `serve` with MCP server, `setup` with full auto-config, `hook session-end`, `worker` |
| `config.py` | Implemented — TOML loading with frozen dataclass schema |
| `formatting.py` | Implemented — shared result formatting (extract_display_learnings, format_results_text) |
| `server.py` | Implemented — FastMCP server factory, lifespan, 4 tools, detailed agent instructions |
| `hooks.py` | Implemented — SessionEnd handler, queue file writing, worker spawning |
| `worker.py` | Implemented — queue processing, retry with attempt tracking, failed job handling |
| `setup.py` | Implemented — MCP config, hook config, prerequisites, JSON merging, global/project scope |
| `ingest/parser.py` | Implemented — parsing, segmentation, discovery |
| `ingest/distiller.py` | Implemented — prompt building, claude -p invocation, learning extraction |
| `ingest/pipeline.py` | Implemented — end-to-end parse → distill → embed → store |
| `embed/base.py` | Implemented — Embedder protocol, factory, EmbeddingError |
| `embed/ollama.py` | Implemented — Ollama provider with dimension lookup |
| `embed/voyage.py` | Implemented — Voyage AI provider with API key validation |
| `embed/openai.py` | Implemented — OpenAI provider with API key validation |
| `storage/models.py` | Implemented — all data models |
| `storage/db.py` | Implemented — LearningStore with CRUD, dedup, has_session, dimension management |
| `retrieve/search.py` | Implemented — query embedding, vector search, metadata filtering |
| `retrieve/rank.py` | Implemented — hotness scoring, dedup, token packing |
