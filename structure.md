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
│   └── plans/
│       ├── architecture.md        # Component architecture and data flow
│       ├── decisions.md           # Design decisions with rationale
│       ├── implementation-phases.md  # High-level phase overview
│       ├── phase-3-detailed.md    # Phase 3 detailed plan
│       └── project-structure.md   # Dependencies, config schema
├── src/
│   └── crowd_control/
│       ├── __init__.py            # Package version
│       ├── cli.py                 # CLI entry point (click) — ingest, list, status commands
│       ├── config.py              # Configuration loading from TOML with dataclass schema
│       ├── server.py              # MCP server definition (stub)
│       ├── hooks.py               # Hook handler logic (stub)
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
│       │   ├── db.py              # LanceDB operations — LearningStore with CRUD and dedup
│       │   └── models.py          # Data models (all Phase 1 models implemented)
│       └── retrieve/
│           ├── __init__.py
│           ├── search.py          # Vector search + metadata filtering (stub)
│           └── rank.py            # Recency decay, dedup, token packing (stub)
└── tests/
    ├── conftest.py                # Shared fixtures (FakeEmbedder)
    ├── test_cli.py                # CLI smoke tests
    ├── test_config.py             # Configuration loading tests
    ├── test_models.py             # Data model construction and serialization tests
    ├── test_parser.py             # JSONL parser and segmentation tests
    ├── test_distiller.py          # Distillation pipeline tests (mocked subprocess)
    ├── test_embedder.py           # Embedding protocol and provider tests
    ├── test_storage.py            # LanceDB storage operation tests
    ├── test_pipeline.py           # End-to-end pipeline tests (mocked distiller + fake embedder)
    └── fixtures/
        ├── sample_session.jsonl   # Multi-segment session with tool calls
        ├── minimal_session.jsonl  # Minimal 1-segment session
        ├── compact_session.jsonl  # Session with compact_boundary split
        └── distillation_response.json  # Canned claude -p response for tests
```

## Status

| Module | Status |
|--------|--------|
| `cli.py` | `ingest` with full pipeline, `list`, `status` with DB stats, other commands stubbed |
| `config.py` | Implemented — TOML loading with frozen dataclass schema |
| `server.py` | Stub |
| `hooks.py` | Stub |
| `ingest/parser.py` | Implemented — parsing, segmentation, discovery |
| `ingest/distiller.py` | Implemented — prompt building, claude -p invocation, learning extraction |
| `ingest/pipeline.py` | Implemented — end-to-end parse → distill → embed → store |
| `embed/base.py` | Implemented — Embedder protocol, factory, EmbeddingError |
| `embed/ollama.py` | Implemented — Ollama provider with dimension lookup |
| `embed/voyage.py` | Implemented — Voyage AI provider with API key validation |
| `embed/openai.py` | Implemented — OpenAI provider with API key validation |
| `storage/models.py` | Implemented — all data models |
| `storage/db.py` | Implemented — LearningStore with CRUD, dedup, dimension management |
| `retrieve/*` | Stubs |
