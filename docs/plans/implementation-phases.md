# Crowd Control - Implementation Phases

## Phase 0: Project scaffolding

- Initialize `pyproject.toml` with metadata, dependencies, entry points
- Create package structure under `src/crowd_control/`
- Set up `ruff` for linting/formatting
- Set up `pytest` with basic conftest
- Create default config template

**Deliverable:** Installable package with `crowd-control --help` working.

## Phase 1: Session parsing and data models

- Define `Learning` and `Session` pydantic models
- Parse Claude Code JSONL session transcripts into structured message lists
- Handle the various message types (user, assistant, tool_use, tool_result)
- Segment conversations into logical chunks (by tool call boundaries or topic shifts)
- Write tests with sample session fixtures

**Deliverable:** `crowd-control ingest --dry-run <path>` shows parsed session structure.

## Phase 2: Distillation pipeline

- Implement the distillation prompt (extract learnings from session segments)
- Call Claude Code CLI (`claude -p`) with `--json-schema` for structured extraction
- Parse structured output into `Learning` objects
- Handle subprocess errors, timeouts, and retries
- Add category classification and tag extraction
- Write tests with mocked API responses

**Deliverable:** `crowd-control ingest <path>` prints extracted learnings to stdout.

## Phase 3: Embedding and storage

- Implement `Embedder` protocol and Ollama provider
- Connect to LanceDB and create the learnings table schema
- Embed learnings and insert into LanceDB
- Implement basic CRUD operations (list, get, delete)
- Add Voyage and OpenAI embedding providers
- Write tests with in-memory or temp-dir LanceDB

**Deliverable:** Full ingestion pipeline works end-to-end. Learnings stored and queryable.

## Phase 4: Retrieval and ranking

- Implement vector search with metadata filtering
- Add recency decay to ranking scores
- Add project-affinity boosting
- Implement deduplication (by text similarity)
- Implement token budget packing
- Write tests with known queries against seeded data

**Deliverable:** `crowd-control search "how does the auth system work"` returns ranked results.

## Phase 5: MCP server

- Define MCP server with FastMCP
- Implement `search_learnings` tool
- Implement `add_learning` tool
- Implement `ingest_session` tool
- Implement `status` tool
- Use lifespan API for DB connection management
- Write integration tests

**Deliverable:** MCP server runs via `crowd-control serve` and responds to tool calls.

## Phase 6: Hooks and automation

- Implement `Stop` hook handler (queue ingestion job)
- Implement `SessionStart` hook handler (retrieve + format context)
- Implement `crowd-control setup` command (auto-configure hooks + MCP)
- Add background worker for processing ingestion queue
- Write tests for hook I/O format

**Deliverable:** Fully automated loop — sessions ingest automatically, new sessions get context.

## Phase 7: Polish and release

- Error handling and graceful degradation throughout
- Logging with configurable verbosity
- Documentation (README, setup guide, configuration reference)
- PyPI packaging and release
- Test on fresh machine with clean install

**Deliverable:** v0.1 release on PyPI.
