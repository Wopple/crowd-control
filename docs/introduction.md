# Introduction

Crowd Control is a learning retention system for Claude Code. It extracts insights from
past coding sessions and makes them searchable in future sessions, giving agents a warm
start instead of rebuilding context from scratch.

## What it does

After each Claude Code session ends, Crowd Control:

1. Parses the session transcript (JSONL)
2. Sends conversation segments to Claude Haiku, which extracts discrete **learnings** —
   architecture decisions, debugging insights, gotchas, conventions, etc.
3. Embeds each learning into a vector (via Ollama, Voyage, or OpenAI)
4. Stores the vectors in a local LanceDB database

During future sessions, the agent searches for relevant learnings via an MCP server tool
(`search_learnings`). Results are ranked by semantic similarity, recency, and usage
frequency.

## Key dependencies

| Dependency | Role |
|------------|------|
| Claude Code CLI (`claude -p`) | Distillation — extracts learnings from transcripts |
| Ollama (`nomic-embed-text`) | Default embedding provider (local, free) |
| LanceDB | Embedded vector database (no server, stored at `~/.crowd-control/db/`) |
| FastMCP | MCP server framework (stdio transport) |
| Click | CLI framework |

## Integration points with Claude Code

Crowd Control connects to Claude Code in two places:

- **MCP server** — registered in `~/.claude.json` (or `.mcp.json` for project scope).
  Claude Code spawns `crowd-control serve` as a subprocess. The agent calls
  `search_learnings`, `add_learning`, `ingest_session`, and `status` tools during
  sessions.
- **SessionEnd hook** — registered in `~/.claude/settings.json`. When a session ends,
  Claude Code runs `crowd-control hook session-end`, which queues the transcript for
  background ingestion.

## Data flow

```
Session ends
  → SessionEnd hook writes queue file → worker spawns in background
  → parser.py reads JSONL → Session with ConversationSegments
  → distiller.py sends segments to Claude Haiku → list of Learning objects
  → embedder converts learning text to vectors
  → LearningStore deduplicates and stores in LanceDB

Agent searches
  → MCP tool search_learnings(query)
  → embedder converts query to vector
  → LanceDB vector search with metadata filters
  → rank_results: similarity × recency × hotness → token-packed results
```

## Where to learn more

| Topic | Document |
|-------|----------|
| File tree and module status | `structure.md` |
| Transcript parsing and LLM extraction | `docs/distillation.md` |
| Embedding providers, LanceDB schema, dedup | `docs/embedding-and-storage.md` |
| Search, scoring formula, token packing | `docs/retrieval.md` |
| SessionEnd hook, queue, background worker | `docs/hooks.md` |
| MCP server tools, lifespan, agent instructions | `docs/mcp-server.md` |
| Config file reference | `docs/configuration.md` |
| End-user installation and troubleshooting | `docs/user-guide.md` |
