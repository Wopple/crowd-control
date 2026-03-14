# Crowd Control

Gives new agents a warm start from past session learnings.

## Quick Start

```bash
pip install crowd-control
crowd-control setup
```

That's it. Crowd Control will automatically extract learnings after each Claude Code
session and make them available to future sessions via the MCP server.

## How It Works

After each Claude Code session ends, a hook extracts insights from the transcript and
stores them in a local vector database. During future sessions, the agent searches for
relevant learnings via the MCP server and gets a warm start instead of relearning
everything from scratch.

## The Problem

LLMs are stateless. Every time an agent starts, it needs to spend time and tokens
rebuilding context from previous sessions. Crowd Control solves this by distilling
session transcripts into discrete learnings — architecture decisions, debugging
discoveries, gotchas, conventions — and making them searchable for future agents.

## Architecture

```
                       Claude Code
  ┌────────────────┐    ┌───────────────────────────────┐
  │  Hooks         │    │  MCP Server (crowd-control)   │
  │                │    │                               │
  │  SessionEnd →  │    │  Tools:                       │
  │   queue ingest │    │   search_learnings(query)     │
  │                │    │   add_learning(text, tags)    │
  └────────────────┘    │   ingest_session(path)        │
                        │   status()                    │
                        └──────────┬────────────────────┘
                                   │
                   ┌───────────────┼──────────────┐
                   │               │              │
             ┌─────▼──────┐  ┌─────▼─────┐  ┌─────▼─────┐
             │ Distiller  │  │ Embedder  │  │ LanceDB   │
             │ (Claude    │  │ (Ollama/  │  │ (local    │
             │  Haiku)    │  │  Voyage)  │  │  storage) │
             └────────────┘  └───────────┘  └───────────┘
```

Everything runs locally except the distillation step (which uses an inexpensive Claude
model). Storage is in `~/.crowd-control/` using LanceDB (embedded, no server). Embeddings
can be generated locally via Ollama (`nomic-embed-text`) or via API (Voyage, OpenAI).

## CLI

```bash
crowd-control setup            # Configure hooks and MCP in Claude Code
crowd-control ingest [path]    # Manually ingest a session transcript
crowd-control search <query>   # Search learnings from the terminal
crowd-control list             # List stored learnings
crowd-control status           # DB stats and index health
crowd-control export           # Export learnings as JSON
crowd-control worker           # Process queued ingestion jobs
crowd-control serve            # Run MCP server (stdio)
```

## Configuration

Configuration lives in `~/.crowd-control/config.toml`. See `docs/configuration.md` for
a complete reference.

Common options:
- Embedding provider: Ollama (default), Voyage AI, or OpenAI
- Token budget for context injection
- Retrieval tuning (similarity threshold, recency decay, result limits)
- Trace logging for debugging

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) with `nomic-embed-text` model (for default embeddings)
- Claude Code CLI installed and authenticated

```bash
ollama pull nomic-embed-text
```

## Design Decisions

**Distillation over raw indexing.** Raw session transcripts are mostly noise. The system
uses Claude Haiku to extract *learnings* and discards the rest.

**One insight per embedding.** Each learning is a single, self-contained insight. Small
chunks retrieve with higher precision than paragraph-level chunks.

**Project affinity + recency decay.** Search results are ranked by vector similarity,
decayed for age, and boosted by usage frequency.

**Don't index what Claude already knows.** Generic programming knowledge is filtered out
during distillation. Only project-specific insights are stored.

## Development

```bash
uv sync
uv run pytest
uv run crowd-control --help
```

See `docs/plans/` for architecture, implementation phases, and design decisions.
