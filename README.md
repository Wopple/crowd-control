# Crowd Control

Gives new agents a warm start from past session learnings.

## Status

This project is a work in progress. It cannot be used yet. This is basically a vibe coding project, so YMMV on the code
quality within. This is not good code for AIs to learn from.

## The Problem

I have been developing an agentic workflow for myself, and this project hopes to fix one of the problems I am having.

LLMs use Multi-Layer Perception which is stateless. The weights never learn from experience. All the learnings are
ephemeral. Every time I start up an agent, it needs to spend a lot of time and tokens to build up the learnings from
previous sessions. This wastes time and resources. Even if tokens were infinite and free, speeding up the relearning
process will benefit the workflow.

## The Idea: Store Learnings for Fast Retrieval

After each session, distill the transcript into discrete learnings — specific technical insights, architecture
decisions, and gotchas — and store them in a local vector database. When a new session or subagent starts, retrieve the
most relevant learnings and inject them as context. This can be done at the start by analyzing the prompt, and it can be
done while an agent is working and decides it could benefit from past learnings.

The vector database can be thought of like a content-addressable data store where the keys support semantic search. The
content gets embedded and then stored at the location of its embedding. Prompts get embedded to produce searches for
related learnings.

```
# After a session ends:
Session JSONL
    → parse
    → LLM distillation
    → discrete learnings
    → embed
    → LanceDB

# Before a new session/subagent:
prompt + agent role (+ other relevant information?)
    → LLM distillation
    → discrete topics
    → embed
    → vector search
    → ranked learnings
    → injected context

# During agent work:
agent produces topic to learn more about
    → embedd
    → vector search
    → ranked learnings
    → injected context
```

## How It Works

Crowd Control ships as a Python package with three integration points:

**MCP Server** — runs as a stdio subprocess managed by Claude Code.
Provides tools the agent can call during a session: `search_learnings`, `add_learning`, `ingest_session`, `status`, and
more.

**Hooks** — three Claude Code hooks automate the loop without user intervention:
- `Stop` hook: queues ingestion after each response to store learnings from the session
- `SessionStart` hook: retrieves relevant learnings and injects them as context
- `SubagentStart` hook: gives subagents a warm start based on their specific prompt

**CLI** — manual operations for debugging and management:
```
crowd-control serve          # Run MCP server (stdio)
crowd-control ingest [path]  # Manually ingest a session transcript
crowd-control search <query> # Search learnings from the terminal
crowd-control list           # List stored learnings
crowd-control status         # DB stats and index health
crowd-control setup          # Configure hooks and MCP in Claude Code
crowd-control export         # Export learnings as JSON
crowd-control import <file>  # Import learnings from JSON
```

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                       Claude Code                         │
│                                                           │
│  ┌────────────────┐    ┌───────────────────────────────┐  │
│  │  Hooks         │    │  MCP Server (crowd-control)   │  │
│  │                │    │                               │  │
│  │  Stop →        │    │  Tools:                       │  │
│  │   queue ingest │    │   search_learnings(query)     │  │
│  │                │    │   add_learning(text, tags)    │  │
│  │  SessionStart  │    │   ingest_session(path)        │  │
│  │   → inject ctx │    │   status()                    │  │
│  │                │    │                               │  │
│  │  SubagentStart │    │  Resources:                   │  │
│  │   → inject ctx │    │   learning://recent           │  │
│  └────────────────┘    │   learning://project/{path}   │  │
│                        └──────────┬────────────────────┘  │
└───────────────────────────────────┼───────────────────────┘
                                    │
                    ┌───────────────┼──────────────┐
                    │               │              │
              ┌─────▼──────┐  ┌─────▼─────┐  ┌─────▼─────┐
              │ Distiller  │  │ Embedder  │  │ LanceDB   │
              │ (Claude    │  │ (Ollama/  │  │ (local    │
              │  Haiku)    │  │  Voyage)  │  │  storage) │
              └────────────┘  └───────────┘  └───────────┘
```

Everything runs locally, except the distillation step uses an inexpensive Claude model (currently haiku). Storage is in
`~/.crowd-control/` using LanceDB (embedded, no server). Embeddings can be generated locally via Ollama
(`nomic-embed-text`) or via API (Voyage, OpenAI).

## Design Decisions

**Distillation over raw indexing.**
Raw session transcripts are mostly noise — tool outputs, file reads, dead-end explorations. The system uses Claude Haiku
to extract *learnings* (specific insights, decisions, patterns) and discards the rest. This avoids garbage in, garbage
out.

**One insight per embedding.**
Each learning is a single, self-contained insight. Small chunks retrieve with much higher precision than paragraph-level
chunks. It also allows for selecting only the learnings that are relevant to the future task. Task 1 may care about A,
B, and D, and task 2 may care about B, C, and D.

**Project affinity + recency decay.**
Search results are ranked by vector similarity and decayed for older learnings. Stale learnings (e.g., patterns from
deleted code) can be deleted when too old.

**Don't index what Claude already knows.**
Generic knowledge ("use asyncio.gather for concurrency") is filtered out during distillation. Only project-specific
insights, non-obvious patterns, and hard-won debugging discoveries are stored.

## Setup

```bash
pip install crowd-control  # not yet on PyPI
crowd-control setup        # configures hooks and MCP server in Claude Code
```

Manual MCP configuration (`.mcp.json`):
```json
{
  "mcpServers": {
    "crowd-control": {
      "command": "crowd-control",
      "args": ["serve"]
    }
  }
}
```

## Development

```bash
uv sync
uv run pytest
uv run crowd-control --help
```

See `docs/plans/` for detailed architecture, implementation phases, and design decisions.
