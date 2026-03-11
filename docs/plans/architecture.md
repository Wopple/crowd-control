# Crowd Control - Architecture Plan

## Overview

Crowd Control is a retrieval-augmented context system for Claude Code that gives new
agents a "warm start" from past session learnings. It captures session transcripts,
distills them into discrete learnings, embeds and stores them in a local vector database,
and retrieves relevant learnings for new sessions automatically.

The system ships as a **Python package** that bundles:
1. An **MCP server** (retrieval + management tools for the agent)
2. **Claude Code hooks** (automatic ingestion and context injection)
3. A **CLI** (manual operations, debugging, status)

## Component Architecture

```
┌───────────────────────────────────────────────────────────┐
│                       Claude Code                         │
│                                                           │
│  ┌────────────────┐    ┌───────────────────────────────┐  │
│  │  Hooks         │    │  MCP Server                   │  │
│  │                │    │  (crowd-control)              │  │
│  │  Stop →        │    │                               │  │
│  │   queue        │    │  Tools:                       │  │
│  │   ingestion    │    │   - search_learnings(query)   │  │
│  │                │    │   - get_learning(id)          │  │
│  │  SubagentStart │    │   - add_learning(text, tags)  │  │
│  │   → inject     │    │   - list_learnings(filters)   │  │
│  │   context      │    │   - delete_learning(id)       │  │
│  │                │    │   - ingest_session(path)      │  │
│  │  SessionStart  │    │   - status()                  │  │
│  │   → inject     │    │                               │  │
│  │   context      │    │  Resources:                   │  │
│  └────────────────┘    │   - learning://recent         │  │
│                        │   - learning://project/{path} │  │
│                        └──────────┬────────────────────┘  │
│                                   │                       │
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

## Data Flow

### Ingestion (post-session)

```
Session JSONL file (~/.claude/projects/*/session.jsonl)
  → Parse structured messages
  → Chunk into conversation segments
  → LLM distillation (extract learnings, discard noise)
  → Embed each learning
  → Store in LanceDB with metadata (project, tags, timestamp, git SHA)
```

### Retrieval (pre-session / on-demand)

```
New prompt + project context
  → Embed query
  → Vector search in LanceDB (filtered by project, recency)
  → Rank by relevance + recency decay
  → Pack into token budget
  → Inject as context (via hook stdout or MCP resource)
```

## Integration Points with Claude Code

### 1. MCP Server (primary interface)

The MCP server is the main integration. It runs as a stdio subprocess managed by
Claude Code and provides tools the agent can call during a session.

**Configuration** (`.mcp.json` or `~/.claude.json`):
```json
{
  "mcpServers": {
    "crowd-control": {
      "command": "crowd-control",
      "args": ["serve"],
      "env": {}
    }
  }
}
```

The MCP server maintains a persistent connection to LanceDB for the session lifetime
via the FastMCP lifespan API.

### 2. Hooks (automation)

Three hooks automate the system without user intervention:

**a) `Stop` hook — queue ingestion after each response**
When Claude finishes responding, trigger async ingestion of any new session data.
This is lightweight — it just signals the MCP server or writes to a queue file.

**b) `SessionStart` hook — inject relevant context**
When a new session starts, retrieve relevant learnings for the current project and
inject them as context via stdout. The hook output becomes part of the agent's context.

**c) `SubagentStart` hook — inject context for subagents**
When a subagent spawns, retrieve learnings relevant to its prompt and inject them.
This is the key "warm start" mechanism from the original design.

**Hook configuration** (`~/.claude/settings.json`):
```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "crowd-control hook stop",
            "timeout": 5
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "crowd-control hook session-start",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

### 3. CLI (manual operations)

```
crowd-control serve          # Run MCP server (stdio)
crowd-control ingest [path]  # Manually ingest a session
crowd-control search <query> # Search learnings from terminal
crowd-control list           # List all learnings
crowd-control status         # Show DB stats, index health
crowd-control setup          # Configure hooks + MCP in Claude Code
crowd-control export         # Export learnings as JSON
crowd-control import <file>  # Import learnings from JSON
```

## Storage Layout

```
~/.crowd-control/
├── config.toml              # User configuration
├── db/                      # LanceDB storage
│   └── learnings.lance/     # Main learnings table
├── queue/                   # Pending ingestion queue
│   └── <session_id>.json    # Queued session references
└── logs/                    # Operation logs
    └── ingestion.log
```

## Learning Data Model

Each learning stored in LanceDB:

| Field        | Type       | Description                                    |
|-------------|------------|------------------------------------------------|
| id          | string     | UUID                                           |
| vector      | float[N]   | Embedding vector                               |
| text        | string     | The learning content                           |
| category    | string     | architecture_decision, debugging_insight, etc. |
| tags        | list[str]  | Languages, frameworks, concepts                |
| project     | string     | Project path or identifier                     |
| session_id  | string     | Source session ID                               |
| git_sha     | string     | Git SHA at time of learning (if available)     |
| timestamp   | datetime   | When the learning was extracted                |
| confidence  | float      | How significant/reliable the learning is       |
| stale       | bool       | Whether the learning has been marked stale     |
