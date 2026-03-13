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
              │ (claude -p │  │ (Ollama/  │  │ (local    │
              │  CLI)      │  │  Voyage)  │  │  storage) │
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
  → Vector search in LanceDB (filtered by project, category, stale)
  → Compute per-result score:
      recency   = exp(-ln(2) / half_life_days * age_days)
      hotness   = sigmoid(log1p(active_count)) * recency
      final     = (1 - hotness_weight) * semantic + hotness_weight * hotness
  → Apply project boost for same-project matches
  → Deduplicate by text similarity
  → Pack into token budget
  → Inject as context (via hook stdout or MCP resource)
  → Increment active_count for returned learnings
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

## Knowledge Scope

Learnings can be scoped in three ways, controlled by `knowledge.scope` in config:

- **`project`** (default): retrieval is filtered to the current project. Learnings from other projects are not surfaced. This prevents irrelevant context from unrelated codebases.
- **`shared`**: retrieval searches all learnings regardless of source project. Useful when working across related repos or when most learnings are broadly applicable.
- **`mixed`** (v0.2+): the distiller classifies each learning as project-specific or universal. At retrieval time, the current project's learnings are merged with the universal pool. This gives each project its own knowledge plus shared insights, without cross-contamination between unrelated projects.

In all modes, every learning records its source `project` for provenance. For `project` and `shared`, the scope setting only changes retrieval behavior — switching between them does not require re-ingesting data.

`mixed` mode adds a `shared: bool` field to each learning, set by the distiller during extraction. Existing learnings default to `shared=False`, so switching to `mixed` is backwards compatible.

When scope is `project`, the `project_boost` retrieval parameter has no effect (all results are already from the current project). When scope is `shared` or `mixed`, `project_boost` up-ranks learnings from the current project.

## Learning Data Model

Each learning stored in LanceDB:

| Field        | Type       | Description                                    |
|-------------|------------|------------------------------------------------|
| id          | string     | UUID                                           |
| vector      | float[N]   | Embedding vector                               |
| text        | string     | The learning content                           |
| category    | string     | architecture_decision, debugging_insight, etc. |
| tags        | list[str]  | Languages, frameworks, concepts                |
| project     | string     | Source project path (always set for provenance) |
| session_id  | string     | Source session ID                               |
| git_sha     | string     | Git SHA at time of learning (if available)     |
| timestamp   | datetime   | When the learning was extracted                |
| confidence  | float      | How significant/reliable the learning is       |
| active_count| int        | Times retrieved in search results (feeds hotness scoring) |
| stale       | bool       | Whether the learning has been marked stale     |
| shared      | bool       | Universal learning visible to all projects (mixed scope only) |
