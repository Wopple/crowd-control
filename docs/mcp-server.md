# MCP Server

The MCP server exposes Crowd Control's functionality to Claude Code via the Model Context
Protocol. When configured, Claude Code spawns the server as a subprocess and calls its tools
during sessions.

## Starting the Server

```bash
crowd-control serve   # starts stdio MCP server
```

Claude Code manages the server's lifecycle. Configure it in `.mcp.json`:

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

Or with `uv` during development:

```json
{
  "mcpServers": {
    "crowd-control": {
      "command": "uv",
      "args": ["run", "crowd-control", "serve"]
    }
  }
}
```

## Architecture

The server is built with FastMCP from the `mcp[cli]` SDK. It uses stdio transport — no
HTTP server, no ports, just stdin/stdout JSON-RPC.

### Lifespan

The server initializes shared resources once at startup via the FastMCP lifespan API:

- **Embedder** — created from config, validates provider connectivity
- **LearningStore** — opens (or creates) the LanceDB database

These are shared across all tool calls for the session lifetime. If the embedder fails
to initialize (e.g., Ollama not running), the server starts with `embedder=None`. Tools
that need the embedder (`search_learnings`, `add_learning`, `ingest_session`) return a
clear error message. Tools that don't need it (`status`) continue to work. This prevents
a missing embedding provider from making the entire server unavailable.

### Error Responses

When a tool encounters an error, it returns a plain-text error string rather than
raising an exception. This ensures the agent sees a useful message instead of a
traceback. Common error responses:

- `"Embedding provider not available. Is ollama running?"` — embedder failed at startup
- `"Learnings database not available. Run 'crowd-control ingest' to initialize."` — no DB
- `"Embedding error during search: ..."` — embedder failed during a tool call
- `"Invalid category '...'. Must be one of: ..."` — bad category in add_learning

### Threading

All tools are async but the underlying LanceDB and embedding operations are synchronous.
Blocking calls are wrapped in `asyncio.to_thread()` so they don't freeze the MCP event loop.

The `ingest_session` tool creates its own `Embedder` and `LearningStore` internally (via
the pipeline module) rather than sharing the lifespan's instances. This avoids cross-thread
mutable state sharing — LanceDB handles concurrent directory access safely.

## Tools

### `search_learnings`

Searches past session learnings by semantic similarity. Calls the same `retrieve_learnings()`
pipeline that the CLI uses — query embedding, vector search, ranking, dedup, token packing.

| Parameter  | Type              | Required | Default        | Description |
|------------|-------------------|----------|----------------|-------------|
| `query`    | `str`             | yes      | —              | Natural language search query |
| `project`  | `str | None`      | no       | `os.getcwd()`  | Filter to specific project |
| `category` | `str | None`      | no       | all categories | Filter by learning category |
| `tags`     | `list[str] | None`| no       | no tag filter  | Filter by tags (match-any, case-insensitive) |
| `limit`    | `int | None`      | no       | config default | Max results to return |

Returns formatted text with scored results, categories, retrieval counts, and ages.

### `add_learning`

Manually stores a learning for future sessions. The agent provides already-formulated
insight text — no distillation step.

| Parameter  | Type              | Required | Default              | Description |
|------------|-------------------|----------|----------------------|-------------|
| `text`     | `str`             | yes      | —                    | Learning content (max 2000 chars) |
| `category` | `str`             | no       | `pattern_discovery`  | Learning category |
| `tags`     | `list[str] | None`| no       | `[]`                 | Relevant tags |
| `project`  | `str | None`      | no       | `os.getcwd()`        | Project path for this learning |

Categories: `architecture_decision`, `debugging_insight`, `pattern_discovery`,
`tool_usage`, `codebase_convention`, `gotcha`.

Returns success/failure message. Duplicates are detected and rejected.

### `ingest_session`

Runs the full ingestion pipeline on a session transcript: parse, LLM distill, embed, store.

| Parameter      | Type         | Required | Default          | Description |
|----------------|-------------|----------|------------------|-------------|
| `session_path` | `str | None` | no       | most recent session | Path to JSONL file |

This can take a minute or more (LLM distillation). Returns counts of segments processed,
learnings distilled, stored, and deduplicated.

### `status`

Shows database status and configuration, scoped to a project.

| Parameter  | Type              | Required | Default        | Description |
|------------|-------------------|----------|----------------|-------------|
| `project`  | `str | None`      | no       | `os.getcwd()`  | Filter stats to specific project |

Returns: database path, project path, project-scoped learning count (with global total),
project-scoped tags (with global tag list), embedding provider/model, scope, retrieval
limits. When all learnings belong to the current project, the output is simplified to
avoid redundant totals.

The tag list is useful for discovering valid values before using the `tags` filter in
`search_learnings`.

## How Tools Map to Existing Modules

The server is a thin adapter — each tool delegates to existing, tested modules:

| Tool | Underlying Module |
|------|-------------------|
| `search_learnings` | `retrieve.retrieve_learnings()` |
| `add_learning` | `storage.models.Learning` + `embed` + `storage.db.LearningStore.add()` |
| `ingest_session` | `ingest.pipeline.ingest_session()` |
| `status` | `storage.db.LearningStore.count()` + config |

## Project Detection

Tools that need a project path default to `os.getcwd()`. This works because Claude Code
sets the MCP server subprocess's working directory to the project root (via `.mcp.json`).

## Server Instructions

The MCP server includes detailed instructions that guide the agent to use
`search_learnings` proactively. These instructions replace the need for a `SessionStart`
hook — the agent decides when to search, and it has the user's prompt to craft precise
queries.

The instructions cover:
- **When to search**: on new prompts, before design decisions, when debugging, when
  building plans, when working with unfamiliar code
- **Search tips**: keep queries concise (short phrase or sentence), one topic per query,
  make multiple calls for multi-topic tasks
- **Query effectiveness**: explains the scoring model (semantic similarity, recency,
  usage frequency), recommends `tags` as the most effective narrowing mechanism, and
  notes `category` for type-based filtering
- **When to store**: non-obvious discoveries specific to this codebase, not generic
  programming knowledge

The `search_learnings` tool docstring also includes inline guidance for the `query`
parameter: good/bad query examples, advice to use domain-specific terms over generic
vocabulary, and a note to narrow with `tags`/`category` before rephrasing.

This is the only retrieval mechanism in the session. There is no automatic injection.
The agent decides what context is worth fetching and when.

## Division of Labor

The MCP server handles **on-demand retrieval** during sessions (agent calls
`search_learnings`). The SessionEnd hook handles **automatic ingestion** after sessions
(queue + background worker). See `docs/hooks.md` for the ingestion pipeline.
