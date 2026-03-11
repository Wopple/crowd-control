# Crowd Control - Design Decisions

## Decision 1: Should we build a custom MCP server?

**Recommendation: Yes — the MCP server is the core delivery mechanism.**

### Rationale

The MCP server is the most natural integration point with Claude Code for several reasons:

1. **On-demand retrieval.** The agent can call `search_learnings(query)` at any point
   during a session when it needs past context. This is more flexible than pre-injecting
   a fixed set of learnings at session start — the agent can search for different things
   as the task evolves.

2. **Tool discoverability.** Claude Code automatically discovers MCP tools and can decide
   when to use them. With a good tool description, the agent will proactively search for
   relevant learnings when tackling unfamiliar code.

3. **State management.** The MCP server process persists for the session lifetime (stdio
   transport), so it can hold an open LanceDB connection and avoid repeated initialization.

4. **Standard distribution.** Users install the package and add one line to their MCP
   config. No custom Claude Code fork or plugin system needed.

5. **Portable.** MCP servers work with any MCP-compatible client, not just Claude Code.
   This future-proofs the project if other tools adopt MCP.

### What the MCP server alone can't do

- **Automatic ingestion** — the agent doesn't call tools after a session ends. We need
  hooks or a separate process for this.
- **Automatic context injection at startup** — the agent has to choose to call the tool.
  Hooks give us guaranteed injection.

**Conclusion:** The MCP server handles retrieval and management. Hooks handle automation
(ingestion triggers, startup injection). The CLI handles manual operations. All three are
part of the same Python package.

---

## Decision 2: Should the vector DB be configurable?

**Recommendation: No — ship with LanceDB as the only supported backend.**

### Rationale

1. **Embedded, no server process.** LanceDB runs in-process. Users don't need to install
   or manage a separate database. This is critical for a "drop-in" solution.

2. **Rust core, Python bindings.** Fast and well-maintained. The Rust foundation aligns
   with the project's values.

3. **Disk-native.** Data persists to disk automatically. No need for serialization or
   manual checkpointing.

4. **Metadata filtering.** Native support for filtering by project, category, timestamp —
   exactly what our retrieval needs.

5. **Simplicity.** One backend means one set of tests, one set of docs, one set of bugs.
   Making the DB configurable adds an abstraction layer, adapter interfaces, and test
   matrices that don't serve users.

### Migration path

If a compelling reason to support other backends arises, LanceDB interactions are isolated
behind a `StorageBackend` protocol class. This makes it possible to add alternatives later
without redesigning the system. But we don't build that abstraction until there's demand.

### Alternatives considered

| DB         | Pros                   | Cons                             |
|------------|------------------------|----------------------------------|
| ChromaDB   | Popular, Python-native | Heavier, server mode recommended |
| Qdrant     | Feature-rich           | Requires server process          |
| FAISS      | Very fast              | Low-level, no metadata support   |
| SQLite+vec | Minimal deps           | Less mature vector support       |

---

## Decision 3: Should the embedding model be configurable?

**Recommendation: Yes — configurable with a sensible default.**

### Rationale

Unlike the vector DB (which is an internal implementation detail), the embedding model
directly affects:

1. **Infrastructure requirements.** Some users have Ollama installed, others don't. Some
   are willing to use API-based embeddings, others want fully local.

2. **Quality vs. cost tradeoff.** Code-specific models (Voyage) outperform general models
   but require API keys and cost money. Users should choose.

3. **Existing setups.** Users who already run Ollama with specific models shouldn't be
   forced to download a new one.

### Supported models (initial release)

| Model                  | Provider | Type  | Dims | Notes                           |
|------------------------|----------|-------|------|---------------------------------|
| nomic-embed-text       | Ollama   | Local | 768  | **Default.** Good quality, free |
| voyage-code-3          | Voyage   | API   | 1024 | Best for code content           |
| text-embedding-3-small | OpenAI   | API   | 1536 | Widely available                |

### Configuration

```toml
# ~/.crowd-control/config.toml

[embedding]
# "ollama", "voyage", "openai"
provider = "ollama"
# Model name for the chosen provider
model = "nomic-embed-text"

# Only needed for API-based providers
[embedding.api]
# Name of env var holding API key
key_env = "VOYAGE_API_KEY"
```

### Implementation

A simple `Embedder` protocol with three implementations:

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
```

This is one of the few places where configurability is warranted because it has a real
impact on user experience and the interface is tiny (one method + one property).

---

## Decision 4: Will there need to be hooks/scripts?

**Recommendation: Yes — two hooks, installed automatically by `crowd-control setup`.**

### Hook 1: `Stop` — Trigger ingestion

**Event:** `Stop` (fires when Claude finishes responding)

**Purpose:** Signal that new session data is available for ingestion.

**Behavior:**
- Reads the session ID from stdin
- Writes a job to `~/.crowd-control/queue/` for async processing
- The MCP server (or a background worker) picks up queued jobs and processes them
- Fast (~10ms) — just writes a small file, no heavy processing in the hook itself

**Why not ingest synchronously?** Ingestion requires LLM API calls for distillation,
which takes seconds to minutes. A synchronous hook would block Claude Code.

### Hook 2: `SessionStart` — Inject warm context

**Event:** `SessionStart` (matcher: `startup`)

**Purpose:** Retrieve and inject relevant learnings at the start of a new session.

**Behavior:**
- Determines current project from `cwd`
- Queries LanceDB for recent, relevant learnings for this project
- Outputs a formatted context block to stdout (which Claude Code injects as context)
- Falls back gracefully if DB is empty or unavailable

**Output format:**
```
## Prior learnings from this codebase

The following insights were extracted from previous sessions. Treat as established context.

- [debugging_insight] The test suite requires PYTHONPATH=src to find modules...
- [architecture_decision] We use the repository pattern with SQLAlchemy...
- [gotcha] The CI pipeline caches pip dependencies by Python version...
```

### Hook 3 (optional): `SubagentStart` — Inject context for subagents

This could provide more targeted context based on the subagent's prompt, but may be
deferred to a later version. The MCP server's `search_learnings` tool covers this case
since subagents have access to MCP tools.

### Installation

`crowd-control setup` will:
1. Add the MCP server config to `~/.claude.json`
2. Add hooks to `~/.claude/settings.json`
3. Validate that prerequisites are met (Ollama installed, etc.)
4. Create the storage directory

---

## Decision 5: What language(s) should we use?

**Recommendation: Python only.**

### Rationale

1. **User preference.** You're familiar with Python, not JavaScript/TypeScript.

2. **Ecosystem fit.** Every dependency has first-class Python support:
   - `mcp` (Python MCP SDK with FastMCP)
   - `lancedb` (Python bindings over Rust core)
   - `ollama` (Python client)
   - `anthropic` (Python SDK for distillation)
   - `voyageai` / `openai` (Python SDKs for alternative embeddings)

3. **Development speed.** This is a tool/glue project — it orchestrates API calls, does
   I/O, and manages data. Python excels here. There's no CPU-bound inner loop that
   would benefit from Rust (LanceDB and Ollama handle the heavy computation).

4. **MCP server support.** The Python MCP SDK (FastMCP) is mature and well-documented.
   It handles protocol compliance, transport, and tool schema generation.

5. **Distribution.** `pip install crowd-control` or `uv pip install crowd-control`. The
   CLI entry point is defined in `pyproject.toml`.

### Python version

Target **Python 3.11+** for:
- `tomllib` in stdlib (for config parsing)
- Better `asyncio` support
- Broad compatibility (ships with macOS, most Linux distros)

---

## Decision 6: What features will the system provide?

### Core features (v0.1 — MVP)

1. **Session ingestion**
   - Parse Claude Code JSONL session transcripts
   - LLM-powered distillation to extract learnings (using Haiku for cost efficiency)
   - Embed learnings and store in LanceDB
   - Tag with project, category, languages, frameworks

2. **Learning retrieval**
   - Vector similarity search
   - Filter by project, category, tags, recency
   - Recency-weighted ranking (newer learnings rank higher)
   - Token budget packing (fit as many learnings as possible within a limit)

3. **MCP server tools**
   - `search_learnings(query, project?, category?, limit?)` — find relevant learnings
   - `ingest_session(session_path?)` — manually trigger ingestion
   - `add_learning(text, category, tags)` — manually add a learning
   - `status()` — show DB stats and system health

4. **Automatic hooks**
   - Post-session ingestion trigger
   - Pre-session context injection

5. **CLI**
   - `crowd-control setup` — configure hooks + MCP in Claude Code
   - `crowd-control ingest [path]` — manually ingest sessions
   - `crowd-control search <query>` — search from terminal
   - `crowd-control status` — show system status

6. **Configuration**
   - TOML config file at `~/.crowd-control/config.toml`
   - Embedding model selection
   - Distillation model selection
   - Token budget for context injection
   - Project-specific overrides

### Extended features (v0.2+)

7. **Staleness management**
   - Recency decay factor in ranking
   - Manual staleness marking
   - Automatic re-validation against current codebase (stretch goal)

8. **Learning management**
   - `list_learnings(filters)` — browse the knowledge base
   - `get_learning(id)` — view a specific learning with full metadata
   - `delete_learning(id)` — remove a learning
   - `export` / `import` — share learnings between machines

9. **Cross-encoder reranking**
   - Optional second-pass ranking with a cross-encoder model
   - Significantly improves retrieval precision at the cost of latency

10. **Deduplication**
    - Detect near-duplicate learnings across sessions
    - Merge or supersede older learnings with newer ones

11. **Analytics**
    - Track which learnings are retrieved most often
    - Track which learnings are retrieved but not useful (agent ignores them)
    - Surface under-utilized learnings

### Non-goals

- **Cloud sync.** This is a local-first tool. Syncing can be done by syncing the
  `~/.crowd-control/db/` directory with any file sync tool.
- **Multi-user.** Designed for a single developer's machine.
- **Real-time indexing.** Ingestion happens after sessions, not during.
- **Supporting non-Claude agents.** MCP is the interface, but the distillation and
  ingestion pipeline is designed around Claude Code session format.

---

## Decision 7: Retrieval should be fully configurable

**Recommendation: Expose all retrieval parameters in config, with sensible defaults.**

### Rationale

Different codebases and workflows produce learnings of varying quality and density.
Users need control over what gets surfaced.

### Configuration

```toml
[retrieval]
max_results = 15              # Maximum number of learnings to retrieve
max_tokens = 4000             # Token budget for context injection
min_similarity = 0.3          # Minimum cosine similarity threshold (0.0-1.0)
recency_decay = 0.95          # Score multiplier per week of age
project_boost = 1.5           # Multiplier for same-project matches
```

- **`min_similarity`** prevents low-quality matches from filling the context budget.
  Without this, a query with no good matches would still return `max_results` irrelevant
  learnings. Default of 0.3 is conservative — users can raise it if they want higher
  precision at the cost of recall.

- All parameters are optional in config. Missing values use defaults defined in code.

---

## Decision 8: Build tooling and PyPI publishing

**Recommendation: Use `uv` for development, `hatchling` as build backend.**

### Rationale

1. **`uv`** — Fast, modern Python package manager. Handles venv creation, dependency
   resolution, lockfiles, and building. Good DX for development.

2. **`hatchling`** — Lightweight, standards-compliant build backend. No runtime
   dependencies. Works with any installer (`pip`, `uv`, `pipx`).

3. **No lock-in.** End users install with `pip install crowd-control` or
   `uv pip install crowd-control`. They don't need `uv` installed.

### pyproject.toml build config

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Publishing workflow

```bash
uv build                   # Creates dist/crowd_control-*.whl and .tar.gz
uv publish                 # Uploads to PyPI (requires PYPI_TOKEN)
```

### Alternatives considered

| Tool       | Pros                          | Cons                              |
|------------|-------------------------------|-----------------------------------|
| Poetry     | Popular, lockfile             | Slower, non-standard build backend|
| Flit       | Very minimal                  | No lockfile, less active          |
| Setuptools | Universal                     | Verbose config, legacy feel       |
| PDM        | Standards-compliant           | Smaller community                 |
