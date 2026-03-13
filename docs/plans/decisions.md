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
# api_key_env = "VOYAGE_API_KEY"
```

### Implementation

A simple `Embedder` protocol with three implementations:

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def max_input_chars(self) -> int: ...
```

Each provider knows its own input limit:

| Model                  | Max tokens | Approx max chars |
|------------------------|------------|------------------|
| nomic-embed-text       | 8192       | 32000            |
| voyage-code-3          | 16000      | 64000            |
| text-embedding-3-small | 8191       | 32000            |

The `max_input_chars` property is used by the distillation prompt to tell the LLM how
large each learning can be. This keeps the distiller and embedder loosely coupled — the
distiller doesn't need to know which embedding model is configured, it just asks for the
limit.

This is one of the few places where configurability is warranted because it has a real
impact on user experience and the interface is tiny (two methods + two properties).

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
   - `ollama` (Python client for local embeddings)
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
   - LLM-powered distillation to extract learnings (via Claude Code CLI)
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

12. **Pipeline telemetry**
    - Structured logging at each pipeline stage (parsing, distillation, embedding, retrieval)
    - Captures input sizes, truncation rates, learning counts, confidence distributions,
      call durations, similarity scores, token budget usage
    - Enables post-hoc analysis to identify tuning opportunities and implementation weaknesses
    - See Decision 11 for full details

13. **Mixed knowledge scope**
    - Distiller classifies learnings as project-specific or universal
    - Retrieval merges project-scoped learnings with the universal pool
    - Requires tuning the classification prompt to avoid over/under-sharing

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
recency_half_life_days = 7    # Exponential decay half-life (days)
hotness_weight = 0.2          # Blend weight: 0.0 = pure semantic, 1.0 = pure hotness
project_boost = 1.5           # Multiplier for same-project matches
```

- **`min_similarity`** prevents low-quality matches from filling the context budget.
  Without this, a query with no good matches would still return `max_results` irrelevant
  learnings. Default of 0.3 is conservative — users can raise it if they want higher
  precision at the cost of recall.

- **`recency_half_life_days`** controls how fast old learnings decay. At the default
  of 7, a learning loses half its recency score each week. This replaces the earlier
  `recency_decay = 0.95` (linear multiplier per week) with exponential decay, which
  is more principled and gives a single intuitive parameter. Inspired by OpenViking's
  `memory_lifecycle.py`.

- **`hotness_weight`** controls how much usage frequency (active count) influences
  ranking vs pure semantic similarity. At 0.2, semantic similarity dominates (80%)
  but frequently-retrieved learnings get a meaningful boost. Inspired by OpenViking's
  `hierarchical_retriever.py`.

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

---

## Decision 9: Distillation via Claude Code CLI, not the Anthropic API

**Recommendation: Use `claude -p` (Claude Code's non-interactive mode) as the only distillation backend.**

### Rationale

Crowd Control is a Claude Code extension. Every user already has Claude Code installed with an
active subscription. Requiring a separate Anthropic API key for distillation would be bad
ergonomics — it adds setup friction and a second billing relationship for a tool that's supposed
to be drop-in.

### How it works

Claude Code's CLI supports non-interactive mode:

```bash
claude -p "prompt" \
  --model haiku \
  --output-format json \
  --json-schema '{"type":"object",...}' \
  --no-session-persistence \
  --max-budget-usd 0.05
```

Key flags:
- `--print` (`-p`): non-interactive mode, reads prompt from argument or stdin
- `--model`: select the model (use `haiku` for cost-effective distillation)
- `--output-format json`: structured JSON response
- `--json-schema`: enforce output schema for reliable parsing into `Learning` objects
- `--no-session-persistence`: don't save the distillation call as a session (avoid recursion)
- `--max-budget-usd`: safety cap per distillation call

### Implementation

The distiller shells out to the `claude` CLI as a subprocess:

```python
import subprocess, json

result = subprocess.run(
    ["claude", "-p", prompt,
     "--model", "haiku",
     "--output-format", "json",
     "--json-schema", schema_json,
     "--no-session-persistence"],
    capture_output=True, text=True, timeout=120
)
learnings = json.loads(result.stdout)
```

### Trade-offs

| Aspect | Claude Code CLI | Anthropic API (not implemented) |
|--------|----------------|--------------------------------|
| Setup | Zero — already installed | Requires API key |
| Billing | User's existing subscription | Separate API billing |
| Structured output | `--json-schema` flag | Native tool_use / JSON mode |
| Model selection | `--model haiku` | Full model ID |
| Rate limits | Shared with interactive use | Separate API limits |
| Latency | Process spawn overhead (~1s) | Direct HTTP, lower latency |

The CLI approach is the right default. Direct API access could be added later as an optional
provider for users who want lower latency or separate rate limits, but it is not in scope for
MVP.

---

## Decision 10: Project-scoped vs shared knowledge

**Recommendation: Configurable, default to project-scoped.**

### Rationale

Not all learnings are project-specific. A debugging technique for async Python applies everywhere.
But most learnings *are* project-specific — codebase conventions, architecture decisions, gotchas
tied to particular dependencies. Mixing project-specific learnings from unrelated codebases adds
noise to retrieval.

### Design

A `knowledge.scope` config option with three values:

- **`project`** (default): retrieval filters by the current project. Only learnings from sessions
  in this project are surfaced. Clean separation, no cross-contamination.
- **`shared`**: all learnings go into a single pool. Retrieval searches everything. The
  `project_boost` config parameter up-ranks matches from the current project while still
  surfacing cross-project results.
- **`mixed`** (v0.2+): the distiller classifies each learning as project-specific or universal
  during extraction. Project-specific learnings are scoped to their source project. Universal
  learnings go into a shared pool. At retrieval time, both the current project's learnings and
  the shared pool are searched and merged. See "Mixed scope design" below.

### Mixed scope design (v0.2+)

In `mixed` mode, the distiller makes a per-learning decision about whether an insight is
project-specific or broadly applicable. This requires changes to the distillation prompt
(Phase 2) and retrieval logic (Phase 4):

**Distillation:** the distillation prompt asks the LLM to classify each learning as
`project_specific` or `universal`. The classification is based on whether the insight depends
on the particular codebase (conventions, architecture, dependency quirks) or is transferable
(debugging techniques, tool usage patterns, language idioms that aren't common knowledge).

**Data model:** the `Learning` model gains a `shared: bool` field (default `False`). When
`shared` is `True`, the learning is visible to all projects during retrieval.

**Retrieval:** queries search for `(project == current_project) OR (shared == True)`. This
gives each project its own learnings plus the universal pool, without cross-contamination
between unrelated projects.

**Why defer to v0.2+:** mixed mode depends on the distiller (Phase 2) being implemented and
tuned. The classification prompt needs iteration — an overly aggressive classifier will dump
project-specific noise into the shared pool, while an overly conservative one makes mixed
mode equivalent to project mode. It's better to ship `project` and `shared` first, get real
usage data, then build `mixed` on top with good examples of what "universal" means in practice.

### Data model impact

For `project` and `shared` modes: none. The `project` field on `Learning` is always set to
the source project path. Scope only changes retrieval query construction. Users can switch
between these modes without re-ingesting.

For `mixed` mode: adds a `shared: bool` field to `Learning`. Existing learnings default to
`shared=False` (project-scoped), so switching to `mixed` mode is backwards compatible — old
learnings behave as project-scoped, and new learnings get classified going forward.

### Configuration

```toml
[knowledge]
scope = "project"   # "project", "shared", or "mixed" (v0.2+)
```

---

## Decision 11: Pipeline telemetry logging

**Recommendation: Log structured telemetry at each pipeline stage for post-hoc analysis and tuning.**

### Rationale

The pipeline has several stages where quality is hard to evaluate in real time: parsing,
truncation, distillation prompt construction, LLM extraction, embedding, retrieval ranking.
Getting these right requires iteration based on real data. Without telemetry, the only way
to identify weaknesses is to re-run sessions manually and inspect output — which doesn't
scale.

Structured logs let us answer questions like:
- How often are tool results being truncated? By how much?
- How large are segments before and after truncation?
- How many learnings is the distiller extracting per segment? Per session?
- Are certain segments producing zero learnings consistently?
- What's the confidence distribution of extracted learnings?
- How long are distillation calls taking?
- How often does retrieval hit the similarity floor?
- What fraction of the token budget is being used at injection time?

### What to log

**Parsing stage:**
- Session file size (bytes) and line count
- Lines skipped (by type: progress, file-history-snapshot, etc.)
- Messages after filtering
- Number of segments produced
- Per-segment: message count, tool result count, truncation count, chars before/after truncation

**Distillation stage:**
- Per-segment: prompt size (chars), segment text size before/after truncation
- Per-segment: number of learnings extracted, their categories and confidence scores
- Per-segment: distillation call duration, model used
- Per-segment: whether the LLM returned an empty list (no learnings)
- Per-session: total learnings before/after the max_learnings cap
- Errors: timeouts, parse failures, validation failures (with the invalid data)

**Embedding stage:**
- Per-learning: text length, embedding duration
- Batch sizes

**Retrieval stage:**
- Query text length
- Number of results before/after similarity filtering
- Similarity score distribution of results
- Token budget usage (how full is the context injection)
- Recency decay impact (how much scores changed)

### Implementation approach

Use Python's `logging` module with a dedicated logger (`crowd_control.telemetry` or similar).
Log telemetry at `INFO` level as structured key-value pairs so they can be parsed
programmatically. Normal operational logs (errors, warnings, progress) use the standard
`crowd_control` logger.

### Configuration

Logging is **off by default**. Most users won't need it and we don't want to create log
files on their system unprompted.

```toml
[general]
# "off" = no logging (default), "file" = log to ~/.crowd-control/logs/
log_level = "off"
```

When enabled, logs write to `~/.crowd-control/logs/`. The `log_level` setting controls
verbosity (`off`, `error`, `warning`, `info`, `debug`). Telemetry data is logged at
`debug` level so users can enable basic error/warning logging without getting flooded
with metrics.

Not in scope for MVP — this is a post-v0.1 feature. The initial implementation should use
Python's `logging` module at key points so the infrastructure is in place. With `log_level`
defaulting to `off`, no handlers are attached and nothing is written to disk.

---

## Decision 12: Retrieval scoring inspired by OpenViking

**Recommendation: Adopt OpenViking's hotness scoring and exponential decay, skip its
LLM-dependent features.**

### Context

OpenViking (github.com/volcengine/OpenViking) is an open-source context database for AI
agents by ByteDance. After a detailed comparison (see `openviking-learnings.md`), we
identified specific algorithms worth adopting and others that don't fit our constraints.

### What we adopt

**1. Hotness scoring** — OpenViking tracks `active_count` (how often each context record
is retrieved) and combines it with recency into a single score:

```
hotness = sigmoid(log1p(active_count)) * exp(-ln(2) / half_life_days * age_days)
```

This is better than pure recency decay because a frequently-used learning from 2 weeks
ago should rank higher than a never-used learning from yesterday. The `sigmoid(log1p(...))`
compression prevents a small number of heavily-used learnings from dominating.

**2. Score blending** — Final ranking blends semantic similarity with hotness:

```
final = (1 - hotness_weight) * semantic + hotness_weight * hotness
```

With `hotness_weight = 0.2`, semantic similarity dominates but usage signals matter.

**3. Exponential recency decay** — `exp(-ln(2) / half_life * age)` with a 7-day half-life
replaces the earlier `0.95^weeks` linear multiplier. Advantages: more principled, faster
decay for stale learnings, single intuitive parameter.

### What we skip (and why)

| Feature | Why not |
|---|---|
| L0/L1/L2 hierarchy | Requires LLM API for summary generation |
| LLM-powered deduplication | Requires async LLM API; cosine 0.95 sufficient at current scale |
| Intent analysis | Requires LLM API; query string + project filtering sufficient |
| Hierarchical directory retrieval | Depends on L0/L1/L2 existing |
| Virtual filesystem (AGFS) | Massive complexity for flat-learning case |

### Why not adopt OpenViking as a platform

OpenViking is a multi-tenant platform designed for teams building agent products. Using
it as a backend for Crowd Control would require:

- A separate LLM API key (OpenViking's memory extraction, dedup, intent analysis, and
  L0/L1 generation all need an async LLM client — incompatible with our `claude -p` approach)
- Running a Go subprocess (AGFS) for its virtual filesystem
- ~1000+ lines of adapter code for Claude Code integration
- Accepting that most platform improvements (multi-tenancy, chat channels, bot framework)
  would be irrelevant to our use case

The algorithms are valuable; the platform coupling is not. See `openviking-learnings.md`
for the full comparison.

### Implementation impact

- Add `active_count: int` to `Learning` model and LanceDB schema
- Implement scoring formulas in `retrieve/rank.py`
- Replace `recency_decay` config with `recency_half_life_days` and `hotness_weight`
- Increment active counts in MCP search tool and hook context injection
