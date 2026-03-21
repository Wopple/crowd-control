# User Guide

This guide covers everything you need to install, configure, use, and troubleshoot
Crowd Control.

## Table of Contents

- [Installation](#installation)
- [CLI Commands](#cli-commands)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Verifying Your Installation](#verifying-your-installation)
- [Debugging](#debugging)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- **Python 3.11+** — uses `tomllib` from the standard library
- **[Claude Code](https://claude.ai/claude-code)** — installed and authenticated (Crowd Control uses `claude -p` for distillation)
- **[Ollama](https://ollama.ai)** — for local embeddings (default provider)

### Step 1: Install Ollama

macOS (Homebrew):

```bash
brew install ollama
```

Or download from [ollama.ai](https://ollama.ai) for macOS, Linux, or Windows.

### Step 2: Pull the embedding model

```bash
ollama pull nomic-embed-text
```

This downloads the `nomic-embed-text` model (~274 MB), which Crowd Control uses to
convert learnings into vectors for semantic search.

### Step 3: Start Ollama

```bash
ollama serve
```

If you installed Ollama via the desktop app, it runs automatically — skip this step.
You can verify Ollama is running with:

```bash
curl -s http://localhost:11434/api/tags | head -c 200
```

You should see JSON output listing your installed models.

### Step 4: Install Crowd Control

```bash
pip install crowd-control[ollama]
```

Or with uv:

```bash
uv tool install crowd-control[ollama]
```

For development from source:

```bash
git clone https://github.com/daniel/crowd-control.git
cd crowd-control
uv sync
```

When developing from source, use `uv run crowd-control` instead of `crowd-control`.

### Step 5: Run setup

```bash
crowd-control setup
```

This configures everything automatically:

1. Creates the `~/.crowd-control/` directory
2. Writes a default `config.toml`
3. Registers the MCP server in Claude Code's config (`~/.claude.json`)
4. Registers the SessionEnd hook in Claude Code's settings (`~/.claude/settings.json`)

You should see output like:

```
Crowd Control configured successfully (global).

MCP server: /Users/you/.claude.json (crowd-control serve)
Hook:
  SessionEnd -> queues ingestion + spawns background worker

Storage: /Users/you/.crowd-control
Embedding: ollama/nomic-embed-text
```

### Alternative embedding providers

If you don't want to use Ollama, you can use a cloud embedding API instead. Install
the provider extra and configure it in `~/.crowd-control/config.toml`:

**Voyage AI:**

```bash
pip install crowd-control[voyage]
export VOYAGE_API_KEY="your-key-here"
```

```toml
[embedding]
provider = "voyage"
model = "voyage-code-3"
api_key_env = "VOYAGE_API_KEY"
```

**OpenAI:**

```bash
pip install crowd-control[openai]
export OPENAI_API_KEY="your-key-here"
```

```toml
[embedding]
provider = "openai"
model = "text-embedding-3-small"
api_key_env = "OPENAI_API_KEY"
```

> **Note:** Switching embedding providers after you've already ingested sessions
> requires deleting and re-creating the database, because vector dimensions differ
> between models. See [Troubleshooting](#dimension-mismatch-after-switching-models).

---

## CLI Commands

### `crowd-control setup`

Configures the MCP server and SessionEnd hook in Claude Code.

```bash
crowd-control setup            # Global (all projects)
crowd-control setup --project  # Current project only
```

| Option | Description |
|--------|-------------|
| `--project` | Write config to `.mcp.json` and `.claude/settings.json` in the current directory instead of the home directory |

Setup is idempotent — running it again updates existing entries without creating
duplicates. Other MCP servers and hooks in your config are left untouched.

---

### `crowd-control ingest [PATH]`

Runs the full ingestion pipeline on a session transcript: parse, distill (via Claude),
embed, and store.

```bash
crowd-control ingest                    # Ingest most recent session for current project
crowd-control ingest /path/to/session.jsonl
crowd-control ingest --dry-run          # Preview session structure without storing
crowd-control ingest --concurrency 4    # Limit parallel distillation requests
```

| Option | Default | Description |
|--------|---------|-------------|
| `PATH` | most recent session | Path to a `.jsonl` session transcript |
| `--dry-run` | off | Parse and display session structure without distilling or storing |
| `--concurrency` | `8` | Maximum parallel `claude -p` distillation requests |

Output on success:

```
Ingested session abc123:
  Segments processed: 5
  Learnings distilled: 12
  Learnings stored: 10
  Duplicates skipped: 2
```

The `--dry-run` flag is useful for inspecting a session before committing to
distillation (which costs API tokens):

```
Session: abc123
Project: /Users/you/code/myapp
Branch:  main
Model:   claude-sonnet-4-20250514
Period:  2026-03-15T10:00:00Z → 2026-03-15T11:30:00Z
Messages: 48 parsed, 42 in segments

Segments (3):
  [1] 10:00:15 — 10:25:30  (14 messages, tools: Read, Edit, Bash)
      User: "Fix the login timeout bug"
  [2] 10:26:00 — 10:50:00  (18 messages, tools: Read, Grep, Edit)
      User: "Now add rate limiting to the auth endpoint"
  [3] 10:51:00 — 11:28:00  (10 messages, tools: Bash)
      User: "Run the test suite and fix any failures"
```

---

### `crowd-control search <QUERY>`

Searches stored learnings by semantic similarity.

```bash
crowd-control search "how does the auth system work"
crowd-control search "debugging database connections" --limit 5
crowd-control search "test patterns" --category codebase_convention
crowd-control search "deployment" --project /Users/you/code/webapp
```

| Option | Default | Description |
|--------|---------|-------------|
| `QUERY` | *(required)* | Natural language search query |
| `--limit` | from config (15) | Maximum results to return |
| `--project` | current directory | Filter to a specific project path |
| `--category` | all | Filter by learning category |

Output:

```
Searching for: "how does the auth system work"

  [1] (score=0.87) [architecture_decision]
      The auth system uses JWT tokens stored in HttpOnly cookies...
      project=/users/you/code/webapp  retrieved=5x  age=3d

  [2] (score=0.72) [debugging_insight]
      Auth middleware must be registered before the CORS middleware...
      project=/users/you/code/webapp  retrieved=2x  age=7d

2 results (searched 142 learnings)
```

Learning categories: `architecture_decision`, `debugging_insight`, `pattern_discovery`,
`tool_usage`, `codebase_convention`, `gotcha`.

---

### `crowd-control list`

Lists stored learnings with optional filtering.

```bash
crowd-control list
crowd-control list --project /Users/you/code/webapp
crowd-control list --category gotcha
crowd-control list --limit 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--project` | all projects | Filter by project path |
| `--category` | all | Filter by learning category |
| `--limit` | `50` | Maximum learnings to display |

---

### `crowd-control status`

Displays database path, learning count, and embedding configuration.

```bash
crowd-control status
```

Output:

```
Database: /Users/you/.crowd-control/db
Learnings: 142
Embedding: ollama/nomic-embed-text
```

If the database hasn't been created yet (no ingestions), you'll see
"Database not initialized" — this is normal.

---

### `crowd-control export`

Exports all learnings as JSON.

```bash
crowd-control export                          # Print to stdout
crowd-control export -o learnings.json        # Write to file
crowd-control export --project /path/to/proj  # Filter by project
crowd-control export --category debugging_insight
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | stdout | Output file path |
| `--project` | all projects | Filter by project path |
| `--category` | all | Filter by learning category |

The output is a JSON object with `version`, `exported_at`, `count`, and `learnings`
fields.

---

### `crowd-control worker`

Processes queued ingestion jobs. Normally spawned automatically by the SessionEnd hook,
but can be run manually to retry failed jobs.

```bash
crowd-control worker
```

The worker:

1. Scans `~/.crowd-control/queue/` for pending jobs
2. Processes each job (parse → distill → embed → store)
3. Removes the queue file on success
4. Retries up to 3 times, then moves failed jobs to `~/.crowd-control/queue/failed/`

---

### `crowd-control prune`

Removes old learnings with low retrieval activity.

```bash
crowd-control prune              # Delete old inactive learnings
crowd-control prune --dry-run    # Preview what would be pruned
```

Uses `max_age_days` (default 90) and `retention_retrieval_interval_days` (default 30)
from the `[ingestion]` config section. A learning older than `max_age_days` must have
been retrieved at least once per interval to survive. For example, a 120-day-old
learning with a 30-day interval needs 4 retrievals to be kept.

This also runs automatically after each ingestion and on MCP server startup, so manual
pruning is rarely needed.

---

### `crowd-control serve`

Starts the MCP server using stdio transport. This is called by Claude Code automatically
— you don't need to run it yourself.

```bash
crowd-control serve
```

The MCP server exposes four tools to the agent:

| Tool | Description |
|------|-------------|
| `search_learnings` | Semantic search over stored learnings |
| `add_learning` | Manually store a learning during a session |
| `ingest_session` | Trigger full ingestion pipeline from within a session |
| `status` | Show database stats and configuration |

---

### `crowd-control hook session-end`

Handles the SessionEnd hook event from Claude Code. Reads a JSON payload from stdin,
writes a queue file, and spawns a background worker. This is called by Claude Code
automatically — you don't need to run it yourself.

---

### Global options

```bash
crowd-control --version     # Show version
crowd-control --help        # Show all commands
crowd-control -v <command>  # Verbose mode (debug output on stderr)
```

---

## Configuration

Configuration lives in `~/.crowd-control/config.toml`. All fields are optional — missing
values use the defaults. The file is created automatically by `crowd-control setup`.

### `[general]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `storage_dir` | string | `"~/.crowd-control"` | Root directory for database, queue, and logs |
| `log_level` | string | `"off"` | Trace logging: `"off"`, `"debug"`, `"info"`, `"warning"`, `"error"`. When not `"off"`, logs are written to `<storage_dir>/logs/crowd-control.log` |

### `[knowledge]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scope` | string | `"project"` | Knowledge scope for retrieval (see below) |

Scope options:

- **`project`** — Learnings are scoped to their source project. Search only returns
  learnings from the current project.
- **`shared`** — All learnings in one pool. Search returns everything regardless of
  project.
- **`mixed`** — *(v0.2+)* Learnings are classified as project-specific or universal during
  distillation. Search returns current-project learnings plus universal ones.

### `[embedding]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | string | `"ollama"` | Embedding provider: `"ollama"`, `"voyage"`, or `"openai"` |
| `model` | string | `"nomic-embed-text"` | Model name for the provider |
| `api_key_env` | string | *(none)* | Environment variable name for API key (API providers only) |

Provider comparison:

| Provider | Default Model | Dimensions | Requires | Cost |
|----------|--------------|------------|----------|------|
| `ollama` | `nomic-embed-text` | 768 | Ollama running locally | Free |
| `voyage` | `voyage-code-3` | 1024 | `VOYAGE_API_KEY` env var | API pricing |
| `openai` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` env var | API pricing |

### `[distillation]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"haiku"` | Claude model alias for distillation (passed to `claude -p --model`) |
| `max_learnings_per_session` | int | `20` | Maximum learnings extracted from a single session |

### `[retrieval]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_results` | int | `15` | Maximum learnings returned per search |
| `max_tokens` | int | `4000` | Token budget for packed results |
| `min_similarity` | float | `0.3` | Minimum cosine similarity threshold (0.0–1.0) |
| `recency_half_life_days` | float | `7.0` | Exponential decay half-life in days |
| `hotness_weight` | float | `0.2` | Blend weight: 0.0 = pure semantic, 1.0 = pure hotness |
| `project_boost` | float | `1.5` | Multiplicative boost for same-project results in non-project scopes |

Tuning tips:

- **`min_similarity`** — Raise if you're getting noisy/irrelevant results. Lower if
  searches return too few results. Enable trace logging to see rejected similarity scores.
- **`recency_half_life_days`** — At 7 days, a learning loses half its recency score each
  week. Increase for long-lived projects where old learnings stay relevant.
- **`hotness_weight`** — At 0.2, semantic similarity dominates (80%) but frequently-used
  learnings get a meaningful boost. Set to 0.0 for pure semantic search.
- **`project_boost`** — Only applies in `shared` or `mixed` scope. Boosts learnings from
  the current project over learnings from other projects.

### `[ingestion]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_ingest` | bool | `true` | Automatically ingest sessions via SessionEnd hook |
| `batch_size` | int | `5` | Embedding batch size |
| `dedup_threshold` | float | `0.90` | Cosine similarity threshold for near-duplicate rejection |
| `max_age_days` | int | `90` | Delete learnings older than this (0 = never prune) |
| `retention_retrieval_interval_days` | int | `30` | Must be retrieved once per this many days to survive past TTL |

With auto-ingestion disabled, the SessionEnd hook will not queue sessions. Use
`crowd-control ingest` to ingest manually.

Pruning runs automatically after each ingestion and on MCP server startup. Learnings
older than `max_age_days` must have been retrieved at least once per
`retention_retrieval_interval_days` — the required count scales with age (e.g., 90 days
old / 30 day interval = 3 retrievals needed). Set `max_age_days = 0` to disable pruning.

### Configuration examples

**Switching to Voyage AI:**

```toml
[embedding]
provider = "voyage"
model = "voyage-code-3"
api_key_env = "VOYAGE_API_KEY"
```

**Increasing token budget for large projects:**

```toml
[retrieval]
max_results = 25
max_tokens = 8000
```

**Disabling auto-ingestion:**

```toml
[ingestion]
auto_ingest = false
```

**Enabling trace logging:**

```toml
[general]
log_level = "debug"
```

Then check `~/.crowd-control/logs/crowd-control.log` after running commands.

**Important:** Switching embedding models requires re-creating the database. Vector
dimensions are fixed at table creation. Back up and delete `~/.crowd-control/db/`, then
re-ingest your sessions. See [Troubleshooting](#dimension-mismatch-after-switching-models).

### Full default config

```toml
[general]
storage_dir = "~/.crowd-control"
log_level = "off"

[knowledge]
scope = "project"

[embedding]
provider = "ollama"
model = "nomic-embed-text"
# api_key_env = "VOYAGE_API_KEY"

[distillation]
model = "haiku"
max_learnings_per_session = 20

[retrieval]
max_results = 15
max_tokens = 4000
min_similarity = 0.4
min_score = 0.4
recency_half_life_days = 7
hotness_weight = 0.2
project_boost = 1.5

[ingestion]
auto_ingest = true
batch_size = 5
dedup_threshold = 0.90
max_age_days = 90
retention_retrieval_interval_days = 30
```

---

## How It Works

### Automatic flow (after setup)

1. You use Claude Code normally in a project
2. When the session ends, the **SessionEnd hook** fires automatically
3. The hook writes a queue file and spawns a **background worker**
4. The worker parses the session transcript, sends segments to **Claude Haiku** for
   distillation, embeds the extracted learnings via **Ollama**, and stores them in
   **LanceDB**
5. In your next session, the agent calls **`search_learnings`** via the MCP server to
   find relevant past insights

You don't need to do anything after setup — the learning loop is fully automatic.

### What gets extracted

The distiller produces six categories of learnings:

| Category | Description |
|----------|-------------|
| `architecture_decision` | Design choices, trade-offs, structural decisions |
| `debugging_insight` | Root causes, diagnostic techniques, fix strategies |
| `pattern_discovery` | Recurring patterns, idioms, approaches that work |
| `tool_usage` | Tool configurations, flags, workflows that were useful |
| `codebase_convention` | Project-specific conventions, naming, file organization |
| `gotcha` | Pitfalls, surprises, things that don't work as expected |

Generic programming knowledge is filtered out — only project-specific insights are kept.

### File layout

```
~/.crowd-control/
├── config.toml              # Your configuration
├── db/                      # LanceDB vector database
├── queue/                   # Pending ingestion jobs
│   ├── <session_id>.json    # Queued sessions
│   └── failed/              # Jobs that failed 3+ times
└── logs/
    ├── worker.err           # Worker stderr (always written)
    └── crowd-control.log    # Trace log (when log_level != "off")
```

---

## Verifying Your Installation

Work through these checks after setup to confirm everything is working.

### 1. Check Ollama is running

```bash
ollama list
```

You should see `nomic-embed-text` in the output. If you get a connection error, start
Ollama with `ollama serve` or launch the desktop app.

### 2. Check the CLI

```bash
crowd-control --version
crowd-control status
```

If the database hasn't been initialized yet, "Database not initialized" is expected.

### 3. Check Claude Code configuration

Verify the MCP server is registered:

```bash
cat ~/.claude.json
```

Look for a `crowd-control` entry under `mcpServers`:

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

Verify the SessionEnd hook is registered:

```bash
cat ~/.claude/settings.json
```

Look for a `crowd-control hook session-end` entry under `hooks.SessionEnd`.

### 4. Run a manual ingestion

If you have an existing Claude Code session, ingest it to seed the database:

```bash
crowd-control ingest --dry-run   # Preview first
crowd-control ingest             # Ingest most recent session
```

Then verify it was stored:

```bash
crowd-control status   # Should show learning count > 0
crowd-control list     # Should show the extracted learnings
```

### 5. Test search

```bash
crowd-control search "what patterns are used in this project"
```

You should see ranked results with scores, categories, and ages.

### 6. Verify the MCP server in Claude Code

Start a new Claude Code session. The MCP server should connect automatically. Ask Claude
to run `search_learnings` to verify the tool is available and returns results.

---

## Debugging

### Enable trace logging

The most powerful debugging tool is trace logging. Enable it in your config:

```toml
[general]
log_level = "debug"
```

Then check the log file after running commands:

```bash
cat ~/.crowd-control/logs/crowd-control.log
```

The trace log includes:

- Embedding batch sizes and timings
- Similarity scores for search results (including rejected ones below `min_similarity`)
- Scoring breakdown (semantic similarity, recency, hotness, final blended score)
- Deduplication decisions
- Token packing details
- Distillation prompt lengths and response parsing

### Use verbose mode

For interactive debugging, add `-v` to any command:

```bash
crowd-control -v search "my query"
crowd-control -v ingest --dry-run
crowd-control -v status
```

This prints debug output to stderr alongside the normal output.

### Check worker logs

The background worker writes stderr to a file (since it runs detached):

```bash
cat ~/.crowd-control/logs/worker.err
```

This captures errors from the automatic ingestion pipeline — distillation failures,
embedding errors, storage issues.

### Check the ingestion queue

See what's pending, in progress, or failed:

```bash
ls ~/.crowd-control/queue/           # Pending jobs
ls ~/.crowd-control/queue/failed/    # Failed jobs (3+ attempts)
```

Each queue file is a small JSON with `session_id`, `session_path`, `project`, and
`queued_at`. Failed jobs also have an `attempts` count.

To retry failed jobs, move them back to the queue and run the worker:

```bash
mv ~/.crowd-control/queue/failed/*.json ~/.crowd-control/queue/
crowd-control worker
```

### Inspect distillation quality

Use `--dry-run` to see what a session looks like before distillation:

```bash
crowd-control ingest --dry-run /path/to/session.jsonl
```

Then ingest and review what was extracted:

```bash
crowd-control ingest /path/to/session.jsonl
crowd-control list --limit 20
```

If learnings are low quality, consider:

- Switching the distillation model (`[distillation] model` in config)
- Checking that sessions are long enough to contain useful insights
- Reviewing the `confidence` scores in the list output

### Export for analysis

Export all learnings for offline analysis or backup:

```bash
crowd-control export -o backup.json
crowd-control export --project /path/to/project -o project-learnings.json
```

---

## Troubleshooting

### `crowd-control: command not found`

The package isn't on your PATH. Common fixes:

- If installed with `pip install --user`, add `~/.local/bin` to your PATH
- If installed in a virtualenv, activate it first
- Try `python -m crowd_control` as an alternative
- If using uv: `uvx crowd-control` or `uv run crowd-control`

### Ollama connection errors

```
Embedding provider error: ...
Is your embedding provider (ollama) running?
```

1. Check if Ollama is running: `ollama list`
2. If not, start it: `ollama serve` (or launch the desktop app)
3. Verify the model is installed: `ollama list` should show `nomic-embed-text`
4. If the model is missing: `ollama pull nomic-embed-text`

### MCP server not connecting in Claude Code

1. Check the config exists: `cat ~/.claude.json`
2. Look for `crowd-control` under `mcpServers`
3. If missing, run `crowd-control setup` again
4. Restart Claude Code after setup (the MCP config is read at startup)
5. Verify the `crowd-control` binary is on the PATH that Claude Code uses

### Sessions not being ingested automatically

1. Check that `auto_ingest` is `true` in `~/.crowd-control/config.toml`
2. Check the hook is registered: look for `crowd-control hook session-end` in
   `~/.claude/settings.json`
3. Check worker logs: `cat ~/.crowd-control/logs/worker.err`
4. Check the queue: `ls ~/.crowd-control/queue/`
5. Try manual ingestion: `crowd-control ingest`

### Dimension mismatch after switching models

```
Embedding dimension mismatch. Table has 768-dim vectors but embedder produces 1024-dim.
```

Vector dimensions are fixed when the database is first created. Switching embedding
models requires re-creating the database:

```bash
# Back up existing data
crowd-control export -o backup.json
cp -r ~/.crowd-control/db ~/.crowd-control/db.bak

# Delete and re-create
rm -rf ~/.crowd-control/db

# Re-ingest your sessions
crowd-control ingest
```

### Invalid config.toml

```
Invalid TOML in /Users/you/.crowd-control/config.toml: ...
```

Fix the syntax error in your config file, or delete it and run `crowd-control setup`
to regenerate the defaults.

### Search returns no results

1. Check that you have ingested learnings: `crowd-control status`
2. If count is 0, run `crowd-control ingest` first
3. If count is > 0 but search finds nothing, your query may not be similar enough.
   Try broader terms or lower `min_similarity` in config
4. Check knowledge scope — if set to `project`, search only returns learnings from
   the current directory's project. Try `crowd-control search "query" --project /path`
   or switch to `shared` scope in config

### Distillation failures

Distillation calls `claude -p` as a subprocess. Common issues:

1. **Claude Code not installed:** The `claude` CLI must be on your PATH
2. **Not authenticated:** Run `claude` interactively to authenticate first
3. **Running inside Claude Code:** Distillation cannot run from within a Claude Code
   session (it detects the `CLAUDECODE` environment variable). Use the SessionEnd hook
   or run `crowd-control ingest` from a regular terminal
4. **Rate limits:** Reduce `--concurrency` if you see retry storms with many segments
