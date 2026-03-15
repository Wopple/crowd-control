# Hooks and Automation

Hooks automate the learning loop: sessions are ingested automatically when they end,
and the agent finds relevant learnings during sessions via the MCP server.

## SessionEnd Hook

When a Claude Code session terminates (exit, clear, logout), the `SessionEnd` hook
fires. It writes a queue file and spawns a background worker to process it.

### Why SessionEnd, not Stop

The `Stop` hook fires after *every* Claude response — potentially dozens of times per
session. This causes partial ingestion and race conditions. `SessionEnd` fires once
when the session actually terminates, eliminating these problems.

### What happens

1. Claude Code fires `SessionEnd` with a JSON payload on stdin
2. `crowd-control hook session-end` reads the payload
3. Validates: required fields present, `auto_ingest` enabled, transcript file exists
4. Writes a queue file to `~/.crowd-control/queue/<session_id>.json`
5. Spawns `crowd-control worker` as a detached background process
6. Exits immediately (does not wait for the worker)

The hook payload includes `transcript_path` — the exact JSONL file for the ended
session. This is more reliable than searching for session files.

### Queue file format

```json
{
  "session_id": "abc123",
  "session_path": "/Users/dan/.claude/projects/-Users-dan-code-webapp/abc123.jsonl",
  "project": "/Users/dan/code/webapp",
  "queued_at": "2026-03-14T10:30:00+00:00"
}
```

### Error handling

The hook never blocks Claude Code. All errors are logged to stderr and the hook
exits 0. If the worker fails to spawn, the queue file persists for manual retry.

## Background Worker

The worker processes queued ingestion jobs. It is normally auto-spawned by the
SessionEnd hook, but can also be run manually.

```bash
crowd-control worker   # Process all queued jobs, then exit
```

### Processing behavior

1. Lists all `.json` files in `~/.crowd-control/queue/`
2. Sorts by `queued_at` (oldest first)
3. For each queue file:
   - Skips if session file no longer exists (deletes queue file)
   - Skips if session already ingested (checks DB by session_id)
   - Runs the full ingestion pipeline (parse, distill, embed, store)
   - Deletes queue file on success
4. On failure: increments attempt count in queue file
5. After 3 failures: moves queue file to `~/.crowd-control/queue/failed/`

### CLAUDECODE environment variable

The worker calls `claude -p` for distillation. Claude Code sets a `CLAUDECODE`
environment variable that prevents recursive invocation. The SessionEnd hook strips
this variable from the spawned worker's environment. This is safe because the session
has already ended and the worker is fully detached.

### Concurrency

Multiple workers can run simultaneously (e.g., two sessions end in quick succession).
This is safe: LanceDB handles concurrent access, and the `has_session()` check
prevents double-ingestion.

## Why No SessionStart Hook

The original architecture planned a `SessionStart` hook to inject learnings at session
start. This was removed because:

1. **The hook has no prompt.** It only receives `cwd` and `branch` — not enough for
   a meaningful search query.
2. **The agent has the prompt.** When the user types their first message, the agent
   can craft a precise `search_learnings` query.
3. **The agent can search multiple times.** As the task evolves, it can search for
   different topics.
4. **Wasted context budget.** Irrelevant learnings injected at startup consume tokens
   for the entire session.

Instead, the MCP server's instructions guide the agent to search proactively. See
`docs/mcp-server.md` for the instruction text.

## Setup

`crowd-control setup` configures everything automatically:

```bash
crowd-control setup            # Global (all projects)
crowd-control setup --project  # Current project only
```

### What setup configures

**MCP server** — adds `crowd-control serve` to Claude Code's MCP config:
- Global: `~/.claude.json`
- Project: `.mcp.json`

**SessionEnd hook** — adds `crowd-control hook session-end` to hook config:
- Global: `~/.claude/settings.json`
- Project: `.claude/settings.json`

### Merge behavior

Setup preserves existing configuration. It only adds or updates crowd-control entries.
Other MCP servers and hooks are left untouched. Running setup multiple times is safe
(idempotent).

### Default config

If `~/.crowd-control/config.toml` doesn't exist, setup copies the default template.
If it already exists, it is not modified.

## Logging

The hook and worker run as background processes, so their output is not visible to the user.

- The hook catches all exceptions and exits 0 to avoid blocking Claude Code.
- The worker's stderr is redirected to `~/.crowd-control/logs/worker.err`.
- When trace logging is enabled (`log_level` in config), both write to
  `~/.crowd-control/logs/crowd-control.log` for debugging.

## File layout

After setup, the relevant files are:

```
~/.crowd-control/
├── config.toml              # User configuration
├── db/                      # LanceDB storage
├── queue/                   # Pending ingestion queue
│   ├── <session_id>.json    # Queued sessions
│   └── failed/              # Jobs that failed 3+ times
└── logs/
    ├── worker.err           # Worker stderr output
    └── crowd-control.log    # Trace log (when enabled)
```
