# Phase 7 Manual Verification Checklist

## Prerequisites
- [ ] Clean Python 3.11+ virtualenv
- [ ] Ollama running with nomic-embed-text model (`ollama pull nomic-embed-text`)
- [ ] Claude Code CLI installed and authenticated

## Installation
- [ ] `pip install .` (or `pip install crowd-control` if published)
- [ ] `crowd-control --version` shows correct version
- [ ] `crowd-control --help` shows all commands

## Setup
- [ ] `crowd-control setup` succeeds
- [ ] Check `~/.claude.json` has crowd-control MCP server entry
- [ ] Check `~/.claude/settings.json` has SessionEnd hook entry
- [ ] `~/.crowd-control/config.toml` exists with defaults
- [ ] Run `crowd-control setup` again — verify idempotent (no duplicates)

## Manual Ingestion
- [ ] Have a Claude Code session JSONL file available
- [ ] `crowd-control ingest <path> --dry-run` shows session structure
- [ ] `crowd-control ingest <path>` runs full pipeline
- [ ] Output shows segments, learnings distilled, stored, deduped
- [ ] `crowd-control status` shows non-zero learning count

## Search
- [ ] `crowd-control search "relevant query"` returns results
- [ ] Results have scores, categories, ages, retrieval counts
- [ ] `crowd-control search "completely irrelevant gibberish"` returns no results
- [ ] `crowd-control search "query" --limit 3` respects limit
- [ ] `crowd-control search "query" --category gotcha` filters correctly

## List and Export
- [ ] `crowd-control list` shows stored learnings
- [ ] `crowd-control list --limit 5` respects limit
- [ ] `crowd-control export` outputs valid JSON to stdout
- [ ] `crowd-control export -o /tmp/test-export.json` writes file
- [ ] Exported JSON has version, count, learnings array
- [ ] Vectors not present in exported data

## MCP Server
- [ ] Start Claude Code in a project with crowd-control configured
- [ ] Verify MCP server connects (check Claude Code's MCP panel)
- [ ] Ask Claude to search for learnings — verify tool call works
- [ ] Ask Claude to add a learning — verify it's stored
- [ ] Run `crowd-control status` to confirm the add_learning persisted

## Automated Hook Loop
- [ ] Start a Claude Code session in a configured project
- [ ] Do some work, then end the session (exit or /clear)
- [ ] Check `~/.crowd-control/queue/` — verify queue file was created
- [ ] Wait ~30s, check that queue file was consumed (deleted)
- [ ] `crowd-control status` shows increased learning count
- [ ] Start a new session — verify MCP server is available

## Error Handling
- [ ] Stop Ollama, run `crowd-control search "test"` — verify clean error
- [ ] Stop Ollama, start Claude Code — verify MCP server starts (status tool works)
- [ ] With Ollama stopped, ask Claude to search — verify agent sees clean error, not crash
- [ ] Start Ollama again — verify MCP tools resume working (new session needed)
- [ ] Run `crowd-control ingest /nonexistent/file` — verify clean error
- [ ] Put invalid TOML in config.toml, run any command — verify clean error
- [ ] Fix config, verify commands work again

## Logging
- [ ] Run `crowd-control ingest <path> -v` — verify verbose output on stderr
- [ ] Set `log_level = "debug"` in config.toml
- [ ] Run `crowd-control ingest <path>`
- [ ] Check `~/.crowd-control/logs/crowd-control.log` exists and has trace data
- [ ] Set `log_level = "off"`, verify no log file is created on next run

## Worker
- [ ] `crowd-control worker` with empty queue exits cleanly
- [ ] `crowd-control worker` with queued job processes it
- [ ] Verify failed jobs move to `queue/failed/` after 3 attempts
