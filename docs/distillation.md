# Distillation Pipeline

The distiller (`src/crowd_control/ingest/distiller.py`) extracts structured learnings
from parsed session transcripts by calling the Claude CLI (`claude -p`).

## Overview

After a session transcript is parsed into a `Session` with `ConversationSegment`s
(Phase 1), the distiller sends each qualifying segment to an LLM with a structured
extraction prompt. The LLM returns a list of learnings — specific technical insights,
decisions, and discoveries — which are validated and returned as `Learning` objects.

## How It Works

### Segment Filtering

Before distillation, `distill_session` filters out trivial segments:
- Fewer than 2 messages
- No assistant messages
- All assistant content is empty thinking blocks

### Prompt Construction

`build_distillation_prompt` renders each segment into a prompt that includes:
- Instructions for the LLM about what constitutes a good learning
- The six learning categories (architecture_decision, debugging_insight, pattern_discovery,
  tool_usage, codebase_convention, gotcha)
- Explicit negative instructions: no generic knowledge, no tool/framework general knowledge,
  no rejected alternatives (only adopted decisions), no file contents or raw logs
- Confidence calibration guidance with examples at each level (1.0 through 0.1), pushing the
  LLM to use the full range rather than clustering all scores at the top
- Project path and git branch as context
- The segment transcript, rendered via `ConversationSegment.to_prompt_text(include_thinking=False)`

Thinking blocks are excluded from the prompt since they are internal model reasoning
and not part of the visible conversation.

Segment text is truncated to 30,000 characters if it exceeds that limit, keeping the
first and last halves with a truncation marker in between. Segments shorter than 50
characters after truncation are skipped entirely.

### Claude CLI Invocation

`call_claude` runs the `claude` CLI as a subprocess:

```
claude -p --model haiku --output-format json --json-schema <schema> --no-session-persistence
```

The prompt is passed via stdin. The JSON schema constrains the output to a list of
learnings with text, category, tags, and confidence fields.

**Retry policy:**
- Up to 3 total attempts (2 retries) with backoff of 2s then 5s
- Retryable: timeouts, non-zero exit codes
- Non-retryable: `claude` not found, `CLAUDECODE` env var set (running inside Claude
  Code), JSON parse failure, missing `structured_output` key in response

The response JSON is expected to contain a `structured_output` key wrapping the actual
schema-conformant output.

### Learning Construction

For each learning returned by the LLM, `distill_segment` constructs a `Learning`
Pydantic model with:
- `text`, `category`, `tags`, `confidence` from the LLM response
- `project` from `session.project_path` (falls back to CWD)
- `session_id` from the session
- `git_sha` from running `git rev-parse HEAD` in the project directory
- Auto-generated `id` (UUID) and `timestamp`

Invalid learnings (e.g., unknown category values) are skipped with a warning.

### Session-Level Orchestration

`distill_session` processes all qualifying segments and:
- Calls an optional `progress_callback(i, total)` before each segment
- Catches `DistillationError` per-segment so one failure doesn't abort the session
- Caps total learnings at `max_learnings` (default 20) by confidence descending,
  preserving original order for ties

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"haiku"` | Claude model to use for distillation |
| `max_learnings` | `20` | Maximum learnings per session |
| `max_learning_chars` | `2000` | Max character length per learning text |
| `timeout` | `120` | Subprocess timeout in seconds |

## Limitations

- Requires the `claude` CLI to be installed and on PATH
- Cannot run from inside Claude Code (detects `CLAUDECODE` env var)
- Segment text longer than 30,000 characters is truncated, which may lose context
- No persistent caching — re-running distillation on the same session will call the
  LLM again
- The git SHA is captured at distillation time, not from the session itself, so it
  reflects the current HEAD rather than the state during the session
