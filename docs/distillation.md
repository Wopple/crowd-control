# Distillation Pipeline

The distiller (`src/crowd_control/ingest/distiller.py`) extracts structured
learnings from parsed session transcripts. The actual LLM calls live in a
swappable provider package (`src/crowd_control/ingest/llm/`); the distiller
itself is provider-agnostic.

## Overview

After a session transcript is parsed into a `Session` with `ConversationSegment`s
(Phase 1), the distiller sends each qualifying segment to an LLM with a structured
extraction prompt. The LLM returns a list of learnings — specific technical insights,
decisions, and discoveries — which are validated and returned as `Learning` objects.

## Provider Abstraction

The `DistillerLLM` protocol in `src/crowd_control/ingest/llm/base.py` exposes a
single method (`generate_structured(prompt, schema) -> dict`) and a
`recommended_concurrency` property. Two implementations ship with Crowd Control:

| Provider     | Class           | Backend                                        |
|--------------|-----------------|------------------------------------------------|
| `ollama`     | `OllamaLLM`     | Local Ollama daemon via the `ollama` Python client |
| `claude-code`| `ClaudeCLILLM`  | Subprocess: `claude -p`                        |

`create_distiller_llm(config.distillation)` builds the right instance based on
the resolved provider on the config. New providers are added by writing a class
that satisfies the protocol and extending the factory.

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

### LLM Invocation — Claude provider

`ClaudeCLILLM.generate_structured` shells out to the `claude` CLI:

```
claude -p --model <model> --output-format json --json-schema <schema> --no-session-persistence
```

The prompt is passed via stdin. The JSON schema constrains the output to a list of
learnings with text, category, tags, and confidence fields.

**Retry policy:**
- Up to 3 total attempts (2 retries) with backoff of 2s then 5s
- Retryable: timeouts, non-zero exit codes
- Non-retryable: `claude` not found, `CLAUDECODE` env var set (running inside
  Claude Code), JSON parse failure, missing `structured_output` key in response

The response JSON is expected to contain a `structured_output` key wrapping the
actual schema-conformant output.

`ClaudeCLILLM` also sets `CROWD_CONTROL_INGESTING=1` on the subprocess
environment so the SessionEnd hook fired by the exiting `claude -p` process
recognises itself and skips queuing. See [hooks.md](hooks.md) for the layered
defence against recursive ingestion.

### LLM Invocation — Ollama provider

`OllamaLLM.generate_structured` calls the local Ollama daemon via the `ollama`
Python client:

```python
client.chat(
    model=<model>,
    messages=[{"role": "user", "content": prompt}],
    format=<schema>,
    options={"temperature": 0.0},
    stream=False,
    keep_alive="10m",
)
```

`format=<schema>` enables Ollama's JSON-schema-constrained decoding (Ollama
≥ 0.5). The response message content is a JSON string that conforms to the
schema directly — **no `structured_output` wrapper to unpack**, unlike the
Claude CLI path.

**HTTP timeout.** `OllamaLLM` constructs `ollama.Client(timeout=300.0)` so CPU
inference of large models doesn't trip the library's short default httpx
timeout. `keep_alive="10m"` is unrelated; it holds the loaded model in memory
between segments to avoid reload overhead.

**No retry policy.** Local failures are user-actionable. Daemon unreachable →
"Ollama not running. Start it with: `ollama serve`". Model not pulled →
"Run: `ollama pull <model>`". Retrying would mask these.

### Subprocess-only guards

Two environment-variable invariants live on `ClaudeCLILLM` only:

- **`CLAUDECODE`** — refuse-to-run guard. `ClaudeCLILLM.generate_structured`
  raises `DistillationError` if `CLAUDECODE` is set, because `claude -p`
  refuses to run inside an interactive Claude Code session. The worker that
  spawns from a SessionEnd hook strips this variable.
- **`CROWD_CONTROL_INGESTING`** — recursion guard. Injected on the `claude -p`
  subprocess env; the SessionEnd hook fired by the exiting subprocess short-
  circuits when it sees it. See [hooks.md](hooks.md).

The Ollama provider runs in-process; it spawns no subprocess and touches
neither variable. Future subprocess-spawning providers must opt into the
`CROWD_CONTROL_INGESTING` guard.

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
- Catches `DistillationError` per-segment so one failure doesn't abort the session
- Caps total learnings at `max_learnings` (default 20) by confidence descending,
  preserving original order for ties

### Concurrency

`distill_session` processes qualifying segments in parallel using a thread pool.
Each segment is submitted as an independent task — segments share no mutable state.

When `max_workers` is not passed, the orchestrator uses
`llm.recommended_concurrency` — `8` for Claude (parallel API calls help) and
`1` for Ollama (the local daemon serves through a single pipeline; concurrent
client calls only queue). The CLI `--concurrency` flag overrides this if the
user wants to tune it.

The actual number of threads is capped at `min(max_workers, segment_count)`.
The git SHA is resolved once before submitting segments, rather than per-segment.

Progress callbacks report `(completed_count, total)` as segments finish, which
may be out of original order. Results are re-sorted by original segment index
before the confidence-based capping step, so the output is deterministic
regardless of completion order.

## Configuration

Config file parameters (`[distillation]` in `config.toml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"ollama:qwen3:8b"` | Provider + model identifier |
| `max_learnings_per_session` | `20` | Maximum learnings per session |

The `model` field encodes both provider and model as a single string. See the
"Choosing a distillation model" section of the [user guide](user-guide.md) for
the resolution table and recommended models.

Internal defaults (not exposed in config file):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_learning_chars` | `2000` | Max character length per learning text |
| `max_workers` | from provider | `8` for Claude, `1` for Ollama (CLI `--concurrency`) |
| `timeout` (claude) | `120s` | Subprocess timeout for `claude -p` |
| `timeout_seconds` (ollama) | `300s` | HTTP timeout on the Ollama client |

## Pipeline Integration

The distillation output feeds into the embedding pipeline. After `distill_session`
returns a list of `Learning` objects, the ingestion pipeline (`ingest/pipeline.py`)
embeds their text into vectors and stores them in LanceDB. See
[embedding-and-storage.md](embedding-and-storage.md) for details.

## Limitations

- **Ollama provider** — requires the `ollama` Python client (`pip install
  crowd-control[ollama]`), a running Ollama daemon, and the configured model
  pulled (`ollama pull <model>`). `crowd-control status` reports readiness.
- **Claude provider** — requires the `claude` CLI on PATH and authenticated.
  Cannot run from inside Claude Code (detects `CLAUDECODE` env var); the
  background worker is launched with this variable stripped.
- Segment text longer than 30,000 characters is truncated, which may lose context.
- No persistent caching — re-running distillation on the same session calls
  the LLM again.
- The git SHA is captured at distillation time, not from the session itself,
  so it reflects the current HEAD rather than the state during the session.
- High concurrency with the Claude provider may hit Claude API rate limits —
  reduce `--concurrency` if you see retry storms.
