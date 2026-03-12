# Phase 2: Distillation Pipeline

## Goal

Take the parsed `ConversationSegment` objects from Phase 1 and extract discrete learnings from them using Claude Code's CLI in non-interactive mode. At the end of this phase, `crowd-control ingest <path>` parses a session, distills it, and prints the extracted learnings to stdout.

## Background: How distillation works

Session transcripts are mostly noise — file reads, tool results, retry loops, boilerplate. The distiller's job is to extract the *learnings* — specific, actionable insights that would help a future agent working on the same codebase.

The distiller receives one `ConversationSegment` at a time (rendered as text via `to_prompt_text()`) and asks Claude to extract learnings from it. Each learning is a self-contained insight classified into a category with tags.

## Background: Claude Code CLI (`claude -p`)

The distiller uses Claude Code's non-interactive mode as its LLM backend. Key details:

**Invocation:**
```bash
claude -p "prompt" \
  --model haiku \
  --output-format json \
  --json-schema '<schema>' \
  --no-session-persistence
```

**JSON output shape** (when using `--output-format json`):
```json
{
  "result": "text response",
  "structured_output": { ... },  // only present when --json-schema is used
  "session_id": "...",
  ...
}
```

When `--json-schema` is provided, the structured data lives in `structured_output`. The `result` field contains the text representation. We parse `structured_output`.

**Key flags:**
- `--model haiku` — cost-effective model for extraction (configurable)
- `--output-format json` — structured response with metadata
- `--json-schema` — enforces output schema, result lands in `structured_output`
- `--no-session-persistence` — don't save distillation calls as sessions (avoids indexing our own distillation calls)

**Constraints:**
- Cannot be run from inside a Claude Code session (detects `CLAUDECODE` env var)
- Process spawn overhead ~1s per call
- Prompt is passed as an argument or via stdin (stdin for long prompts)

---

## Step 1: Define the JSON schema for distillation output

The `--json-schema` flag enforces the output structure. Define a schema that maps directly to our `Learning` model.

### Schema

```json
{
  "type": "object",
  "properties": {
    "learnings": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "text": {
            "type": "string",
            "description": "A specific, self-contained technical insight or decision. Should be understandable without the full session context."
          },
          "category": {
            "type": "string",
            "enum": [
              "architecture_decision",
              "debugging_insight",
              "pattern_discovery",
              "tool_usage",
              "codebase_convention",
              "gotcha"
            ]
          },
          "tags": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Languages, frameworks, libraries, or concepts involved."
          },
          "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How significant and reliable this learning is. 1.0 = certain and important, 0.5 = likely useful, 0.1 = marginal."
          }
        },
        "required": ["text", "category", "tags", "confidence"]
      }
    }
  },
  "required": ["learnings"]
}
```

### Where to define it

Create a constant `LEARNING_EXTRACTION_SCHEMA` in `ingest/distiller.py` as a Python dict. Serialize it to JSON when passing to `claude -p --json-schema`.

The schema does NOT include `id`, `project`, `session_id`, `git_sha`, `timestamp`, `stale`, or `shared` — those are populated by the distiller after the LLM call using session metadata.

### Acceptance criteria

- [ ] Schema is defined as a Python dict constant
- [ ] Schema serializes to valid JSON
- [ ] Schema matches what `Learning` expects (category enum values match `LearningCategory`)

---

## Step 2: Write the distillation prompt

The prompt is the most important part of the distiller. It tells Claude what to extract and what to ignore.

### Prompt template

```
You are analyzing a coding session transcript to extract learnings. Extract as many
distinct learnings as you can find — most segments contain multiple insights.

A "learning" is a specific technical insight, decision, or discovery that would help
a future AI agent working on this codebase. Each learning should be:

- Self-contained: understandable without reading the full session
- Specific: references concrete code, patterns, or decisions (not generic advice)
- Actionable: a future agent could apply this insight directly
- Concise: each learning must be under {max_learning_chars} characters, but don't pad
  to fill that limit — most learnings should be a few sentences. If an insight is complex,
  break it into multiple smaller learnings rather than writing one large one.

Look for all learnings present in the segment — there may be several across different
categories. Consider: what decisions were made? What was discovered? What would a
future agent need to know?

Categories:
- architecture_decision: A deliberate choice about code structure, patterns, or organization
- debugging_insight: A bug's root cause and how it was identified or fixed
- pattern_discovery: A recurring pattern, idiom, or convention found in the codebase
- tool_usage: How a specific tool, library, or API should be used in this project
- codebase_convention: A naming, style, or structural convention specific to this project
- gotcha: A non-obvious pitfall, edge case, or surprising behavior

Do NOT extract:
- Generic programming knowledge (e.g., "use is instead of == for None comparison")
- File contents or raw error logs
- Exploratory dead ends that produced no insight
- Information that's obvious from reading the code itself
- Anything already well-known about the language or framework

If the segment contains no learnings worth extracting, return an empty list.

Project: {project_path}
Git branch: {git_branch}

--- Session transcript ---
{segment_text}
--- End transcript ---
```

### Implementation

Define a function `build_distillation_prompt(segment: ConversationSegment, session: Session, max_learning_chars: int) -> str` that fills in the template with:
- `project_path` from `session.project_path` (fall back to CWD if empty — see Step 4)
- `git_branch` from `session.git_branch` (or "unknown")
- `segment_text` from `segment.to_prompt_text(include_thinking=False)` — thinking blocks are excluded because they are internal model reasoning, not outcomes. Including them would waste tokens and could confuse the distiller.
- `max_learning_chars` from the configured embedding model's `max_input_chars` property

### Thinking block exclusion

Add an `include_thinking: bool = True` parameter to `ConversationSegment.to_prompt_text()`. When `False`, skip `ThinkingBlock` entries during rendering. The default is `True` for backwards compatibility, but the distiller always passes `False`. Only the outcomes of the session (text, tool use, tool results) are relevant for learning extraction.

### Empty prompt guard

After building the segment text (and truncating), check if the result is empty or trivially short (< 50 chars). If so, skip the segment — there's nothing to distill. This avoids wasting an LLM call on segments that passed the message-count filter in Step 5 but produced no renderable content (e.g., all messages were meta).

### Where `max_learning_chars` comes from

The embedding model determines how large a learning can be — if a learning is too long to embed, it's useless. The `Embedder` protocol exposes a `max_input_chars` property that each provider sets based on its model's token limit.

The distiller does not import or instantiate an embedder. Instead, `max_learning_chars` is passed down from the caller:

```
config → embedding provider → max_input_chars → distill_session(max_learning_chars=...) → prompt
```

For the default `nomic-embed-text` model, `max_input_chars` is 32,000. This is generous — we want learnings much smaller than this in practice (a few sentences each). But the prompt enforces the hard limit so the LLM never produces a learning that can't be embedded.

As a fallback for Phase 2 (before embedding is implemented in Phase 3), use a default of 2000 characters. This is conservative and produces good-sized learnings. Phase 3 will replace this default with the actual value from the configured embedder.

### Prompt size management

A single segment's `to_prompt_text()` output can be very large (many tool call/result cycles). The prompt needs to fit within the model's context window with room for the response.

Implement a `truncate_segment_text(text: str, max_chars: int = 30000) -> str` function that:
1. If `len(text) <= max_chars`, return as-is
2. Otherwise, keep the first `max_chars // 2` chars and last `max_chars // 2` chars with a `\n...[segment truncated]...\n` marker in between

30,000 characters is roughly 7,500 tokens — well within Haiku's context with room for the system prompt, schema overhead, and response.

### Acceptance criteria

- [ ] Prompt template renders correctly with session metadata
- [ ] Empty/None git_branch is handled gracefully
- [ ] Long segments are truncated to stay within limits
- [ ] Prompt explicitly instructs empty list for no-learning segments
- [ ] Prompt tells the LLM to extract multiple learnings per segment
- [ ] Prompt enforces a per-learning character limit derived from `max_learning_chars`
- [ ] Prompt instructs the LLM to break large insights into multiple smaller learnings
- [ ] Thinking blocks are excluded from the segment text
- [ ] Empty/trivial segment text is detected and skipped before making the LLM call

---

## Step 3: Implement the Claude CLI subprocess wrapper

Build a general-purpose wrapper for calling `claude -p` that handles the subprocess lifecycle.

### Function: `call_claude(prompt: str, json_schema: dict, model: str = "haiku", timeout: int = 120) -> dict`

Location: `ingest/distiller.py`

**Behavior:**

1. Serialize `json_schema` to a compact JSON string (`json.dumps(json_schema, separators=(",", ":"))` — no extra whitespace, since this goes in a CLI argument).
2. Build the command:
   ```python
   cmd = [
       "claude", "-p",
       "--model", model,
       "--output-format", "json",
       "--json-schema", schema_json,
       "--no-session-persistence",
   ]
   ```
3. Run via `subprocess.run()` with:
   - `input=prompt` (pass prompt via stdin, not as an argument — avoids shell argument length limits for large prompts)
   - `capture_output=True`
   - `text=True`
   - `timeout=timeout`
4. Check `returncode`. If non-zero, raise `DistillationError` with stderr.
5. Parse `stdout` as JSON using robust extraction (see below).
6. Extract and return `structured_output` from the response dict. If `structured_output` is missing, raise `DistillationError`.

### Robust JSON extraction

The `claude -p` command may sometimes emit non-JSON content on stdout (deprecation warnings, version notices, etc.) before the actual JSON response. To handle this:

Implement a `_extract_json(raw: str) -> dict` function:
1. Find the first `{` in the string and attempt `json.loads()` from that position to the end.
2. If parsing fails, find the next `{` after the previous position and try again.
3. Repeat for up to 4 total attempts (4 `{` positions).
4. If all 4 attempts fail, raise `DistillationError`. # TODO: log the raw output for debugging

The function should try `json.loads(raw[pos:])` at each position, which handles the common case of text prefix before JSON. It does NOT need to handle JSON followed by trailing text — `json.loads` already tolerates that via `JSONDecodeError` and we try the next `{`.

### Error handling

Define a `DistillationError(Exception)` class in `ingest/distiller.py`.

Errors to handle:
- **`FileNotFoundError`**: `claude` binary not found — raise `DistillationError("claude CLI not found. Is Claude Code installed?")`
- **`subprocess.TimeoutExpired`**: call took too long — raise `DistillationError(f"Distillation timed out after {timeout}s")`
- **Non-zero exit code**: raise `DistillationError(f"claude exited with code {code}: {stderr}")`
- **JSON parse failure on stdout**: raise `DistillationError(f"Failed to parse claude output as JSON: {stdout[:200]}")`
- **Missing `structured_output`**: raise `DistillationError("No structured_output in claude response")`
- **`CLAUDECODE` env var set**: detect this upfront and raise `DistillationError("Cannot run distillation from inside a Claude Code session. Run crowd-control from a regular terminal.")` — this is a common gotcha.

### Retry policy

Wrap `call_claude` with a retry mechanism for transient failures. The retry logic lives in `call_claude` itself (not in the caller).

**Policy:**
- Max 2 retries (3 total attempts)
- Backoff: 2s after first failure, 5s after second failure
- Retryable errors: `subprocess.TimeoutExpired`, non-zero exit codes
- Non-retryable errors: `FileNotFoundError` (claude not found), `CLAUDECODE` env var, JSON parse failure after all extraction attempts, missing `structured_output`
- Use `time.sleep()` for backoff — this is acceptable because `call_claude` is already blocking (subprocess), and the caller (`distill_session`) processes segments sequentially

Log each retry attempt at warning level: `"Retrying distillation (attempt {n}/3): {error}"`

### Acceptance criteria

- [ ] Passes prompt via stdin
- [ ] Parses `structured_output` from JSON response using robust extraction
- [ ] Raises `DistillationError` with clear messages for all failure modes
- [ ] Detects and rejects running inside Claude Code
- [ ] Timeout is configurable
- [ ] Retries transient failures up to 2 times with backoff
- [ ] Non-retryable errors fail immediately without retry

---

## Step 4: Implement the distillation function

This is the core function that takes a segment and returns learnings.

### Function: `distill_segment(segment: ConversationSegment, session: Session, model: str = "haiku", max_learning_chars: int = 2000) -> list[Learning]`

Location: `ingest/distiller.py`

**Behavior:**

1. Resolve `project_path`: use `session.project_path` if non-empty, otherwise fall back to `str(Path.cwd())`. This ensures the `Learning.project` field is always meaningful — an empty project would break project-scoped retrieval in Phase 4.
2. Build the prompt via `build_distillation_prompt(segment, session, max_learning_chars)`. If the rendered segment text is empty or trivially short (< 50 chars), return an empty list immediately without calling the LLM.
3. Call `call_claude(prompt, LEARNING_EXTRACTION_SCHEMA, model=model)`.
4. Extract the `learnings` list from the response.
5. For each raw learning dict, construct a `Learning` object:
   - `text`, `category`, `tags`, `confidence` — from the LLM response
   - `id` — auto-generated (default)
   - `project` — from resolved `project_path` (see step 1)
   - `session_id` — from `session.session_id`
   - `git_sha` — attempt to read from git: `git rev-parse HEAD` in resolved `project_path` (best-effort, None on failure)
   - `timestamp` — `datetime.now()`
   - `stale` — `False`
   - `shared` — `False`
6. Validate each learning via Pydantic. Skip any that fail validation (log a warning via `logging.warning`) rather than crashing.
7. Return the list of valid `Learning` objects.

### Git SHA helper

```python
def _get_git_sha(project_path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=project_path,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
```

### Acceptance criteria

- [ ] Returns `list[Learning]` with all metadata fields populated
- [ ] Category values from the LLM match `LearningCategory` enum
- [ ] Invalid learnings are skipped with a warning, not a crash
- [ ] Git SHA is best-effort (None if unavailable)

---

## Step 5: Implement the session-level distillation orchestrator

Distill an entire session by iterating over its segments.

### Function: `distill_session(session: Session, model: str = "haiku", max_learnings: int = 20, max_learning_chars: int = 2000, progress_callback: Callable[[int, int], None] | None = None) -> list[Learning]`

Location: `ingest/distiller.py`

**Behavior:**

1. Filter `session.segments` to qualifying segments (see "Segment filtering" below).
2. For each qualifying segment:
   a. If `progress_callback` is provided, call `progress_callback(i, total)` where `i` is 1-indexed and `total` is the number of qualifying segments.
   b. Call `distill_segment(max_learning_chars=max_learning_chars)`.
3. Collect all learnings.
4. If total learnings exceed `max_learnings`, sort by confidence descending (break ties by preserving original order) and keep the top `max_learnings`.
5. Return the final list.

### Segment filtering

Not every segment is worth distilling. Skip segments where:
- `len(segment.messages) < 2` — a lone user message with no response
- No assistant messages at all (only user/system messages)
- All assistant content is empty thinking blocks (model produced no output)

Additionally, `distill_segment` (Step 4) checks if the rendered prompt text is empty or trivially short (< 50 chars) and returns an empty list without calling the LLM. This catches segments that pass the above filters but produce no renderable content (e.g., all messages are meta).

This avoids wasting LLM calls on segments that can't possibly contain learnings.

### Progress and logging

Each segment = one `claude -p` call. For a large session with 15+ segments, that's 15+ calls. Two types of output:

**Library-level logging** (via `logging` module, `crowd_control.ingest.distiller` logger):
- `logger.info("Distilling %d segments from session %s", n, session_id)`
- `logger.info("Distilling segment %d/%d...", i, n)`
- `logger.warning(...)` for skipped segments, invalid learnings, retries
- These are invisible unless the user configures logging — appropriate for library code

**CLI-level progress** (via `click.echo`, in `cli.py` only):
- The CLI layer is responsible for user-facing progress (see Step 6)
- Library functions (`distill_session`, `distill_segment`) NEVER use `click.echo`

The `max_learnings` cap (default 20, from config `distillation.max_learnings_per_session`) prevents runaway extraction.

### Error handling

If `distill_segment()` raises `DistillationError` for one segment, log the error (`logger.warning`) and continue to the next segment. Don't let one bad segment abort the entire session.

### Acceptance criteria

- [ ] Skips trivial segments
- [ ] Caps total learnings at `max_learnings`
- [ ] Continues on per-segment errors
- [ ] Logs progress

---

## Step 6: Wire up `crowd-control ingest <path>` (without `--dry-run`)

Update `cli.py` so that `crowd-control ingest <path>` (without `--dry-run`) runs the full parse → distill pipeline and prints the extracted learnings.

### Behavior

The `ingest` command (without `--dry-run`) is a user-facing CLI command. It should provide simple progress updates so the user knows work is happening.

1. Resolve the session path (same as `--dry-run`).
2. Parse the session via `parse_session_file()`.
3. Print session summary: `"Session {session_id}: {n} segments to distill"`
4. Distill the session, showing per-segment progress. Since `distill_session()` is a library function that returns the final result, the CLI should iterate segments itself or receive a callback. Two options:
   - **Option A (preferred):** Call `distill_session()` which does the work internally, then print the result. Add a `progress_callback: Callable[[int, int], None] | None = None` parameter to `distill_session()` that is called with `(current, total)` before each segment. The CLI passes `lambda cur, tot: click.echo(f"  Distilling segment {cur}/{tot}...", nl=True)`.
   - **Option B:** Iterate segments in the CLI and call `distill_segment()` per segment. This duplicates the segment-filtering logic from `distill_session()`.

   Use Option A — it keeps logic in the library and just adds a progress hook.
5. Print each learning in a human-readable format:

```
Session abc123: 5 segments to distill
  Distilling segment 1/5...
  Distilling segment 2/5...
  Distilling segment 3/5...
  Distilling segment 4/5... (skipped: too short)
  Distilling segment 5/5...

Extracted 7 learnings:

[1] [debugging_insight] (confidence: 0.9)
    The auth middleware silently swallows ConnectionResetError exceptions,
    causing requests to hang instead of failing fast.
    Tags: python, auth, error-handling

[2] [codebase_convention] (confidence: 0.7)
    All database models use UUID primary keys generated by the application,
    not auto-increment IDs from the database.
    Tags: python, sqlalchemy, database

...
```

6. If `DistillationError` is raised at the session level (e.g., not inside Claude Code check), print the error and exit with code 1.

### CLI changes

The existing `ingest` command already has `path` and `--dry-run` arguments. The change is:
- `--dry-run`: parse only (existing behavior)
- No flag: parse + distill + print learnings (new behavior, with progress output)
- Future (Phase 3): parse + distill + embed + store

Note: When `ingest` is called programmatically by the system (hooks, MCP server) in future phases, it will go through `pipeline.py`, not the CLI. The CLI is the user-facing entry point and is appropriate for progress output.

### Acceptance criteria

- [ ] `crowd-control ingest <path>` prints extracted learnings
- [ ] `crowd-control ingest` (no path) auto-discovers most recent session
- [ ] Clear error message when run from inside Claude Code
- [ ] Clear error message when `claude` binary not found

---

## Step 7: Write tests

### Testing strategy

The distiller calls an external process (`claude -p`). Tests must not make real LLM calls. Use two approaches:

1. **Unit tests with mocked subprocess** — patch `subprocess.run` to return canned JSON responses. Tests the parsing, validation, and error handling logic.
2. **Integration test fixture** — a pre-recorded JSON response file that represents what `claude -p --output-format json --json-schema ...` returns. Tests the end-to-end flow from segment to learnings.

### Test file: `tests/test_distiller.py`

**Test cases:**

`TestBuildPrompt`:
- `test_prompt_contains_project_path` — project path appears in the prompt
- `test_prompt_contains_segment_text` — segment transcript appears in the prompt
- `test_prompt_handles_none_branch` — None git_branch doesn't crash
- `test_long_segment_truncated` — segment text over 30k chars is truncated
- `test_prompt_contains_max_learning_chars` — the character limit appears in the prompt
- `test_prompt_encourages_multiple_learnings` — prompt contains language about extracting multiple learnings
- `test_prompt_excludes_thinking_blocks` — thinking block content does not appear in the rendered segment text
- `test_empty_project_path_falls_back_to_cwd` — empty project_path uses CWD

`TestExtractJson`:
- `test_clean_json` — pure JSON string parses correctly
- `test_json_with_prefix` — text before the JSON `{` is handled
- `test_json_with_multiple_braces` — first valid JSON object is extracted, even if earlier `{` chars fail
- `test_no_valid_json` — raises `DistillationError` after 4 attempts

`TestCallClaude`:
- `test_successful_call` — mock subprocess returns valid JSON with `structured_output`, function returns it
- `test_claude_not_found` — mock raises `FileNotFoundError`, function raises `DistillationError` (no retry)
- `test_timeout` — mock raises `TimeoutExpired`, function raises `DistillationError` after retries
- `test_nonzero_exit` — mock returns exit code 1, function raises `DistillationError` after retries
- `test_invalid_json_output` — mock returns non-JSON stdout, function raises `DistillationError` (no retry)
- `test_missing_structured_output` — mock returns JSON without `structured_output`, function raises `DistillationError` (no retry)
- `test_rejects_inside_claude_code` — set `CLAUDECODE` env var, function raises `DistillationError` (no retry)
- `test_retry_succeeds_on_second_attempt` — mock fails once then succeeds, function returns the successful result
- `test_retry_exhausted` — mock fails 3 times, function raises `DistillationError`

`TestDistillSegment`:
- `test_returns_learning_objects` — mock `call_claude` to return valid learnings, verify `Learning` objects have correct fields
- `test_populates_metadata` — verify `project`, `session_id`, `timestamp` are set from session
- `test_skips_invalid_learnings` — include one learning with an invalid category, verify it's skipped
- `test_empty_learnings_list` — LLM returns empty list, function returns empty list
- `test_empty_prompt_text_skips_llm_call` — segment that renders to empty text returns empty list without calling `call_claude`

`TestDistillSession`:
- `test_skips_trivial_segments` — session with a 1-message segment, verify it's not distilled
- `test_caps_at_max_learnings` — mock returns many learnings, verify capped
- `test_continues_on_segment_error` — mock raises error for one segment, verify others still processed
- `test_progress_callback_called` — verify callback is invoked with correct (current, total) for each segment

### Fixture file: `tests/fixtures/distillation_response.json`

A canned JSON response matching what `claude -p --output-format json --json-schema ...` returns:

```json
{
  "result": "...",
  "structured_output": {
    "learnings": [
      {
        "text": "The auth module uses a custom middleware chain...",
        "category": "architecture_decision",
        "tags": ["python", "auth", "middleware"],
        "confidence": 0.85
      },
      {
        "text": "Tests require PYTHONPATH=src to resolve imports...",
        "category": "gotcha",
        "tags": ["python", "pytest", "imports"],
        "confidence": 0.9
      }
    ]
  },
  "session_id": "distill-session-001"
}
```

### Acceptance criteria

- [ ] All tests pass without making real LLM calls
- [ ] Error paths are tested
- [ ] Subprocess is mocked correctly (stdin, stdout, returncode, timeout)

---

## Step 8: Write implementation docs

Per CLAUDE.md, each phase must produce durable documentation in `docs/` (not `docs/plans/`).

Create `docs/distillation.md` covering:
- What the distiller does (high-level)
- How it calls Claude Code CLI (`claude -p` flags, stdin prompt, structured output)
- The distillation prompt (what it asks for, what it filters out)
- How learnings are constructed from the LLM response (which fields come from the LLM, which from session metadata)
- Configuration: model, max_learnings_per_session, max_learning_chars
- Limitations: cannot run from inside Claude Code, one subprocess per segment

This doc should be enough for an agent to understand the distillation system without reading `distiller.py`.

---

## Files modified in this phase

| File | Change |
|------|--------|
| `src/crowd_control/ingest/distiller.py` | Full implementation: schema, prompt, subprocess wrapper, retry logic, robust JSON extraction, distillation functions |
| `src/crowd_control/storage/models.py` | Add `include_thinking` parameter to `ConversationSegment.to_prompt_text()` |
| `src/crowd_control/cli.py` | Wire up `ingest` (non-dry-run) to distill and print with progress |
| `tests/test_distiller.py` | All distillation tests (including retry, JSON extraction, progress callback) |
| `tests/fixtures/distillation_response.json` | Canned claude -p response |
| `docs/distillation.md` | Implementation documentation for the distillation system |

---

## Dependencies on other phases

- **Phase 1 (done):** `ConversationSegment.to_prompt_text()`, `Session`, `Learning` model, `parse_session_file()`
  - Note: Phase 1 was completed before the implementation docs requirement was added. A `docs/parsing.md` should be backfilled as part of Phase 2 work.
- **Phase 3 (future):** will consume `list[Learning]` from `distill_session()` and embed + store them

## What Phase 3 expects from Phase 2

Phase 3 will call `distill_session()` and receive `list[Learning]` objects ready to be embedded. The `Learning` objects will have all fields populated except for `vector` (which Phase 3 adds). The `ingest/pipeline.py` module (currently a stub) will eventually orchestrate: parse → distill → embed → store.
