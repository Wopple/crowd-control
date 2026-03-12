# Phase 1: Session Parsing and Data Models

## Goal

Parse Claude Code JSONL session transcripts into structured data and define the data models that the rest of the system builds on. At the end of this phase, `crowd-control ingest --dry-run <path>` reads a session file and prints a structured summary of the parsed conversation.

## Background: JSONL Format

Claude Code stores session transcripts as JSONL files at `~/.claude/projects/<encoded-project-path>/<session-id>.jsonl`. Each line is a JSON object with a `type` field. The observed message types are:

| Type | Description |
|------|-------------|
| `user` | User messages and tool results |
| `assistant` | Model responses (text, tool_use, thinking) |
| `system` | System messages (subtypes: `local_command`, `turn_duration`, `compact_boundary`) |
| `progress` | Progress updates (subtypes: `agent_progress`, `hook_progress`, `bash_progress`) |
| `file-history-snapshot` | File state snapshots for undo/redo |
| `queue-operation` | Background task notifications |

### Common envelope fields

Every message has these fields (not all are always present):

```
parentUuid       # UUID of the parent message in the conversation tree
isSidechain      # Whether this is a sidechain (branched) message
userType         # "external" for user-initiated
cwd              # Working directory
sessionId        # Session UUID
version          # Claude Code version
gitBranch        # Current git branch
type             # Message type (see table above)
uuid             # This message's UUID
timestamp        # ISO 8601 timestamp
```

### User messages

User messages have a `message` field with `role: "user"` and `content` that is either:
- A plain string (user typed text)
- A list of content blocks, each with a `type` field:
  - `text` — user text
  - `tool_result` — result of a tool call, with fields: `tool_use_id`, `type`, `content`

Additional fields: `isMeta` (true for system-injected messages like command caveats).

### Assistant messages

Assistant messages have a `message` field with `role: "assistant"` and a `content` list of blocks:
- `thinking` — model's chain-of-thought (has `thinking` text field and `signature`)
- `text` — model's text response (has `text` field)
- `tool_use` — tool call (has `name`, `id`, `input` fields)

The `message` also includes `model`, `usage` (token counts), and `stop_reason`.

### System messages

Subtypes:
- `local_command` — output from a local CLI command, has `content` field
- `turn_duration` — timing metadata, has `durationMs` field
- `compact_boundary` — marks where conversation was compacted, has `content: "Conversation compacted"`

### Progress messages

High-frequency status updates with a `data` field. Subtypes via `data.type`:
- `agent_progress` — subagent execution updates
- `hook_progress` — hook execution updates
- `bash_progress` — shell command progress

### Messages to skip during parsing

These types are noise for learning extraction and should be discarded:
- `file-history-snapshot` — internal undo/redo state
- `progress` — transient status updates
- `queue-operation` — background task plumbing
- `system` with subtype `turn_duration` — timing metadata only

---

## Step 1: Define data models in `storage/models.py`

Define Pydantic models for the parsed data and for learnings. These models are used throughout the system — parsing, distillation, storage, and retrieval all share them.

### Models to define

**`MessageRole`** — enum: `user`, `assistant`, `system`

**`ContentBlock`** — union type for the content blocks inside a message:
- `TextBlock(type="text", text: str)`
- `ToolUseBlock(type="tool_use", id: str, name: str, input: dict)`
- `ToolResultBlock(type="tool_result", tool_use_id: str, content: str)`
- `ThinkingBlock(type="thinking", thinking: str)`

**`Message`** — a single parsed message:
- `role: MessageRole`
- `content: list[ContentBlock]`
- `uuid: str`
- `parent_uuid: str | None`
- `timestamp: datetime`
- `model: str | None` (only on assistant messages)
- `usage: dict | None` (only on assistant messages, raw token counts)

**`ConversationSegment`** — a group of related messages forming one "turn" of interaction:
- `messages: list[Message]` — the user prompt, assistant response(s), and tool results
- `tool_names: list[str]` — tools used in this segment (for tagging)
- `start_time: datetime`
- `end_time: datetime`

A segment represents a logical unit: user says something → assistant responds (possibly with multiple tool calls and results) → before the next user message. This is the unit we hand to the distiller.

**`Session`** — a full parsed session:
- `session_id: str`
- `project_path: str` (extracted from `cwd`)
- `git_branch: str | None`
- `claude_version: str | None`
- `segments: list[ConversationSegment]`
- `start_time: datetime`
- `end_time: datetime`
- `message_count: int` (total messages before filtering)
- `model: str | None` (primary model used, from first assistant message)

**`LearningCategory`** — enum: `architecture_decision`, `debugging_insight`, `pattern_discovery`, `tool_usage`, `codebase_convention`, `gotcha`

**`KnowledgeScope`** — enum: `project`, `shared`, `mixed`

Controls how learnings are scoped and retrieved.

- `project` (default): retrieval is filtered to the current project. Learnings from other projects are not surfaced.
- `shared`: all learnings go into a single pool. Retrieval searches everything regardless of source project.
- `mixed` (v0.2+): the distiller classifies each learning as project-specific or universal. Retrieval merges the current project's learnings with the universal pool.

For `project` and `shared`, the `Learning` model doesn't change — scope only affects retrieval query construction (Phase 4). A user can switch between these modes without re-ingesting.

For `mixed`, the `Learning` model's `shared` field (see below) is set by the distiller during extraction. This requires the distillation prompt (Phase 2) to classify each learning. Existing learnings default to `shared=False`, so switching to `mixed` is backwards compatible.

**`Learning`** — a single extracted learning (output of distillation, input to embedding/storage):
- `id: str` (UUID, generated on creation)
- `text: str`
- `category: LearningCategory`
- `tags: list[str]` (languages, frameworks, concepts)
- `project: str` (source project path — always set for provenance)
- `session_id: str`
- `git_sha: str | None`
- `timestamp: datetime`
- `confidence: float` (0.0–1.0)
- `stale: bool` (default False)
- `shared: bool` (default False — only meaningful in `mixed` scope, where `True` means the learning is universal and visible to all projects)

Note: `Learning` does not contain the vector. The vector is added at storage time by the embedding step. The LanceDB table schema will combine `Learning` fields with a `vector` column.

### Implementation notes

- Use `pydantic.BaseModel` for all models. Pydantic is already a dependency via `mcp`.
- Use discriminated unions for `ContentBlock` (discriminator: `type` field).
- `Learning.id` should default to `uuid.uuid4().hex` via `Field(default_factory=...)`.
- All timestamps should be `datetime` with timezone info preserved from the JSONL.
- Keep models serializable — they'll be passed to the distiller as context and stored in LanceDB.
- `KnowledgeScope` is a simple `str` enum. It lives in `models.py` alongside the other enums but is primarily consumed by config loading and retrieval — Phase 1 just defines it.

### Acceptance criteria

- [ ] All models defined with type annotations and docstrings
- [ ] Models can round-trip through JSON: `Model.model_validate_json(m.model_dump_json())` works
- [ ] `Learning` generates a UUID on creation if not provided
- [ ] `KnowledgeScope` enum has `project`, `shared`, and `mixed` variants
- [ ] Tests in `tests/test_models.py` covering construction and serialization

---

## Step 2: Build the JSONL parser in `ingest/parser.py`

### Functions to implement

**`parse_session_file(path: Path) -> Session`**

Top-level entry point. Opens the JSONL file, reads line by line, filters out noise, parses messages, segments them, and returns a `Session`.

**`parse_jsonl_line(line: str) -> dict | None`**

Parse a single JSONL line. Returns `None` for lines that should be skipped (malformed JSON, `file-history-snapshot`, `progress`, `queue-operation`, `turn_duration`).

**`parse_message(raw: dict) -> Message | None`**

Convert a raw JSONL dict into a `Message`. Handles:
- `user` type: extract `message.content` — handle both string and list-of-blocks forms. For `tool_result` blocks, extract and store the `content` field (which can be very long — truncate at a configurable limit for distillation, but store the original length so we know it was truncated).
- `assistant` type: extract `message.content` blocks. Parse `tool_use`, `text`, and `thinking` blocks. Extract `model` and `usage` from `message`.
- `system` type with subtype `local_command` or `compact_boundary`: convert to a `Message` with role `system` and the content string.
- Return `None` for any message that doesn't fit these patterns (defensive — new types may be added to Claude Code).

**`segment_messages(messages: list[Message]) -> list[ConversationSegment]`**

Group messages into conversation segments. The segmentation rule:

1. Start a new segment at each user message that is **not** a tool result. (User text = new intent. Tool results are continuations of the previous assistant turn.)
2. Accumulate all messages until the next segment boundary.
3. A segment contains: the user prompt, all assistant responses (may include multiple tool_use → tool_result cycles), and any system messages in between.

This means a single segment captures one complete "task" or "question" — the user asks something, the assistant works through it (possibly using many tools), and then the user speaks again.

Edge cases:
- The first message(s) may be system messages (meta/command output) before any user message. Group these into a preamble segment.
- `compact_boundary` system messages indicate the conversation was compressed. Start a new segment after a compact boundary since context before it was summarized away.
- Messages with `isMeta: true` are system-injected (e.g., command caveats). Include them in the current segment but mark them so the distiller can deprioritize them.

**`extract_session_metadata(messages: list[dict]) -> dict`**

Pull session-level metadata from the raw JSONL dicts before full parsing. Looks at the first few messages to extract `sessionId`, `cwd`, `gitBranch`, `version`. Returns a dict that gets unpacked into `Session` fields.

### Tool result truncation

Tool results (file reads, bash output, glob results) can be enormous. The distiller doesn't need the full output of a 2000-line file read — it needs to know that the file was read and a brief summary of what was in it. Implement a truncation strategy:

- If a `tool_result` content is longer than `MAX_TOOL_RESULT_CHARS` (default: 500), truncate to the first 200 chars + `\n...[truncated {n} chars]...\n` + last 200 chars.
- Store the original length in the `ToolResultBlock` as `original_length: int | None`.
- This keeps the parser output reasonably sized for distillation while preserving enough context to understand what happened.

### Implementation notes

- Use `pathlib.Path` for all file paths.
- Parse timestamps with `datetime.fromisoformat()`.
- Be defensive: any JSONL line that fails to parse should be logged and skipped, not crash the pipeline.
- The parser should work on both absolute paths and `~`-expanded paths.

### Acceptance criteria

- [ ] `parse_session_file` successfully parses the real JSONL files in `~/.claude/projects/`
- [ ] Noise types are filtered out (`progress`, `file-history-snapshot`, `queue-operation`, `turn_duration`)
- [ ] Messages are correctly segmented into conversation turns
- [ ] Tool results are truncated to a reasonable size
- [ ] Metadata (session_id, project, branch, model) is extracted
- [ ] Tests in `tests/test_parser.py` with sample JSONL fixtures

---

## Step 3: Create test fixtures

Build sample JSONL data for testing. Do **not** use real session data (it may contain sensitive content). Instead, create synthetic fixtures that cover the message shapes observed.

### Fixture file: `tests/fixtures/sample_session.jsonl`

A minimal but complete session with:
1. A `file-history-snapshot` line (should be skipped)
2. A `user` message with `isMeta: true` (command caveat, should be included but marked)
3. A `system` message with subtype `local_command`
4. A `user` message with plain text content ("Fix the login bug")
5. An `assistant` message with `thinking` + `tool_use` (Read file)
6. A `user` message with `tool_result` content (file contents, long enough to test truncation)
7. An `assistant` message with `thinking` + `text` + `tool_use` (Edit file)
8. A `user` message with `tool_result` (edit confirmation)
9. An `assistant` message with `text` (summary of what was done)
10. A `progress` line (should be skipped)
11. A `system` message with subtype `turn_duration` (should be skipped)
12. A second `user` message with plain text ("Now run the tests")
13. An `assistant` message with `tool_use` (Bash)
14. A `user` message with `tool_result` (test output)
15. An `assistant` message with `text` (results summary)

This should produce 2 conversation segments:
- Segment 1: messages 4–9 (fix the login bug)
- Segment 2: messages 12–15 (run the tests)

### Fixture file: `tests/fixtures/minimal_session.jsonl`

Just 3 lines: a user message, an assistant text response, and a turn_duration system message. Tests the minimum viable session.

### Fixture file: `tests/fixtures/compact_session.jsonl`

A session that includes a `compact_boundary` system message in the middle. Tests that segmentation correctly splits at the boundary.

---

## Step 4: Wire up the `ingest --dry-run` CLI command

Update `cli.py` so that `crowd-control ingest --dry-run <path>` calls the parser and prints a human-readable summary.

### Output format

```
Session: 4fbe8e02-6895-4057-950f-8e21090d9bd0
Project: /Users/daniel/git/crowd-control
Branch:  main
Model:   claude-sonnet-4-6
Period:  2026-03-11T19:07:36Z → 2026-03-11T19:45:12Z
Messages: 34 total, 18 after filtering

Segments (3):
  [1] 19:07:36 — 19:15:42  (6 messages, tools: Read, Glob)
      User: "Read structure.md and then create the initial project..."
  [2] 19:15:42 — 19:32:10  (8 messages, tools: Read, Edit, Write)
      User: "Now set up the CLI entry point with click..."
  [3] 19:32:10 — 19:45:12  (4 messages, tools: Bash)
      User: "Run the tests"
```

The user prompt preview should show the first 70 characters of the first text content block in the first user message of each segment.

### Implementation

- Import `parse_session_file` from `ingest.parser`
- Resolve the path argument (expand `~`, resolve relative paths)
- If no path given, find the most recent JSONL file in `~/.claude/projects/` for the current working directory
- Print the formatted output using `click.echo`
- On parse errors, print the error and exit with code 1

### Acceptance criteria

- [ ] `crowd-control ingest --dry-run tests/fixtures/sample_session.jsonl` prints the expected output
- [ ] `crowd-control ingest --dry-run` (no path) finds and parses the most recent session for `cwd`
- [ ] Parse errors are reported cleanly

---

## Step 5: Session discovery

Implement a utility for finding session JSONL files, since multiple commands will need this (ingest, status, etc.).

### Function: `find_sessions(project_path: str | None = None) -> list[Path]`

Location: `ingest/parser.py` (or a new `ingest/discovery.py` if parser.py is getting long)

Logic:
1. Determine the Claude Code projects directory: `~/.claude/projects/`
2. If `project_path` is given, encode it the way Claude Code does (replace `/` with `-`, strip leading `-`) to find the matching subdirectory. If not given, try to match the current working directory.
3. List all `.jsonl` files in the matching directory.
4. Sort by modification time, most recent first.
5. Return the list of paths.

### Encoding rule

Claude Code encodes project paths by replacing `/` with `-`. For example:
- `/Users/daniel/git/crowd-control` → `-Users-daniel-git-crowd-control`

The function should handle this encoding.

### Acceptance criteria

- [ ] Correctly encodes project paths to match Claude Code's directory naming
- [ ] Returns sessions sorted by recency
- [ ] Returns empty list (not error) if project directory doesn't exist

---

## Files modified in this phase

| File | Change |
|------|--------|
| `src/crowd_control/storage/models.py` | All data models |
| `src/crowd_control/ingest/parser.py` | JSONL parsing and segmentation |
| `src/crowd_control/cli.py` | Wire up `ingest --dry-run` |
| `tests/test_models.py` | Model construction and serialization tests |
| `tests/test_parser.py` | Parser unit tests |
| `tests/fixtures/sample_session.jsonl` | Synthetic test data |
| `tests/fixtures/minimal_session.jsonl` | Minimal test data |
| `tests/fixtures/compact_session.jsonl` | Compact boundary test data |

---

## Dependencies on other phases

- **None.** Phase 1 has no dependency on embedding, storage, or retrieval. The models defined here will be used by all subsequent phases.

## What Phase 2 expects from Phase 1

Phase 2 (distillation) will receive `ConversationSegment` objects and produce `Learning` objects. The distiller needs:
- Segments with readable content (tool results truncated, thinking blocks available for context)
- A way to serialize a segment to a string that can be included in an LLM prompt
- The `Learning` model ready to be instantiated from the distiller's structured output

Add a `to_prompt_text() -> str` method on `ConversationSegment` that renders the segment as a readable transcript suitable for inclusion in a distillation prompt. Format:

```
User: Fix the login bug in auth.py