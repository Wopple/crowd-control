# Code Quality Fixes

Five targeted improvements identified during code review. Each is independent and can be
implemented in any order.

## Phase 1: Deduplicate project path fallback

**Problem:** `distill_segment` and `build_distillation_prompt` both independently compute
`project_path = session.project_path if session.project_path else str(Path.cwd())`. If
they diverge, the git SHA metadata won't match the project path in the prompt.

**Change:** Compute the resolved project path once in `distill_segment` and pass it as a
parameter to `build_distillation_prompt`. Remove the fallback logic from
`build_distillation_prompt`.

**Files:** `distiller.py`, `test_distiller.py`

## Phase 2: Remove JSON extraction heuristics

**Problem:** `_extract_json` scans for the first valid JSON object by trying up to 4 `{`
positions. Since `call_claude` uses `--output-format json`, output should be valid JSON.
The heuristic masks real errors.

**Change:** Replace `_extract_json` with `json.loads`. On failure, raise
`DistillationError` immediately (non-retryable). Remove `_extract_json` and its tests.

**Files:** `distiller.py`, `test_distiller.py`

## Phase 3: Refactor segment_messages mutation pattern

**Problem:** `segment_messages` uses an inner `_flush()` function that mutates outer scope
via `nonlocal`. The preamble carry-forward is subtle — trailing non-substantive messages
are silently dropped.

**Change:** Replace with a stateful approach that makes the accumulation explicit. Document
(via test) what happens to trailing non-substantive messages.

**Files:** `parser.py`, `test_parser.py`

## Phase 4: Learning text length validation

**Problem:** The `Learning` model has no max length on `text`. Oversized learnings from
the LLM degrade embedding quality and waste storage.

**Change:** Add a Pydantic field validator on `Learning.text` that truncates to
`max_learning_chars` (a class-level constant, default 2000). Log a warning when
truncation occurs. This must not raise — individual oversized learnings are truncated,
not rejected, so batch processing continues.

**Files:** `models.py`, `test_models.py`

## Phase 5: Extract segment qualification predicate

**Problem:** `distill_session` has 28 lines of inline filtering logic (min messages, has
assistant, non-empty thinking check) that violates single responsibility and is hard to
test directly.

**Change:** Extract to a pure function `is_segment_worth_distilling(segment) -> bool`.
Add direct unit tests for each filter condition. Simplify `distill_session` to a
list comprehension filter.

**Files:** `distiller.py`, `test_distiller.py`
