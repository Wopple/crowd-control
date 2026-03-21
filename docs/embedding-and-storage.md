# Embedding and Storage

After distillation produces `Learning` objects, the embedding and storage layer converts
them into vectors and persists them in LanceDB for later retrieval.

## Overview

```
Learning objects (from distiller)
    → embed text into vectors (Ollama/Voyage/OpenAI)
    → deduplicate against existing learnings
    → store in LanceDB
```

## Embedding Providers

The system supports three embedding providers, configured via `[embedding]` in
`~/.crowd-control/config.toml`:

| Provider | Model | Dimensions | Requires |
|----------|-------|------------|----------|
| `ollama` (default) | `nomic-embed-text` | 768 | Ollama running locally |
| `voyage` | `voyage-code-3` | 1024 | `VOYAGE_API_KEY` env var |
| `openai` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` env var |

All providers are optional dependencies. Install with:
```bash
pip install crowd-control[ollama]   # or [voyage] or [openai]
```

### Embedder Protocol

All providers implement the same interface (`embed/base.py`):

- `embed(texts: list[str]) -> list[list[float]]` — batch embed texts into vectors
- `dimensions: int` — vector dimensionality (model-dependent)
- `max_input_chars: int` — per-text character limit

The factory function `create_embedder(config)` creates the appropriate provider based
on `config.embedding.provider`. Lazy imports ensure only the configured provider's
package needs to be installed.

### Text Handling

Texts exceeding `max_input_chars` are silently truncated. In practice, learning text
is already capped at 2000 characters by the distiller, well within any provider's limit.
Truncation is a safety net, not a normal code path.

## LanceDB Storage

### Table Schema

Learnings are stored in a single `learnings` table with this schema:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | UUID hex |
| `vector` | fixed_size_list(float32, N) | Embedding vector (N = model dimensions) |
| `text` | string | Learning text |
| `category` | string | One of the `LearningCategory` values |
| `tags` | list(string) | Languages, frameworks, concepts |
| `project` | string | Source project path |
| `session_id` | string | Source session ID |
| `git_sha` | string | Git HEAD at distillation time |
| `timestamp` | timestamp(us, UTC) | When the learning was created |
| `confidence` | float32 | Distiller's confidence score (0.0–1.0) |
| `active_count` | int32 | Retrieval count (used by pruning) |
| `stale` | bool | Marked stale (for future use) |
| `shared` | bool | Cross-project learning (for future use) |

The vector column dimension is fixed at table creation and must match the embedding
model. Switching models requires deleting and re-creating the database.

### Dimension Mismatch Detection

When opening an existing table, `LearningStore` reads the vector dimension from the
table schema. If a different `vector_dimensions` is passed (because the user switched
embedding models), it raises a clear error with backup and re-ingestion instructions.

### Schema Versioning

The database tracks its schema version in a `_metadata` table (key-value pairs).
When `LearningStore` opens an existing table, it compares the stored version to the
code's expected version. If the stored version is behind, incremental migrations run
automatically:

- Each migration uses LanceDB's native `add_columns` / `alter_columns` APIs — no
  drop-and-recreate.
- Version is updated after each successful step. A partial failure leaves the DB at
  the last successful version.
- Migrations are idempotent: safe to re-run if interrupted.
- For pre-migration databases (no `_metadata` table), the system assumes version 1
  and creates the metadata table on first access.

Migration logic lives in `storage/migration.py`. The migration registry
(`_MIGRATIONS`) is empty until the schema changes from v1.

### Deduplication

The `add()` method performs multi-stage deduplication before inserting. It returns an
`AddResult` with `stored` (count of inserted learnings) and `duplicates` (list of
`DuplicateInfo` with `new_text`, `matched_text`, and `similarity` for each rejection).

**Against existing DB rows** (skipped when table is empty):

1. **Exact text match** — rejects learnings with identical text (SQL `WHERE` clause).
   Single quotes in text are escaped to prevent SQL injection.

2. **Near-duplicate by embedding similarity** — `_find_near_duplicate()` searches for
   the closest vector using cosine distance. If similarity ≥ `dedup_threshold`, the
   learning is rejected. Returns the matched text and similarity for informative
   rejection messages.

**Within the current batch** (always active, including on empty tables):

3. **Exact text dedup** — a `seen_texts` set catches identical text within the batch.

4. **Near-duplicate by vector similarity** — each learning's vector is compared against
   vectors of previously-accepted batch items using `_find_similar_in_batch()` (pure
   cosine similarity in Python). This catches near-duplicates in the first ingestion
   when the table is empty and DB-level dedup is skipped.

### Dedup Threshold Calibration

The default `dedup_threshold` is **0.90** (cosine similarity). This was determined
empirically by analyzing 221 learnings embedded with `nomic-embed-text`:

| Category | Similarity range | Examples |
|----------|-----------------|----------|
| True duplicates (tight paraphrases) | 0.91–0.95 | Same insight with minor wording changes |
| **Threshold** | **0.90** | |
| Distinct but related | 0.85–0.90 | Same topic, different insights |
| Clearly different | < 0.85 | Different topics |

The gap between the lowest true duplicate (0.9071) and the highest distinct pair
(0.8970) is narrow, so 0.90 was chosen as the midpoint. This catches all observed
duplicates while preserving genuinely distinct insights about similar topics.

Synthetic test pairs confirmed: heavily-reworded versions of the same insight score
0.70–0.89 (below threshold), while genuinely different insights about similar topics
score 0.47–0.74 (well-separated). The real duplicates in the DB are tighter
paraphrases than synthetically generated ones.

**Cosine metric note:** LanceDB defaults to L2 distance. `.metric("cosine")` must be
chained on every `.search()` call — it's a per-query setting, not a table property.

## Configuration

Relevant config sections in `~/.crowd-control/config.toml`:

```toml
[embedding]
provider = "ollama"              # "ollama", "voyage", or "openai"
model = "nomic-embed-text"       # Model name for the provider
# api_key_env = "VOYAGE_API_KEY" # Env var name for API key (API providers only)

[ingestion]
dedup_threshold = 0.90           # Cosine similarity threshold for near-duplicate rejection
max_age_days = 90                # Prune learnings older than this (0 = never prune)
retention_retrieval_interval_days = 30  # Must be retrieved once per this many days to survive
```

The database path is derived from `[general].storage_dir` (default `~/.crowd-control`),
stored at `<storage_dir>/db/`.

## Ingestion Pipeline

The pipeline (`ingest/pipeline.py`) orchestrates the full flow:

1. **Parse** — `parse_session_file(path)` → `Session`
2. **Distill** — `distill_session(session, ...)` → `list[Learning]`
3. **Embed** — `embedder.embed([l.text for l in learnings])` → vectors
4. **Store** — `LearningStore.add(records)` with dedup
5. **Prune** — `LearningStore.prune()` removes old low-activity learnings

If distillation returns zero learnings, the pipeline exits early without creating
an embedder or touching the database.

The pipeline returns an `IngestResult` with counts:
- `segments_processed` — total segments in the session
- `learnings_distilled` — learnings produced by the distiller
- `learnings_stored` — actually inserted after dedup
- `learnings_deduplicated` — rejected by dedup

## CLI Commands

### `crowd-control ingest [path]`

Runs the full pipeline. Without `--dry-run`, it parses, distills, embeds, and stores.

### `crowd-control list [--project P] [--category C] [--limit N]`

Lists stored learnings with optional filtering. Results ordered by timestamp descending.

### `crowd-control prune [--dry-run]`

Removes old learnings with insufficient retrieval activity. A learning older than
`max_age_days` must have been retrieved at least once per
`retention_retrieval_interval_days` to survive (e.g., a 120-day-old learning with a
30-day interval needs 4 retrievals). With `--dry-run`, shows what would be pruned
without deleting.

Pruning also runs automatically after each ingestion and on MCP server startup.

### `crowd-control status`

Shows database path, learning count, and configured embedding provider/model.
