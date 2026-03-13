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
| `stale` | bool | Marked stale (for future use) |
| `shared` | bool | Cross-project learning (for future use) |

The vector column dimension is fixed at table creation and must match the embedding
model. Switching models requires deleting and re-creating the database.

### Dimension Mismatch Detection

When opening an existing table, `LearningStore` reads the vector dimension from the
table schema. If a different `vector_dimensions` is passed (because the user switched
embedding models), it raises a clear error with backup and re-ingestion instructions.

### Deduplication

The `add()` method performs two-stage deduplication before inserting:

1. **Exact text match** — rejects learnings with identical text (SQL `WHERE` clause).
   Single quotes in text are escaped to prevent SQL injection.

2. **Near-duplicate by embedding similarity** — searches for the closest vector using
   cosine distance. If `_distance < (1.0 - dedup_threshold)`, the learning is rejected.
   The default threshold of 0.95 means learnings with ≥ 95% cosine similarity are
   considered duplicates.

Dedup queries are skipped entirely when the table is empty (first ingestion), since
there's nothing to deduplicate against.

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
dedup_threshold = 0.95           # Cosine similarity threshold for near-duplicate rejection
```

The database path is derived from `[general].storage_dir` (default `~/.crowd-control`),
stored at `<storage_dir>/db/`.

## Ingestion Pipeline

The pipeline (`ingest/pipeline.py`) orchestrates the full flow:

1. **Parse** — `parse_session_file(path)` → `Session`
2. **Distill** — `distill_session(session, ...)` → `list[Learning]`
3. **Embed** — `embedder.embed([l.text for l in learnings])` → vectors
4. **Store** — `LearningStore.add(records)` with dedup

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

### `crowd-control status`

Shows database path, learning count, and configured embedding provider/model.
