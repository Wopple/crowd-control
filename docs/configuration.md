# Configuration Reference

Crowd Control is configured via `~/.crowd-control/config.toml`. All fields are optional —
missing values use the defaults shown below.

The config file is created automatically by `crowd-control setup` with sensible defaults.

## `[general]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `storage_dir` | string | `"~/.crowd-control"` | Root directory for all data (DB, queue, logs) |
| `log_level` | string | `"off"` | Trace logging level: `"off"`, `"debug"`, `"info"`, `"warning"`, `"error"` |

When `log_level` is not `"off"`, trace logs are written to `<storage_dir>/logs/crowd-control.log`.
This is useful for profiling distillation quality, retrieval scoring, and similarity thresholds.

## `[embedding]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | string | `"ollama"` | Embedding provider: `"ollama"`, `"voyage"`, or `"openai"` |
| `model` | string | `"nomic-embed-text"` | Model name for the provider |
| `api_key_env` | string | `null` | Environment variable name for API key (API providers only) |

### Supported providers

| Provider | Default Model | Dimensions | Requires |
|----------|--------------|------------|----------|
| `ollama` | `nomic-embed-text` | 768 | Ollama running locally |
| `voyage` | `voyage-code-3` | 1024 | `VOYAGE_API_KEY` env var |
| `openai` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` env var |

Install provider packages with extras:
```bash
pip install crowd-control[ollama]
pip install crowd-control[voyage]
pip install crowd-control[openai]
```

### Example: Switching to Voyage AI

```toml
[embedding]
provider = "voyage"
model = "voyage-code-3"
api_key_env = "VOYAGE_API_KEY"
```

Set the API key:
```bash
export VOYAGE_API_KEY="your-key-here"
```

**Important:** Switching embedding models requires re-creating the database. The vector
dimensions are fixed at table creation. Back up and delete `~/.crowd-control/db/`, then
re-ingest your sessions.

## `[distillation]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"haiku"` | Claude model alias for distillation (passed to `claude -p --model`) |
| `max_learnings_per_session` | int | `20` | Maximum learnings extracted per session |

## `[retrieval]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_results` | int | `15` | Maximum learnings returned per search |
| `max_tokens` | int | `4000` | Token budget for packed results |
| `min_similarity` | float | `0.3` | Minimum cosine similarity threshold (0.0–1.0) |
| `recency_half_life_days` | float | `7.0` | Exponential decay half-life in days |
| `hotness_weight` | float | `0.2` | Blend weight: 0.0 = pure semantic, 1.0 = pure hotness |
| `project_boost` | float | `1.5` | Multiplicative boost for same-project results in non-project scopes |

### Tuning retrieval

- **`min_similarity`** — Raise this if you're getting noisy results. Lower it if searches
  return too few results. Enable trace logging to see rejected similarity scores.
- **`recency_half_life_days`** — At 7 days, a learning loses half its recency score each
  week. Increase for long-lived projects where old learnings stay relevant.
- **`hotness_weight`** — At 0.2, semantic similarity dominates (80%) but frequently-used
  learnings get a meaningful boost.

### Example: Increasing token budget for large projects

```toml
[retrieval]
max_results = 25
max_tokens = 8000
```

## `[ingestion]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_ingest` | bool | `true` | Automatically ingest sessions via SessionEnd hook |
| `batch_size` | int | `5` | Embedding batch size |
| `dedup_threshold` | float | `0.95` | Cosine similarity threshold for near-duplicate rejection |

### Example: Disabling auto-ingestion

```toml
[ingestion]
auto_ingest = false
```

With auto-ingestion disabled, the SessionEnd hook will not queue sessions. Use
`crowd-control ingest` to ingest manually.

## `[knowledge]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scope` | string | `"project"` | Knowledge scope: `"project"`, `"shared"`, or `"mixed"` |

- **`project`** — Learnings are scoped to their source project. Retrieval only returns
  learnings from the current project.
- **`shared`** — All learnings in one pool. Retrieval searches everything.
- **`mixed`** — (v0.2+) Distiller classifies each learning as project-specific or universal.

## Example: Enabling trace logging

```toml
[general]
log_level = "debug"
```

Then check `~/.crowd-control/logs/crowd-control.log` after running commands. The trace log
includes similarity scores, rejected results, embedding batch sizes, and scoring details.

## Full default config

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
min_similarity = 0.3
recency_half_life_days = 7
hotness_weight = 0.2
project_boost = 1.5

[ingestion]
auto_ingest = true
batch_size = 5
dedup_threshold = 0.95
```
