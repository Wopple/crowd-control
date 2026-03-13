# Crowd Control - Project Structure

## Dependencies

### Core

| Package     | Purpose                            | Notes                    |
|-------------|------------------------------------|--------------------------|
| `mcp`       | MCP server framework (FastMCP)     | `mcp[cli]` for dev tools |
| `lancedb`   | Vector database                    | Embedded, Rust-backed    |
| `ollama`    | Local embedding (default provider) | Optional if using API    |
| `click`     | CLI framework                      | Clean, composable CLI    |
| `pydantic`  | Data validation and models         | Already a dep of `mcp`   |

Note: distillation uses the Claude Code CLI (`claude -p`), not the `anthropic` Python SDK.
This avoids requiring a separate API key — the user's existing Claude Code subscription is
used.

### Optional (for alternative embedding providers)

| Package    | Purpose              |
|------------|----------------------|
| `voyageai` | Voyage AI embeddings |
| `openai`   | OpenAI embeddings    |

### Development

| Package  | Purpose                |
|----------|------------------------|
| `pytest` | Testing                |
| `ruff`   | Linting and formatting |

## Entry Points

Defined in `pyproject.toml`:

```toml
[project.scripts]
crowd-control = "crowd_control.cli:main"
```

This gives users a `crowd-control` command after `pip install`.

## Configuration File

Default location: `~/.crowd-control/config.toml`

```toml
[general]
storage_dir = "~/.crowd-control"
log_level = "off"                      # "off", "error", "warning", "info", "debug"

[knowledge]
scope = "project"                      # "project", "shared", or "mixed" (v0.2+)

[embedding]
provider = "ollama"                # "ollama", "voyage", "openai"
model = "nomic-embed-text"

# api_key_env = "VOYAGE_API_KEY"  # Uncomment for API providers

[distillation]
model = "haiku"                        # Claude Code model alias (passed to claude -p --model)
max_learnings_per_session = 20

[retrieval]
max_results = 15
max_tokens = 4000
min_similarity = 0.3               # Minimum cosine similarity to include
recency_half_life_days = 7         # Exponential decay half-life (days)
hotness_weight = 0.2               # Blend: 0.0 = pure semantic, 1.0 = pure hotness
project_boost = 1.5                # Boost for same-project matches

[ingestion]
auto_ingest = true                  # Whether Stop hook triggers ingestion
batch_size = 5                      # Sessions to process per batch
```
