# Crowd Control - Implementation Phases

## Phase 0: Project scaffolding ✅

Complete. Package structure, `pyproject.toml`, ruff, pytest, default config template.

## Phase 1: Session parsing and data models ✅

Complete. See `docs/distillation.md` for parsing details. Models in `storage/models.py`.

## Phase 2: Distillation pipeline ✅

Complete. See `docs/distillation.md`. Prompt construction, `claude -p` invocation,
retry policy, parallel segment processing, confidence-based capping.

## Phase 3: Embedding and storage ✅

Complete. See `docs/embedding-and-storage.md`. Three embedding providers (Ollama,
Voyage, OpenAI), LanceDB storage with two-stage dedup, full ingestion pipeline.

## Unscheduled: Within-session dedup

Text-based deduplication in `distill_session` (see `learning-deduplication.md`).
Independent of Phase 4 — a refinement to the distillation pipeline.

## Phase 4: Retrieval and ranking

The retrieval and ranking system, informed by algorithms from OpenViking
(see `openviking-learnings.md` for rationale and source references).

### Data model changes

- Add `active_count: int = 0` to `Learning` model and LanceDB schema
- Add `LearningStore.increment_active_count(learning_id)` method

### Search (`retrieve/search.py`)

- Embed query text via configured embedder
- LanceDB vector search with cosine distance
- Metadata filters: project, category, tags, stale
- Minimum similarity threshold from config
- Return raw results with similarity scores

### Ranking (`retrieve/rank.py`)

Scoring formula (from OpenViking's `memory_lifecycle.py`):

- Recency: `exp(-ln(2) / half_life_days * age_days)` (half-life default 7 days)
- Hotness: `sigmoid(log1p(active_count)) * recency_factor`
- Final: `(1 - hotness_weight) * semantic_score + hotness_weight * hotness_score`
- Apply project boost for same-project matches
- Deduplicate by text similarity (post-search pass)
- Pack into token budget

### Config changes

Replace `recency_decay = 0.95` with:
- `recency_half_life_days = 7` — exponential decay half-life
- `hotness_weight = 0.2` — blend weight for hotness vs semantic similarity

### Active count tracking

Increment `active_count` when learnings are returned by search (MCP tool or hook
injection). This creates the feedback loop that makes hotness scoring meaningful.

**Deliverable:** `crowd-control search "how does the auth system work"` returns ranked
results. Frequently-retrieved learnings rise in ranking over time.

## Phase 5: MCP server

- Define MCP server with FastMCP
- Implement `search_learnings` tool (calls retrieval pipeline, increments active counts)
- Implement `add_learning` tool
- Implement `ingest_session` tool
- Implement `status` tool
- Use lifespan API for DB connection management
- Write integration tests

**Deliverable:** MCP server runs via `crowd-control serve` and responds to tool calls.

## Phase 6: Hooks and automation

- Implement `Stop` hook handler (queue ingestion job)
- Implement `SessionStart` hook handler (retrieve + format context, increment active counts)
- Implement `crowd-control setup` command (auto-configure hooks + MCP)
- Add background worker for processing ingestion queue
- Write tests for hook I/O format

**Deliverable:** Fully automated loop — sessions ingest automatically, new sessions get context.

## Phase 7: Polish and release

- Error handling and graceful degradation throughout
- Logging with configurable verbosity
- Documentation (README, setup guide, configuration reference)
- PyPI packaging and release
- Test on fresh machine with clean install

**Deliverable:** v0.1 release on PyPI.
