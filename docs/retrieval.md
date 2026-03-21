# Retrieval and Ranking

After learnings are ingested and stored in LanceDB, the retrieval system finds and
ranks them for context injection. The system has two modules with distinct
responsibilities:

- `retrieve/__init__.py` — `retrieve_learnings()` orchestrator: the single entry point
  that the CLI, MCP server, and hooks all call (search → rank → increment active counts)
- `retrieve/search.py` — embeds a query and runs vector search against LanceDB; defines
  `BaseResult`, `SearchResult`, `Scope` type, and `validate_scope()`
- `retrieve/rank.py` — scores, deduplicates, and packs results into a token budget

## Overview

```
Query text
    → embed via configured embedder
    → vector search in LanceDB (with metadata filters)
    → raw results with similarity scores
    → scoring: blend semantic similarity with hotness
    → post-search deduplication
    → token packing
    → ranked results
```

## Orchestration

`retrieve_learnings()` in `retrieve/__init__.py` is the top-level entry point. It
runs the full pipeline: search → rank → increment active counts. Returns a
`RetrievalResult` containing `ranked`, `search_results`, and `total_learnings`.

The CLI `search` command, MCP server, and hooks should all call this function
rather than assembling the pipeline themselves.

## Scope Validation

The `scope` parameter is constrained to `Literal["project", "shared", "mixed"]`
via the `Scope` type alias. `validate_scope()` raises `ValueError` for invalid
values. Both `search_learnings()` and `rank_results()` validate on entry.

## Search Module

`search_learnings()` embeds the query and runs vector search. It:

1. Embeds the query using the configured embedder.
2. Over-fetches from the DB: `limit = max(max_results * 2, 30)` to ensure enough
   candidates survive dedup and token packing.
3. Calls `LearningStore.vector_search()` with metadata filters.
4. Returns `SearchResults` containing `SearchResult` objects with similarity scores.

### Metadata Filtering

Filtering is done at the database level via WHERE clauses:

| Scope | Filter |
|-------|--------|
| `project` | `project = current_project` |
| `shared` | No project filter (all learnings) |
| `mixed` | `project = current_project OR shared = true` |

Additional filters:
- `category` — optional, filters to a specific learning category
- `tags` — optional list of tags, match-any semantics (OR). Uses `array_contains()`
  in LanceDB's DuckDB SQL dialect. Tags are lowercased at search time to match the
  normalized storage format.
- `exclude_stale` — defaults to true, excludes `stale = true` learnings

### Distance-to-Similarity Conversion

LanceDB cosine distance is `1 - cosine_similarity`. The search converts this:
```
similarity = 1.0 - distance
```
Results below `min_similarity` (default 0.3) are filtered out.

## Ranking Module

`rank_results()` takes raw search results and produces a final ranked list. It
runs three pure-function stages: scoring, deduplication, and token packing.

### Scoring Formula

Adapted from OpenViking's `memory_lifecycle.py` (ByteDance):

**1. Recency factor:**
```
decay_constant = ln(2) / half_life_days
recency = exp(-decay_constant * age_days)
```
- `age_days` is clamped to >= 0 to handle clock skew
- Recency is 1.0 for brand-new learnings, 0.5 at `half_life_days`, 0.25 at
  `2 * half_life_days`
- Default half-life: 7 days

**2. Hotness score:**
```
hotness = sigmoid(log1p(active_count)) * recency
```
- `sigmoid(log1p(...))` compresses usage counts so heavily-used learnings don't dominate
- `active_count=0` → `sigmoid(0) = 0.5` (cold start gets mild recency boost)
- `active_count=10` → `sigmoid(2.4) ≈ 0.92`
- `active_count=100` → `sigmoid(4.6) ≈ 0.99`
- Multiplied by recency, so old heavily-used learnings still decay

**3. Final score (blending):**
```
blended = (1 - hotness_weight) * similarity + hotness_weight * hotness
```
- Default `hotness_weight=0.2`: 80% semantic similarity, 20% hotness

**4. Project boost (conditional):**
```
if scope != "project" and result.project == current_project:
    blended *= project_boost
```
- Default `project_boost=1.5` (50% boost for same-project results)
- Skipped in `project` scope since all results already match the project

### Worked Example

A learning with:
- `similarity = 0.85`, `active_count = 5`, `age = 3 days`, `half_life = 7`

Computes:
- `recency = exp(-ln(2)/7 * 3) = exp(-0.297) ≈ 0.743`
- `hotness = sigmoid(log1p(5)) * 0.743 = sigmoid(1.79) * 0.743 ≈ 0.857 * 0.743 ≈ 0.637`
- `final = 0.8 * 0.85 + 0.2 * 0.637 = 0.680 + 0.127 = 0.807`

### Post-Search Deduplication

After scoring, results are deduplicated by text similarity using
`SequenceMatcher.ratio()` with a 0.85 threshold. This is a safety net — storage-time
dedup (cosine >= 0.95) already prevents most duplicates, but edge cases exist where
semantically similar but textually distinct learnings both match a query.

The algorithm walks results sorted by score descending. Each result is compared against
all previously-kept results. If any have ratio >= 0.85, the current result is dropped.
O(n²) but n is small (typically 30–100).

### Token Packing

After dedup, results are packed into the token budget using `len(text) / 4` as a
rough token estimate. Results are walked in score order; the packer stops when
either `max_tokens` (default 4000) or `max_results` (default 15) is reached,
avoiding unnecessary iteration over remaining candidates.

### Data Model Hierarchy

`BaseResult` defines the fields common to all result types (`id`, `text`,
`category`, `tags`, `project`, `similarity`). `SearchResult` extends it with
storage metadata (`session_id`, `timestamp`, `confidence`, `active_count`).
`RankedResult` extends it with scoring fields (`hotness`, `final_score`).

## Active Count Tracking

`active_count` tracks how many times a learning has been returned in search results.
This feeds the hotness scoring formula, creating a feedback loop:

1. Learning is returned in search results
2. `active_count` is incremented
3. Future searches rank it higher via hotness score
4. Frequently-useful learnings rise; rarely-retrieved ones decay

The increment happens via `LearningStore.increment_active_count()`, called by
`retrieve_learnings()` after ranking. The method fetches all matching rows in a
single query, then applies individual updates.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_results` | `15` | Maximum learnings returned |
| `max_tokens` | `4000` | Token budget for packed results |
| `min_similarity` | `0.3` | Minimum cosine similarity threshold |
| `recency_half_life_days` | `7.0` | Exponential decay half-life in days |
| `hotness_weight` | `0.2` | Blend weight: 0.0 = pure semantic, 1.0 = pure hotness |
| `project_boost` | `1.5` | Multiplicative boost for same-project results |

## CLI Usage

```bash
crowd-control search "how does the auth system work"
```

Output:
```
Searching for: "how does the auth system work"

  [1] (score=0.87) [architecture_decision]
      The auth system uses JWT tokens stored in HttpOnly cookies...
      project=/users/dan/code/webapp  retrieved=5x  age=3d

  [2] (score=0.72) [debugging_insight]
      Auth middleware must be registered before the CORS middleware...
      project=/users/dan/code/webapp  retrieved=2x  age=7d

3 results (searched 142 learnings)
```

Options:
- `--limit N` — override max results
- `--project PATH` — filter by project path
- `--category CAT` — filter by learning category
- `--tag TAG` — filter by tag (repeatable, match-any). Case-insensitive.


