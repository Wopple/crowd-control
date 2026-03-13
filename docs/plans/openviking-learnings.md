# Plan: Introduce OpenViking Learnings into Crowd Control

## Background

OpenViking (github.com/volcengine/OpenViking) is an open-source context database for AI
agents by ByteDance. It solves the same core problem as Crowd Control — giving agents a warm
start from past session learnings — but at platform scale with multi-tenant support, a virtual
filesystem (AGFS), and LLM-powered features throughout.

After a thorough comparison, we identified specific algorithms and design ideas that would
improve Crowd Control's retrieval and ranking quality without importing OpenViking's platform
complexity or its dependency on an async LLM API.

This plan covers Phase 4 (Retrieval and ranking) from `implementation-phases.md`, informed
by OpenViking's battle-tested approaches.

## What We're Taking From OpenViking

### 1. Hotness Scoring

OpenViking tracks how often each piece of context is actually used (retrieved and consumed
by an agent) and blends usage frequency with recency into a single "hotness" score:

```
hotness = sigmoid(log1p(active_count)) * exp(-decay_constant * age_days)
```

Where `decay_constant = ln(2) / half_life_days` (7-day half-life).

This is better than pure recency decay because a frequently-retrieved learning from 2 weeks
ago should rank higher than a never-used learning from yesterday.

**What this requires in Crowd Control:**
- Add `active_count` field to the Learning model and DB schema
- Increment `active_count` when a learning is returned in search results
- Implement the hotness formula in `retrieve/rank.py`

### 2. Score Blending

OpenViking combines semantic similarity with hotness using a weighted blend:

```
final_score = (1 - hotness_weight) * semantic_score + hotness_weight * hotness_score
```

With `hotness_weight = 0.2` (semantic similarity dominates, hotness is a tiebreaker/boost).

**What this requires in Crowd Control:**
- Add a `hotness_weight` config parameter (default 0.2)
- Apply blending in `retrieve/rank.py` after vector search

### 3. Exponential Recency Decay (replacing linear decay)

The current plan uses `recency_decay = 0.95` (multiplier per week). OpenViking uses
exponential decay with a configurable half-life, which is more principled:

```
recency_factor = exp(-ln(2) / half_life_days * age_days)
```

With half-life = 7 days: a 1-week-old learning has 50% recency factor, 2-week has 25%, etc.
This decays faster than the current `0.95^weeks` plan and is more tunable.

**What this requires in Crowd Control:**
- Replace `recency_decay` config with `recency_half_life_days` (default 7)
- Implement exponential decay in `retrieve/rank.py`

### 4. Active Count Tracking

OpenViking's session `used()` method tracks which contexts were accessed during a session
and increments their active counts. This feeds the hotness scoring.

**What this requires in Crowd Control:**
- `LearningStore.increment_active_count(learning_id)` method
- The MCP server's `search_learnings` tool increments counts for returned results
- The hook-based context injection increments counts for injected learnings

## What We're NOT Taking (and Why)

| OpenViking Feature | Why Not |
|----|-----|
| L0/L1/L2 hierarchy | Requires LLM API for summary generation. Crowd Control's flat one-insight-per-embedding design retrieves well for the single-user case. |
| LLM-powered deduplication | Requires async LLM API. Cosine similarity (0.95) is sufficient at current scale. Revisit if DB grows past ~1000 learnings. |
| Intent analysis before search | Requires LLM API. The MCP tool's query string and project filtering provide adequate targeting for now. |
| Hierarchical directory retrieval | Depends on L0/L1/L2 existing. Without the hierarchy, this is just flat vector search — which we already do. |
| Convergence detection | Useful for iterative deepening in a tree. Not applicable to flat vector search. |
| Virtual filesystem (AGFS) | Massive complexity for no benefit in the single-user, flat-learning case. |

## Implementation Steps

### Step 1: Data Model Changes

Add `active_count: int = 0` to the `Learning` model and LanceDB schema. This is a
non-breaking addition — existing records default to 0.

Update `LearningStore` with an `increment_active_count(learning_id)` method.

### Step 2: Implement `retrieve/search.py`

Vector search with metadata filtering against LanceDB. This is the basic retrieval that
the current stub needs:

- Query embedding via the configured embedder
- LanceDB vector search with cosine distance
- Metadata filters: project, category, tags, stale
- Minimum similarity threshold (`min_similarity` config)
- Returns raw results with similarity scores

### Step 3: Implement `retrieve/rank.py`

Post-search ranking using OpenViking-inspired scoring:

- Compute recency factor: `exp(-ln(2) / half_life_days * age_days)`
- Compute hotness: `sigmoid(log1p(active_count)) * recency_factor`
- Blend: `(1 - hotness_weight) * semantic_score + hotness_weight * hotness`
- Apply project boost for same-project matches
- Sort by final score descending
- Pack into token budget using `len(text) / 4` as a token count approximation

**Cold start note:** When `active_count` is 0, `sigmoid(log1p(0)) = sigmoid(0) = 0.5`,
so hotness is `0.5 * recency`, not zero. This means fresh learnings get a mild recency
boost through the hotness channel before any usage data exists. This is intentional — it
avoids a discontinuity between 0 and 1 uses and provides a gentle recency signal from
day one.

**Post-search dedup:** Storage-time dedup (cosine ≥ 0.95) already prevents near-identical
learnings from being stored. Post-search dedup is a safety net for edge cases where
semantically similar but textually distinct learnings both match a query. Use
`SequenceMatcher.ratio()` with a 0.85 threshold, same as `learning-deduplication.md`.
At most `max_results` items, so O(n²) is fine.

### Step 4: Wire Active Count Tracking

When learnings are returned by search (either via MCP tool or hook injection), increment
their active counts. This creates the feedback loop that makes hotness scoring meaningful
over time.

Batch the updates — collect all returned learning IDs and update in a single pass rather
than one DB round-trip per learning. Race conditions on concurrent increments are
acceptable since the counter is approximate.

### Step 5: Config Changes

Update `[retrieval]` config section:

```toml
[retrieval]
max_results = 15
max_tokens = 4000
min_similarity = 0.3
recency_half_life_days = 7    # replaces recency_decay = 0.95
hotness_weight = 0.2          # blend weight for hotness vs semantic
project_boost = 1.5
```

**Half-life tradeoff:** 7 days is borrowed from OpenViking, which targets daily-use
agents. At 7 days, a 2-week-old learning has 25% recency and a 1-month-old has ~5%.
This may feel aggressive for projects worked on intermittently. Users who switch between
projects on longer cycles should increase this value. The default is a starting point
for tuning, not a researched optimum.

## Future Considerations

If the `claude -p` constraint is ever relaxed (i.e., an API key becomes acceptable), the
following OpenViking features become available without architectural changes:

- **LLM-powered dedup** in `LearningStore.add()` — ask the LLM whether a new learning
  should SKIP, CREATE, or MERGE with existing similar learnings
- **Intent analysis** in `retrieve/search.py` — classify the query type before searching
  to improve precision
- **L0/L1 grouping** — cluster related learnings under topic abstracts to reduce injection
  token cost (would require schema changes)

These are natural extensions of the architecture, not rewrites.
