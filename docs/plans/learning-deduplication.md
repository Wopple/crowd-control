# Plan: Learning Deduplication

## Context

When distilling a session with multiple segments, independent LLM calls can extract
the same learning from different segments. This produces duplicates in the output.
The parallel distillation plan (see `parallel-distillation.md`) makes this worse since
each segment is processed without knowledge of others.

Phase 3 (`implementation-phases.md`) already plans embedding-based dedup at storage time.
This plan covers a lighter-weight dedup step that runs immediately after distillation,
before storage exists.

## Scope

**In scope:** Deduplicating learnings within a single `distill_session` call — i.e.,
across segments of the same session.

**Out of scope:** Cross-session dedup (that's Phase 3, embedding similarity against
the stored corpus).

## Approach

Add a pure function `deduplicate_learnings(learnings: list[Learning]) -> list[Learning]`
that removes duplicates from a flat list. Call it in `distill_session` after collecting
all learnings and before the confidence-based capping step.

### Dedup strategy: text similarity (no embeddings)

Since we don't have an embedding model available at this stage (and tests must not
query one per CLAUDE.md), use text-based similarity:

1. **Exact match** — identical `text` after normalization (lowercase, strip whitespace).
   Keep the one with higher confidence.

2. **High-overlap match** — two learnings whose normalized text shares a high ratio of
   overlap. Use `difflib.SequenceMatcher.ratio()` with a threshold (e.g., 0.85). This
   catches near-duplicates like rephrasings or minor wording differences. Keep the one
   with higher confidence; on tie, keep the one that appeared first (earlier segment).

### Why not embeddings?

- No embedding model is available yet (Phase 3 dependency)
- Tests cannot call an embedding model (CLAUDE.md rule)
- `SequenceMatcher` is stdlib, zero dependencies, fast for the volume we handle
  (max ~60 learnings per session = ~1,800 pairs)
- For within-session dedup, text overlap catches the dominant case (same insight
  extracted from overlapping context). Semantic dedup (different words, same meaning)
  is better handled at storage time with real embeddings.

## Integration point

In `distill_session`, after the `for`/thread-pool loop collects `all_learnings` and
before the `max_learnings` capping step:

```
all_learnings = deduplicate_learnings(all_learnings)
```

## Decisions

1. **Threshold:** 0.85 `SequenceMatcher.ratio()`. No near-miss logging — keep it simple.

2. **Cross-category dedup:** Yes — merge regardless of category. Text similarity is the
   signal; different categories on near-identical text is just LLM inconsistency, not a
   meaningful distinction.
