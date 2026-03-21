"""End-to-end ingestion pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from crowd_control.config import CrowdControlConfig
from crowd_control.embed.base import create_embedder
from crowd_control.ingest.distiller import distill_session
from crowd_control.ingest.parser import parse_session_file
from crowd_control.storage.db import LearningStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of an ingestion run."""

    session_id: str
    segments_processed: int
    learnings_distilled: int
    learnings_stored: int
    learnings_deduplicated: int


def _wrap_progress(
    callback: Callable[[str, int, int], None] | None, stage: str
) -> Callable[[int, int], None] | None:
    if callback is None:
        return None
    return lambda completed, total: callback(stage, completed, total)


def ingest_session(
    session_path: Path,
    config: CrowdControlConfig,
    max_workers: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> IngestResult:
    """Run the full ingestion pipeline on a session file.

    Steps:
        1. Parse the session file into a Session object
        2. Distill qualifying segments into Learning objects
        3. Embed learning texts into vectors
        4. Store in LanceDB with dedup

    Args:
        session_path: Path to the JSONL session file.
        config: Loaded configuration.
        max_workers: Override for distillation concurrency.
            If None, uses the distiller's default (8).
        progress_callback: Called with (stage_name, completed, total) at each step.

    Returns:
        IngestResult with counts of what was processed.
    """
    # 1. Parse
    logger.info("Parsing session %s", session_path)
    session = parse_session_file(session_path)

    # 2. Distill
    distill_kwargs: dict = {
        "model": config.distillation.model,
        "max_learnings": config.distillation.max_learnings_per_session,
        "progress_callback": _wrap_progress(progress_callback, "distilling"),
    }
    if max_workers is not None:
        distill_kwargs["max_workers"] = max_workers

    logger.info("Distilling %d segments", len(session.segments))
    learnings = distill_session(session, **distill_kwargs)

    if not learnings:
        return IngestResult(
            session_id=session.session_id,
            segments_processed=len(session.segments),
            learnings_distilled=0,
            learnings_stored=0,
            learnings_deduplicated=0,
        )

    # 3. Embed
    logger.info("Embedding %d learnings", len(learnings))
    embedder = create_embedder(config.embedding)
    texts = [learning.text for learning in learnings]
    total_chars = sum(len(t) for t in texts)
    logger.debug("Embedding batch size: %d, total chars: %d", len(texts), total_chars)
    vectors = embedder.embed(texts)

    # 4. Build records and store
    records = []
    for learning, vector in zip(learnings, vectors):
        record = learning.model_dump(mode="python")
        record["vector"] = vector
        records.append(record)

    store = LearningStore(
        config.db_path,
        embedder.dimensions,
        config.ingestion.dedup_threshold,
    )
    add_result = store.add(records)
    logger.info(
        "Stored %d learnings (%d duplicates skipped)",
        add_result.stored,
        len(learnings) - add_result.stored,
    )

    return IngestResult(
        session_id=session.session_id,
        segments_processed=len(session.segments),
        learnings_distilled=len(learnings),
        learnings_stored=add_result.stored,
        learnings_deduplicated=len(learnings) - add_result.stored,
    )
