"""Background ingestion worker.

Processes queued ingestion jobs written by the SessionEnd hook.
Normally auto-spawned by the hook, but can also be run manually.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from crowd_control.config import CrowdControlConfig
from crowd_control.ingest.pipeline import ingest_session
from crowd_control.storage.db import LearningStore

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


@dataclass
class _QueueEntry:
    """Parsed queue file with its path."""

    path: Path
    data: dict
    queued_at: str


def process_queue(config: CrowdControlConfig) -> int:
    """Process all queued ingestion jobs.

    Returns the number of sessions successfully ingested.
    """
    queue_dir = Path(config.storage_dir).expanduser() / "queue"
    if not queue_dir.exists():
        return 0

    entries = _load_queue_entries(queue_dir)
    if not entries:
        return 0

    ingested = 0
    for entry in entries:
        if _process_one(entry, config):
            ingested += 1

    return ingested


def _load_queue_entries(queue_dir: Path) -> list[_QueueEntry]:
    """Read and parse all queue files once, sorted by queued_at."""
    entries = []
    for path in queue_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            queued_at = data.get("queued_at", path.name)
            entries.append(_QueueEntry(path=path, data=data, queued_at=queued_at))
        except (json.JSONDecodeError, OSError):
            logger.warning("Malformed queue file: %s", path)
            _move_to_failed(path)
    entries.sort(key=lambda e: e.queued_at)
    return entries


def _process_one(entry: _QueueEntry, config: CrowdControlConfig) -> bool:
    """Process a single queue entry. Returns True if ingestion succeeded."""
    data = entry.data
    queue_file = entry.path

    session_id = data.get("session_id")
    session_path_str = data.get("session_path")

    if not session_id or not session_path_str:
        logger.warning("Queue file missing required fields: %s", queue_file)
        _move_to_failed(queue_file)
        return False

    session_path = Path(session_path_str)

    # Check if session file still exists
    if not session_path.exists():
        logger.info("Session file no longer exists, removing queue file: %s", session_path)
        queue_file.unlink(missing_ok=True)
        return False

    # Check if already ingested
    try:
        store = LearningStore(config.db_path)
        if store.has_session(session_id):
            logger.info("Session already ingested, removing queue file: %s", session_id)
            queue_file.unlink(missing_ok=True)
            return False
    except ValueError:
        # DB doesn't exist yet — not ingested
        pass

    # Run ingestion
    try:
        result = ingest_session(session_path, config)
        logger.info(
            "Ingested session %s: %d learnings stored",
            result.session_id,
            result.learnings_stored,
        )
        queue_file.unlink(missing_ok=True)
        return True
    except Exception as exc:
        logger.exception("Ingestion failed for %s", session_path)
        _handle_failure(queue_file, data, str(exc))
        return False


def _handle_failure(queue_file: Path, data: dict, error_message: str) -> None:
    """Increment attempt count; move to failed/ after MAX_ATTEMPTS."""
    attempts = data.get("attempts", 0) + 1
    if attempts >= MAX_ATTEMPTS:
        _move_to_failed(queue_file)
    else:
        updated = {**data, "attempts": attempts, "last_error": error_message}
        queue_file.write_text(json.dumps(updated, indent=2))


def _move_to_failed(queue_file: Path) -> None:
    """Move a queue file to the failed/ subdirectory."""
    failed_dir = queue_file.parent / "failed"
    failed_dir.mkdir(exist_ok=True)
    dest = failed_dir / queue_file.name
    shutil.move(str(queue_file), str(dest))
