"""Hook handler logic for Claude Code integration.

Pure functions that CLI hook subcommands call. Keeping logic separate
from the CLI makes it testable without invoking click.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from crowd_control.config import CrowdControlConfig

logger = logging.getLogger(__name__)


_MAX_SESSION_ID_LENGTH = 200


@dataclass
class QueueResult:
    """Result of queuing a session for ingestion."""

    session_path: Path | None
    queue_file: Path | None
    skipped_reason: str | None = None
    worker_spawned: bool = False


def handle_session_end_hook(
    event: dict,
    config: CrowdControlConfig,
) -> QueueResult:
    """Process a SessionEnd hook event: validate the session file and queue it.

    Args:
        event: Parsed JSON from stdin (the SessionEnd hook payload).
        config: Loaded configuration.

    Returns:
        QueueResult indicating what happened.
    """
    # 1. Validate required fields
    required = ["session_id", "transcript_path", "cwd"]
    missing = [f for f in required if f not in event]
    if missing:
        return QueueResult(
            session_path=None,
            queue_file=None,
            skipped_reason=f"missing fields: {', '.join(missing)}",
        )

    # 2. Check auto_ingest config
    if not config.ingestion.auto_ingest:
        return QueueResult(
            session_path=None,
            queue_file=None,
            skipped_reason="auto_ingest disabled",
        )

    session_id = event["session_id"]
    transcript_path = Path(event["transcript_path"])
    cwd = event["cwd"]

    # 3. Verify transcript exists
    if not transcript_path.exists():
        return QueueResult(
            session_path=None,
            queue_file=None,
            skipped_reason=f"transcript file not found: {transcript_path}",
        )

    # 4. Sanitize session_id for filename use
    sanitized_id = _sanitize_session_id(session_id)

    # 5. Build queue directory and write queue file
    queue_dir = Path(config.storage_dir).expanduser() / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    queue_file = queue_dir / f"{sanitized_id}.json"
    queue_data = {
        "session_id": session_id,
        "session_path": str(transcript_path),
        "project": cwd,
        "queued_at": datetime.now(UTC).isoformat(),
    }
    queue_file.write_text(json.dumps(queue_data, indent=2))
    logger.info("Queued session %s for ingestion", session_id)

    # 6. Spawn worker
    worker_spawned = spawn_worker(config)

    return QueueResult(
        session_path=transcript_path,
        queue_file=queue_file,
        worker_spawned=worker_spawned,
    )


def spawn_worker(config: CrowdControlConfig) -> bool:
    """Spawn a detached worker process to process the ingestion queue.

    Returns True if the worker was spawned, False on failure.
    The worker runs with CLAUDECODE removed from the environment
    so it can call claude -p for distillation.
    """
    try:
        worker_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        log_dir = Path(config.storage_dir).expanduser() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "worker.err"

        stderr_file = open(log_path, "a")  # noqa: SIM115
        try:
            subprocess.Popen(
                [sys.executable, "-m", "crowd_control.cli", "worker"],
                env=worker_env,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
        finally:
            stderr_file.close()
        logger.info("Spawned background worker")
        return True
    except OSError:
        logger.exception("Failed to spawn worker")
        return False


def _sanitize_session_id(session_id: str) -> str:
    """Sanitize a session ID for safe use as a filename."""
    sanitized = session_id.replace("\x00", "")
    sanitized = sanitized.replace("/", "_").replace("\\", "_")
    sanitized = sanitized.replace("..", "_")
    sanitized = sanitized.replace(":", "_")
    if len(sanitized) > _MAX_SESSION_ID_LENGTH:
        sanitized = sanitized[:_MAX_SESSION_ID_LENGTH]
    return sanitized
