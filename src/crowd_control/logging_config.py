"""Logging configuration for Crowd Control.

Two distinct concerns:
1. Operational logging — stderr output during interactive CLI commands.
2. Trace logging — optional file-based logging for profiling and debugging.

A single configure_logging() call sets up both based on context.
"""

from __future__ import annotations

import logging
from pathlib import Path

from crowd_control.config import CrowdControlConfig

_PACKAGE_LOGGER = "crowd_control"

_STDERR_FORMAT = "%(levelname)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

_LOG_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(
    config: CrowdControlConfig,
    *,
    interactive: bool = True,
    verbose: bool = False,
) -> None:
    """Set up logging handlers for the crowd_control package.

    Args:
        config: Loaded configuration (used for log_level and storage_dir).
        interactive: If True, attach a stderr StreamHandler for operational
            messages. Set to False for background processes (worker, hooks).
        verbose: If True, lower the stderr handler to DEBUG level.
    """
    root = logging.getLogger(_PACKAGE_LOGGER)
    # Clear any existing handlers to avoid duplication on repeated calls
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    if interactive:
        stderr_handler = logging.StreamHandler()
        stderr_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        stderr_handler.setFormatter(logging.Formatter(_STDERR_FORMAT))
        root.addHandler(stderr_handler)

    log_level = config.log_level.lower()
    if log_level != "off" and log_level in _LOG_LEVEL_MAP:
        log_dir = Path(config.storage_dir).expanduser() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "crowd-control.log"

        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(_LOG_LEVEL_MAP[log_level])
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)

    # If no handlers were added, prevent messages from propagating to the
    # root logger (which would print them with Python's default format).
    if not root.handlers:
        root.addHandler(logging.NullHandler())
