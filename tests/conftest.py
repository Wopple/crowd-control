"""Shared test fixtures for crowd-control."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_logging():
    """Prevent tests from writing to the production log file.

    Code paths like the CLI call configure_logging() with the user's real
    config (which may have log_level="debug"), attaching a FileHandler to
    ~/.crowd-control/logs/crowd-control.log.

    This fixture patches configure_logging at its definition module so that
    lazy imports in CLI/server code get the mock. Tests that import the real
    function at module level (test_logging_config.py) keep the direct
    reference and are unaffected.
    """
    with patch("crowd_control.logging_config.configure_logging"):
        yield
    logger = logging.getLogger("crowd_control")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def insert_learning(store, embedder, text, **overrides):
    """Insert a learning into a store using the fake embedder.

    Provides sensible defaults for all required fields. Any field
    can be overridden via keyword arguments.
    """
    vector = embedder.embed([text])[0]
    record = {
        "id": overrides.pop("id", f"id-{hash(text) % 10000}"),
        "vector": vector,
        "text": text,
        "category": overrides.pop("category", "debugging_insight"),
        "tags": overrides.pop("tags", []),
        "project": overrides.pop("project", "/test/project"),
        "session_id": overrides.pop("session_id", "sess-001"),
        "git_sha": overrides.pop("git_sha", "abc123"),
        "timestamp": overrides.pop("timestamp", datetime(2025, 1, 15, 10, 0, tzinfo=UTC)),
        "confidence": overrides.pop("confidence", 0.8),
        "active_count": overrides.pop("active_count", 0),
        "stale": overrides.pop("stale", False),
        "shared": overrides.pop("shared", False),
    }
    record.update(overrides)
    store.add([record])
    return record


class FakeEmbedder:
    """Deterministic embedder for tests. Returns hash-based vectors."""

    def __init__(self, dimensions: int = 8):
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._hash_vector(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def max_input_chars(self) -> int:
        return 32000

    def _hash_vector(self, text: str) -> list[float]:
        """Generate a deterministic, normalized vector from text using hashing."""
        h = hashlib.shake_256(text.encode()).digest(self._dimensions * 2)
        raw = [
            int.from_bytes(h[i : i + 2], "big") / 65535.0 for i in range(0, self._dimensions * 2, 2)
        ]
        norm = sum(x**2 for x in raw) ** 0.5
        return [x / norm for x in raw]
