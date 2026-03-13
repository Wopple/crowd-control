"""Shared test fixtures for crowd-control."""

from __future__ import annotations

import hashlib


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
            int.from_bytes(h[i : i + 2], "big") / 65535.0
            for i in range(0, self._dimensions * 2, 2)
        ]
        norm = sum(x**2 for x in raw) ** 0.5
        return [x / norm for x in raw]
