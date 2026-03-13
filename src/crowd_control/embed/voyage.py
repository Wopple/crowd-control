"""Voyage AI embedding provider."""

from __future__ import annotations

import logging
import os

from crowd_control.embed.base import EmbeddingError

logger = logging.getLogger(__name__)

_MODEL_DIMENSIONS = {
    "voyage-code-3": 1024,
    "voyage-3": 1024,
}

_MODEL_MAX_CHARS = {
    "voyage-code-3": 64000,
    "voyage-3": 32000,
}


class VoyageEmbedder:
    def __init__(self, model: str = "voyage-code-3", api_key_env: str | None = None):
        self._model = model
        self._dimensions = _MODEL_DIMENSIONS.get(model)
        self._max_chars = _MODEL_MAX_CHARS.get(model, 32000)

        key_var = api_key_env or "VOYAGE_API_KEY"
        api_key = os.environ.get(key_var)
        if not api_key:
            raise ValueError(f"Voyage API key not found. Set the {key_var} environment variable.")

        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "Voyage AI package not installed. Run: pip install crowd-control[voyage]"
            )

        self._client = voyageai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        truncated = [t[: self._max_chars] for t in texts]
        try:
            result = self._client.embed(truncated, model=self._model)
        except Exception as e:
            raise EmbeddingError(f"Voyage API error: {e}") from e
        embeddings = result.embeddings

        if self._dimensions is None and embeddings:
            self._dimensions = len(embeddings[0])

        return embeddings

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError(f"Dimensions unknown for model '{self._model}'.")
        return self._dimensions

    @property
    def max_input_chars(self) -> int:
        return self._max_chars
