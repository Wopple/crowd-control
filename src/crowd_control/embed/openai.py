"""OpenAI embedding provider."""

from __future__ import annotations

import logging
import os

from crowd_control.embed.base import EmbeddingError

logger = logging.getLogger(__name__)

_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_MODEL_MAX_CHARS = {
    "text-embedding-3-small": 32000,
    "text-embedding-3-large": 32000,
    "text-embedding-ada-002": 32000,
}


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", api_key_env: str | None = None):
        self._model = model
        self._dimensions = _MODEL_DIMENSIONS.get(model)
        self._max_chars = _MODEL_MAX_CHARS.get(model, 32000)

        key_var = api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(key_var)
        if not api_key:
            raise ValueError(f"OpenAI API key not found. Set the {key_var} environment variable.")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Run: pip install crowd-control[openai]"
            )

        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        truncated = [t[: self._max_chars] for t in texts]
        try:
            response = self._client.embeddings.create(input=truncated, model=self._model)
        except Exception as e:
            raise EmbeddingError(f"OpenAI API error: {e}") from e
        embeddings = [item.embedding for item in response.data]

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
