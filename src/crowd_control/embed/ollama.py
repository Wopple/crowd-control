"""Ollama embedding provider."""

from __future__ import annotations

import logging

from crowd_control.embed.base import EmbeddingError

logger = logging.getLogger(__name__)

_MODEL_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}

_MODEL_MAX_CHARS = {
    "nomic-embed-text": 32000,
    "mxbai-embed-large": 32000,
    "all-minilm": 16000,
}


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text"):
        self._model = model
        self._dimensions = _MODEL_DIMENSIONS.get(model)
        self._max_chars = _MODEL_MAX_CHARS.get(model, 32000)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Ollama.

        Calls ollama.embed() which handles batching internally.
        Truncates texts that exceed max_input_chars.

        Raises EmbeddingError if Ollama is not reachable.
        """
        if not texts:
            return []

        try:
            import ollama as ollama_client
        except ImportError:
            raise ImportError(
                "Ollama package not installed. Run: pip install crowd-control[ollama]"
            )

        truncated = [t[: self._max_chars] for t in texts]
        try:
            response = ollama_client.embed(model=self._model, input=truncated)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to connect to Ollama. Is it running? (ollama serve): {e}"
            ) from e

        embeddings = response.embeddings

        if self._dimensions is None and embeddings:
            self._dimensions = len(embeddings[0])

        return embeddings

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError(
                f"Dimensions unknown for model '{self._model}'. "
                "Call embed() first or add the model to _MODEL_DIMENSIONS."
            )
        return self._dimensions

    @property
    def max_input_chars(self) -> int:
        return self._max_chars
