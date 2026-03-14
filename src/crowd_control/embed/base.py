"""Embedder protocol and factory."""

from __future__ import annotations

from typing import Protocol

from crowd_control.config import EmbeddingConfig


class EmbeddingError(Exception):
    """Raised when embedding fails (connection error, API error, etc.)."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of embedding vectors, same length as texts.
            Each vector has length == self.dimensions.
        """
        ...

    @property
    def dimensions(self) -> int:
        """The dimensionality of the embedding vectors."""
        ...

    @property
    def max_input_chars(self) -> int:
        """Maximum character length per input text."""
        ...


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Create an embedder from config. Raises EmbeddingError on failure."""
    try:
        match config.provider:
            case "ollama":
                from crowd_control.embed.ollama import OllamaEmbedder

                return OllamaEmbedder(model=config.model)
            case "voyage":
                from crowd_control.embed.voyage import VoyageEmbedder

                return VoyageEmbedder(model=config.model, api_key_env=config.api_key_env)
            case "openai":
                from crowd_control.embed.openai import OpenAIEmbedder

                return OpenAIEmbedder(model=config.model, api_key_env=config.api_key_env)
            case _:
                raise EmbeddingError(f"Unknown embedding provider: {config.provider}")
    except ImportError as e:
        package_hint = {"ollama": "ollama", "voyage": "voyage", "openai": "openai"}.get(
            config.provider, config.provider
        )
        raise EmbeddingError(
            f"{config.provider} package not installed. "
            f"Run: pip install crowd-control[{package_hint}]"
        ) from e
    except ValueError as e:
        raise EmbeddingError(str(e)) from e
