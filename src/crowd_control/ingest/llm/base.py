"""DistillerLLM protocol and factory."""

from __future__ import annotations

import logging
from typing import Protocol

from crowd_control.config import DistillationConfig

logger = logging.getLogger(__name__)


class DistillationError(Exception):
    """Raised when distillation fails (subprocess error, API error, schema violation)."""


class DistillerLLM(Protocol):
    def generate_structured(self, prompt: str, schema: dict) -> dict:
        """Run the LLM with JSON-schema-constrained output.

        Returns the parsed schema-conformant object. Raises DistillationError
        on any failure; messages should be user-actionable.
        """
        ...

    @property
    def recommended_concurrency(self) -> int:
        """Default thread-pool size that suits this backend."""
        ...

    @property
    def provider_name(self) -> str:
        """External provider name as it appears in config (e.g. 'ollama', 'claude-code')."""
        ...

    @property
    def model_id(self) -> str:
        """The model identifier passed to the backend (e.g. 'qwen3:8b', 'haiku')."""
        ...


def create_distiller_llm(config: DistillationConfig) -> DistillerLLM:
    """Construct the LLM implementation indicated by the resolved provider."""
    provider = config.resolved_provider
    model = config.resolved_model_id

    if provider == "claude":
        from crowd_control.ingest.llm.claude import ClaudeCLILLM

        return ClaudeCLILLM(model=model)

    if provider == "ollama":
        try:
            from crowd_control.ingest.llm.ollama import OllamaLLM
        except ImportError as e:
            raise DistillationError(
                "ollama package not installed. Run: pip install crowd-control[ollama]"
            ) from e
        return OllamaLLM(model=model)

    raise DistillationError(f"Unknown distillation provider: {provider!r}")
