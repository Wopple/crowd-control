"""Ollama distiller implementation."""

from __future__ import annotations

import json
import logging
import time

# Module-level import (not lazy). If `ollama` isn't installed, importing this
# module raises ImportError — which the factory in llm/base.py catches and
# converts to a DistillationError with actionable instructions.
import httpx
import ollama

from crowd_control.ingest.llm.base import DistillationError

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Distiller that calls the local Ollama daemon via the `ollama` Python client.

    Uses Ollama's `format=<schema>` for JSON-schema-constrained decoding
    (Ollama >= 0.5). The response is the schema-conformant JSON directly as
    the message content — no `structured_output` wrapper to strip.
    """

    DEFAULT_TIMEOUT_SECONDS: float = 300.0

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        # `ollama` uses httpx with a short default timeout. CPU inference of an
        # 8B model can run 30–60s per segment; the default would trip well
        # before generation completes. We build a Client with an explicit
        # timeout once and reuse it.
        self._client = ollama.Client(timeout=timeout_seconds)
        # `keep_alive` controls how long Ollama keeps the model loaded between
        # calls. A full session ingestion on CPU can take 30+ minutes; using
        # "1h" prevents the model from being unloaded between segments and
        # incurring a multi-second reload penalty on each subsequent call.
        self._keep_alive = "1h"

    @property
    def recommended_concurrency(self) -> int:
        return 1

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_id(self) -> str:
        return self._model

    def generate_structured(self, prompt: str, schema: dict) -> dict:
        started = time.monotonic()
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                format=schema,
                options={"temperature": self._temperature},
                stream=False,
                keep_alive=self._keep_alive,
            )
        except ollama.ResponseError as e:
            raise self._map_response_error(e) from e
        except (httpx.ConnectError, ConnectionError) as e:
            raise DistillationError(
                "Ollama not running. Start it with: ollama serve"
            ) from e
        except httpx.TimeoutException as e:
            raise DistillationError(
                f"Ollama call timed out after {self._timeout_seconds}s. "
                "If generation is genuinely this slow, increase OllamaLLM "
                "timeout_seconds or choose a smaller model."
            ) from e
        except Exception as e:
            # Final-fallback hedge: some `ollama-python` versions wrap
            # underlying errors in their own exceptions whose class isn't part
            # of the public API. Substring-match the message as a last resort
            # so users still get the actionable "ollama serve" hint.
            msg = str(e).lower()
            if "connection" in msg or "refused" in msg or "connect" in msg:
                raise DistillationError(
                    "Ollama not running. Start it with: ollama serve"
                ) from e
            raise DistillationError(f"Ollama chat() failed: {e}") from e

        content = response.message.content if hasattr(response, "message") else None
        if not content:
            raise DistillationError("Ollama returned an empty response body")

        elapsed = time.monotonic() - started
        logger.debug(
            "ollama chat returned %d bytes (model=%s, elapsed=%.1fs)",
            len(content),
            self._model,
            elapsed,
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise DistillationError(
                f"Ollama returned non-JSON content despite format=schema: {e}"
            ) from e

    def _map_response_error(self, err: ollama.ResponseError) -> DistillationError:
        msg = str(err).lower()
        if "not found" in msg or "no such model" in msg or "pull" in msg:
            return DistillationError(
                f"Model '{self._model}' not pulled. "
                f"Run: ollama pull {self._model}"
            )
        return DistillationError(f"Ollama returned an error: {err}")
