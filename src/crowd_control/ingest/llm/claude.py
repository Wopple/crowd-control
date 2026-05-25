"""Claude CLI distiller implementation."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

from crowd_control.hooks import INGEST_MARKER_ENV
from crowd_control.ingest.llm.base import DistillationError

logger = logging.getLogger(__name__)


def _extract_structured_output(parsed: dict | list) -> dict | None:
    """Pull `structured_output` out of a claude -p response.

    Handles two shapes:
    - JSON array of streaming events (current): take the final `result` event.
    - Single JSON object (legacy): read the top-level `structured_output`.
    """
    if isinstance(parsed, list):
        for event in reversed(parsed):
            if isinstance(event, dict) and event.get("type") == "result":
                return event.get("structured_output")
        return None
    if isinstance(parsed, dict):
        return parsed.get("structured_output")
    return None


class ClaudeCLILLM:
    """Distiller that shells out to `claude -p`.

    Preserves both subprocess-shaped guards:
    - CLAUDECODE refuse-to-run check.
    - CROWD_CONTROL_INGESTING marker injected on subprocess env so the
      SessionEnd hook that fires for the exiting `claude -p` recognises itself
      and refuses to queue another ingestion.
    """

    DEFAULT_TIMEOUT = 120

    def __init__(self, model: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._model = model
        self._timeout = timeout

    @property
    def recommended_concurrency(self) -> int:
        return 8

    @property
    def provider_name(self) -> str:
        return "claude-code"

    @property
    def model_id(self) -> str:
        return self._model

    def generate_structured(self, prompt: str, schema: dict) -> dict:
        if os.environ.get("CLAUDECODE"):
            raise DistillationError(
                "Cannot call claude -p from inside Claude Code (CLAUDECODE env var is set)"
            )

        cmd = [
            "claude",
            "-p",
            "--model",
            self._model,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, separators=(",", ":")),
            "--no-session-persistence",
        ]

        max_retries = 2
        backoff = [2, 5]
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                result = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env={**os.environ, INGEST_MARKER_ENV: "1"},
                )
            except FileNotFoundError:
                raise DistillationError("claude CLI not found. Is it installed and on PATH?")
            except subprocess.TimeoutExpired as e:
                last_error = DistillationError(f"claude CLI timed out after {self._timeout}s")
                if attempt < max_retries:
                    logger.warning(
                        "claude CLI timed out (attempt %d/%d), retrying in %ds",
                        attempt + 1,
                        max_retries + 1,
                        backoff[attempt],
                    )
                    time.sleep(backoff[attempt])
                    continue
                raise last_error from e

            if result.returncode != 0:
                last_error = DistillationError(
                    f"claude CLI exited with code {result.returncode}: {result.stderr[:200]}"
                )
                if attempt < max_retries:
                    logger.warning(
                        "claude CLI failed with exit code %d (attempt %d/%d), retrying in %ds",
                        result.returncode,
                        attempt + 1,
                        max_retries + 1,
                        backoff[attempt],
                    )
                    time.sleep(backoff[attempt])
                    continue
                raise last_error

            logger.debug("claude CLI returned %d bytes of output", len(result.stdout))
            try:
                parsed = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise DistillationError(
                    f"claude CLI returned invalid JSON (length={len(result.stdout)}): {e}"
                ) from e

            structured = _extract_structured_output(parsed)
            if structured is None:
                raise DistillationError("claude CLI response missing 'structured_output' key")
            return structured

        raise last_error or DistillationError("Unexpected retry exhaustion")
