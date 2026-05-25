"""Readiness check for the distillation provider, used by `status`."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from crowd_control.config import INTERNAL_TO_EXTERNAL_PROVIDER, DistillationConfig

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class DistillationStatus:
    """Resolved distillation provider/model plus readiness check result."""

    provider: str        # external name: "ollama" or "claude-code"
    model: str
    ready: bool
    hint: str | None     # actionable next step when not ready, else None


def check_distillation_status(cfg: DistillationConfig) -> DistillationStatus:
    """Probe the configured distillation backend for readiness.

    For Ollama: confirm the daemon is reachable and the model is pulled.
    For Claude: confirm the `claude` CLI is on PATH.

    Never raises — all failure modes resolve to ready=False with a hint.
    """
    internal = cfg.resolved_provider
    external = INTERNAL_TO_EXTERNAL_PROVIDER.get(internal, internal)
    model = cfg.resolved_model_id

    if internal == "ollama":
        return _check_ollama(external, model)
    if internal == "claude":
        return _check_claude(external, model)
    return DistillationStatus(
        provider=external,
        model=model,
        ready=False,
        hint=f"Unknown provider {internal!r}",
    )


def _check_ollama(external: str, model: str) -> DistillationStatus:
    try:
        import ollama
    except ImportError:
        return DistillationStatus(
            provider=external,
            model=model,
            ready=False,
            hint=(
                "ollama package not installed. "
                "Run: pip install crowd-control[ollama]"
            ),
        )
    try:
        client = ollama.Client(timeout=_PROBE_TIMEOUT_SECONDS)
        listed = client.list()
    except Exception as e:
        logger.debug("Ollama probe failed: %s", e, exc_info=True)
        return DistillationStatus(
            provider=external,
            model=model,
            ready=False,
            hint="Ollama not reachable. Start it with: ollama serve",
        )

    tags = {m.get("name", m.get("model", "")) for m in (listed.get("models") or [])}
    if model in tags or any(t.startswith(f"{model}:") for t in tags):
        return DistillationStatus(provider=external, model=model, ready=True, hint=None)
    return DistillationStatus(
        provider=external,
        model=model,
        ready=False,
        hint=f"Model {model!r} not pulled. Run: ollama pull {model}",
    )


def _check_claude(external: str, model: str) -> DistillationStatus:
    from shutil import which

    if which("claude") is None:
        return DistillationStatus(
            provider=external,
            model=model,
            ready=False,
            hint="claude CLI not found on PATH. Install Claude Code and authenticate.",
        )
    return DistillationStatus(provider=external, model=model, ready=True, hint=None)
