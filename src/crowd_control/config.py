"""Configuration loading and defaults."""

from __future__ import annotations

import dataclasses
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("~/.crowd-control/config.toml")


class ConfigError(Exception):
    """Raised when configuration loading fails (e.g., invalid TOML)."""


# Distillation provider / model resolution.

# Legacy bare aliases (no provider prefix). These exist for backwards
# compatibility with pre-0.0.5 configs and as user-friendly shorthand.
_CLAUDE_MODEL_ALIASES: frozenset[str] = frozenset({"haiku", "sonnet", "opus"})

# Per-provider default models, used when the value is just the provider name.
_DEFAULT_MODEL_FOR_PROVIDER: dict[str, str] = {
    "claude": "haiku",
    "ollama": "qwen3:8b",
}

# Map external provider names (in config.toml) to internal names.
EXTERNAL_TO_INTERNAL_PROVIDER: dict[str, str] = {
    "ollama": "ollama",
    "claude-code": "claude",
}
INTERNAL_TO_EXTERNAL_PROVIDER: dict[str, str] = {
    "ollama": "ollama",
    "claude": "claude-code",
}


def parse_distillation_model(raw: str) -> tuple[str, str]:
    """Resolve a distillation `model` string to (internal_provider, model_id).

    Resolution rules (first match wins):
    1. Empty/whitespace → ConfigError.
    2. Contains ':' → split on first colon; prefix must be a valid external
       provider name, suffix non-empty.
    3. Bare external provider name ('ollama' / 'claude-code') → provider's
       default model.
    4. Legacy bare Claude alias (haiku/sonnet/opus) or starts with 'claude-'
       → ('claude', raw).
    5. Otherwise → ConfigError listing valid forms.

    The parser does NOT validate that a given (provider, model_id) pair is
    coherent — e.g. `ollama:haiku` is accepted because we cannot enumerate
    every Ollama tag a user might pull. Such mismatches surface at first
    LLM call with the backend's own actionable error (e.g. "Run: ollama
    pull haiku").
    """
    if raw is None:
        raise ConfigError(
            "distillation.model is missing. "
            "Expected e.g. 'ollama:qwen3:8b' or 'claude-code:haiku'."
        )
    value = raw.strip()
    if not value:
        raise ConfigError(
            "distillation.model is empty. "
            "Expected e.g. 'ollama:qwen3:8b' or 'claude-code:haiku'."
        )

    if ":" in value:
        prefix, _, suffix = value.partition(":")
        provider_external = prefix.strip()
        model_id = suffix.strip()
        if not model_id:
            raise ConfigError(
                f"distillation.model {raw!r} has empty model part after ':'"
            )
        if provider_external not in EXTERNAL_TO_INTERNAL_PROVIDER:
            raise ConfigError(
                f"distillation.model {raw!r} uses unknown provider "
                f"{provider_external!r}. Valid providers: 'ollama', 'claude-code'."
            )
        return (EXTERNAL_TO_INTERNAL_PROVIDER[provider_external], model_id)

    if value in EXTERNAL_TO_INTERNAL_PROVIDER:
        internal = EXTERNAL_TO_INTERNAL_PROVIDER[value]
        return (internal, _DEFAULT_MODEL_FOR_PROVIDER[internal])

    if value in _CLAUDE_MODEL_ALIASES or value.startswith("claude-"):
        return ("claude", value)

    raise ConfigError(
        f"distillation.model {raw!r} is not recognised. "
        f"Expected: 'ollama:<model>', 'claude-code:<alias>', "
        f"'ollama', 'claude-code', or one of: "
        f"{', '.join(sorted(_CLAUDE_MODEL_ALIASES))}."
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    api_key_env: str | None = None


@dataclass(frozen=True)
class DistillationConfig:
    """Distillation provider configuration.

    `model` is the user-facing source of truth (e.g. 'ollama:qwen3:8b').
    `resolved_provider` and `resolved_model_id` are derived in __post_init__
    so downstream code never re-parses the identifier.
    """

    model: str = "ollama:qwen3:8b"
    max_learnings_per_session: int = 20
    resolved_provider: str = dataclasses.field(init=False)
    resolved_model_id: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        provider, model_id = parse_distillation_model(self.model)
        object.__setattr__(self, "resolved_provider", provider)
        object.__setattr__(self, "resolved_model_id", model_id)


@dataclass(frozen=True)
class RetrievalConfig:
    max_results: int = 15
    max_tokens: int = 4000
    min_similarity: float = 0.4
    min_score: float = 0.4
    recency_half_life_days: float = 7.0
    hotness_weight: float = 0.2
    project_boost: float = 1.5

    def __post_init__(self):
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")
        if not (0.0 <= self.hotness_weight <= 1.0):
            raise ValueError("hotness_weight must be between 0.0 and 1.0")


@dataclass(frozen=True)
class IngestionConfig:
    auto_ingest: bool = True
    agent_ingest: bool = True
    agent_delete: bool = True
    batch_size: int = 5
    dedup_threshold: float = 0.90
    max_age_days: int = 90
    retention_retrieval_interval_days: int = 30


_VALID_SCOPES = frozenset({"project", "shared", "mixed"})


@dataclass(frozen=True)
class KnowledgeConfig:
    scope: str = "project"

    def __post_init__(self):
        if self.scope not in _VALID_SCOPES:
            valid = ", ".join(sorted(_VALID_SCOPES))
            raise ValueError(f"scope must be one of: {valid}")


@dataclass(frozen=True)
class CrowdControlConfig:
    storage_dir: str = "~/.crowd-control"
    log_level: str = "off"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)

    @property
    def db_path(self) -> str:
        return str(Path(self.storage_dir).expanduser() / "db")


# Mapping from TOML section names to their dataclass types.
_SECTION_MAP: dict[str, type] = {
    "embedding": EmbeddingConfig,
    "distillation": DistillationConfig,
    "retrieval": RetrievalConfig,
    "ingestion": IngestionConfig,
    "knowledge": KnowledgeConfig,
}


def load_config(config_path: Path | None = None) -> CrowdControlConfig:
    """Load config from file, falling back to defaults for missing keys.

    If config_path is None, looks for ~/.crowd-control/config.toml.
    If the file doesn't exist, returns all defaults.
    """
    path = (config_path or _DEFAULT_CONFIG_PATH).expanduser()

    if not path.exists():
        return CrowdControlConfig()

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e

    # Extract top-level (general) fields
    general = raw.get("general", {})
    top_kwargs: dict = {}
    if "storage_dir" in general:
        top_kwargs["storage_dir"] = general["storage_dir"]
    if "log_level" in general:
        top_kwargs["log_level"] = general["log_level"]

    # Build each section dataclass from its TOML section
    for section_name, cls in _SECTION_MAP.items():
        section_data = raw.get(section_name, {})
        if section_data:
            valid_fields = {f.name for f in dataclasses.fields(cls) if f.init}
            filtered = {k: v for k, v in section_data.items() if k in valid_fields}
            try:
                top_kwargs[section_name] = cls(**filtered)
            except ConfigError as e:
                raise ConfigError(f"{path}: [{section_name}] {e}") from e

    cfg = CrowdControlConfig(**top_kwargs)

    # INFO log when a legacy bare distillation alias resolved to a prefixed form,
    # so users know what's in effect and how to write it explicitly.
    raw_distillation_model = raw.get("distillation", {}).get("model")
    if raw_distillation_model and ":" not in raw_distillation_model:
        canonical = (
            f"{INTERNAL_TO_EXTERNAL_PROVIDER[cfg.distillation.resolved_provider]}:"
            f"{cfg.distillation.resolved_model_id}"
        )
        logger.info(
            "distillation.model %r resolved to %s. "
            "Write it as %r in config.toml to silence this message.",
            raw_distillation_model,
            canonical,
            canonical,
        )

    return cfg
