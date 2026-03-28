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


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    api_key_env: str | None = None


@dataclass(frozen=True)
class DistillationConfig:
    model: str = "haiku"
    max_learnings_per_session: int = 20


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


@dataclass(frozen=True)
class KnowledgeConfig:
    scope: str = "project"


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
            valid_fields = {f.name for f in dataclasses.fields(cls)}
            filtered = {k: v for k, v in section_data.items() if k in valid_fields}
            top_kwargs[section_name] = cls(**filtered)

    return CrowdControlConfig(**top_kwargs)
