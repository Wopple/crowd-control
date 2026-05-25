"""Tests for configuration loading."""

import pytest

from crowd_control.config import (
    CrowdControlConfig,
    EmbeddingConfig,
    IngestionConfig,
    RetrievalConfig,
    load_config,
)


def test_load_defaults_when_no_file(tmp_path):
    config = load_config(tmp_path / "nonexistent.toml")
    assert config == CrowdControlConfig()
    assert config.storage_dir == "~/.crowd-control"
    assert config.embedding.provider == "ollama"
    assert config.embedding.model == "nomic-embed-text"
    assert config.ingestion.auto_ingest is True
    assert config.ingestion.agent_ingest is True
    assert config.ingestion.dedup_threshold == 0.90
    assert config.ingestion.max_age_days == 90
    assert config.ingestion.retention_retrieval_interval_days == 30


def test_load_partial_config(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[embedding]\nprovider = "voyage"\nmodel = "voyage-code-3"\n')
    config = load_config(config_file)
    assert config.embedding.provider == "voyage"
    assert config.embedding.model == "voyage-code-3"
    # Unspecified sections use defaults
    assert config.distillation.model == "ollama:qwen3:8b"
    assert config.distillation.resolved_provider == "ollama"
    assert config.distillation.resolved_model_id == "qwen3:8b"
    assert config.ingestion.dedup_threshold == 0.90


def test_load_full_config(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("""\
[general]
storage_dir = "/tmp/cc"
log_level = "debug"

[embedding]
provider = "openai"
model = "text-embedding-3-small"
api_key_env = "MY_KEY"

[distillation]
model = "sonnet"
max_learnings_per_session = 10

[retrieval]
max_results = 5
max_tokens = 2000
min_similarity = 0.5
recency_half_life_days = 14.0
hotness_weight = 0.3
project_boost = 2.0

[ingestion]
auto_ingest = false
agent_ingest = false
batch_size = 3
dedup_threshold = 0.9
max_age_days = 60
retention_retrieval_interval_days = 15

[knowledge]
scope = "shared"
""")
    config = load_config(config_file)
    assert config.storage_dir == "/tmp/cc"
    assert config.log_level == "debug"
    assert config.embedding == EmbeddingConfig(
        provider="openai", model="text-embedding-3-small", api_key_env="MY_KEY"
    )
    assert config.distillation.model == "sonnet"
    assert config.distillation.resolved_provider == "claude"
    assert config.distillation.resolved_model_id == "sonnet"
    assert config.distillation.max_learnings_per_session == 10
    assert config.retrieval.max_results == 5
    assert config.retrieval.recency_half_life_days == 14.0
    assert config.retrieval.hotness_weight == 0.3
    assert config.retrieval.project_boost == 2.0
    assert config.ingestion == IngestionConfig(
        auto_ingest=False,
        agent_ingest=False,
        agent_delete=True,
        batch_size=3,
        dedup_threshold=0.9,
        max_age_days=60,
        retention_retrieval_interval_days=15,
    )
    assert config.knowledge.scope == "shared"


def test_legacy_distillation_alias_logs_info(tmp_path, caplog):
    """Bare legacy `model = 'haiku'` should still work and emit an INFO line."""
    import logging

    config_file = tmp_path / "config.toml"
    config_file.write_text('[distillation]\nmodel = "haiku"\n')
    with caplog.at_level(logging.INFO, logger="crowd_control.config"):
        config = load_config(config_file)
    assert config.distillation.resolved_provider == "claude"
    assert config.distillation.resolved_model_id == "haiku"
    assert any("resolved to claude-code:haiku" in r.message for r in caplog.records)


def test_prefixed_distillation_model_no_info_log(tmp_path, caplog):
    """Explicit prefixed form should not emit the legacy-alias INFO line."""
    import logging

    config_file = tmp_path / "config.toml"
    config_file.write_text('[distillation]\nmodel = "ollama:qwen3:8b"\n')
    with caplog.at_level(logging.INFO, logger="crowd_control.config"):
        load_config(config_file)
    assert not any("resolved to" in r.message for r in caplog.records)


def test_invalid_distillation_model_raises_with_path(tmp_path):
    """ConfigError from DistillationConfig should be re-raised with the file path."""
    from crowd_control.config import ConfigError

    config_file = tmp_path / "config.toml"
    config_file.write_text('[distillation]\nmodel = "gemini"\n')
    with pytest.raises(ConfigError, match=str(config_file)):
        load_config(config_file)


def test_expand_tilde_in_db_path():
    config = CrowdControlConfig(storage_dir="~/my-data")
    assert "~" not in config.db_path
    assert config.db_path.endswith("/my-data/db")


def test_db_path_derived_from_storage_dir():
    config = CrowdControlConfig(storage_dir="/opt/cc")
    assert config.db_path == "/opt/cc/db"


def test_retrieval_defaults():
    config = RetrievalConfig()
    assert config.recency_half_life_days == 7.0
    assert config.hotness_weight == 0.2
    assert config.project_boost == 1.5
    assert config.max_results == 15
    assert config.max_tokens == 4000
    assert config.min_similarity == 0.4
    assert config.min_score == 0.4


def test_retrieval_half_life_zero_raises():
    with pytest.raises(ValueError, match="recency_half_life_days must be positive"):
        RetrievalConfig(recency_half_life_days=0)


def test_retrieval_half_life_negative_raises():
    with pytest.raises(ValueError, match="recency_half_life_days must be positive"):
        RetrievalConfig(recency_half_life_days=-1)


def test_retrieval_hotness_weight_out_of_range():
    with pytest.raises(ValueError, match="hotness_weight must be between"):
        RetrievalConfig(hotness_weight=1.5)
    with pytest.raises(ValueError, match="hotness_weight must be between"):
        RetrievalConfig(hotness_weight=-0.1)


def test_old_config_with_recency_decay_loads(tmp_path):
    """Old configs with recency_decay should load without error (key is filtered)."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""\
[retrieval]
max_results = 10
recency_decay = 0.95
""")
    config = load_config(config_file)
    assert config.retrieval.max_results == 10
    # recency_decay is silently ignored, defaults apply
    assert config.retrieval.recency_half_life_days == 7.0
    assert config.retrieval.hotness_weight == 0.2


def test_agent_delete_defaults_to_true():
    config = CrowdControlConfig()
    assert config.ingestion.agent_delete is True


def test_agent_delete_from_toml(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[ingestion]\nagent_delete = false\n")
    config = load_config(config_file)
    assert config.ingestion.agent_delete is False


def test_unknown_keys_in_any_section_filtered(tmp_path):
    """Unknown keys in any config section should be silently filtered."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""\
[embedding]
provider = "ollama"
some_future_key = "value"
""")
    config = load_config(config_file)
    assert config.embedding.provider == "ollama"
