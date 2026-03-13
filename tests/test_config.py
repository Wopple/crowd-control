"""Tests for configuration loading."""

from crowd_control.config import (
    CrowdControlConfig,
    EmbeddingConfig,
    IngestionConfig,
    load_config,
)


def test_load_defaults_when_no_file(tmp_path):
    config = load_config(tmp_path / "nonexistent.toml")
    assert config == CrowdControlConfig()
    assert config.storage_dir == "~/.crowd-control"
    assert config.embedding.provider == "ollama"
    assert config.embedding.model == "nomic-embed-text"
    assert config.ingestion.dedup_threshold == 0.95


def test_load_partial_config(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[embedding]\nprovider = "voyage"\nmodel = "voyage-code-3"\n')
    config = load_config(config_file)
    assert config.embedding.provider == "voyage"
    assert config.embedding.model == "voyage-code-3"
    # Unspecified sections use defaults
    assert config.distillation.model == "haiku"
    assert config.ingestion.dedup_threshold == 0.95


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
recency_decay = 0.9
project_boost = 2.0

[ingestion]
auto_ingest = false
batch_size = 3
dedup_threshold = 0.9

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
    assert config.distillation.max_learnings_per_session == 10
    assert config.retrieval.max_results == 5
    assert config.ingestion == IngestionConfig(auto_ingest=False, batch_size=3, dedup_threshold=0.9)
    assert config.knowledge.scope == "shared"


def test_expand_tilde_in_db_path():
    config = CrowdControlConfig(storage_dir="~/my-data")
    assert "~" not in config.db_path
    assert config.db_path.endswith("/my-data/db")


def test_db_path_derived_from_storage_dir():
    config = CrowdControlConfig(storage_dir="/opt/cc")
    assert config.db_path == "/opt/cc/db"
