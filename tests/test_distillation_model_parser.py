"""Tests for parse_distillation_model and DistillationConfig __post_init__."""

import pytest

from crowd_control.config import (
    ConfigError,
    DistillationConfig,
    parse_distillation_model,
)


class TestParseDistillationModel:
    def test_prefixed_ollama_with_colon_in_tag(self):
        assert parse_distillation_model("ollama:qwen3:8b") == ("ollama", "qwen3:8b")

    def test_prefixed_claude_code(self):
        assert parse_distillation_model("claude-code:haiku") == ("claude", "haiku")

    def test_prefixed_claude_code_full_name(self):
        assert parse_distillation_model("claude-code:claude-3-5-sonnet") == (
            "claude",
            "claude-3-5-sonnet",
        )

    @pytest.mark.parametrize("alias", ["haiku", "sonnet", "opus"])
    def test_legacy_bare_alias(self, alias):
        assert parse_distillation_model(alias) == ("claude", alias)

    def test_legacy_claude_prefix_bare(self):
        assert parse_distillation_model("claude-3-5-sonnet") == (
            "claude",
            "claude-3-5-sonnet",
        )

    def test_bare_ollama_uses_default_model(self):
        assert parse_distillation_model("ollama") == ("ollama", "qwen3:8b")

    def test_bare_claude_code_uses_default_model(self):
        assert parse_distillation_model("claude-code") == ("claude", "haiku")

    @pytest.mark.parametrize("raw", ["", "   ", "\t"])
    def test_empty_or_whitespace_rejected(self, raw):
        with pytest.raises(ConfigError, match="empty"):
            parse_distillation_model(raw)

    def test_empty_model_after_colon_rejected(self):
        with pytest.raises(ConfigError, match="empty model part"):
            parse_distillation_model("ollama:")

    def test_unknown_provider_prefix_rejected(self):
        with pytest.raises(ConfigError, match="unknown provider"):
            parse_distillation_model("voyage:foo")

    def test_unknown_bare_value_rejected(self):
        with pytest.raises(ConfigError, match="not recognised"):
            parse_distillation_model("gemini")

    def test_whitespace_around_value_stripped(self):
        assert parse_distillation_model(" claude-code:haiku ") == ("claude", "haiku")

    def test_case_sensitive_provider_prefix(self):
        with pytest.raises(ConfigError, match="unknown provider"):
            parse_distillation_model("Ollama:qwen3:8b")


class TestDistillationConfigPostInit:
    def test_default_resolves_to_ollama(self):
        cfg = DistillationConfig()
        assert cfg.resolved_provider == "ollama"
        assert cfg.resolved_model_id == "qwen3:8b"

    def test_legacy_haiku_resolves_to_claude(self):
        cfg = DistillationConfig(model="haiku")
        assert cfg.resolved_provider == "claude"
        assert cfg.resolved_model_id == "haiku"

    def test_invalid_model_raises_config_error(self):
        with pytest.raises(ConfigError):
            DistillationConfig(model="gemini")

    def test_resolved_fields_are_immutable(self):
        cfg = DistillationConfig(model="ollama:qwen3:8b")
        with pytest.raises(Exception):
            cfg.resolved_provider = "claude"  # frozen dataclass
