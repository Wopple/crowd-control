"""Tests for error handling and graceful degradation."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from crowd_control.config import ConfigError, CrowdControlConfig, EmbeddingConfig, load_config
from crowd_control.embed.base import EmbeddingError, create_embedder
from crowd_control.server import ServerDeps, handle_search_learnings, handle_status


class TestCreateEmbedderWrapsExceptions:
    """create_embedder wraps ImportError and ValueError into EmbeddingError."""

    def test_wraps_import_error(self):
        config = EmbeddingConfig(provider="ollama")
        # Setting sys.modules entry to None causes import to raise ImportError
        with patch.dict(
            "sys.modules",
            {"crowd_control.embed.ollama": None},
        ):
            with pytest.raises(EmbeddingError, match="not installed"):
                create_embedder(config)

    def test_wraps_value_error(self):
        config = EmbeddingConfig(provider="voyage")
        # Create a fake module whose VoyageEmbedder raises ValueError
        fake_module = ModuleType("crowd_control.embed.voyage")
        fake_module.VoyageEmbedder = MagicMock(
            side_effect=ValueError("Voyage API key not found")
        )
        with patch.dict("sys.modules", {"crowd_control.embed.voyage": fake_module}):
            with pytest.raises(EmbeddingError, match="API key not found"):
                create_embedder(config)


class TestLoadConfigInvalidToml:
    """load_config raises ConfigError with the file path on invalid TOML."""

    def test_invalid_toml_raises_config_error(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("this is not valid [ toml ===")

        with pytest.raises(ConfigError, match=str(config_file)):
            load_config(config_file)

    def test_config_error_message_contains_path(self, tmp_path):
        config_file = tmp_path / "bad.toml"
        config_file.write_text("[broken\nkey = ???")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)

        assert str(config_file) in str(exc_info.value)


class TestMcpLifespanBrokenEmbedder:
    """MCP server lifespan yields ServerDeps with embedder=None when embedder fails."""

    @pytest.mark.anyio
    async def test_lifespan_with_broken_embedder(self):
        from unittest.mock import AsyncMock

        from crowd_control.server import _default_lifespan

        mock_server = AsyncMock()

        with (
            patch(
                "crowd_control.server.load_config",
                return_value=CrowdControlConfig(),
            ),
            patch(
                "crowd_control.server.create_embedder",
                side_effect=EmbeddingError("provider down"),
            ),
            patch(
                "crowd_control.server.LearningStore",
                side_effect=ValueError("no dimensions"),
            ),
        ):
            async with _default_lifespan(mock_server) as deps:
                assert isinstance(deps, ServerDeps)
                assert deps.embedder is None


class TestStatusWithNoDeps:
    """status tool works gracefully when embedder and store are None."""

    @pytest.mark.anyio
    async def test_status_embedder_none(self):
        deps = ServerDeps(
            config=CrowdControlConfig(),
            store=None,
            embedder=None,
        )
        result = await handle_status(deps)
        assert isinstance(result, str)
        assert "unavailable" in result

    @pytest.mark.anyio
    async def test_status_does_not_crash(self):
        deps = ServerDeps(
            config=CrowdControlConfig(),
            store=None,
            embedder=None,
        )
        # Should return a string, never raise
        result = await handle_status(deps)
        assert "Learnings:" in result


class TestSearchLearningsNoEmbedder:
    """search_learnings returns an error string when embedder is None."""

    @pytest.mark.anyio
    async def test_returns_error_string(self):
        deps = ServerDeps(
            config=CrowdControlConfig(),
            store=None,
            embedder=None,
        )
        result = await handle_search_learnings(deps, "test query")
        assert isinstance(result, str)
        assert "not available" in result.lower() or "not running" in result.lower()

    @pytest.mark.anyio
    async def test_does_not_raise(self):
        deps = ServerDeps(
            config=CrowdControlConfig(),
            store=None,
            embedder=None,
        )
        # Must not raise, must return a string
        result = await handle_search_learnings(deps, "anything")
        assert isinstance(result, str)


class TestInteractiveCommandsExitOnBrokenConfig:
    """Interactive commands should exit 1 when config is broken."""

    def test_status_exits_nonzero_on_config_error(self):
        from crowd_control.cli import main

        runner = CliRunner()
        with patch("crowd_control.cli.load_config", side_effect=ConfigError("bad toml")):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 1
        assert "bad toml" in result.output


class TestHookSessionEndExitsZero:
    """hook session-end always exits 0, even on errors."""

    def test_exits_zero_with_empty_stdin(self):
        from crowd_control.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["hook", "session-end"], input="")
        assert result.exit_code == 0

    def test_exits_zero_with_invalid_json_stdin(self):
        from crowd_control.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["hook", "session-end"], input="not json{{{")
        assert result.exit_code == 0

    def test_exits_zero_with_no_stdin(self):
        from crowd_control.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["hook", "session-end"], input="{}")
        assert result.exit_code == 0

    def test_exits_zero_with_broken_config(self, tmp_path):
        """Hook must exit 0 even when config.toml is invalid."""
        from crowd_control.cli import main

        bad_config = tmp_path / "config.toml"
        bad_config.write_text("this is [[ not valid toml ===")

        runner = CliRunner()
        with patch("crowd_control.cli.load_config", side_effect=ConfigError("bad toml")):
            result = runner.invoke(main, ["hook", "session-end"], input="{}")
        assert result.exit_code == 0
