"""Tests for the setup command logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crowd_control.config import CrowdControlConfig
from crowd_control.setup import (
    configure_hooks,
    configure_mcp_server,
    ensure_default_config,
    ensure_storage_dir,
)


@pytest.fixture
def config(tmp_path):
    return CrowdControlConfig(storage_dir=str(tmp_path / "cc"))


class TestConfigureMcpServer:
    def test_creates_mcp_config(self, tmp_path):
        path = configure_mcp_server(tmp_path, project_scope=False)
        assert path == tmp_path / ".claude.json"
        data = json.loads(path.read_text())
        assert "crowd-control" in data["mcpServers"]
        assert data["mcpServers"]["crowd-control"]["command"] == "crowd-control"

    def test_preserves_existing_mcp_servers(self, tmp_path):
        mcp_path = tmp_path / ".claude.json"
        mcp_path.write_text(
            json.dumps({"mcpServers": {"other-server": {"command": "other", "args": ["run"]}}})
        )

        configure_mcp_server(tmp_path, project_scope=False)

        data = json.loads(mcp_path.read_text())
        assert "other-server" in data["mcpServers"]
        assert "crowd-control" in data["mcpServers"]

    def test_project_scope_writes_mcp_json(self, tmp_path):
        path = configure_mcp_server(tmp_path, project_scope=True)
        assert path == tmp_path / ".mcp.json"
        assert path.exists()


class TestConfigureHooks:
    def test_creates_hook_config(self, tmp_path):
        path = configure_hooks(tmp_path)
        assert path == tmp_path / ".claude" / "settings.json"
        data = json.loads(path.read_text())
        assert "SessionEnd" in data["hooks"]
        entries = data["hooks"]["SessionEnd"]
        assert len(entries) == 1
        assert entries[0]["hooks"][0]["command"] == "crowd-control hook session-end"

    def test_preserves_existing_hooks(self, tmp_path):
        hooks_path = tmp_path / ".claude" / "settings.json"
        hooks_path.parent.mkdir(parents=True)
        hooks_path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "Stop": [{"hooks": [{"type": "command", "command": "other-tool stop"}]}]
                    }
                }
            )
        )

        configure_hooks(tmp_path)

        data = json.loads(hooks_path.read_text())
        assert "Stop" in data["hooks"]
        assert "SessionEnd" in data["hooks"]

    def test_updates_existing_crowd_control_hooks(self, tmp_path):
        hooks_path = tmp_path / ".claude" / "settings.json"
        hooks_path.parent.mkdir(parents=True)
        hooks_path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [
                            {
                                "hooks": [
                                    {"type": "command", "command": "crowd-control hook old-command"}
                                ]
                            },
                            {"hooks": [{"type": "command", "command": "other-tool session-end"}]},
                        ]
                    }
                }
            )
        )

        configure_hooks(tmp_path)

        data = json.loads(hooks_path.read_text())
        entries = data["hooks"]["SessionEnd"]
        assert len(entries) == 2
        commands = [e["hooks"][0]["command"] for e in entries]
        assert "crowd-control hook session-end" in commands
        assert "other-tool session-end" in commands
        assert "crowd-control hook old-command" not in commands

    def test_project_scope_hooks(self, tmp_path):
        path = configure_hooks(tmp_path)
        assert path == tmp_path / ".claude" / "settings.json"
        assert path.exists()


class TestEnsureStorageDir:
    def test_creates_storage_dir(self, config):
        storage = Path(config.storage_dir)
        assert not storage.exists()
        ensure_storage_dir(config)
        assert storage.exists()


class TestEnsureDefaultConfig:
    def test_writes_default_config(self, config):
        ensure_storage_dir(config)
        ensure_default_config(config)
        config_path = Path(config.storage_dir) / "config.toml"
        assert config_path.exists()
        assert "ollama" in config_path.read_text()

    def test_preserves_existing_config(self, config):
        ensure_storage_dir(config)
        config_path = Path(config.storage_dir) / "config.toml"
        config_path.write_text("custom = true\n")
        ensure_default_config(config)
        assert config_path.read_text() == "custom = true\n"


class TestIdempotent:
    def test_setup_idempotent(self, tmp_path):
        config = CrowdControlConfig(storage_dir=str(tmp_path / "cc"))
        ensure_storage_dir(config)

        configure_mcp_server(tmp_path, project_scope=False)
        configure_hooks(tmp_path)

        # Run again
        configure_mcp_server(tmp_path, project_scope=False)
        configure_hooks(tmp_path)

        # Should still be valid, with single entries
        mcp_data = json.loads((tmp_path / ".claude.json").read_text())
        assert len(mcp_data["mcpServers"]) == 1

        hooks_data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert len(hooks_data["hooks"]["SessionEnd"]) == 1
