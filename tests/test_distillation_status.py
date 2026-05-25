"""Tests for the distillation readiness check."""

from unittest.mock import patch

import pytest

from crowd_control.config import DistillationConfig
from crowd_control.ingest.llm.status import check_distillation_status


class TestCheckDistillationStatus:
    def test_ollama_ready_when_model_pulled(self):
        pytest.importorskip("ollama")
        cfg = DistillationConfig(model="ollama:qwen3:8b")
        with patch("ollama.Client") as mock_client_cls:
            client = mock_client_cls.return_value
            client.list.return_value = {"models": [{"name": "qwen3:8b"}]}
            ds = check_distillation_status(cfg)
        assert ds.provider == "ollama"
        assert ds.model == "qwen3:8b"
        assert ds.ready is True
        assert ds.hint is None

    def test_ollama_not_ready_when_model_missing(self):
        pytest.importorskip("ollama")
        cfg = DistillationConfig(model="ollama:qwen3:8b")
        with patch("ollama.Client") as mock_client_cls:
            client = mock_client_cls.return_value
            client.list.return_value = {"models": [{"name": "nomic-embed-text"}]}
            ds = check_distillation_status(cfg)
        assert ds.ready is False
        assert "ollama pull qwen3:8b" in ds.hint

    def test_ollama_matches_when_tag_includes_suffix(self):
        """A pulled model 'qwen3:8b:latest' should satisfy a request for 'qwen3:8b'."""
        pytest.importorskip("ollama")
        cfg = DistillationConfig(model="ollama:qwen3:8b")
        with patch("ollama.Client") as mock_client_cls:
            client = mock_client_cls.return_value
            client.list.return_value = {"models": [{"name": "qwen3:8b:latest"}]}
            ds = check_distillation_status(cfg)
        assert ds.ready is True

    def test_ollama_daemon_down(self):
        pytest.importorskip("ollama")
        cfg = DistillationConfig(model="ollama:qwen3:8b")
        with patch("ollama.Client") as mock_client_cls:
            client = mock_client_cls.return_value
            client.list.side_effect = ConnectionError("refused")
            ds = check_distillation_status(cfg)
        assert ds.ready is False
        assert "ollama serve" in ds.hint

    def test_claude_ready_when_on_path(self):
        cfg = DistillationConfig(model="claude-code:haiku")
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            ds = check_distillation_status(cfg)
        assert ds.provider == "claude-code"
        assert ds.ready is True
        assert ds.hint is None

    def test_claude_not_ready_when_missing(self):
        cfg = DistillationConfig(model="claude-code:haiku")
        with patch("shutil.which", return_value=None):
            ds = check_distillation_status(cfg)
        assert ds.ready is False
        assert "claude CLI not found" in ds.hint
