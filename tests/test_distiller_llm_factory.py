"""Tests for the create_distiller_llm factory."""

import builtins

import pytest

from crowd_control.config import DistillationConfig
from crowd_control.ingest.llm.base import (
    DistillationError,
    create_distiller_llm,
)
from crowd_control.ingest.llm.claude import ClaudeCLILLM


class TestCreateDistillerLLM:
    def test_routes_claude_code(self):
        cfg = DistillationConfig(model="claude-code:haiku")
        llm = create_distiller_llm(cfg)
        assert isinstance(llm, ClaudeCLILLM)
        assert llm.recommended_concurrency == 8

    def test_routes_legacy_bare_haiku_to_claude(self):
        cfg = DistillationConfig(model="haiku")
        llm = create_distiller_llm(cfg)
        assert isinstance(llm, ClaudeCLILLM)

    def test_routes_ollama(self):
        pytest.importorskip("ollama")
        from crowd_control.ingest.llm.ollama import OllamaLLM

        cfg = DistillationConfig(model="ollama:qwen3:8b")
        llm = create_distiller_llm(cfg)
        assert isinstance(llm, OllamaLLM)
        assert llm.recommended_concurrency == 1

    def test_ollama_missing_package_error(self, monkeypatch):
        cfg = DistillationConfig(model="ollama:qwen3:8b")

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "crowd_control.ingest.llm.ollama":
                raise ImportError("simulated missing ollama")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(DistillationError, match="ollama package not installed"):
            create_distiller_llm(cfg)
