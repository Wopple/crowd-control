"""Tests for OllamaLLM."""

import pytest

ollama = pytest.importorskip("ollama")

from unittest.mock import MagicMock, patch  # noqa: E402

from crowd_control.ingest.llm.base import DistillationError  # noqa: E402
from crowd_control.ingest.llm.ollama import OllamaLLM  # noqa: E402


def _fake_response(content: str):
    r = MagicMock()
    r.message.content = content
    return r


class TestOllamaLLM:
    def test_generate_structured_happy_path(self):
        llm = OllamaLLM(model="qwen3:8b")
        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = _fake_response('{"learnings": []}')
            out = llm.generate_structured("prompt", {"type": "object"})
        assert out == {"learnings": []}

    def test_passes_schema_as_format(self):
        schema = {"type": "object", "required": ["learnings"]}
        llm = OllamaLLM(model="qwen3:8b")
        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = _fake_response('{"learnings": []}')
            llm.generate_structured("p", schema)
            kwargs = mock_chat.call_args.kwargs
        assert kwargs["format"] == schema
        assert kwargs["model"] == "qwen3:8b"
        assert kwargs["options"] == {"temperature": 0.0}
        assert kwargs["messages"] == [{"role": "user", "content": "p"}]
        assert kwargs["stream"] is False

    def test_model_not_pulled_maps_to_actionable_error(self):
        llm = OllamaLLM(model="qwen3:8b")
        err = ollama.ResponseError("model 'qwen3:8b' not found, try `ollama pull`")
        with patch.object(llm._client, "chat", side_effect=err):
            with pytest.raises(DistillationError, match="ollama pull qwen3:8b"):
                llm.generate_structured("p", {})

    def test_connection_error_maps_to_actionable_error(self):
        llm = OllamaLLM(model="qwen3:8b")
        with patch.object(llm._client, "chat", side_effect=ConnectionError("refused")):
            with pytest.raises(DistillationError, match="ollama serve"):
                llm.generate_structured("p", {})

    def test_empty_response_rejected(self):
        llm = OllamaLLM(model="qwen3:8b")
        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = _fake_response("")
            with pytest.raises(DistillationError, match="empty response"):
                llm.generate_structured("p", {})

    def test_non_json_content_rejected(self):
        llm = OllamaLLM(model="qwen3:8b")
        with patch.object(llm._client, "chat") as mock_chat:
            mock_chat.return_value = _fake_response("not json")
            with pytest.raises(DistillationError, match="non-JSON"):
                llm.generate_structured("p", {})

    def test_recommended_concurrency(self):
        assert OllamaLLM(model="qwen3:8b").recommended_concurrency == 1

    def test_provider_and_model_properties(self):
        llm = OllamaLLM(model="qwen3:8b")
        assert llm.provider_name == "ollama"
        assert llm.model_id == "qwen3:8b"

    def test_timeout_error_maps_to_actionable(self):
        import httpx
        llm = OllamaLLM(model="qwen3:8b", timeout_seconds=42.0)
        with patch.object(llm._client, "chat", side_effect=httpx.ReadTimeout("slow")):
            with pytest.raises(DistillationError, match="timed out after 42"):
                llm.generate_structured("p", {})
